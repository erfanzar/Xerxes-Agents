mod app;
mod bridge;
mod diff;
mod events;
mod input;
mod markdown;
mod render;
mod slash;
mod spinner;
mod theme;

use std::env;
use std::io::{self, Write};
use std::time::Duration;

use clap::Parser;
use crossterm::{
    cursor,
    event::{self, DisableBracketedPaste, EnableBracketedPaste, Event as CEvent},
    execute,
    terminal::{disable_raw_mode, enable_raw_mode},
};
use ratatui::{backend::CrosstermBackend, Terminal, TerminalOptions, Viewport};

use app::{App, Cell, Mode, ToolResult};
use bridge::Bridge;
use events::{Event, Request};
use input::{handle_key, Action};

#[derive(Parser)]
#[command(name = "calute", about = "Calute — agent CLI")]
struct Cli {
    /// Model name (e.g. gpt-4o, claude-sonnet-4-6)
    #[arg(short, long)]
    model: Option<String>,

    /// API base URL
    #[arg(long)]
    base_url: Option<String>,

    /// API key for the provider
    #[arg(long)]
    api_key: Option<String>,

    /// Python executable
    #[arg(long, default_value = "python")]
    python: String,

    /// Project directory
    #[arg(long)]
    project_dir: Option<String>,

    /// Permission mode: auto, accept-all, manual
    #[arg(long, default_value = "auto")]
    permission_mode: String,

    /// Non-interactive: run a single prompt and exit
    #[arg(short, long)]
    print: Option<String>,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();

    let project_dir = cli
        .project_dir
        .unwrap_or_else(|| env::current_dir().unwrap().to_string_lossy().to_string());


    let model = cli.model.unwrap_or_default();

    let (mut bridge, mut event_rx) = Bridge::spawn(&cli.python, &project_dir).await?;

    bridge
        .send(Request::init(
            &model,
            &cli.permission_mode,
            cli.base_url.as_deref(),
            cli.api_key.as_deref(),
        ))
        .await?;


    if let Some(prompt) = cli.print {
        return run_noninteractive(&mut bridge, &mut event_rx, &prompt).await;
    }


    run_interactive(&mut bridge, &mut event_rx).await
}


async fn run_noninteractive(
    bridge: &mut Bridge,
    event_rx: &mut tokio::sync::mpsc::UnboundedReceiver<Event>,
    prompt: &str,
) -> anyhow::Result<()> {

    while let Some(event) = event_rx.recv().await {
        if matches!(event, Event::Ready { .. }) {
            break;
        }
    }

    bridge.send(Request::query(prompt)).await?;

    while let Some(event) = event_rx.recv().await {
        match event {
            Event::TextChunk { text } => {
                print!("{text}");
                io::stdout().flush()?;
            }
            Event::QueryDone => break,
            Event::Error { message } => eprintln!("Error: {message}"),
            _ => {}
        }
    }
    println!();
    Ok(())
}


async fn run_interactive(
    bridge: &mut Bridge,
    event_rx: &mut tokio::sync::mpsc::UnboundedReceiver<Event>,
) -> anyhow::Result<()> {

    enable_raw_mode()?;
    let mut stderr = io::stderr();
    execute!(stderr, EnableBracketedPaste, cursor::Hide)?;

    let backend = CrosstermBackend::new(stderr);
    let viewport_height = crossterm::terminal::size()
        .map(|(_, h)| h.saturating_sub(2).max(6))
        .unwrap_or(20);
    let mut terminal = Terminal::with_options(
        backend,
        TerminalOptions {
            viewport: Viewport::Inline(viewport_height),
        },
    )?;

    let mut app = App::new();


    app.cells.push(Cell::System {
        message: "Connecting...".into(),
    });

    let result = event_loop(&mut terminal, &mut app, bridge, event_rx).await;


    disable_raw_mode()?;
    execute!(
        terminal.backend_mut(),
        DisableBracketedPaste,
        cursor::Show,
    )?;


    terminal.insert_before(1, |buf| {
        buf.set_string(0, 0, "", ratatui::style::Style::default());
    })?;

    result
}

async fn event_loop(
    terminal: &mut Terminal<CrosstermBackend<io::Stderr>>,
    app: &mut App,
    bridge: &mut Bridge,
    event_rx: &mut tokio::sync::mpsc::UnboundedReceiver<Event>,
) -> anyhow::Result<()> {
    loop {
        terminal.draw(|f| render::draw(f, app))?;

        tokio::select! {
            _ = tokio::task::yield_now() => {

                let poll_ms = if app.mode == Mode::Streaming { 50 } else { 100 };
                if event::poll(Duration::from_millis(poll_ms))? {
                    match event::read()? {
                    CEvent::Paste(text) => {
                        input::handle_paste(app, &text);
                    }
                    CEvent::Key(key) => {
                        match handle_key(app, key) {
                            Action::None => {}
                            Action::Submit(text) => {
                                app.cells.push(Cell::User { text: text.clone() });
                                app.mode = Mode::Streaming;
                                app.spinner.reset();
                                bridge.send(Request::query(&text)).await?;
                            }
                            Action::Slash(text) => {
                                if text.starts_with("/provider") {

                                    app.cells.push(Cell::System {
                                        message: "Loading provider profiles...".into(),
                                    });
                                    app.provider_selected = 0;
                                    bridge.send(Request::provider_list()).await?;
                                } else {
                                    app.cells.push(Cell::System {
                                        message: text.clone(),
                                    });
                                    bridge.send(Request::slash(&text)).await?;
                                }
                            }
                            Action::PermissionResponse(granted) => {
                                app.cells.push(Cell::System {
                                    message: if granted { "Allowed".into() } else { "Denied".into() },
                                });
                                app.mode = Mode::Streaming;
                                bridge.send(Request::permission_response(granted)).await?;
                            }
                            Action::Cancel => {
                                bridge.send(Request::cancel()).await?;
                                app.flush_streaming();
                                app.active_tool = None;
                                app.cells.push(Cell::System {
                                    message: "(cancelled)".into(),
                                });
                                app.mode = Mode::Input;
                            }
                            Action::ProviderList => {
                                bridge.send(Request::provider_list()).await?;
                            }
                            Action::ProviderSelect(name) => {
                                bridge.send(Request::provider_select(&name)).await?;
                            }
                            Action::ProviderFetchModels(url, key) => {
                                bridge.send(Request::fetch_models(&url, &key)).await?;
                            }
                            Action::ProviderSave(name, url, key, model) => {
                                bridge.send(Request::provider_save(&name, &url, &key, &model)).await?;
                            }
                            Action::ProviderStepInput(_) => {

                            }
                            Action::Quit => {
                                app.should_quit = true;
                            }
                        }
                    }
                    _ => {}
                    }
                }
            }

            Some(event) = event_rx.recv() => {
                handle_bridge_event(app, event);
            }
        }

        if app.should_quit {
            break;
        }
    }
    Ok(())
}

fn handle_bridge_event(app: &mut App, event: Event) {
    match event {
        Event::Ready {
            model,
            provider,
            tools,
            permission_mode,
            has_profile,
        } => {
            app.model = model.clone();
            app.provider = provider.clone();
            app.tool_count = tools;
            app.permission_mode = permission_mode;
            app.has_profile = has_profile;

            app.cells.clear();
            let cwd = std::env::current_dir()
                .map(|p| {
                    let home = std::env::var("HOME").unwrap_or_default();
                    let s = p.to_string_lossy().to_string();
                    if !home.is_empty() && s.starts_with(&home) {
                        format!("~{}", &s[home.len()..])
                    } else {
                        s
                    }
                })
                .unwrap_or_default();

            let header = render::welcome_lines(&model, &provider, tools, &cwd);
            let header_text = header
                .iter()
                .map(|l| {
                    l.spans.iter().map(|s| s.content.as_ref()).collect::<String>()
                })
                .collect::<Vec<_>>()
                .join("\n");
            app.cells.push(Cell::System { message: header_text });

            if !has_profile {
                app.cells.push(Cell::Error {
                    message: "No provider configured. Run /provider to set up a provider profile.".into(),
                });
            }
        }

        Event::TextChunk { text } => {
            app.streaming_text.push_str(&text);
        }

        Event::ThinkingChunk { text } => {
            app.streaming_thinking.push_str(&text);
        }

        Event::ToolStart { name, inputs, .. } => {

            app.flush_streaming();

            let (verb, detail) = App::tool_verb_detail(&name, &inputs);
            app.active_tool = Some(if detail.is_empty() {
                verb.clone()
            } else {
                format!("{verb} {detail}")
            });
            app.spinner.reset();
            app.cells.push(Cell::ToolCall {
                name: name.clone(),
                verb,
                detail,
                result: None,
                collapsed: true,
            });
        }

        Event::ToolEnd {
            name,
            result,
            permitted,
            duration_ms,
            ..
        } => {
            app.active_tool = None;
            let is_diff_result =
                matches!(name.as_str(), "Edit" | "Write" | "FileEditTool" | "WriteFile")
                    && diff::is_diff(&result);


            if let Some(Cell::ToolCall {
                result: ref mut r,
                collapsed,
                ..
            }) = app.cells.last_mut()
            {
                *r = Some(ToolResult {
                    output: result,
                    permitted,
                    duration_ms,
                    is_diff: is_diff_result,
                });

                if is_diff_result {
                    *collapsed = false;
                }
            }
        }

        Event::PermissionRequest {
            description,
            inputs,
            ..
        } => {
            app.perm_description = description;
            app.perm_inputs = inputs;
            app.mode = Mode::Permission;
        }

        Event::TurnDone {
            input_tokens,
            output_tokens,
            ..
        } => {
            app.total_in += input_tokens;
            app.total_out += output_tokens;
            app.turn_count += 1;
        }

        Event::QueryDone => {
            app.flush_streaming();
            app.active_tool = None;
            app.mode = Mode::Input;
        }

        Event::SlashResult { output } => {
            app.cells.push(Cell::System { message: output });
        }

        Event::State {
            turn_count,
            total_input_tokens,
            total_output_tokens,
            cost_usd,
            ..
        } => {
            app.turn_count = turn_count;
            app.total_in = total_input_tokens;
            app.total_out = total_output_tokens;
            app.cost_usd = cost_usd;
        }

        Event::Error { message } => {
            app.cells.push(Cell::Error { message });
            if app.mode == Mode::Streaming {
                app.flush_streaming();
                app.mode = Mode::Input;
            }
        }

        Event::Exit => {
            app.should_quit = true;
        }

        Event::ProviderList { profiles } => {
            app.provider_profiles = profiles.clone();
            app.provider_selected = 0;

            let mut lines = vec!["Select a provider profile (Up/Down + Enter):".to_string()];
            for (i, p) in profiles.iter().enumerate() {
                let name = p.get("name").and_then(|v| v.as_str()).unwrap_or("?");
                let model = p.get("model").and_then(|v| v.as_str()).unwrap_or("?");
                let active = p.get("active").and_then(|v| v.as_bool()).unwrap_or(false);
                let marker = if active { " (active)" } else { "" };
                lines.push(format!("  {}: {} — {}{}", i + 1, name, model, marker));
            }
            lines.push(format!("  {}: + New profile", profiles.len() + 1));
            app.cells.push(Cell::System {
                message: lines.join("\n"),
            });
            app.mode = Mode::ProviderSetup(app::ProviderStep::SelectProfile);
        }

        Event::ModelsList { models, base_url } => {
            app.provider_models = models.clone();
            app.provider_selected = 0;
            if models.is_empty() {
                app.cells.push(Cell::System {
                    message: format!(
                        "No models found at {base_url}/models.\nEnter model name manually:"
                    ),
                });

                app.mode = Mode::Input;


                app.mode = Mode::ProviderSetup(app::ProviderStep::SelectModel);
            } else {
                let mut lines = vec![format!(
                    "Found {} models. Select one (Up/Down + Enter):",
                    models.len()
                )];
                for (i, m) in models.iter().enumerate().take(20) {
                    lines.push(format!("  {}: {}", i + 1, m));
                }
                if models.len() > 20 {
                    lines.push(format!("  ... and {} more", models.len() - 20));
                }
                lines.push(format!("  {}: Custom model name", models.len() + 1));
                app.cells.push(Cell::System {
                    message: lines.join("\n"),
                });
                app.mode = Mode::ProviderSetup(app::ProviderStep::SelectModel);
            }
        }

        Event::ProviderSaved { message, model, provider } => {
            if !model.is_empty() {
                app.model = model;
            }
            if !provider.is_empty() {
                app.provider = provider;
            }
            app.has_profile = true;
            app.cells.push(Cell::System { message });
            app.mode = Mode::Input;
        }

        Event::Unknown(_) => {}
    }
}
