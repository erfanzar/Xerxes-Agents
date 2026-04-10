// Copyright 2025 The EasyDeL/Calute Author @erfanzar (Erfan Zare Chavoshi).
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//! Inline viewport rendering matching Codex CLI visual design.
//!
/// Cell rendering patterns:
///   User:       › text              (bold+dim prefix, blended background)
///   Agent:      • text              (dim bullet, plain text)
///   ExecCell:   • Ran cmd           (green bullet, bold verb, cyan detail)
///               ✓                   (green) or ✗ N (red) exit status
///     output:   └ first line        (dim, tree connector)
///               continuation        (dim, 4-space indent)
///   Spinner:    • Working (Ns • esc to interrupt)
///   Header:     ╭──────╮ bordered box with model + directory

use ratatui::{
    layout::{Constraint, Direction, Layout, Rect},
    style::{Modifier, Style},
    text::{Line, Span},
    widgets::{Block, Borders, Paragraph, Wrap},
    Frame,
};

use crate::app::{App, Cell, Mode};
use crate::{diff, markdown, slash, theme};

/// Max lines to show for tool output before truncating.
const TOOL_OUTPUT_MAX_LINES: usize = 5;

pub fn draw(f: &mut Frame, app: &App) {
    let area = f.area();
    let width = area.width as usize;

    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Min(1),
            Constraint::Length(1),
            Constraint::Length(1),
        ])
        .split(area);

    draw_conversation(f, app, chunks[0], width);
    draw_status_bar(f, app, chunks[1]);
    draw_input(f, app, chunks[2]);
}


fn draw_conversation(f: &mut Frame, app: &App, area: Rect, width: usize) {
    let mut lines: Vec<Line<'static>> = Vec::new();

    for cell in &app.cells {
        render_cell(cell, &mut lines, width);
    }


    if !app.streaming_text.is_empty() {
        let md_lines = markdown::render(&app.streaming_text, width.saturating_sub(4));
        for (i, ml) in md_lines.iter().enumerate() {
            let prefix = if i == 0 { "• " } else { "  " };
            let mut spans = vec![Span::styled(prefix, theme::assistant_bullet())];
            spans.extend(ml.spans.clone());
            lines.push(Line::from(spans));
        }
    }


    if app.mode == Mode::Streaming {
        let label = if app.active_tool.is_some() {
            "Working"
        } else {
            "Thinking"
        };
        lines.push(Line::from(vec![]));
        lines.push(Line::from(vec![
            Span::styled(format!("{} ", app.spinner.frame()), theme::dim()),
            Span::styled(label, theme::spinner()),
            Span::styled(
                format!(" ({} • esc to interrupt)", app.spinner.elapsed_str()),
                theme::dim(),
            ),
        ]));
    }


    if app.mode == Mode::Permission {
        lines.push(Line::from(vec![]));
        lines.push(Line::from(vec![
            Span::styled("• ", theme::warning()),
            Span::styled("Allow? ", theme::warning()),
            Span::styled(app.perm_description.clone(), theme::text()),
        ]));
        lines.push(Line::from(vec![
            Span::raw("  "),
            Span::styled("[y]", Style::default().fg(theme::GREEN).add_modifier(Modifier::BOLD)),
            Span::styled(" Allow  ", theme::dim()),
            Span::styled("[n]", Style::default().fg(theme::RED).add_modifier(Modifier::BOLD)),
            Span::styled(" Deny", theme::dim()),
        ]));
    }


    use crate::app::ProviderStep;
    match &app.mode {
        Mode::ProviderSetup(ProviderStep::SelectProfile) => {


            lines.push(Line::from(Span::styled(
                "  Select a provider profile:",
                theme::dim(),
            )));
            for (i, p) in app.provider_profiles.iter().enumerate() {
                let name = p.get("name").and_then(|v| v.as_str()).unwrap_or("?");
                let model = p.get("model").and_then(|v| v.as_str()).unwrap_or("?");
                let active = p.get("active").and_then(|v| v.as_bool()).unwrap_or(false);
                let marker = if active { " (active)" } else { "" };
                let (prefix, style) = if i == app.provider_selected {
                    ("  › ", theme::tool_verb())
                } else {
                    ("    ", theme::dim())
                };
                lines.push(Line::from(Span::styled(
                    format!("{prefix}{name} — {model}{marker}"),
                    style,
                )));
            }

            let new_idx = app.provider_profiles.len();
            let (prefix, style) = if app.provider_selected == new_idx {
                ("  › ", theme::tool_verb())
            } else {
                ("    ", theme::dim())
            };
            lines.push(Line::from(Span::styled(
                format!("{prefix}+ New profile"),
                style,
            )));
            lines.push(Line::from(Span::styled(
                "  Enter to confirm · Esc to cancel",
                theme::dim(),
            )));
        }
        Mode::ProviderSetup(ProviderStep::SelectModel) => {
            lines.push(Line::from(Span::styled(
                format!("  Select model ({} available):", app.provider_models.len()),
                theme::dim(),
            )));

            let total = app.provider_models.len();
            let window = 15usize;
            let start = if app.provider_selected > window / 2 {
                (app.provider_selected - window / 2).min(total.saturating_sub(window))
            } else {
                0
            };
            let end = (start + window).min(total);

            if start > 0 {
                lines.push(Line::from(Span::styled(
                    format!("    ↑ {} more", start),
                    theme::dim(),
                )));
            }
            for (i, m) in app.provider_models.iter().enumerate().skip(start).take(end - start) {
                let (prefix, style) = if i == app.provider_selected {
                    ("  › ", theme::tool_verb())
                } else {
                    ("    ", theme::dim())
                };
                lines.push(Line::from(Span::styled(
                    format!("{prefix}{m}"),
                    style,
                )));
            }
            if end < total {
                lines.push(Line::from(Span::styled(
                    format!("    ↓ {} more", total - end),
                    theme::dim(),
                )));
            }

            let custom_idx = total;
            let (prefix, style) = if app.provider_selected == custom_idx {
                ("  › ", theme::tool_verb())
            } else {
                ("    ", theme::dim())
            };
            lines.push(Line::from(Span::styled(
                format!("{prefix}Custom model name..."),
                style,
            )));
            lines.push(Line::from(Span::styled(
                "  Enter to confirm · Esc to cancel",
                theme::dim(),
            )));
        }
        Mode::ProviderSetup(ProviderStep::EnterName)
        | Mode::ProviderSetup(ProviderStep::EnterBaseUrl)
        | Mode::ProviderSetup(ProviderStep::EnterApiKey)
        | Mode::ProviderSetup(ProviderStep::EnterModel) => {

            lines.push(Line::from(vec![
                Span::styled("  › ", theme::input_prompt()),
                Span::styled(app.input.clone(), theme::text()),
            ]));
        }
        _ => {}
    }


    let w = area.width.max(1) as usize;
    let wrapped = prewrap_lines(lines, w);

    let visible = area.height as usize;
    let total = wrapped.len();
    let scroll = total.saturating_sub(visible) as u16;


    let paragraph = Paragraph::new(wrapped).scroll((scroll, 0));
    f.render_widget(paragraph, area);
}


fn render_cell(cell: &Cell, lines: &mut Vec<Line<'static>>, width: usize) {
    match cell {


        Cell::User { text } => {
            lines.push(Line::from(vec![]));

            let content = format!("› {text}");
            let pad_len = width.saturating_sub(content.len());
            let pad = " ".repeat(pad_len);
            lines.push(Line::from(vec![
                Span::styled(content, theme::user_prefix().bg(theme::USER_BG)),
                Span::styled(pad, theme::user_bg()),
            ]));
            lines.push(Line::from(vec![]));
        }


        Cell::Assistant { markdown: md } => {
            let md_lines = markdown::render(md, width.saturating_sub(4));
            for (i, ml) in md_lines.iter().enumerate() {
                let prefix = if i == 0 { "• " } else { "  " };
                let mut spans = vec![Span::styled(prefix, theme::assistant_bullet())];
                spans.extend(ml.spans.clone());
                lines.push(Line::from(spans));
            }
        }


        Cell::Thinking { text } => {
            let preview: String = text.chars().take(80).collect();
            let suffix = if text.len() > 80 { "…" } else { "" };
            lines.push(Line::from(vec![
                Span::styled("• ", theme::dim()),
                Span::styled(format!("{preview}{suffix}"), theme::italic()),
            ]));
        }


        Cell::ToolCall {
            verb,
            detail,
            result,
            collapsed,
            ..
        } => {
            if let Some(res) = result {

                let (bullet, status_suffix) = if !res.permitted {
                    (
                        Span::styled("• ", theme::tool_failure()),
                        vec![Span::styled(" ✗ denied", theme::tool_failure())],
                    )
                } else if res.output.starts_with("Error") {
                    (
                        Span::styled("• ", theme::tool_failure()),
                        vec![Span::styled(" ✗", theme::tool_failure())],
                    )
                } else {
                    (
                        Span::styled("• ", theme::tool_success()),
                        vec![Span::styled(" ✓", theme::tool_success())],
                    )
                };

                let mut header = vec![bullet, Span::styled(verb.clone(), theme::tool_verb())];
                if !detail.is_empty() {
                    header.push(Span::raw(" "));
                    header.push(Span::styled(detail.clone(), theme::tool_detail()));
                }
                header.extend(status_suffix);
                lines.push(Line::from(header));


                if res.permitted {
                    if res.is_diff && !collapsed {

                        let diff_lines = diff::render(&res.output, 25);
                        for (i, dl) in diff_lines.iter().enumerate() {
                            let prefix = if i == 0 { "  └ " } else { "    " };
                            let mut spans = vec![Span::styled(prefix, theme::dim())];
                            spans.extend(dl.spans.clone());
                            lines.push(Line::from(spans));
                        }
                    } else if res.output.starts_with("Error") {
                        let preview: String = res.output.lines().next().unwrap_or("").chars().take(80).collect();
                        lines.push(Line::from(vec![
                            Span::styled("  └ ", theme::dim()),
                            Span::styled(preview, theme::error()),
                        ]));
                    } else if res.output.trim().is_empty() {
                        lines.push(Line::from(vec![
                            Span::styled("  └ ", theme::dim()),
                            Span::styled("(no output)", theme::dim()),
                        ]));
                    } else {
                        let output_lines: Vec<&str> = res.output.lines().collect();
                        let show_count = output_lines.len().min(TOOL_OUTPUT_MAX_LINES);
                        let omitted = output_lines.len().saturating_sub(TOOL_OUTPUT_MAX_LINES);

                        for (i, oline) in output_lines.iter().take(show_count).enumerate() {
                            let prefix = if i == 0 { "  └ " } else { "    " };
                            let short: String = oline.chars().take(width.saturating_sub(6)).collect();
                            lines.push(Line::from(vec![
                                Span::styled(prefix, theme::dim()),
                                Span::styled(short, theme::tool_result()),
                            ]));
                        }
                        if omitted > 0 {
                            lines.push(Line::from(vec![
                                Span::styled("    ", theme::dim()),
                                Span::styled(format!("… +{omitted} lines"), theme::dim()),
                            ]));
                        }
                    }
                }
            } else {

                let mut header = vec![
                    Span::styled("• ", theme::dim()),
                    Span::styled(verb.clone(), theme::tool_verb()),
                ];
                if !detail.is_empty() {
                    header.push(Span::raw(" "));
                    header.push(Span::styled(detail.clone(), theme::tool_detail()));
                }
                lines.push(Line::from(header));
            }
        }


        Cell::Error { message } => {
            for sline in message.lines() {
                lines.push(Line::from(vec![
                    Span::styled("• ", theme::tool_failure()),
                    Span::styled(sline.to_string(), theme::error()),
                ]));
            }
        }


        Cell::System { message } => {
            for sline in message.lines() {
                lines.push(Line::from(Span::styled(
                    format!("  {sline}"),
                    theme::dim(),
                )));
            }
        }

        Cell::AgentActivity {
            agent_name,
            agent_type,
            status,
            tool_calls,
            text_preview,
        } => {
            let (bullet_style, status_icon) = match status.as_str() {
                "completed" => (theme::tool_success(), "✓"),
                "failed" => (theme::tool_failure(), "✗"),
                _ => (theme::dim(), "◦"),
            };

            let type_label = if agent_type.is_empty() {
                String::new()
            } else {
                format!(" [{}]", agent_type)
            };

            lines.push(Line::from(vec![
                Span::styled("  ↳ ", theme::dim()),
                Span::styled(status_icon, bullet_style),
                Span::styled(format!(" {agent_name}"), theme::tool_verb()),
                Span::styled(type_label, theme::dim()),
            ]));

            if !tool_calls.is_empty() {
                let count = tool_calls.len();
                let recent: String = tool_calls
                    .iter()
                    .rev()
                    .take(3)
                    .map(|s| s.as_str())
                    .collect::<Vec<_>>()
                    .join(", ");
                lines.push(Line::from(vec![
                    Span::styled("      ", theme::dim()),
                    Span::styled(format!("{count} tool calls"), theme::dim()),
                    Span::styled(format!(" ({recent})"), theme::tool_detail()),
                ]));
            }

            if !text_preview.is_empty() {
                let preview: String = text_preview.chars().take(60).collect();
                lines.push(Line::from(vec![
                    Span::styled("      ", theme::dim()),
                    Span::styled(preview, theme::tool_result()),
                ]));
            }
        }

        Cell::Handoff { from, to, reason } => {
            lines.push(Line::from(vec![]));
            lines.push(Line::from(vec![
                Span::styled("  → ", Style::default().fg(theme::CYAN).add_modifier(Modifier::BOLD)),
                Span::styled("Handoff ", Style::default().fg(theme::CYAN)),
                Span::styled(format!("{from} → {to}"), theme::tool_verb()),
            ]));
            if !reason.is_empty() {
                lines.push(Line::from(vec![
                    Span::styled("    ", theme::dim()),
                    Span::styled(reason.clone(), theme::dim()),
                ]));
            }
        }

        Cell::PlanHeader { objective, steps } => {
            lines.push(Line::from(vec![]));
            lines.push(Line::from(vec![
                Span::styled("  ◈ ", Style::default().fg(theme::CYAN).add_modifier(Modifier::BOLD)),
                Span::styled("Plan: ", Style::default().fg(theme::CYAN).add_modifier(Modifier::BOLD)),
                Span::styled(objective.clone(), theme::text()),
            ]));
            for (id, agent, desc) in steps {
                let short_desc: String = desc.chars().take(50).collect();
                lines.push(Line::from(vec![
                    Span::styled(format!("    {id}. "), theme::dim()),
                    Span::styled(format!("[{agent}] "), theme::tool_detail()),
                    Span::styled(short_desc, theme::dim()),
                ]));
            }
            lines.push(Line::from(vec![]));
        }

        Cell::PlanStep {
            step_id,
            agent,
            description,
            status,
        } => {
            let (icon, style) = match status.as_str() {
                "completed" => ("✓", theme::tool_success()),
                "failed" => ("✗", theme::tool_failure()),
                _ => ("◦", theme::dim()),
            };

            let short_desc: String = description.chars().take(50).collect();
            lines.push(Line::from(vec![
                Span::styled(format!("    {icon} Step {step_id} "), style),
                Span::styled(format!("[{agent}] "), theme::tool_detail()),
                Span::styled(short_desc, theme::dim()),
            ]));
        }
    }
}


fn draw_status_bar(f: &mut Frame, app: &App, area: Rect) {
    let model = if app.model.is_empty() {
        "(no model)"
    } else {
        &app.model
    };

    let tokens = format!(
        "{}in/{}out",
        fmt_tokens(app.total_in),
        fmt_tokens(app.total_out)
    );

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

    let mut parts = vec![
        Span::styled(format!(" {model}"), theme::status()),
    ];

    if app.cost_usd > 0.0 {
        parts.push(Span::styled(" · ", theme::dim()));
        parts.push(Span::styled(format!("${:.4}", app.cost_usd), theme::success()));
    }

    parts.push(Span::styled(" · ", theme::dim()));
    parts.push(Span::styled(tokens, theme::status()));
    parts.push(Span::styled(" · ", theme::dim()));
    parts.push(Span::styled(cwd, theme::status()));

    f.render_widget(Paragraph::new(Line::from(parts)), area);
}


fn draw_input(f: &mut Frame, app: &App, area: Rect) {
    let prompt = if app.mode == Mode::Input { "› " } else { "  " };
    let prompt_style = if app.mode == Mode::Input {
        theme::input_prompt()
    } else {
        theme::dim()
    };

    let input_line = Line::from(vec![
        Span::styled(prompt, prompt_style),
        Span::styled(app.input.clone(), theme::text()),
    ]);
    f.render_widget(Paragraph::new(input_line), area);


    if app.slash_popup && app.mode == Mode::Input {
        let matches = slash::filter(&app.slash_filter);
        if !matches.is_empty() {
            let max_visible: usize = 10;
            let visible_count = matches.len().min(max_visible);
            let popup_height = (visible_count + 2) as u16;

            let scroll_offset = if app.slash_selected >= visible_count {
                app.slash_selected - visible_count + 1
            } else {
                0
            };

            let popup_area = Rect {
                x: area.x + 2,
                y: area.y.saturating_sub(popup_height),
                width: area.width.min(55),
                height: popup_height,
            };

            let popup_lines: Vec<Line<'static>> = matches
                .iter()
                .enumerate()
                .skip(scroll_offset)
                .take(visible_count)
                .map(|(i, &cmd_idx)| {
                    let cmd = &slash::COMMANDS[cmd_idx];
                    let (name_style, desc_style) = if i == app.slash_selected {
                        (
                            Style::default().fg(theme::CYAN).add_modifier(Modifier::BOLD),
                            theme::dim(),
                        )
                    } else {
                        (theme::dim(), theme::dim())
                    };
                    Line::from(vec![
                        Span::styled(format!(" {:<16}", cmd.name), name_style),
                        Span::styled(cmd.description, desc_style),
                    ])
                })
                .collect();

            let popup = Paragraph::new(popup_lines).block(
                Block::default()
                    .borders(Borders::ALL)
                    .border_style(theme::border()),
            );
            f.render_widget(popup, popup_area);
        }
    }


    if app.mode == Mode::Input {
        f.set_cursor_position((area.x + 2 + app.cursor as u16, area.y));
    }
}


pub fn welcome_lines(model: &str, provider: &str, _tools: usize, cwd: &str) -> Vec<Line<'static>> {
    let version = "v0.2.0";
    let title = format!(">_ Calute ({version})");
    let model_line = if model.is_empty() {
        "model:     (none)   /provider to configure".to_string()
    } else {
        format!("model:     {model} ({provider})   /model to change")
    };
    let dir_line = format!("directory: {cwd}");

    let inner_width = 56usize;
    let top = format!("╭{}╮", "─".repeat(inner_width));
    let bot = format!("╰{}╯", "─".repeat(inner_width));

    let pad = |s: &str| -> String {
        let w = unicode_display_width(s);
        let padding = inner_width.saturating_sub(w);
        format!("│{s}{}│", " ".repeat(padding))
    };

    let blank = pad(&" ".repeat(inner_width.min(56)));

    vec![
        Line::from(Span::styled(top, theme::dim())),
        Line::from(Span::styled(pad(&format!(" {title}")), theme::dim())),
        Line::from(Span::styled(blank.clone(), theme::dim())),
        Line::from(Span::styled(pad(&format!(" {model_line}")), theme::dim())),
        Line::from(Span::styled(pad(&format!(" {dir_line}")), theme::dim())),
        Line::from(Span::styled(bot, theme::dim())),
        Line::from(vec![]),
    ]
}

fn unicode_display_width(s: &str) -> usize {
    unicode_width::UnicodeWidthStr::width(s)
}

/// Break lines that exceed `max_width` into multiple lines.
/// Each output Line fits in exactly one terminal row.
fn prewrap_lines(lines: Vec<Line<'static>>, max_width: usize) -> Vec<Line<'static>> {
    let mut out: Vec<Line<'static>> = Vec::new();
    for line in lines {
        let line_w: usize = line
            .spans
            .iter()
            .map(|s| unicode_display_width(s.content.as_ref()))
            .sum();
        if line_w <= max_width {
            out.push(line);
            continue;
        }


        let style = line.spans.first().map(|s| s.style).unwrap_or_default();
        let full: String = line.spans.iter().map(|s| s.content.as_ref()).collect();
        let mut pos = 0;
        let chars: Vec<char> = full.chars().collect();
        while pos < chars.len() {
            let mut end = pos;
            let mut w = 0usize;
            while end < chars.len() {
                let cw = unicode_width::UnicodeWidthChar::width(chars[end]).unwrap_or(1);
                if w + cw > max_width {
                    break;
                }
                w += cw;
                end += 1;
            }
            if end == pos {

                end = pos + 1;
            }
            let chunk: String = chars[pos..end].iter().collect();
            out.push(Line::from(Span::styled(chunk, style)));
            pos = end;
        }
    }
    out
}

fn fmt_tokens(n: u64) -> String {
    if n >= 1_000_000 {
        format!("{:.1}M", n as f64 / 1_000_000.0)
    } else if n >= 1_000 {
        format!("{:.1}k", n as f64 / 1_000.0)
    } else {
        n.to_string()
    }
}
