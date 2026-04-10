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

//! Keyboard input handling.
//!
//! Maps crossterm key events to [`Action`] variants for each UI mode
//! (input, streaming, permission, provider setup). Supports emacs-style
//! editing, slash command popup navigation, and paste detection.
use crossterm::event::{KeyCode, KeyEvent, KeyModifiers};

use crate::app::{App, Mode};
use crate::slash;

pub enum Action {
    None,
    Submit(String),
    Slash(String),
    PermissionResponse(bool),
    Cancel,
    Quit,
    /// Provider setup actions.
    ProviderList,
    ProviderSelect(String),
    ProviderFetchModels(String, String),
    ProviderSave(String, String, String, String),
    ProviderStepInput(String),
}

pub fn handle_key(app: &mut App, key: KeyEvent) -> Action {
    match app.mode {
        Mode::Input => handle_input(app, key),
        Mode::Streaming => handle_streaming(key),
        Mode::Permission => handle_permission(key),
        Mode::ProviderSetup(_) => handle_provider_setup(app, key),
    }
}

fn handle_input(app: &mut App, key: KeyEvent) -> Action {

    if app.slash_popup {
        match (key.modifiers, key.code) {
            (_, KeyCode::Esc) => {
                app.slash_popup = false;
                return Action::None;
            }
            (_, KeyCode::Up) => {
                app.slash_selected = app.slash_selected.saturating_sub(1);
                return Action::None;
            }
            (_, KeyCode::Down) => {
                let matches = slash::filter(&app.slash_filter);
                if !matches.is_empty() {
                    app.slash_selected = (app.slash_selected + 1).min(matches.len() - 1);
                }
                return Action::None;
            }
            (_, KeyCode::Tab) => {

                let matches = slash::filter(&app.slash_filter);
                if let Some(&idx) = matches.get(app.slash_selected) {
                    app.input = slash::COMMANDS[idx].name.to_string();
                    app.cursor = app.input.len();
                    app.slash_popup = false;
                }
                return Action::None;
            }
            _ => {}
        }
    }

    match (key.modifiers, key.code) {

        (KeyModifiers::CONTROL, KeyCode::Char('c')) => Action::Quit,
        (KeyModifiers::CONTROL, KeyCode::Char('d')) => {
            if app.input.is_empty() {
                Action::Quit
            } else {
                app.delete();
                Action::None
            }
        }


        (_, KeyCode::Enter) => {

            if app.slash_popup {
                let matches = slash::filter(&app.slash_filter);
                if let Some(&idx) = matches.get(app.slash_selected) {
                    let cmd = slash::COMMANDS[idx].name.to_string();

                    if matches!(cmd.as_str(), "/model") {
                        app.input = format!("{cmd} ");
                        app.cursor = app.input.len();
                        app.slash_popup = false;
                        return Action::None;
                    }
                    app.input = cmd;
                    app.cursor = app.input.len();
                    app.slash_popup = false;
                }
            }

            let text = app.take_input();
            if text.is_empty() {
                return Action::None;
            }
            if text.starts_with('/') {
                match text.split_whitespace().next().unwrap_or("") {
                    "/exit" | "/quit" | "/q" => return Action::Quit,
                    _ => return Action::Slash(text),
                }
            }
            Action::Submit(text)
        }


        (_, KeyCode::Backspace) => {
            app.backspace();
            Action::None
        }
        (_, KeyCode::Delete) => {
            app.delete();
            Action::None
        }
        (KeyModifiers::CONTROL, KeyCode::Char('w')) => {
            app.delete_word();
            Action::None
        }
        (KeyModifiers::CONTROL, KeyCode::Char('u')) => {
            app.clear_line();
            Action::None
        }


        (_, KeyCode::Left) => {
            app.move_left();
            Action::None
        }
        (_, KeyCode::Right) => {
            app.move_right();
            Action::None
        }
        (KeyModifiers::CONTROL, KeyCode::Char('a')) | (_, KeyCode::Home) => {
            app.home();
            Action::None
        }
        (KeyModifiers::CONTROL, KeyCode::Char('e')) | (_, KeyCode::End) => {
            app.end();
            Action::None
        }


        (_, KeyCode::Up) => {
            app.history_up();
            Action::None
        }
        (_, KeyCode::Down) => {
            app.history_down();
            Action::None
        }


        (KeyModifiers::NONE | KeyModifiers::SHIFT, KeyCode::Char(c)) => {
            app.insert_char(c);
            Action::None
        }

        _ => Action::None,
    }
}

fn handle_streaming(key: KeyEvent) -> Action {
    match (key.modifiers, key.code) {
        (KeyModifiers::CONTROL, KeyCode::Char('c')) => Action::Cancel,
        _ => Action::None,
    }
}

fn handle_permission(key: KeyEvent) -> Action {
    match (key.modifiers, key.code) {
        (_, KeyCode::Char('y') | KeyCode::Char('Y')) => Action::PermissionResponse(true),
        (_, KeyCode::Char('n') | KeyCode::Char('N') | KeyCode::Enter) => {
            Action::PermissionResponse(false)
        }
        (KeyModifiers::CONTROL, KeyCode::Char('c')) => Action::PermissionResponse(false),
        _ => Action::None,
    }
}

fn handle_provider_setup(app: &mut App, key: KeyEvent) -> Action {
    use crate::app::ProviderStep;


    if key.code == KeyCode::Esc {
        app.mode = Mode::Input;
        app.clear_line();
        app.cells.push(crate::app::Cell::System {
            message: "Provider setup cancelled.".into(),
        });
        return Action::None;
    }

    match app.mode.clone() {
        Mode::ProviderSetup(ProviderStep::SelectProfile) => {

            let count = app.provider_profiles.len() + 1;
            match key.code {
                KeyCode::Up => {
                    app.provider_selected = app.provider_selected.saturating_sub(1);
                    Action::None
                }
                KeyCode::Down => {
                    app.provider_selected = (app.provider_selected + 1).min(count - 1);
                    Action::None
                }
                KeyCode::Enter => {
                    if app.provider_selected < app.provider_profiles.len() {

                        let name = app.provider_profiles[app.provider_selected]
                            .get("name")
                            .and_then(|v| v.as_str())
                            .unwrap_or("")
                            .to_string();
                        app.mode = Mode::Input;
                        app.clear_line();
                        Action::ProviderSelect(name)
                    } else {

                        app.mode = Mode::ProviderSetup(ProviderStep::EnterName);
                        app.clear_line();
                        app.cells.push(crate::app::Cell::System {
                            message: "Enter profile name:".into(),
                        });
                        Action::None
                    }
                }
                _ => Action::None,
            }
        }

        Mode::ProviderSetup(ProviderStep::EnterName) => {
            match key.code {
                KeyCode::Enter => {
                    let name = app.take_input();
                    if name.is_empty() {
                        return Action::None;
                    }
                    app.provider_name = name.clone();
                    app.mode = Mode::ProviderSetup(ProviderStep::EnterBaseUrl);
                    app.cells.push(crate::app::Cell::System {
                        message: format!("Profile: {name}\nEnter base URL (e.g. http://localhost:11434/v1):"),
                    });
                    Action::None
                }
                KeyCode::Backspace => { app.backspace(); Action::None }
                KeyCode::Char(c) => { app.insert_char(c); Action::None }
                _ => Action::None,
            }
        }

        Mode::ProviderSetup(ProviderStep::EnterBaseUrl) => {
            match key.code {
                KeyCode::Enter => {
                    let url = app.take_input();
                    if url.is_empty() {
                        return Action::None;
                    }
                    app.provider_base_url = url.clone();
                    app.mode = Mode::ProviderSetup(ProviderStep::EnterApiKey);
                    app.cells.push(crate::app::Cell::System {
                        message: format!("Base URL: {url}\nEnter API key (or press Enter to skip):"),
                    });
                    Action::None
                }
                KeyCode::Backspace => { app.backspace(); Action::None }
                KeyCode::Left => { app.move_left(); Action::None }
                KeyCode::Right => { app.move_right(); Action::None }
                KeyCode::Char(c) => { app.insert_char(c); Action::None }
                _ => Action::None,
            }
        }

        Mode::ProviderSetup(ProviderStep::EnterApiKey) => {
            match key.code {
                KeyCode::Enter => {
                    let api_key = app.take_input();
                    app.provider_api_key = api_key;
                    app.mode = Mode::ProviderSetup(ProviderStep::FetchingModels);
                    app.cells.push(crate::app::Cell::System {
                        message: "Fetching available models...".into(),
                    });

                    Action::ProviderFetchModels(
                        app.provider_base_url.clone(),
                        app.provider_api_key.clone(),
                    )
                }
                KeyCode::Backspace => { app.backspace(); Action::None }
                KeyCode::Left => { app.move_left(); Action::None }
                KeyCode::Right => { app.move_right(); Action::None }
                KeyCode::Char(c) => { app.insert_char(c); Action::None }
                _ => Action::None,
            }
        }

        Mode::ProviderSetup(ProviderStep::FetchingModels) => {

            Action::None
        }

        Mode::ProviderSetup(ProviderStep::SelectModel) => {
            let count = app.provider_models.len().max(1);
            match key.code {
                KeyCode::Up => {
                    app.provider_selected = app.provider_selected.saturating_sub(1);
                    Action::None
                }
                KeyCode::Down => {

                    app.provider_selected = (app.provider_selected + 1).min(count);
                    Action::None
                }
                KeyCode::Enter => {
                    if app.provider_selected < app.provider_models.len() {
                        let model = app.provider_models[app.provider_selected].clone();
                        app.mode = Mode::Input;
                        app.clear_line();
                        Action::ProviderSave(
                            app.provider_name.clone(),
                            app.provider_base_url.clone(),
                            app.provider_api_key.clone(),
                            model,
                        )
                    } else {

                        app.clear_line();
                        app.cells.push(crate::app::Cell::System {
                            message: "Enter model name:".into(),
                        });
                        app.mode = Mode::ProviderSetup(ProviderStep::EnterModel);
                        Action::None
                    }
                }

                KeyCode::Char(c) => {

                    let typed = format!("{c}");
                    if let Some(pos) = app.provider_models.iter().position(|m| {
                        m.to_lowercase().starts_with(&typed.to_lowercase())
                    }) {
                        app.provider_selected = pos;
                    }
                    Action::None
                }
                _ => Action::None,
            }
        }

        Mode::ProviderSetup(ProviderStep::EnterModel) => {
            match key.code {
                KeyCode::Enter => {
                    let model = app.take_input();
                    if model.is_empty() {
                        return Action::None;
                    }
                    app.mode = Mode::Input;
                    Action::ProviderSave(
                        app.provider_name.clone(),
                        app.provider_base_url.clone(),
                        app.provider_api_key.clone(),
                        model,
                    )
                }
                KeyCode::Backspace => { app.backspace(); Action::None }
                KeyCode::Left => { app.move_left(); Action::None }
                KeyCode::Right => { app.move_right(); Action::None }
                KeyCode::Char(c) => { app.insert_char(c); Action::None }
                _ => Action::None,
            }
        }

        _ => Action::None,
    }
}

/// Handle pasted text — insert into the input buffer.
pub fn handle_paste(app: &mut App, text: &str) {
    for c in text.chars() {
        if c == '\n' || c == '\r' {
            continue;
        }
        app.insert_char(c);
    }
}
