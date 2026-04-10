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

//! Application state machine and cell model.
//!
//! Defines the conversation cell types ([`Cell`]), UI mode FSM ([`Mode`]),
//! input editing, history navigation, and provider setup state.

use std::collections::HashMap;

use crate::spinner::Spinner;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Mode {
    Input,
    Streaming,
    Permission,
    /// Interactive provider setup — multi-step prompt.
    ProviderSetup(ProviderStep),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ProviderStep {
    /// Choosing from existing profiles or "new".
    SelectProfile,
    /// Entering base URL for a new profile.
    EnterBaseUrl,
    /// Entering API key.
    EnterApiKey,
    /// Entering profile name.
    EnterName,
    /// Choosing model from fetched list or typing custom.
    SelectModel,
    /// Typing a custom model name.
    EnterModel,
    /// Waiting for model list to load.
    FetchingModels,
}

/// A single renderable unit in the conversation.
#[derive(Debug, Clone)]
pub enum Cell {
    User {
        text: String,
    },
    Assistant {
        markdown: String,
    },
    Thinking {
        text: String,
    },
    ToolCall {
        name: String,
        verb: String,
        detail: String,
        result: Option<ToolResult>,
        collapsed: bool,
    },
    Error {
        message: String,
    },
    System {
        message: String,
    },
    /// A sub-agent was spawned and is working.
    AgentActivity {
        agent_name: String,
        agent_type: String,
        status: String,        // "running", "completed", "failed"
        tool_calls: Vec<String>, // brief tool call descriptions
        text_preview: String,  // first ~80 chars of output
    },
    /// Handoff between agents.
    Handoff {
        from: String,
        to: String,
        reason: String,
    },
    /// An execution plan header.
    PlanHeader {
        objective: String,
        steps: Vec<(String, String, String)>, // (id, agent, description)
    },
    /// A plan step status update.
    PlanStep {
        step_id: String,
        agent: String,
        description: String,
        status: String,
    },
}

#[derive(Debug, Clone)]
pub struct ToolResult {
    pub output: String,
    pub permitted: bool,
    pub duration_ms: f64,
    pub is_diff: bool,
}

pub struct App {
    pub mode: Mode,


    pub input: String,
    pub cursor: usize,
    pub history: Vec<String>,
    pub history_idx: Option<usize>,


    pub cells: Vec<Cell>,


    pub streaming_text: String,
    pub streaming_thinking: String,


    pub active_tool: Option<String>,


    pub model: String,
    pub provider: String,
    pub permission_mode: String,
    pub tool_count: usize,


    pub turn_count: u64,
    pub total_in: u64,
    pub total_out: u64,
    pub cost_usd: f64,


    pub perm_description: String,
    pub perm_inputs: HashMap<String, serde_json::Value>,


    pub spinner: Spinner,


    pub slash_popup: bool,
    pub slash_filter: String,
    pub slash_selected: usize,


    pub provider_profiles: Vec<serde_json::Value>,
    pub provider_models: Vec<String>,
    pub provider_base_url: String,
    pub provider_api_key: String,
    pub provider_name: String,
    pub provider_selected: usize,

    pub has_profile: bool,
    pub should_quit: bool,
}

impl App {
    pub fn new() -> Self {
        Self {
            mode: Mode::Input,
            input: String::new(),
            cursor: 0,
            history: Vec::new(),
            history_idx: None,
            cells: Vec::new(),
            streaming_text: String::new(),
            streaming_thinking: String::new(),
            active_tool: None,
            model: String::new(),
            provider: String::new(),
            permission_mode: "auto".into(),
            tool_count: 0,
            turn_count: 0,
            total_in: 0,
            total_out: 0,
            cost_usd: 0.0,
            perm_description: String::new(),
            perm_inputs: HashMap::new(),
            spinner: Spinner::new(),
            slash_popup: false,
            slash_filter: String::new(),
            slash_selected: 0,
            provider_profiles: Vec::new(),
            provider_models: Vec::new(),
            provider_base_url: String::new(),
            provider_api_key: String::new(),
            provider_name: String::new(),
            provider_selected: 0,
            has_profile: false,
            should_quit: false,
        }
    }


    pub fn insert_char(&mut self, c: char) {
        self.input.insert(self.cursor, c);
        self.cursor += c.len_utf8();
        self.update_slash_state();
    }

    pub fn backspace(&mut self) {
        if self.cursor > 0 {
            let prev = self.input[..self.cursor]
                .chars()
                .last()
                .map(|c| c.len_utf8())
                .unwrap_or(0);
            self.cursor -= prev;
            self.input.remove(self.cursor);
            self.update_slash_state();
        }
    }

    pub fn delete(&mut self) {
        if self.cursor < self.input.len() {
            self.input.remove(self.cursor);
        }
    }

    pub fn move_left(&mut self) {
        if self.cursor > 0 {
            let prev = self.input[..self.cursor]
                .chars()
                .last()
                .map(|c| c.len_utf8())
                .unwrap_or(0);
            self.cursor -= prev;
        }
    }

    pub fn move_right(&mut self) {
        if self.cursor < self.input.len() {
            let next = self.input[self.cursor..]
                .chars()
                .next()
                .map(|c| c.len_utf8())
                .unwrap_or(0);
            self.cursor += next;
        }
    }

    pub fn home(&mut self) {
        self.cursor = 0;
    }

    pub fn end(&mut self) {
        self.cursor = self.input.len();
    }

    pub fn delete_word(&mut self) {
        if self.cursor == 0 {
            return;
        }
        let before = &self.input[..self.cursor];
        let trimmed = before.trim_end();
        let new_end = trimmed
            .rfind(|c: char| c.is_whitespace())
            .map(|i| i + 1)
            .unwrap_or(0);
        self.input.drain(new_end..self.cursor);
        self.cursor = new_end;
        self.update_slash_state();
    }

    pub fn clear_line(&mut self) {
        self.input.clear();
        self.cursor = 0;
        self.slash_popup = false;
    }

    pub fn take_input(&mut self) -> String {
        let text = self.input.clone();
        if !text.is_empty() {
            self.history.push(text.clone());
        }
        self.clear_line();
        self.history_idx = None;
        text
    }

    pub fn history_up(&mut self) {
        if self.history.is_empty() {
            return;
        }
        let idx = match self.history_idx {
            Some(0) => 0,
            Some(i) => i - 1,
            None => self.history.len() - 1,
        };
        self.history_idx = Some(idx);
        self.input = self.history[idx].clone();
        self.cursor = self.input.len();
    }

    pub fn history_down(&mut self) {
        match self.history_idx {
            Some(i) if i + 1 < self.history.len() => {
                let idx = i + 1;
                self.history_idx = Some(idx);
                self.input = self.history[idx].clone();
                self.cursor = self.input.len();
            }
            Some(_) => {
                self.history_idx = None;
                self.clear_line();
            }
            None => {}
        }
    }


    pub fn flush_streaming(&mut self) {
        if !self.streaming_thinking.is_empty() {
            self.cells.push(Cell::Thinking {
                text: std::mem::take(&mut self.streaming_thinking),
            });
        }
        if !self.streaming_text.is_empty() {
            self.cells.push(Cell::Assistant {
                markdown: std::mem::take(&mut self.streaming_text),
            });
        }
    }


    fn update_slash_state(&mut self) {
        if self.input.starts_with('/') && self.mode == Mode::Input {
            self.slash_popup = true;
            self.slash_filter = self.input[1..].to_string();
            self.slash_selected = 0;
        } else {
            self.slash_popup = false;
        }
    }


    pub fn tool_verb_detail(name: &str, inputs: &HashMap<String, serde_json::Value>) -> (String, String) {
        match name {
            "Read" | "ReadFile" => {
                let path = inputs.get("file_path").and_then(|v| v.as_str()).unwrap_or("");
                ("Read".into(), shorten_path(path))
            }
            "Write" | "WriteFile" => {
                let path = inputs.get("file_path").and_then(|v| v.as_str()).unwrap_or("");
                ("Wrote".into(), shorten_path(path))
            }
            "Edit" | "FileEditTool" => {
                let path = inputs.get("file_path").and_then(|v| v.as_str()).unwrap_or("");
                ("Edited".into(), shorten_path(path))
            }
            "Bash" | "ExecuteShell" => {
                let cmd = inputs.get("command").and_then(|v| v.as_str()).unwrap_or("");
                let short: String = cmd.chars().take(60).collect();
                ("Ran".into(), short)
            }
            "Glob" | "GlobTool" => {
                let pat = inputs.get("pattern").and_then(|v| v.as_str()).unwrap_or("");
                ("List".into(), pat.to_string())
            }
            "Grep" | "GrepTool" => {
                let pat = inputs.get("pattern").and_then(|v| v.as_str()).unwrap_or("");
                ("Searched".into(), format!("/{pat}/"))
            }
            "ListDir" => {
                let path = inputs.get("directory_path").and_then(|v| v.as_str()).unwrap_or(".");
                ("List".into(), path.to_string())
            }
            _ => (name.to_string(), String::new()),
        }
    }
}

fn shorten_path(path: &str) -> String {

    let parts: Vec<&str> = path.split('/').filter(|s| !s.is_empty()).collect();
    if parts.len() <= 2 {
        path.to_string()
    } else {
        parts[parts.len() - 2..].join("/")
    }
}
