// Copyright 2025 The EasyDeL/Xerxes Author @erfanzar (Erfan Zare Chavoshi).
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

//! Slash command definitions and prefix-based filtering.
//!
//! Provides the static command list shown in the autocomplete popup
//! when the user types `/` in the input line. The [`filter`] function
//! returns matching command indices for progressive narrowing.

/// Slash command definitions and filtering.

pub struct SlashCommand {
    pub name: &'static str,
    pub description: &'static str,
}

pub const COMMANDS: &[SlashCommand] = &[
    SlashCommand { name: "/provider", description: "Setup or switch provider profile" },
    SlashCommand { name: "/sampling", description: "View or set sampling parameters" },
    SlashCommand { name: "/compact", description: "Summarize conversation to free context" },
    SlashCommand { name: "/plan", description: "Plan and execute a multi-step task" },
    SlashCommand { name: "/agents", description: "List agent types and running agents" },
    SlashCommand { name: "/skills", description: "List available skills" },
    SlashCommand { name: "/skill", description: "Invoke a skill by name" },
    SlashCommand { name: "/skill-create", description: "Create a new skill" },
    SlashCommand { name: "/model", description: "Switch or show current model" },
    SlashCommand { name: "/help", description: "Show available commands" },
    SlashCommand { name: "/cost", description: "Show cost summary" },
    SlashCommand { name: "/context", description: "Show context info" },
    SlashCommand { name: "/clear", description: "Clear conversation" },
    SlashCommand { name: "/tools", description: "List available tools" },
    SlashCommand { name: "/thinking", description: "Toggle thinking display" },
    SlashCommand { name: "/verbose", description: "Toggle verbose mode" },
    SlashCommand { name: "/debug", description: "Toggle debug mode" },
    SlashCommand { name: "/permissions", description: "Cycle permission mode" },
    SlashCommand { name: "/config", description: "Show config" },
    SlashCommand { name: "/history", description: "Show message count" },
    SlashCommand { name: "/exit", description: "Exit Xerxes" },
];

/// Filter commands by prefix. Returns indices into COMMANDS.
pub fn filter(query: &str) -> Vec<usize> {
    let q = query.to_lowercase();
    COMMANDS
        .iter()
        .enumerate()
        .filter(|(_, cmd)| {

            cmd.name[1..].starts_with(&q) || cmd.name.starts_with(&format!("/{q}"))
        })
        .map(|(i, _)| i)
        .collect()
}
