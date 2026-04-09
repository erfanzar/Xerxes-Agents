/// Slash command definitions and filtering.

pub struct SlashCommand {
    pub name: &'static str,
    pub description: &'static str,
}

pub const COMMANDS: &[SlashCommand] = &[
    SlashCommand { name: "/provider", description: "Setup or switch provider profile" },
    SlashCommand { name: "/sampling", description: "View or set sampling parameters" },
    SlashCommand { name: "/compact", description: "Summarize conversation to free context" },
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
    SlashCommand { name: "/exit", description: "Exit Calute" },
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
