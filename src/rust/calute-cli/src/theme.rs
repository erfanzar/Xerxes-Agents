use ratatui::style::{Color, Modifier, Style};


pub const WHITE: Color = Color::Rgb(230, 230, 230);
pub const DIM: Color = Color::Rgb(110, 110, 110);
pub const CYAN: Color = Color::Cyan;
pub const GREEN: Color = Color::Green;
pub const RED: Color = Color::Red;
pub const YELLOW: Color = Color::Yellow;
pub const MAGENTA: Color = Color::Magenta;

pub const USER_BG: Color = Color::Rgb(35, 35, 38);
pub const CODE_FG: Color = Color::Cyan;
pub const BORDER: Color = Color::Rgb(80, 80, 85);


/// Normal text.
pub fn text() -> Style {
    Style::default()
}

/// Dimmed secondary text.
pub fn dim() -> Style {
    Style::default().add_modifier(Modifier::DIM)
}

/// Bold text.
pub fn bold() -> Style {
    Style::default().add_modifier(Modifier::BOLD)
}

/// Dim + italic (for reasoning/thinking).
pub fn italic() -> Style {
    Style::default().add_modifier(Modifier::DIM | Modifier::ITALIC)
}

/// User message: `› ` prefix — bold + dim.
pub fn user_prefix() -> Style {
    Style::default().add_modifier(Modifier::BOLD | Modifier::DIM)
}

/// User message text with background.
pub fn user() -> Style {
    Style::default().bg(USER_BG)
}

/// User message background only (for padding).
pub fn user_bg() -> Style {
    Style::default().bg(USER_BG)
}

/// Assistant bullet: `• ` — dim.
pub fn assistant_bullet() -> Style {
    Style::default().add_modifier(Modifier::DIM)
}

/// Tool call bullet: `• ` — green, bold.
pub fn tool_bullet() -> Style {
    Style::default().fg(GREEN).add_modifier(Modifier::BOLD)
}

/// Tool call success: `✓` — green, bold.
pub fn tool_success() -> Style {
    Style::default().fg(GREEN).add_modifier(Modifier::BOLD)
}

/// Tool call failure: `✗` — red, bold.
pub fn tool_failure() -> Style {
    Style::default().fg(RED).add_modifier(Modifier::BOLD)
}

/// Tool verb: "Ran", "Read", "Edited" — bold.
pub fn tool_verb() -> Style {
    Style::default().add_modifier(Modifier::BOLD)
}

/// Tool detail: command/path — cyan.
pub fn tool_detail() -> Style {
    Style::default().fg(CYAN)
}

/// Tool output/result: dim text.
pub fn tool_result() -> Style {
    Style::default().add_modifier(Modifier::DIM)
}

pub fn success() -> Style {
    Style::default().fg(GREEN)
}

pub fn error() -> Style {
    Style::default().fg(RED)
}

pub fn warning() -> Style {
    Style::default().fg(YELLOW)
}

/// Inline code: cyan.
pub fn code() -> Style {
    Style::default().fg(CYAN)
}

/// Heading: bold + underlined.
pub fn heading() -> Style {
    Style::default().add_modifier(Modifier::BOLD | Modifier::UNDERLINED)
}

/// Heading level 2: bold only.
pub fn heading2() -> Style {
    Style::default().add_modifier(Modifier::BOLD)
}

/// Accent (for headings lower than h2).
pub fn accent() -> Style {
    Style::default().add_modifier(Modifier::ITALIC)
}

/// Spinner text: bold.
pub fn spinner() -> Style {
    Style::default().add_modifier(Modifier::BOLD)
}

/// Input prompt `› `.
pub fn input_prompt() -> Style {
    Style::default().add_modifier(Modifier::BOLD | Modifier::DIM)
}

/// Border color for popups/boxes.
pub fn border() -> Style {
    Style::default().fg(BORDER)
}

/// Status bar.
pub fn status() -> Style {
    Style::default().add_modifier(Modifier::DIM)
}
