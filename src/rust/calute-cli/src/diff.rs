/// Render unified diffs with colored +/- lines.
use ratatui::text::{Line, Span};

use crate::theme;

pub fn render(diff_text: &str, max_lines: usize) -> Vec<Line<'static>> {
    let mut lines = Vec::new();
    let mut count = 0;

    for raw in diff_text.lines() {
        if count >= max_lines {
            lines.push(Line::from(Span::styled(
                format!("  ... ({} more lines)", diff_text.lines().count() - count),
                theme::dim(),
            )));
            break;
        }

        let styled = if raw.starts_with("+++") || raw.starts_with("---") {
            Line::from(Span::styled(
                format!("  {raw}"),
                theme::bold(),
            ))
        } else if raw.starts_with('+') {
            Line::from(Span::styled(
                format!("  {raw}"),
                theme::success(),
            ))
        } else if raw.starts_with('-') {
            Line::from(Span::styled(
                format!("  {raw}"),
                theme::error(),
            ))
        } else if raw.starts_with("@@") {
            Line::from(Span::styled(
                format!("  {raw}"),
                theme::tool_detail(),
            ))
        } else {
            continue;
        };

        lines.push(styled);
        count += 1;
    }

    lines
}

/// Check if a tool result string looks like a unified diff.
pub fn is_diff(text: &str) -> bool {
    text.contains("---") && text.contains("+++") && text.contains("@@")
}
