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

//! Unified diff rendering with colored +/- lines.
//!
//! Parses tool result strings that contain unified diff output and
//! renders them as styled ratatui [`Line`]s with green additions,
//! red deletions, and cyan hunk headers.

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
