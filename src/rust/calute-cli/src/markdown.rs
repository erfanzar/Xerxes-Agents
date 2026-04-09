/// Convert markdown text to styled ratatui Lines.
use pulldown_cmark::{Event as MdEvent, Parser, Tag, TagEnd};
use ratatui::{
    style::{Modifier, Style},
    text::{Line, Span},
};

use crate::theme;

pub fn render(md: &str, width: usize) -> Vec<Line<'static>> {
    let parser = Parser::new(md);
    let mut lines: Vec<Line<'static>> = Vec::new();
    let mut current_spans: Vec<Span<'static>> = Vec::new();
    let mut style_stack: Vec<Style> = vec![theme::text()];
    let mut in_code_block = false;
    let mut code_buf = String::new();
    let mut list_depth: usize = 0;
    let mut ordered_index: Option<u64> = None;

    for event in parser {
        match event {
            MdEvent::Start(tag) => match tag {
                Tag::Heading { level, .. } => {
                    flush_line(&mut current_spans, &mut lines);
                    let style = match level {
                        pulldown_cmark::HeadingLevel::H1 => theme::heading().add_modifier(Modifier::UNDERLINED),
                        pulldown_cmark::HeadingLevel::H2 => theme::heading(),
                        _ => theme::accent().add_modifier(Modifier::ITALIC),
                    };
                    style_stack.push(style);
                }
                Tag::Emphasis => {
                    style_stack.push(current_style(&style_stack).add_modifier(Modifier::ITALIC));
                }
                Tag::Strong => {
                    style_stack.push(current_style(&style_stack).add_modifier(Modifier::BOLD));
                }
                Tag::CodeBlock(_) => {
                    flush_line(&mut current_spans, &mut lines);
                    in_code_block = true;
                    code_buf.clear();
                }
                Tag::List(start) => {
                    flush_line(&mut current_spans, &mut lines);
                    list_depth += 1;
                    ordered_index = start;
                }
                Tag::Item => {
                    let indent = "  ".repeat(list_depth);
                    let marker = match ordered_index {
                        Some(n) => {
                            let s = format!("{indent}{n}. ");
                            ordered_index = Some(n + 1);
                            s
                        }
                        None => format!("{indent}- "),
                    };
                    current_spans.push(Span::styled(marker, theme::dim()));
                }
                Tag::Link { dest_url, .. } => {
                    style_stack.push(
                        current_style(&style_stack)
                            .fg(theme::CYAN)
                            .add_modifier(Modifier::UNDERLINED),
                    );

                    let _ = dest_url;
                }
                Tag::BlockQuote(_) => {
                    style_stack.push(Style::default().fg(theme::GREEN));
                    current_spans.push(Span::styled("  > ", theme::dim()));
                }
                Tag::Paragraph => {}
                _ => {}
            },

            MdEvent::End(tag_end) => match tag_end {
                TagEnd::Heading(_) => {
                    style_stack.pop();
                    flush_line(&mut current_spans, &mut lines);
                }
                TagEnd::Emphasis | TagEnd::Strong | TagEnd::Link | TagEnd::BlockQuote(_) => {
                    style_stack.pop();
                }
                TagEnd::CodeBlock => {
                    in_code_block = false;

                    for code_line in code_buf.lines() {
                        lines.push(Line::from(Span::styled(
                            format!("  {code_line}"),
                            theme::code(),
                        )));
                    }
                    code_buf.clear();
                }
                TagEnd::List(_) => {
                    list_depth = list_depth.saturating_sub(1);
                    if list_depth == 0 {
                        ordered_index = None;
                    }
                }
                TagEnd::Item => {
                    flush_line(&mut current_spans, &mut lines);
                }
                TagEnd::Paragraph => {
                    flush_line(&mut current_spans, &mut lines);

                    if list_depth == 0 {
                        lines.push(Line::from(vec![]));
                    }
                }
                _ => {}
            },

            MdEvent::Text(text) => {
                if in_code_block {
                    code_buf.push_str(&text);
                } else {
                    let style = current_style(&style_stack);

                    let wrapped = textwrap::wrap(&text, width.max(20));
                    for (i, chunk) in wrapped.iter().enumerate() {
                        current_spans.push(Span::styled(chunk.to_string(), style));
                        if i + 1 < wrapped.len() {
                            flush_line(&mut current_spans, &mut lines);
                        }
                    }
                }
            }

            MdEvent::Code(code) => {
                current_spans.push(Span::styled(
                    format!("`{code}`"),
                    theme::code(),
                ));
            }

            MdEvent::SoftBreak | MdEvent::HardBreak => {
                flush_line(&mut current_spans, &mut lines);
            }

            MdEvent::Rule => {
                flush_line(&mut current_spans, &mut lines);
                let rule: String = "─".repeat(width.min(60));
                lines.push(Line::from(Span::styled(rule, theme::dim())));
            }

            _ => {}
        }
    }

    flush_line(&mut current_spans, &mut lines);


    while lines.last().map(|l| l.spans.is_empty()).unwrap_or(false) {
        lines.pop();
    }


    let mut deduped: Vec<Line<'static>> = Vec::with_capacity(lines.len());
    let mut prev_empty = false;
    for line in lines {
        let is_empty = line.spans.is_empty();
        if is_empty && prev_empty {
            continue;
        }
        prev_empty = is_empty;
        deduped.push(line);
    }

    deduped
}

fn current_style(stack: &[Style]) -> Style {
    stack.last().copied().unwrap_or(theme::text())
}

fn flush_line(spans: &mut Vec<Span<'static>>, lines: &mut Vec<Line<'static>>) {
    if !spans.is_empty() {
        lines.push(Line::from(std::mem::take(spans)));
    }
}
