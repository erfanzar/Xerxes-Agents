/// Subprocess bridge to the Python agent runtime.
///
/// Spawns `python -m calute.bridge` as a child process and communicates
/// via newline-delimited JSON over stdin/stdout.
use std::process::Stdio;

use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::process::{Child, ChildStdin, ChildStdout};
use tokio::sync::mpsc;

use crate::events::{Event, RawEvent, Request};

pub struct Bridge {
    stdin: ChildStdin,
    _child: Child,
}

impl Bridge {
    /// Spawn the Python bridge server and return (Bridge, event_rx).
    pub async fn spawn(
        python: &str,
        project_dir: &str,
    ) -> anyhow::Result<(Self, mpsc::UnboundedReceiver<Event>)> {
        let mut child = tokio::process::Command::new(python)
            .args(["-m", "calute.bridge"])
            .current_dir(project_dir)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .spawn()
            .map_err(|e| anyhow::anyhow!("Failed to spawn Python bridge: {e}"))?;

        let stdin = child.stdin.take().expect("stdin piped");
        let stdout = child.stdout.take().expect("stdout piped");

        let (tx, rx) = mpsc::unbounded_channel();


        tokio::spawn(read_events(stdout, tx));

        Ok((
            Bridge {
                stdin,
                _child: child,
            },
            rx,
        ))
    }

    /// Send a request to the Python bridge.
    pub async fn send(&mut self, req: Request) -> anyhow::Result<()> {
        let mut line = serde_json::to_string(&req)?;
        line.push('\n');
        self.stdin.write_all(line.as_bytes()).await?;
        self.stdin.flush().await?;
        Ok(())
    }
}

/// Read newline-delimited JSON events from the Python process stdout.
async fn read_events(stdout: ChildStdout, tx: mpsc::UnboundedSender<Event>) {
    let reader = BufReader::new(stdout);
    let mut lines = reader.lines();

    while let Ok(Some(line)) = lines.next_line().await {
        if line.trim().is_empty() {
            continue;
        }
        match serde_json::from_str::<RawEvent>(&line) {
            Ok(raw) => {
                let event = Event::parse(raw);
                if tx.send(event).is_err() {
                    break;
                }
            }
            Err(e) => {

                let _ = tx.send(Event::Error {
                    message: format!("Bridge parse error: {e} — line: {}", &line[..line.len().min(200)]),
                });
            }
        }
    }
}
