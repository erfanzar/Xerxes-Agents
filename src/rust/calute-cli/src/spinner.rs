use std::time::Instant;

pub struct Spinner {
    start: Instant,
}

impl Spinner {
    pub fn new() -> Self {
        Self {
            start: Instant::now(),
        }
    }

    pub fn reset(&mut self) {
        self.start = Instant::now();
    }

    /// Returns `•` or `◦` based on 600ms blink cycle.
    pub fn frame(&self) -> &'static str {
        let elapsed = self.start.elapsed().as_millis();
        if (elapsed / 600) % 2 == 0 {
            "•"
        } else {
            "◦"
        }
    }

    pub fn elapsed_str(&self) -> String {
        let total_secs = self.start.elapsed().as_secs();
        if total_secs < 60 {
            format!("{total_secs}s")
        } else if total_secs < 3600 {
            let mins = total_secs / 60;
            let secs = total_secs % 60;
            format!("{mins}m {secs:02}s")
        } else {
            let hours = total_secs / 3600;
            let mins = (total_secs % 3600) / 60;
            let secs = total_secs % 60;
            format!("{hours}h {mins:02}m {secs:02}s")
        }
    }
}
