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

//! Animated status indicator.
//!
//! Blinks between `•` and `◦` every 600ms (matching Codex CLI's spinner)
//! and tracks elapsed time for display as `Ns` / `Nm NNs` / `Nh NNm NNs`.

use std::time::Instant;

/// Blinking dot spinner with elapsed time tracking.
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
