# Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Push-to-talk key handler for the TUI.

The handler owns:

    * a record state machine (idle → recording → transcribing → idle)
    * a callback that injects the transcribed text into the TUI buffer
    * a "continuous" mode that auto-restarts recording after the agent
      finishes a response

The actual audio backend (sounddevice / faster-whisper) is injected so
tests can drive the state machine without hardware."""

from __future__ import annotations

import enum
import logging
import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from .._compat_shims import xerxes_subdir_safe
from ..tools.voice_mode import NullRecorder, VoiceRecorder

logger = logging.getLogger(__name__)


class VoiceState(enum.Enum):
    """Lifecycle states for the push-to-talk handler.

    Transitions: ``IDLE → RECORDING → TRANSCRIBING → IDLE`` (and the
    optional ``PLAYING`` reserved for TTS playback feedback)."""

    IDLE = "idle"
    RECORDING = "recording"
    TRANSCRIBING = "transcribing"
    PLAYING = "playing"


TranscribeFn = Callable[[Path], str]
SubmitFn = Callable[[str], None]


@dataclass
class VoiceKeyHandler:
    """Push-to-talk + continuous-listen helper.

    Attributes:
        recorder_factory: callable returning a new ``VoiceRecorder``
            for each press. Default uses ``NullRecorder`` so unit
            tests work without sounddevice.
        transcribe: callable that converts the recorded WAV into text.
            Real wiring uses ``xerxes.tools.transcription_tool.transcribe``.
        submit: callable that receives the transcribed text. The TUI
            wires this to the input buffer's accept handler.
        continuous: when True, ``stop_recording`` immediately re-enters
            RECORDING after delivering text. Toggle via ``set_continuous``."""

    recorder_factory: Callable[[Path], VoiceRecorder] = field(default=lambda path: NullRecorder(path))
    transcribe: TranscribeFn = field(default=lambda path: f"[stub transcript of {path.name}]")
    submit: SubmitFn = field(default=lambda text: None)
    continuous: bool = False
    _state: VoiceState = field(default=VoiceState.IDLE, init=False)
    _recorder: VoiceRecorder | None = field(default=None, init=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)
    _output_dir: Path = field(default_factory=lambda: xerxes_subdir_safe("voice"))

    # ---------------------------- state queries

    @property
    def state(self) -> VoiceState:
        """Return the current handler state under the internal lock."""
        with self._lock:
            return self._state

    def is_recording(self) -> bool:
        """Return ``True`` while audio capture is in progress."""
        return self.state is VoiceState.RECORDING

    # ---------------------------- transitions

    def start_recording(self) -> None:
        """Begin a new recording session.

        If currently RECORDING, this is a no-op (push-and-hold has no
        repeat semantics)."""
        with self._lock:
            if self._state is VoiceState.RECORDING:
                return
            if self._state is VoiceState.TRANSCRIBING:
                # Don't restart on top of a pending transcribe; queue.
                return
            path = self._output_dir / f"clip-{int(threading.get_ident())}.wav"
            self._output_dir.mkdir(parents=True, exist_ok=True)
            self._recorder = self.recorder_factory(path)
            self._state = VoiceState.RECORDING
            recorder = self._recorder
        recorder.start()

    def stop_recording(self) -> str:
        """Stop the active recording, transcribe, and submit the text.

        Returns the transcribed string (also delivered via ``submit``).
        If not currently recording, returns ``""``."""
        with self._lock:
            if self._state is not VoiceState.RECORDING:
                return ""
            recorder = self._recorder
            self._state = VoiceState.TRANSCRIBING
        if recorder is None:
            with self._lock:
                self._state = VoiceState.IDLE
            return ""
        path = recorder.stop()
        try:
            text = self.transcribe(path)
        except Exception as exc:
            logger.warning("transcribe failed: %s", exc)
            text = ""
        try:
            if text:
                self.submit(text)
        finally:
            with self._lock:
                self._recorder = None
                self._state = VoiceState.IDLE
            if self.continuous and text:
                # Auto-restart recording right after delivering.
                self.start_recording()
        return text

    def toggle(self) -> str:
        """Press-and-release: starts a recording if idle, stops + transcribes if recording."""
        if self.state is VoiceState.IDLE:
            self.start_recording()
            return ""
        if self.state is VoiceState.RECORDING:
            return self.stop_recording()
        return ""

    def set_continuous(self, value: bool) -> None:
        """Toggle the auto-restart behavior after each transcription."""
        with self._lock:
            self.continuous = bool(value)


__all__ = ["SubmitFn", "TranscribeFn", "VoiceKeyHandler", "VoiceState"]
