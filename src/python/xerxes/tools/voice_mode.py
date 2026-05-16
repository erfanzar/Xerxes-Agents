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
"""Voice-mode recording primitives for the TUI.

Records a push-to-talk WAV using ``sounddevice`` (when available), then hands
the recording to :func:`xerxes.tools.transcription_tool.transcribe` to produce
text the TUI can submit as the next turn.

The recorder is structured around an abstract :class:`VoiceRecorder` so the
TUI key binding can ``start()`` on key-down and ``stop()`` on key-up, and so
tests can swap in :class:`NullRecorder` for a deterministic, dependency-free
backend.
"""

from __future__ import annotations

import importlib.util
import logging
import threading
from abc import ABC, abstractmethod
from pathlib import Path

logger = logging.getLogger(__name__)


class VoiceRecorder(ABC):
    """Abstract recorder interface used by the TUI voice mode."""

    @abstractmethod
    def start(self) -> None:
        """Begin capturing audio asynchronously; idempotent for active recorders."""

    @abstractmethod
    def stop(self) -> Path:
        """Stop capture, flush the WAV to disk, and return its path."""

    @property
    def recording(self) -> bool:
        """Return ``True`` while a capture is in progress."""
        return False


class NullRecorder(VoiceRecorder):
    """Recorder that writes a 44-byte placeholder WAV.

    Used as a deterministic stand-in in tests so callers can exercise the
    full record/transcribe pipeline without requiring an audio device.
    """

    def __init__(self, output_path: Path) -> None:
        """Bind the recorder to ``output_path`` and start in an inactive state."""
        self._path = output_path
        self._active = False

    def start(self) -> None:
        """Mark the recorder as active without producing audio."""
        self._active = True

    def stop(self) -> Path:
        """Write a minimal valid PCM WAV header to the output path and return it."""
        self._active = False
        self._path.parent.mkdir(parents=True, exist_ok=True)
        # Smallest possible PCM WAV (44-byte header + zero data).
        header = bytes.fromhex(
            "52494646 24000000 57415645 666d7420 10000000 01000100 401f0000 803e0000 02001000 64617461 00000000".replace(
                " ", ""
            )
        )
        self._path.write_bytes(header)
        return self._path

    @property
    def recording(self) -> bool:
        """Return ``True`` between :meth:`start` and :meth:`stop`."""
        return self._active


class SoundDeviceRecorder(VoiceRecorder):
    """Recorder backed by ``sounddevice`` and ``numpy``.

    Captures 16 kHz mono int16 PCM by default into an in-memory buffer and
    flushes to a WAV file on :meth:`stop`. Audio dependencies are imported
    lazily so the class can be constructed without ``sounddevice`` installed;
    :meth:`start` raises ``RuntimeError`` when the optional ``voice`` extra
    is missing.
    """

    def __init__(
        self,
        output_path: Path,
        *,
        samplerate: int = 16000,
        channels: int = 1,
    ) -> None:
        """Configure the output WAV path and audio format.

        Args:
            output_path: Destination WAV file; parent directories are created
                on :meth:`stop`.
            samplerate: Capture rate in Hz, defaulting to 16 kHz (the rate
                most STT models expect).
            channels: Number of input channels — 1 for mono microphone input.
        """
        self._path = output_path
        self._samplerate = samplerate
        self._channels = channels
        self._lock = threading.Lock()
        self._frames: list = []
        self._stream = None
        self._active = False

    def start(self) -> None:
        """Open the input stream and begin appending frames to the buffer.

        Idempotent: calling ``start`` while already recording is a no-op.

        Raises:
            RuntimeError: ``sounddevice`` or ``numpy`` is not installed.
        """
        if self._active:
            return
        if importlib.util.find_spec("sounddevice") is None or importlib.util.find_spec("numpy") is None:
            raise RuntimeError("sounddevice + numpy required; install xerxes-agent[voice]")
        import numpy as _np  # type: ignore  # noqa: F401
        import sounddevice as sd  # type: ignore

        with self._lock:
            self._frames = []

            def callback(indata, _frames, _time, _status):
                with self._lock:
                    self._frames.append(indata.copy())

            self._stream = sd.InputStream(
                samplerate=self._samplerate,
                channels=self._channels,
                dtype="int16",
                callback=callback,
            )
            self._stream.start()
            self._active = True

    def stop(self) -> Path:
        """Close the input stream and write the buffered audio to disk.

        Returns:
            Path to the freshly written WAV file. When no frames were captured
            a single silent frame is written so downstream tools always receive
            a valid file.
        """
        if not self._active:
            return self._path
        import wave

        import numpy as np  # type: ignore
        import sounddevice as sd  # type: ignore  # noqa: F401

        with self._lock:
            assert self._stream is not None
            self._stream.stop()
            self._stream.close()
            self._stream = None
            self._active = False
            frames = self._frames or [np.zeros((1, self._channels), dtype="int16")]
            data = np.concatenate(frames, axis=0)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with wave.open(str(self._path), "wb") as wf:
            wf.setnchannels(self._channels)
            wf.setsampwidth(2)  # int16
            wf.setframerate(self._samplerate)
            wf.writeframes(data.tobytes())
        return self._path

    @property
    def recording(self) -> bool:
        """Return ``True`` while the underlying ``sounddevice`` stream is open."""
        return self._active


def record_to_file(
    output_path: Path,
    *,
    duration_seconds: float = 5.0,
    samplerate: int = 16000,
) -> Path:
    """Record a fixed-duration clip via :class:`SoundDeviceRecorder`.

    Convenience helper for non-TUI callers; the TUI uses push-to-talk and
    drives the recorder directly. Blocks the current thread for
    ``duration_seconds`` before stopping.

    Raises:
        RuntimeError: ``sounddevice`` or ``numpy`` are not installed.
    """
    rec = SoundDeviceRecorder(output_path, samplerate=samplerate)
    rec.start()
    threading.Event().wait(duration_seconds)
    return rec.stop()


__all__ = ["NullRecorder", "SoundDeviceRecorder", "VoiceRecorder", "record_to_file"]
