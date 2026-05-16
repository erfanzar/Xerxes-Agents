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
"""Tests for tts_tool, transcription_tool, voice_mode."""

from __future__ import annotations

import pytest
from xerxes.tools import transcription_tool, tts_tool
from xerxes.tools.voice_mode import NullRecorder, SoundDeviceRecorder


class TestTTS:
    def test_empty_text_raises(self):
        with pytest.raises(ValueError):
            tts_tool.speak("")

    def test_unknown_provider_raises(self, tmp_path):
        with pytest.raises(ValueError):
            tts_tool.speak("hello", provider="bogus", out_path=tmp_path / "x.mp3")

    def test_registered_provider_invoked(self, tmp_path):
        calls = []

        def fake(text, voice, path):
            calls.append((text, voice, path))
            path.write_bytes(b"fake-audio")
            return path

        tts_tool.register_provider("test", fake)
        result = tts_tool.speak("hello world", provider="test", voice="V", out_path=tmp_path / "out.mp3")
        assert result["provider"] == "test"
        assert result["voice"] == "V"
        assert result["bytes"] == len(b"fake-audio")
        assert (tmp_path / "out.mp3").read_bytes() == b"fake-audio"

    def test_default_voice_label(self, tmp_path):
        def fake(text, voice, path):
            path.write_bytes(b"x")
            return path

        tts_tool.register_provider("test2", fake)
        result = tts_tool.speak("h", provider="test2", out_path=tmp_path / "a.mp3")
        assert result["voice"] == "(default)"


class TestTranscription:
    def test_missing_audio_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            transcription_tool.transcribe(tmp_path / "doesnt.wav")

    def test_unknown_backend_raises(self, tmp_path):
        (tmp_path / "a.wav").write_bytes(b"fake")
        with pytest.raises(ValueError):
            transcription_tool.transcribe(tmp_path / "a.wav", backend="bogus")

    def test_registered_backend_invoked(self, tmp_path):
        (tmp_path / "a.wav").write_bytes(b"fake")
        seen = {}

        def fake(path, model):
            seen["path"] = path
            seen["model"] = model
            return "hello world"

        transcription_tool.register_backend("test", fake)
        out = transcription_tool.transcribe(tmp_path / "a.wav", backend="test", model="m")
        assert out["text"] == "hello world"
        assert out["backend"] == "test"
        assert seen["model"] == "m"


class TestNullRecorder:
    def test_lifecycle(self, tmp_path):
        rec = NullRecorder(tmp_path / "out.wav")
        assert rec.recording is False
        rec.start()
        assert rec.recording is True
        out = rec.stop()
        assert rec.recording is False
        assert out.exists()
        # Smallest valid PCM WAV begins with RIFF.
        assert out.read_bytes()[:4] == b"RIFF"


class TestSoundDeviceRecorderUnavailable:
    def test_start_raises_without_libs(self, tmp_path, monkeypatch):
        """When sounddevice isn't installed, start() must raise clearly."""
        import importlib.util as iu

        original = iu.find_spec

        def patched(name, *a, **kw):
            if name in {"sounddevice", "numpy"}:
                return None
            return original(name, *a, **kw)

        monkeypatch.setattr(iu, "find_spec", patched)
        rec = SoundDeviceRecorder(tmp_path / "x.wav")
        with pytest.raises(RuntimeError):
            rec.start()
