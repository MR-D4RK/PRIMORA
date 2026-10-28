"""Voice MVP tests: Windows services are mocked; no mic or model needed."""
from __future__ import annotations

import base64

import pytest

from PRIMORA import voice_demo


def test_voice_identity_does_not_change_with_tone(monkeypatch):
    monkeypatch.setattr(
        voice_demo, "installed_voices",
        lambda: ("Voice A", "Voice B"),
    )
    calls = []
    monkeypatch.setattr(
        voice_demo, "_powershell",
        lambda script, **kw: calls.append(script) or "",
    )
    speaker = voice_demo.WindowsVoice("Voice B")
    assert speaker.speak("Hello.", "neutral") == "neutral"
    assert speaker.speak("This is wonderful!", "excited") == "excited"
    assert speaker.speak("I'm sorry that happened.", "gentle") == "gentle"
    assert speaker.voice_id == "Voice B"
    assert len(calls) == 3
    # The chosen exact same voice is embedded in each Windows synth call.
    encoded = base64.b64encode(b"Voice B").decode("ascii")
    assert all(encoded in script for script in calls)
    assert "$s.Rate=0;" in calls[0]
    assert "$s.Rate=2;" in calls[1]
    assert "$s.Rate=-2;" in calls[2]


def test_unknown_voice_is_rejected(monkeypatch):
    monkeypatch.setattr(voice_demo, "installed_voices", lambda: ("Voice A",))
    with pytest.raises(RuntimeError, match="not installed"):
        voice_demo.WindowsVoice("Voice Z")


def test_synth_text_cannot_be_powershell_source(monkeypatch):
    monkeypatch.setattr(voice_demo, "installed_voices", lambda: ("A",))
    scripts = []
    monkeypatch.setattr(
        voice_demo, "_powershell",
        lambda script, **kwargs: scripts.append(script) or "",
    )
    payload = "hello'); Remove-Item C:\\\\SENSITIVE; #"
    voice_demo.WindowsVoice("A").speak(payload)
    assert "Remove-Item" not in scripts[0]
    assert base64.b64encode(payload.encode("utf-8")).decode("ascii") in scripts[0]


def test_tones_vary_without_speaker_changes():
    assert voice_demo.choose_tone("That's wonderful!") == "excited"
    assert voice_demo.choose_tone("I'm sorry to hear that.") == "gentle"
    assert voice_demo.choose_tone("Be careful with that.") == "serious"
    assert voice_demo.choose_tone("Hello.") == "neutral"


def test_voice_session_reuses_aurora_and_conversation_history(monkeypatch):
    from PRIMORA import main as main_mod

    monkeypatch.setattr(
        main_mod, "_restore_cli_history",
        lambda aurora, sid: (
            [{"role": "user", "content": "Last time we talked about art.", "seq": 7}],
            8,
        ),
    )
    class Reply:
        success = True
        content = "That sounds wonderful!"
    class StubAurora:
        def __init__(self):
            self.calls = []
        def respond(self, messages, session_id):
            self.calls.append((messages, session_id))
            return Reply()
    class StubVoice:
        voice_id = "Permanent Voice"
        def __init__(self):
            self.utterances = []
        def listen(self):
            return "Can we discuss art?"
        def speak(self, text):
            self.utterances.append(text)

    commands = iter(("", "q"))
    output = []
    aurora, speaker = StubAurora(), StubVoice()
    voice_demo._voice_session(
        aurora, speaker, input_fn=lambda prompt: next(commands),
        output_fn=output.append, session_id="fixed-session",
    )
    assert len(aurora.calls) == 1
    messages, sid = aurora.calls[0]
    assert sid == "fixed-session"
    assert messages[-1] == {
        "role": "user", "content": "Can we discuss art?", "seq": 8,
    }
    assert messages[0]["seq"] == 7
    assert speaker.utterances == [Reply.content]
    assert any("Aurora: " in line for line in output)


def test_unsupported_platform_has_actionable_error(monkeypatch):
    monkeypatch.setattr(voice_demo.os, "name", "posix")
    with pytest.raises(RuntimeError, match="Windows"):
        voice_demo._powershell("irrelevant")
