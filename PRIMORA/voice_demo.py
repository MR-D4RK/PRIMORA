"""Optional Windows voice demo; the existing Aurora and memory remain unchanged.

Uses installed Windows speech recognition and synthesis. No cloud APIs,
downloads, or microphone activity until the user explicitly starts --voice.
A voice identity is selected once and never switched on a tone change.
"""
from __future__ import annotations

import base64
import os
import re
import subprocess
from dataclasses import dataclass
from typing import Optional


def _powershell(script: str, *, timeout: float = 30.0) -> str:
    if os.name != "nt":
        raise RuntimeError("The first voice demo requires Windows (System.Speech).")
    command = base64.b64encode(script.encode("utf-16-le")).decode("ascii")
    try:
        completed = subprocess.run(
            ["powershell.exe", "-NoProfile", "-NonInteractive",
             "-EncodedCommand", command],
            capture_output=True, text=True, encoding="utf-8",
            errors="replace", timeout=timeout, check=False,
        )
    except FileNotFoundError as exc:
        raise RuntimeError("Windows PowerShell is not installed.") from exc
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError("Windows speech operation timed out.") from exc
    if completed.returncode:
        problem = (completed.stderr or completed.stdout).strip()
        raise RuntimeError(
            "Windows speech engine failed: " + problem[-500:]
        )
    return completed.stdout.strip()


def _ps_safe(text: str) -> str:
    """Create an inert string literal (not user-controlled PowerShell code)."""
    return "[Text.Encoding]::UTF8.GetString([Convert]::FromBase64String('" + (
        base64.b64encode(text.encode("utf-8")).decode("ascii")
    ) + "'))"


def installed_voices() -> tuple[str, ...]:
    raw = _powershell(
        "[Console]::OutputEncoding=[Text.Encoding]::UTF8;"
        "Add-Type -AssemblyName System.Speech;"
        "$s=New-Object System.Speech.Synthesis.SpeechSynthesizer;"
        "try { foreach($v in $s.GetInstalledVoices()) {"
        "[Console]::WriteLine($v.VoiceInfo.Name) } }"
        "finally { $s.Dispose() }",
    )
    return tuple(line.strip() for line in raw.splitlines() if line.strip())


@dataclass(frozen=True)
class SpeakingTone:
    rate: int = 0
    volume: int = 90


TONES = {
    "neutral": SpeakingTone(0, 90),
    "happy": SpeakingTone(1, 95),
    "excited": SpeakingTone(2, 100),
    "gentle": SpeakingTone(-2, 80),
    "serious": SpeakingTone(-1, 90),
}


def choose_tone(text: str) -> str:
    """Small demo-only delivery cue; not a claim about Aurora's emotions."""
    value = text.strip()
    if re.search(r"\b(?:sorry|i understand this is hard|my condolences)\b", value, re.I):
        return "gentle"
    if re.search(r"\b(?:warning|critical|be careful|important safety)\b", value, re.I):
        return "serious"
    if re.search(r"\b(?:amazing|wonderful|fantastic|hooray|awesome)\b", value, re.I):
        return "excited"
    if "!" in value:
        return "happy"
    return "neutral"


class WindowsVoice:
    """One voice identity with adjustable rate and volume."""

    def __init__(self, voice_id: Optional[str] = None):
        available = installed_voices()
        if not available:
            raise RuntimeError(
                "No Windows desktop speech voices are installed. "
                "Install a Windows speech voice and try again."
            )
        selected = voice_id or os.environ.get("PRIMORA_VOICE_ID") or available[0]
        if selected not in available:
            raise RuntimeError(
                f"Voice {selected!r} is not installed. Available: "
                + ", ".join(available)
            )
        self.voice_id = selected

    def listen(self, seconds: int = 12, language: str = "en-US") -> str:
        """Recognize one utterance from the default microphone, locally.

        Uses the installed Windows dictation recognizer for this locale;
        a missing English recognition pack gives an actionable error.
        """
        seconds = max(3, min(int(seconds), 30))
        language = language or "en-US"
        script = (
            "[Console]::OutputEncoding=[Text.Encoding]::UTF8;"
            "Add-Type -AssemblyName System.Speech;"
            "$r=$null;"
            "try {"
            "$culture=[Globalization.CultureInfo]::GetCultureInfo("
            + _ps_safe(language) + ");"
            "$r=New-Object System.Speech.Recognition.SpeechRecognitionEngine($culture);"
            "$r.LoadGrammar((New-Object System.Speech.Recognition.DictationGrammar));"
            "$r.SetInputToDefaultAudioDevice();"
            "$heard=$r.Recognize([TimeSpan]::FromSeconds(" + str(seconds) + "));"
            "if ($null -ne $heard) { [Console]::WriteLine($heard.Text) }"
            "} finally { if($null -ne $r){$r.Dispose()} }"
        )
        try:
            return _powershell(script, timeout=seconds + 12)
        except RuntimeError as exc:
            raise RuntimeError(
                f"Microphone recognition unavailable. Check the default "
                f"microphone and Windows {language} speech recognition pack. {exc}"
            ) from exc

    def speak(self, text: str, tone: Optional[str] = None) -> str:
        if not text.strip():
            return "neutral"
        mode = tone or choose_tone(text)
        if mode not in TONES:
            raise ValueError("Unknown speaking tone: " + str(mode))
        settings = TONES[mode]
        script = (
            "[Console]::OutputEncoding=[Text.Encoding]::UTF8;"
            "Add-Type -AssemblyName System.Speech;"
            "$s=New-Object System.Speech.Synthesis.SpeechSynthesizer;"
            "try {"
            "$s.SelectVoice(" + _ps_safe(self.voice_id) + ");"
            "$s.Rate=" + str(settings.rate) + ";"
            "$s.Volume=" + str(settings.volume) + ";"
            "$s.Speak(" + _ps_safe(text) + ");"
            "} finally { $s.Dispose() }"
        )
        _powershell(script, timeout=max(30, min(180, len(text) / 8 + 25)))
        return mode


def _voice_session(aurora, voice: WindowsVoice, *,
                   input_fn=input, output_fn=print,
                   session_id: Optional[str] = None):
    """Run voice conversation via the same Aurora.respond path as text CLI."""
    from PRIMORA.main import _restore_cli_history

    sid = session_id or os.environ.get("PRIMORA_SESSION_ID", "default")
    history, next_seq = _restore_cli_history(aurora, sid)
    output_fn(
        f"Aurora voice demo. Voice: {voice.voice_id}. "
        "Press Enter to talk; type q to quit."
    )
    while True:
        try:
            command = input_fn("Talk [Enter / q]: ").strip().casefold()
        except (KeyboardInterrupt, EOFError):
            output_fn("Goodbye!")
            return
        if command in ("q", "quit", "exit"):
            output_fn("Goodbye!")
            return
        if command:
            output_fn("Press Enter to listen, or q to quit.")
            continue
        try:
            text = voice.listen()
        except RuntimeError as exc:
            output_fn(f"Microphone error: {exc}")
            return
        if not text.strip():
            output_fn("Didn't catch that. Try again.")
            continue
        output_fn(f"You: {text}")
        if text.strip().casefold() in ("quit", "exit", "goodbye aurora"):
            output_fn("Goodbye!")
            return
        entry = {"role": "user", "content": text}
        if next_seq is not None:
            entry["seq"] = next_seq
            next_seq += 1
        result = aurora.respond(history + [entry], session_id=sid)
        history.append(entry)
        if not result.success:
            output_fn(f"Aurora error: {result.error}")
            continue
        reply = {"role": "assistant", "content": result.content}
        if next_seq is not None:
            reply["seq"] = next_seq
            next_seq += 1
        history.append(reply)
        output_fn("Aurora: " + result.content)
        try:
            voice.speak(result.content)
        except RuntimeError as exc:
            output_fn(f"Speaker error: {exc}")
            return


def run_voice(aurora) -> None:
    voice = WindowsVoice()
    _voice_session(aurora, voice)
