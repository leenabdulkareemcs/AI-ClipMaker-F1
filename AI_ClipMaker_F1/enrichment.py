"""
AI ClipMaker F1 — Enrichment Pipeline
Phase 4: LLM clip title generation (Claude API) + voice commentary (gTTS)
"""

import os
import json
import time


# ─────────────────────────────────────────────────────────────────────────────
# LLM TITLE GENERATOR (Claude API)
# ─────────────────────────────────────────────────────────────────────────────

def generate_clip_titles(events, race_name, session_type, api_key, log=None):
    """
    Call the Claude API to generate a vivid title for each clip.
    Returns the events list with 'title' field added to each.
    """
    def _log(msg):
        if log:
            log(msg)

    if not api_key or not api_key.strip():
        _log("      [!] No Claude API key — skipping title generation")
        for ev in events:
            ev["title"] = ev.get("label", ev.get("type", "F1 Moment")).replace("_", " ").title()
        return events

    try:
        import anthropic
    except ImportError:
        _log("      [!] anthropic package not installed — pip install anthropic")
        for ev in events:
            ev["title"] = ev.get("label", ev.get("type", "F1 Moment")).replace("_", " ").title()
        return events

    client = anthropic.Anthropic(api_key=api_key)

    # Build a batch prompt — one call for all events to save API calls
    events_list = []
    for i, ev in enumerate(events):
        t = ev.get("video_timestamp", 0)
        mins = int(t // 60)
        secs = int(t % 60)
        events_list.append({
            "id": i,
            "type": ev.get("type", "unknown"),
            "label": ev.get("label", ""),
            "score": round(ev.get("score", 0.5), 2),
            "time": f"{mins}:{secs:02d}",
            "driver": ev.get("driver", ""),
        })

    system_prompt = (
        "You are an expert F1 broadcast journalist. Generate vivid, punchy clip titles "
        "for Formula 1 highlight moments. Titles should be exciting, specific, and under "
        "8 words. Match the energy to the excitement score (1.0 = maximum drama). "
        "Use F1 terminology naturally. Do not use hashtags or emojis."
    )

    user_prompt = (
        f"Race: {race_name}\n"
        f"Session: {session_type}\n\n"
        f"Generate a title for each of these {len(events_list)} F1 moments. "
        f"Return ONLY a JSON array of objects with 'id' and 'title' fields. "
        f"No other text, no markdown, no explanation.\n\n"
        f"Moments:\n{json.dumps(events_list, indent=2)}"
    )

    _log(f"      Calling Claude API for {len(events)} clip titles...")
    try:
        response = client.messages.create(
            model="claude-sonnet-4-5",
            max_tokens=1500,
            messages=[{"role": "user", "content": user_prompt}],
            system=system_prompt,
        )
        raw = response.content[0].text.strip()

        # Strip markdown fences if present
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        raw = raw.strip()

        titles_data = json.loads(raw)
        title_map = {item["id"]: item["title"] for item in titles_data}

        for i, ev in enumerate(events):
            ev["title"] = title_map.get(i, ev.get("label", "F1 Moment"))

        _log(f"      ✓ {len(title_map)} titles generated")

    except Exception as e:
        _log(f"      [!] Claude API error: {e}")
        _log("          Using fallback titles")
        for ev in events:
            ev["title"] = _fallback_title(ev)

    return events


def _fallback_title(ev):
    """Generate a basic title without LLM."""
    ev_type = ev.get("type", "moment")
    driver = ev.get("driver", "")
    t = ev.get("video_timestamp", 0)
    mins = int(t // 60)
    secs = int(t % 60)
    time_str = f"T+{mins}:{secs:02d}"

    titles = {
        "pit_stop": f"Pit Stop — {driver} {time_str}" if driver else f"Pit Stop {time_str}",
        "safety_car": f"Safety Car Deployed {time_str}",
        "virtual_safety_car": f"Virtual Safety Car {time_str}",
        "position_change": ev.get("label", f"Position Change {time_str}"),
        "fastest_lap": f"Fastest Lap {time_str}",
        "flag": f"Flag — {time_str}",
        "audio_crowd": f"Crowd Erupts {time_str}",
        "audio_impact": f"Impact / Incident {time_str}",
        "audio_excitement": f"Action {time_str}",
    }
    return titles.get(ev_type, f"F1 Moment {time_str}")


# ─────────────────────────────────────────────────────────────────────────────
# ARABIC / ENGLISH VOICE COMMENTARY (gTTS)
# ─────────────────────────────────────────────────────────────────────────────

COMMENTARY_TEMPLATES_EN = {
    "pit_stop":           "And {driver} dives into the pits! The crew springs into action.",
    "safety_car":         "The safety car is out! The race is neutralised.",
    "virtual_safety_car": "Virtual safety car deployed. All drivers must reduce speed.",
    "position_change":    "{label}. What a moment in this race!",
    "fastest_lap":        "That is the fastest lap of the race! Incredible pace.",
    "flag":               "The flag is out. A significant moment in this Grand Prix.",
    "audio_crowd":        "Listen to that crowd! Something spectacular just happened.",
    "audio_impact":       "A big moment! The crowd reacts immediately.",
    "audio_excitement":   "The excitement is at fever pitch right now.",
}

COMMENTARY_TEMPLATES_AR = {
    "pit_stop":           "يدخل {driver} إلى منطقة التوقف! فريق الميكانيكيين ينطلق على الفور.",
    "safety_car":         "سيارة الأمان في المضمار! السباق يدخل في حالة تحييد.",
    "virtual_safety_car": "سيارة الأمان الافتراضية! يجب على جميع السائقين تخفيض سرعتهم.",
    "position_change":    "تغيير في الترتيب! ما هذه اللحظة المثيرة في هذا السباق!",
    "fastest_lap":        "أسرع جولة في السباق! أداء لا يُصدق!",
    "flag":               "تلوح الراية. لحظة محورية في جائزة هذا الموسم الكبرى.",
    "audio_crowd":        "اسمع الجمهور! شيء رائع حدث للتو!",
    "audio_impact":       "لحظة كبيرة! الجمهور يتفاعل بشكل فوري.",
    "audio_excitement":   "الإثارة في أوجها الآن!",
}


def _fill_template(template, ev):
    driver = ev.get("driver", "the driver") or "the driver"
    label = ev.get("label", "").split("[")[0].strip()  # strip vision annotation
    return template.format(driver=driver, label=label)


def generate_commentary_audio(events, output_dir, language="both", log=None):
    """
    Generate voice commentary using edge-tts (Microsoft Neural voices).
    Falls back to gTTS if edge-tts not available.
    Arabic voice: ar-SA-HamedNeural (natural Saudi Arabic male)
    English voice: en-GB-RyanNeural (broadcast-style English)
    """
    def _log(msg):
        if log:
            log(msg)

    comm_dir = os.path.join(output_dir, "commentary")
    os.makedirs(comm_dir, exist_ok=True)

    # Try edge-tts first (much better quality)
    use_edge = False
    try:
        import edge_tts
        use_edge = True
        _log("      Using edge-tts (Microsoft Neural voices)")
    except ImportError:
        try:
            from gtts import gTTS
            _log("      Using gTTS (install edge-tts for better Arabic: pip install edge-tts)")
        except ImportError:
            _log("      [!] No TTS available — pip install edge-tts")
            return events

    # Voice map for edge-tts
    EDGE_VOICES = {
        "ar": "ar-SA-HamedNeural",   # Natural Saudi Arabic male voice
        "en": "en-GB-RyanNeural",     # Broadcast-style English
    }

    _log(f"      Generating voice commentary for {len(events)} clips...")

    langs = []
    if language in ("en", "both"):
        langs.append(("en", COMMENTARY_TEMPLATES_EN))
    if language in ("ar", "both"):
        langs.append(("ar", COMMENTARY_TEMPLATES_AR))

    for i, ev in enumerate(events):
        ev_type = ev.get("type", "audio_excitement")

        for lang_code, templates in langs:
            template = templates.get(ev_type, templates.get("audio_excitement", ""))
            text = _fill_template(template, ev)
            out_path = os.path.join(comm_dir, f"clip_{i+1:02d}_{lang_code}.mp3")

            if use_edge:
                try:
                    import asyncio, edge_tts
                    voice = EDGE_VOICES.get(lang_code, "en-GB-RyanNeural")

                    async def _speak(text, voice, path):
                        communicate = edge_tts.Communicate(text, voice)
                        await communicate.save(path)

                    asyncio.run(_speak(text, voice, out_path))
                    ev[f"commentary_{lang_code}"] = out_path
                except Exception as e:
                    _log(f"      [!] edge-tts error clip {i+1} ({lang_code}): {e}")
                    # Fallback to gTTS
                    try:
                        from gtts import gTTS
                        tts = gTTS(text=text, lang=lang_code, slow=False)
                        tts.save(out_path)
                        ev[f"commentary_{lang_code}"] = out_path
                    except Exception:
                        pass
            else:
                try:
                    from gtts import gTTS
                    tts = gTTS(text=text, lang=lang_code, slow=False)
                    tts.save(out_path)
                    ev[f"commentary_{lang_code}"] = out_path
                except Exception as e:
                    _log(f"      [!] gTTS error clip {i+1} ({lang_code}): {e}")

    _log(f"      ✓ Commentary audio saved to: {comm_dir}/")
    return events


def _get_windows_font():
    """Find a working font file on Windows for FFmpeg drawtext."""
    candidates = [
        "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/calibri.ttf",
        "C:/Windows/Fonts/segoeui.ttf",
        "C:/Windows/Fonts/tahoma.ttf",
        "C:/Windows/Fonts/verdana.ttf",
    ]
    for f in candidates:
        if os.path.exists(f):
            return f
    return None


def burn_title_into_clip(ffmpeg_bin, src_path, out_path, title, clip_number,
                          font_size=32, log=None):
    """
    Use FFmpeg drawtext filter to burn the clip title into the video.
    Handles Windows font paths automatically. Falls back to copy on failure.
    """
    import subprocess, platform

    def _log(msg):
        if log:
            log(msg)

    # Escape special characters for FFmpeg drawtext
    safe_title = (title
        .replace("\\", "")
        .replace("'", "")
        .replace(":", " -")
        .replace("[", "(")
        .replace("]", ")")
    )

    # Build drawtext filter
    font_part = ""
    if platform.system() == "Windows":
        font_path = _get_windows_font()
        if font_path:
            # FFmpeg on Windows needs forward slashes and escaped colons in font path
            font_path_ff = font_path.replace("\\", "/").replace(":", "\\\\:")
            font_part = f"fontfile='{font_path_ff}':"

    vf = (
        f"drawtext={font_part}"
        f"text='{safe_title}':"
        f"fontsize={font_size}:"
        f"fontcolor=white:"
        f"borderw=3:bordercolor=black@0.9:"
        f"x=40:y=h-th-40:"
        f"box=1:boxcolor=black@0.6:boxborderw=10:"
        f"line_spacing=4"
    )

    cmd = [
        ffmpeg_bin, "-y",
        "-i", src_path,
        "-vf", vf,
        "-c:v", "libx264",
        "-preset", "ultrafast",
        "-c:a", "copy",
        out_path
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        _log(f"      [!] drawtext failed clip {clip_number} — using plain copy")
        _log(f"          {result.stderr[-200:]}")
        import shutil
        shutil.copy2(src_path, out_path)


def mix_commentary_into_clip(ffmpeg_bin, video_path, audio_path, out_path, log=None):
    """
    Mix commentary audio over the clip's existing audio.
    Commentary plays from the start of the clip at a lower volume.
    """
    import subprocess

    def _log(msg):
        if log:
            log(msg)

    cmd = [
        ffmpeg_bin, "-y",
        "-i", video_path,
        "-i", audio_path,
        "-filter_complex",
        "[0:a]volume=0.7[orig];[1:a]volume=1.0[comm];[orig][comm]amix=inputs=2:duration=first[aout]",
        "-map", "0:v",
        "-map", "[aout]",
        "-c:v", "copy",
        "-c:a", "aac",
        "-shortest",
        out_path
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        _log(f"      [!] Commentary mix failed: {result.stderr[-200:]}")
        import shutil
        shutil.copy2(video_path, out_path)
