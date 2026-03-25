# 🏎️ AI ClipMaker F1

> **Automatic Formula 1 Highlight Reel Generator**  
> Inspired by the Saudi Arabian Grand Prix — Jeddah Street Circuit  
> Sports Hackathon 2026

---

## What Is This?

AI ClipMaker F1 takes any F1 race broadcast video and automatically generates a highlight reel — no manual editing, no CSV files, no frame-by-frame scrubbing.

It combines three AI signals to find the most exciting moments:

| Signal | What It Does |
|--------|-------------|
| **OpenF1 API** | Fetches pit stops, safety cars, overtakes, fastest laps — official F1 data, free |
| **Audio AI** (librosa) | Analyses broadcast audio for crowd roars, crash impacts, engine spikes |
| **Vision AI** (YOLOv8) | Scans frames around each moment to confirm car proximity and on-track action |

Then enriches each clip with:
- **AI-generated titles** via Claude API — *"Verstappen Storms Back to the Lead — Lap 34"*
- **Arabic + English voice commentary** via Microsoft Neural TTS
- **Title burned into video** via FFmpeg drawtext
- **Crossfade transitions** between clips

---

## Demo

**2025 Saudi Arabian GP — Jeddah Street Circuit**

The app detected 284 OpenF1 events + 25 audio excitement peaks, fused them into 52 moments, and assembled a 5-minute highlight reel with Arabic commentary — fully automatically.

---

## Full Pipeline

```
Race video (.mp4)
      │
      ▼
┌─────────────────────────────────────────┐
│  Phase 1 — OpenF1 API                  │
│  Pit stops · Safety car · Overtakes    │
│  Fastest lap · Position changes        │
└──────────────────┬──────────────────────┘
                   │
┌──────────────────▼──────────────────────┐
│  Phase 2 — Audio AI (librosa)          │
│  Crowd roars · Crash impacts           │
│  Engine spikes · Commentary peaks      │
└──────────────────┬──────────────────────┘
                   │
┌──────────────────▼──────────────────────┐
│  Phase 3 — Vision AI (YOLOv8)          │
│  Car proximity · Pit lane activity     │
│  Safety car lights · Debris detection  │
└──────────────────┬──────────────────────┘
                   │
┌──────────────────▼──────────────────────┐
│  Signal Fusion + Excitement Scorer     │
│  Multi-signal agreement bonus          │
│  Rank by score · Filter top N          │
└──────────────────┬──────────────────────┘
                   │
┌──────────────────▼──────────────────────┐
│  Phase 4 — LLM Enrichment             │
│  Claude API → vivid clip titles        │
│  edge-tts → Arabic + English voice     │
└──────────────────┬──────────────────────┘
                   │
┌──────────────────▼──────────────────────┐
│  Phase 5 — FFmpeg Video Engine         │
│  Cut clips · Burn titles               │
│  Mix commentary · Crossfade assembly   │
└──────────────────┬──────────────────────┘
                   │
                   ▼
        F1_Highlights.mp4 ✓
```

---

## Installation

### Requirements
- Python 3.9+
- FFmpeg

### Install FFmpeg
**Windows:**
```bash
# Download from https://www.gyan.dev/ffmpeg/builds/
# Extract and add bin/ folder to PATH
```

**Mac:**
```bash
brew install ffmpeg
```

### Install Python packages

**Core (required):**
```bash
pip install streamlit moviepy pandas requests plotly anthropic gtts
```

**AI features (optional but recommended):**
```bash
pip install librosa scipy        # Phase 2 — Audio AI
pip install ultralytics opencv-python  # Phase 3 — Vision AI
pip install edge-tts             # Better Arabic/English voice
```

---

## Usage

### Windows
```bash
# Double-click:
Launch_ClipMaker_F1.bat
```

### Mac
```bash
# Move folder to Desktop first, then double-click:
ClipMaker_F1.app
```

### Manual (any OS)
```bash
cd AI_ClipMaker_F1
python -m streamlit run app_streamlit.py
```

---

## How To Use

### 1. Race & Video tab
- Select your race from the dropdown (Saudi GP listed first)
- Click **Fetch Race Data** — gets the OpenF1 session key automatically
- Click **Preview Events** — shows all detected moments as coloured badges
- Browse to your race broadcast video (`.mp4` or `.mkv`)
- Set the **Race Start Offset** — how far into the video the formation lap begins

### 2. AI Pipeline tab
- **Audio AI** — tick to detect crowd roars and crashes from the audio track
- **Vision AI** — tick for frame-by-frame confirmation (slow, use with Top N filter)
- **LLM Titles** — paste your Claude API key for AI-generated clip titles
- **Voice Commentary** — tick for Arabic + English commentary mixed into each clip

### 3. Output tab
- Always do a **Dry Run** first — previews the clip list without rendering
- Set Top N (e.g. 10) for a tight highlight reel, or 0 for everything
- Untick Dry Run and click **Run** for the real render

### 4. Heatmap tab
- Generates an interactive Plotly excitement curve across the full race
- OpenF1 events overlaid as dotted lines
- Where audio peaks and API events overlap = the most exciting moments

---

## Race Start Offset

The most important setting. OpenF1 times are measured from session start. Your video likely has pre-race coverage before the formation lap.

**How to find it:**
1. Open your video in any player
2. Find when the formation lap begins
3. Note the timestamp (e.g. `0:08:24`)
4. Enter that in the app — Minutes: 8, Seconds: 24

---

## Project Structure

```
AI_ClipMaker_F1/
├── app_streamlit.py      Main application — UI + pipeline orchestration
├── detection.py          Phase 2 (Audio AI) + Phase 3 (Vision AI)
├── enrichment.py         Phase 4 (Claude API titles + edge-tts voice)
├── Launch_ClipMaker_F1.bat   Windows launcher
├── ClipMaker_F1.app/         Mac launcher
└── README.md
```

---

## Tech Stack

| Layer | Tools |
|-------|-------|
| UI | Streamlit, Plotly |
| Race data | OpenF1 API (free, no auth) |
| Audio AI | librosa, numpy, scipy |
| Vision AI | YOLOv8 (Ultralytics), OpenCV |
| LLM | Claude API (Anthropic) |
| Voice | edge-tts (Microsoft Neural TTS) |
| Video | FFmpeg, MoviePy |
| Language | Python 3.11 |

---

## Supported Races

Any F1 race available on OpenF1 (all races from 2023 onwards). Pre-loaded in the dropdown:

- 🇸🇦 Saudi Arabian GP — Jeddah (2023, 2024, 2025)
- 🇧🇭 Bahrain GP — Sakhir
- 🇦🇺 Australian GP — Melbourne
- 🇲🇨 Monaco GP — Monte Carlo
- 🇬🇧 British GP — Silverstone
- 🇮🇹 Italian GP — Monza
- 🇺🇸 Las Vegas GP
- 🇦🇪 Abu Dhabi GP — Yas Marina

---

## Built On

This project is directly inspired by and built on top of **ClipMaker 1.1** by [@B03GHB4L1](https://twitter.com/B03GHB4L1) — a football highlight tool using pandas + FFmpeg. We kept the proven FFmpeg video engine and replaced the manual CSV input with a full AI detection pipeline.

| | ClipMaker 1.1 | AI ClipMaker F1 |
|--|--|--|
| Sport | Football | Formula 1 |
| Input | Video + CSV file | Video only |
| Moment detection | Manual CSV | AI + OpenF1 API |
| Titles | None | Claude API |
| Voice | None | Arabic + English |

---

## API Keys

| Key | Where to get it | Required? |
|-----|----------------|-----------|
| Claude API | [console.anthropic.com](https://console.anthropic.com) | Optional (for titles) |
| OpenF1 | No key needed | Free, automatic |

API keys are entered in the UI at runtime — never stored in code or config files.

---

## License

MIT License — free to use, modify, and distribute.

---

*AI ClipMaker F1 — Saudi Arabian GP Hackathon 2026*
