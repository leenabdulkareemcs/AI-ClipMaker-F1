================================================================
  AI CLIPMAKER F1 — Full Pipeline (All 5 Phases)
  Automatic Formula 1 Highlight Reel Generator
  Built on ClipMaker 1.1 by B4L1
================================================================

PHASES
----------------------------------------------------------------
  Phase 1  OpenF1 API       Pit stops, safety car, overtakes,
                             fastest lap — fetched automatically
  Phase 2  Audio AI         librosa finds crowd roars + crashes
  Phase 3  Vision AI        YOLOv8 confirms action visually
  Phase 4  LLM Titles       Claude API writes vivid clip titles
  Phase 4  Voice Commentary gTTS generates Arabic + English audio
  Phase 5  Output           FFmpeg assembles the final reel
                             with title overlay + commentary mix


REQUIRED
----------------------------------------------------------------
  Python   https://www.python.org/downloads
  FFmpeg   https://ffmpeg.org/download.html
  Video    Any .mp4 / .mkv race broadcast

Core packages (auto-installed on first launch):
  streamlit  moviepy  pandas  requests  plotly  anthropic  gtts


OPTIONAL AI PACKAGES
----------------------------------------------------------------
Install these manually for the AI phases:

  pip install librosa scipy              # Phase 2 — Audio AI
  pip install ultralytics opencv-python  # Phase 3 — Vision AI


INSTALL FFMPEG
----------------------------------------------------------------
  Windows:
    1. https://www.gyan.dev/ffmpeg/builds/
    2. Download ffmpeg-release-essentials.zip
    3. Extract → copy ffmpeg.exe to C:\Windows\System32\

  Mac:
    brew install ffmpeg
    (Get Homebrew first: https://brew.sh)


LAUNCH
----------------------------------------------------------------
  Windows:  Double-click Launch_ClipMaker_F1.bat

  Mac:
    1. Move entire folder to Desktop or Documents first
    2. Double-click ClipMaker_F1.app
    3. First time: Right-click → Open → Open


HOW TO USE
----------------------------------------------------------------
TAB 1 — Race & Video
  a. Select race (Saudi GP listed first)
  b. Click "Fetch Race Data" to get session key from OpenF1
  c. Click "Preview Events" to see pit stops, SC, overtakes
  d. Browse to your video file
  e. Set Race Start Offset (see below)
  f. Optional: filter event types or set min excitement score

TAB 2 — AI Pipeline
  a. Audio AI:  tick to find crowd roars and crashes
  b. Vision AI: tick if ultralytics installed (slow — use Top N)
  c. LLM Titles: paste Claude API key for vivid titles
  d. Commentary: tick for Arabic + English voice files

TAB 3 — Output
  a. Choose output folder
  b. Tick "Dry Run" first — always. Preview clips without rendering.
  c. Once happy: untick Dry Run → click Run

TAB 4 — Heatmap
  Click "Generate Heatmap" to see audio excitement across the
  full race with OpenF1 events overlaid as dotted lines.


THE RACE START OFFSET — MOST IMPORTANT SETTING
----------------------------------------------------------------
OpenF1 times are measured from session start. Your video likely
has pre-race coverage before the formation lap.

Set the offset to: how many seconds into the video the
formation lap / lights-out actually happens.

Example: 4 minutes of pre-race build-up → set offset to 4:00.

TO FIND THE RIGHT VALUE
  1. Do a Dry Run
  2. Find a known event (a safety car you remember)
  3. Check when the app says it happens vs. where it is in
     your video player
  4. Adjust the offset by that difference and re-run


OUTPUT STRUCTURE
----------------------------------------------------------------
After a full render:

  F1_Highlights.mp4      Combined highlight reel
  clips_raw/             Raw individual clips
  clips_enriched/        Clips with titles / commentary mixed in
  commentary/            .mp3 audio files
                           clip_01_en.mp3  (English)
                           clip_01_ar.mp3  (Arabic)


TROUBLESHOOTING
----------------------------------------------------------------
"No events from OpenF1"
  → Race data takes a few days after a recent race.
    Check openf1.org/v1/sessions?year=2024 and enter key manually.

"FFmpeg not found"
  → Install FFmpeg and add it to system PATH (see above).

"librosa not found"    → pip install librosa scipy
"ultralytics not found"→ pip install ultralytics opencv-python
"gTTS not installed"   → pip install gTTS

Clips are too early / late
  → Race start offset is wrong. Use Dry Run to diagnose.

Vision detection is very slow
  → Use Top N = 10 filter + nano (n) model size.

Mac: browse button doesn't work
  → brew install python-tk

Mac: app won't open
  → Move folder to Desktop/Documents. Right-click → Open.

Claude API error
  → Check key at console.anthropic.com. Ensure credits available.


FILES IN THIS FOLDER
----------------------------------------------------------------
  app_streamlit.py           Main app (all phases)
  detection.py               Audio AI + Vision AI modules
  enrichment.py              LLM titles + voice commentary
  Launch_ClipMaker_F1.bat    Windows launcher
  ClipMaker_F1.app/          Mac launcher
  README.txt                 This file

================================================================
  AI ClipMaker F1 — Full Pipeline
  Built on ClipMaker 1.1 by B4L1  |  Saudi Arabian GP 2026 Hackathon
================================================================
