import sys
import os
import threading
import queue
import time
import platform
import requests
import pandas as pd
import streamlit as st
from datetime import datetime

sys.path.insert(0, os.path.dirname(__file__))
from detection import (
    extract_audio_excitement, get_audio_energy_curve,
    run_vision_detection, fuse_signals,
)
from enrichment import (
    generate_clip_titles, generate_commentary_audio,
    burn_title_into_clip, mix_commentary_into_clip, _fallback_title,
)

# =============================================================================
# PAGE CONFIG
# =============================================================================
st.set_page_config(page_title="AI ClipMaker F1", page_icon="🏎️", layout="wide")
st.markdown("""
<style>
.block-container{padding-top:1.5rem;padding-bottom:1rem}
.log-box{background:#0e1117;color:#00ff88;font-family:'Courier New',monospace;
 font-size:12px;padding:14px;border-radius:8px;height:300px;overflow-y:auto;
 white-space:pre-wrap;border:1px solid #2a2a2a}
h1{font-size:1.9rem!important}
.progress-label{font-size:13px;color:#aaa;margin-bottom:4px}
.footer{text-align:center;color:#555;font-size:11px;padding-top:8px}
.phase-header{background:linear-gradient(90deg,#1a1a2e,#16213e);border-left:3px solid #e94560;
 padding:6px 12px;border-radius:0 6px 6px 0;font-size:13px;font-weight:600;
 color:#e94560;margin-bottom:8px}
.badge{display:inline-block;padding:2px 8px;border-radius:4px;font-size:11px;font-weight:600;margin:1px}
</style>
""", unsafe_allow_html=True)

# =============================================================================
# FILE DIALOG HELPERS
# =============================================================================
def _pick_file_thread(rq, filetypes):
    import tkinter as tk; from tkinter import filedialog
    root=tk.Tk(); root.withdraw()
    try:
        if platform.system()=="Windows": root.wm_attributes("-topmost",True)
        elif platform.system()=="Darwin": os.system("osascript -e 'tell application \"Python\" to activate'")
    except: pass
    path=filedialog.askopenfilename(filetypes=filetypes); root.destroy(); rq.put(path)

def _pick_folder_thread(rq):
    import tkinter as tk; from tkinter import filedialog
    root=tk.Tk(); root.withdraw()
    try:
        if platform.system()=="Windows": root.wm_attributes("-topmost",True)
        elif platform.system()=="Darwin": os.system("osascript -e 'tell application \"Python\" to activate'")
    except: pass
    path=filedialog.askdirectory(); root.destroy(); rq.put(path)

def browse_file(ft):
    q=queue.Queue(); t=threading.Thread(target=_pick_file_thread,args=(q,ft),daemon=True)
    t.start(); t.join(timeout=60)
    try: return q.get_nowait()
    except: return ""

def browse_folder():
    q=queue.Queue(); t=threading.Thread(target=_pick_folder_thread,args=(q,),daemon=True)
    t.start(); t.join(timeout=60)
    try: return q.get_nowait()
    except: return ""

# =============================================================================
# OPENF1 API
# =============================================================================
OPENF1_BASE = "https://api.openf1.org/v1"

KNOWN_RACES = {
    "2025 Saudi Arabian GP — Jeddah (Round 4)": {"year":2025,"round":4},
    "2024 Saudi Arabian GP — Jeddah (Round 2)": {"year":2024,"round":2},
    "2023 Saudi Arabian GP — Jeddah (Round 2)": {"year":2023,"round":2},
    "2024 Bahrain GP — Sakhir (Round 1)":        {"year":2024,"round":1},
    "2024 Australian GP — Melbourne (Round 3)":  {"year":2024,"round":3},
    "2024 Monaco GP — Monte Carlo (Round 8)":    {"year":2024,"round":8},
    "2024 British GP — Silverstone (Round 12)":  {"year":2024,"round":12},
    "2024 Italian GP — Monza (Round 16)":        {"year":2024,"round":16},
    "2024 Las Vegas GP (Round 21)":              {"year":2024,"round":21},
    "2024 Abu Dhabi GP — Yas Marina (Round 24)": {"year":2024,"round":24},
    "2023 Monaco GP — Monte Carlo (Round 8)":    {"year":2023,"round":8},
    "2023 Italian GP — Monza (Round 15)":        {"year":2023,"round":15},
}

DRIVER_ABBREV = {
    1:"VER",11:"PER",44:"HAM",63:"RUS",16:"LEC",55:"SAI",4:"NOR",81:"PIA",
    14:"ALO",18:"STR",10:"GAS",31:"OCO",23:"ALB",2:"SAR",77:"BOT",24:"ZHO",
    20:"MAG",27:"HUL",22:"TSU",3:"RIC",40:"LAW",43:"COL",87:"BEA",50:"ANT",
}

EVENT_EXCITEMENT = {
    "pit_stop":0.55,"safety_car":0.80,"virtual_safety_car":0.60,
    "position_change":0.65,"fastest_lap":0.50,"race_start":0.90,"flag":0.70,
}

BADGE_COLORS = {
    "pit_stop":"#264653","safety_car":"#e76f51","virtual_safety_car":"#f4a261",
    "position_change":"#2a9d8f","fastest_lap":"#e9c46a","flag":"#e63946",
    "audio_crowd":"#457b9d","audio_impact":"#6a0572","audio_excitement":"#1d3557",
}

@st.cache_data(ttl=3600,show_spinner=False)
def fetch_session_key(year, round_number, session_name):
    try:
        mr = requests.get(f"{OPENF1_BASE}/meetings", params={"year":year}, timeout=10)
        meetings = mr.json()
        if not isinstance(meetings,list) or len(meetings)<round_number: return None,""
        meeting = meetings[round_number-1]
        mk = meeting.get("meeting_key"); mname = meeting.get("meeting_name","")
        sr = requests.get(f"{OPENF1_BASE}/sessions", params={"meeting_key":mk}, timeout=10)
        sessions = sr.json()
        for s in sessions:
            if s.get("session_name")==session_name: return s.get("session_key"), mname
    except: pass
    return None,""

@st.cache_data(ttl=3600,show_spinner=False)
def fetch_f1_events(session_key):
    events = []
    try:
        r = requests.get(f"{OPENF1_BASE}/pit", params={"session_key":session_key}, timeout=15)
        for p in (r.json() if r.status_code==200 else []):
            d=p.get("date"); dn=p.get("driver_number"); dur=p.get("pit_duration")
            if d:
                ab=DRIVER_ABBREV.get(dn,f"#{dn}")
                events.append({"type":"pit_stop","date":d,"source":"openf1",
                    "label":f"Pit stop — {ab} ({dur:.1f}s)" if dur else f"Pit stop — {ab}",
                    "driver":ab,"score":EVENT_EXCITEMENT["pit_stop"]})
    except: pass
    try:
        r = requests.get(f"{OPENF1_BASE}/race_control", params={"session_key":session_key}, timeout=15)
        for rc in (r.json() if r.status_code==200 else []):
            msg=rc.get("message","").upper(); d=rc.get("date"); flag=rc.get("flag","")
            if not d: continue
            if "SAFETY CAR DEPLOYED" in msg or ("SAFETY CAR" in msg and "VIRTUAL" not in msg):
                events.append({"type":"safety_car","date":d,"label":"Safety car deployed",
                    "driver":None,"score":EVENT_EXCITEMENT["safety_car"],"source":"openf1"})
            elif "VIRTUAL SAFETY CAR" in msg or "VSC" in msg:
                events.append({"type":"virtual_safety_car","date":d,"label":"Virtual safety car",
                    "driver":None,"score":EVENT_EXCITEMENT["virtual_safety_car"],"source":"openf1"})
            elif flag in ("RED","CHEQUERED") or "RED FLAG" in msg or "CHEQUERED" in msg:
                events.append({"type":"flag","date":d,"label":f"{flag.title()} flag" if flag else "Flag",
                    "driver":None,"score":EVENT_EXCITEMENT["flag"],"source":"openf1"})
    except: pass
    try:
        r = requests.get(f"{OPENF1_BASE}/position", params={"session_key":session_key}, timeout=20)
        pos = r.json() if r.status_code==200 else []
        if isinstance(pos,list) and pos:
            df = pd.DataFrame(pos).sort_values("date")
            for dn, grp in df.groupby("driver_number"):
                grp = grp.reset_index(drop=True)
                for i in range(1,len(grp)):
                    pp=grp.at[i-1,"position"]; cp=grp.at[i,"position"]
                    if pd.notna(pp) and pd.notna(cp):
                        pp,cp=int(pp),int(cp)
                        if cp!=pp:
                            ab=DRIVER_ABBREV.get(dn,f"#{dn}")
                            dir_="overtake" if cp<pp else "loses position"
                            sc=EVENT_EXCITEMENT["position_change"]
                            if cp<=3 or pp<=3: sc=min(sc+0.2,1.0)
                            if cp==1: sc=0.92
                            events.append({"type":"position_change","date":grp.at[i,"date"],
                                "label":f"{ab} {dir_}: P{pp}→P{cp}","driver":ab,
                                "score":sc,"source":"openf1"})
    except: pass
    try:
        r = requests.get(f"{OPENF1_BASE}/laps", params={"session_key":session_key}, timeout=20)
        laps = r.json() if r.status_code==200 else []
        if isinstance(laps,list) and laps:
            df=pd.DataFrame(laps)
            if "lap_duration" in df.columns:
                df["lap_duration"]=pd.to_numeric(df["lap_duration"],errors="coerce")
                df=df.dropna(subset=["lap_duration"])
                if len(df):
                    row=df.loc[df["lap_duration"].idxmin()]
                    dn=int(row["driver_number"]) if pd.notna(row.get("driver_number")) else 0
                    ab=DRIVER_ABBREV.get(dn,f"#{dn}")
                    ds=row.get("date_start")
                    if ds:
                        lt=row["lap_duration"]; m=int(lt//60); s=lt%60
                        events.append({"type":"fastest_lap","date":ds,
                            "label":f"Fastest lap — {ab} ({m}:{s:06.3f})","driver":ab,
                            "score":EVENT_EXCITEMENT["fastest_lap"],"source":"openf1"})
    except: pass
    return events

def parse_openf1_date(ds):
    if not ds: return None
    for fmt in ["%Y-%m-%dT%H:%M:%S.%f+00:00","%Y-%m-%dT%H:%M:%S+00:00",
                "%Y-%m-%dT%H:%M:%S.%fZ","%Y-%m-%dT%H:%M:%SZ","%Y-%m-%dT%H:%M:%S"]:
        try: return datetime.strptime(ds,fmt)
        except: continue
    return None

def events_to_timestamps(events, offset_seconds):
    parsed = [(parse_openf1_date(e.get("date","")),e) for e in events]
    parsed = [(dt,e) for dt,e in parsed if dt]
    if not parsed: return []
    parsed.sort(key=lambda x:x[0])
    t0 = parsed[0][0]
    result = []
    for dt,e in parsed:
        vt = (dt-t0).total_seconds() + offset_seconds
        if vt>=0: result.append({**e,"video_timestamp":vt})
    return result

# =============================================================================
# CLIP ENGINE
# =============================================================================
def merge_overlapping_windows(windows, min_gap):
    if not windows: return []
    merged = [list(windows[0])]
    for s,e,lbl,sc in windows[1:]:
        prev=merged[-1]
        if s<=prev[1]+min_gap:
            prev[1]=max(prev[1],e)
            if lbl not in prev[2]: prev[2]=prev[2]+" + "+lbl
            prev[3]=max(prev[3],sc)
        else: merged.append([s,e,lbl,sc])
    return [tuple(w) for w in merged]

def get_ffmpeg_binary():
    import shutil
    cmd=shutil.which("ffmpeg")
    if cmd: return cmd
    try:
        from moviepy.config import FFMPEG_BINARY
        if os.path.exists(FFMPEG_BINARY): return FFMPEG_BINARY
    except: pass
    raise ValueError("FFmpeg not found. Install from https://ffmpeg.org/download.html")

def get_video_duration(path, ffb):
    import subprocess, re
    r=subprocess.run([ffb,"-i",path],capture_output=True,text=True)
    m=re.search(r"Duration:\s*(\d+):(\d+):([\d.]+)",r.stdout+r.stderr)
    if not m: raise ValueError(f"Cannot read duration of {path}")
    return int(m.group(1))*3600+int(m.group(2))*60+float(m.group(3))

def cut_clip_ffmpeg(ffb, src, start, end, out):
    import subprocess
    r=subprocess.run([ffb,"-y","-ss",str(start),"-i",src,"-t",str(end-start),
        "-map","0:v:0","-map","0:a:0?","-c:v","libx264","-preset","ultrafast",
        "-c:a","aac","-avoid_negative_ts","make_zero",out],capture_output=True,text=True)
    if r.returncode!=0: raise ValueError(f"FFmpeg error: {r.stderr[-500:]}")

def concat_clips_ffmpeg(ffb, clip_paths, out_path):
    import subprocess, tempfile
    with tempfile.NamedTemporaryFile(mode="w",suffix=".txt",delete=False) as f:
        for p in clip_paths: f.write(f"file '{p}'\n")
        lp=f.name
    r=subprocess.run([ffb,"-y","-f","concat","-safe","0","-i",lp,"-c","copy",out_path],
        capture_output=True,text=True)
    try: os.remove(lp)
    except: pass
    if r.returncode!=0: raise ValueError(f"Concat error: {r.stderr[-500:]}")



def concat_clips_with_crossfade(ffb, clip_paths, out_path, fade_duration=0.5, log=None):
    """Concatenate clips with smooth crossfade transitions. Falls back to simple concat."""
    import subprocess, tempfile, re, shutil

    def _log(msg):
        if log: log(msg)

    if len(clip_paths) == 1:
        shutil.copy2(clip_paths[0], out_path); return

    def get_dur(path):
        r = subprocess.run([ffb,"-i",path], capture_output=True, text=True)
        m = re.search(r"Duration:\s*(\d+):(\d+):([\d.]+)", r.stdout+r.stderr)
        return int(m.group(1))*3600+int(m.group(2))*60+float(m.group(3)) if m else 15.0

    tmp_dir = tempfile.mkdtemp()
    normalized = []
    _log(f"      Normalising {len(clip_paths)} clips for crossfade...")
    for i, p in enumerate(clip_paths):
        np_ = os.path.join(tmp_dir, f"n{i:03d}.mp4")
        r = subprocess.run([
            ffb,"-y","-i",p,
            "-vf","scale=1920:1080:force_original_aspect_ratio=decrease,pad=1920:1080:(ow-iw)/2:(oh-ih)/2",
            "-r","25","-c:v","libx264","-preset","ultrafast","-crf","23",
            "-c:a","aac","-ar","44100","-ac","2", np_
        ], capture_output=True, text=True)
        normalized.append(np_ if r.returncode==0 else p)

    durations = [get_dur(p) for p in normalized]
    n = len(normalized)
    filter_parts = []
    cumulative = max(0.1, durations[0] - fade_duration)
    filter_parts.append(f"[0:v][1:v]xfade=transition=fade:duration={fade_duration}:offset={cumulative:.3f}[v1]")
    filter_parts.append(f"[0:a][1:a]acrossfade=d={fade_duration}[a1]")
    for i in range(2, n):
        cumulative += durations[i-1] - fade_duration
        pv=f"v{i-1}"; pa=f"a{i-1}"; cv=f"v{i}"; ca=f"a{i}"
        filter_parts.append(f"[{pv}][{i}:v]xfade=transition=fade:duration={fade_duration}:offset={max(0.1,cumulative):.3f}[{cv}]")
        filter_parts.append(f"[{pa}][{i}:a]acrossfade=d={fade_duration}[{ca}]")
    lv=f"v{n-1}"; la=f"a{n-1}"
    cmd=[ffb,"-y"]
    for p in normalized: cmd+=["-i",p]
    cmd+=["-filter_complex",";".join(filter_parts),
          "-map",f"[{lv}]","-map",f"[{la}]",
          "-c:v","libx264","-preset","ultrafast","-crf","20","-c:a","aac",out_path]
    _log(f"      Applying {fade_duration}s crossfade between clips...")
    result = subprocess.run(cmd, capture_output=True, text=True)
    for p in normalized:
        try: os.remove(p)
        except: pass
    try: os.rmdir(tmp_dir)
    except: pass
    if result.returncode != 0:
        _log("      [!] Crossfade failed — using simple concat")
        concat_clips_ffmpeg(ffb, clip_paths, out_path)
    else:
        _log("      Crossfade transitions applied")

# =============================================================================
# MAIN PIPELINE RUNNER
# =============================================================================
def run_clipmaker_f1(config, log_queue, progress_queue):
    def log(msg): log_queue.put({"type":"log","msg":msg})
    def prog(c,t,e): progress_queue.put({"current":c,"total":t,"elapsed":e,"phase":"clips"})

    try:
        log("══════════════════════════════════════════════")
        log("  AI ClipMaker F1 — Full Pipeline Starting")
        log("══════════════════════════════════════════════\n")

        race_name    = config.get("race_name","F1 Race")
        session_type = config.get("session_type","Race")
        session_key  = config["session_key"]
        video_path   = config.get("video_file","").strip().strip("\"'")
        video_offset = config["video_offset_seconds"]

        # ── Phase 1: OpenF1 ───────────────────────────────────────────────
        log("━━━ Phase 1: OpenF1 API ━━━")
        openf1_raw = fetch_f1_events(session_key)
        if not openf1_raw:
            raise ValueError("No events from OpenF1 API. Check session key and network access.")
        log(f"  {len(openf1_raw)} events fetched")
        timed_openf1 = events_to_timestamps(openf1_raw, video_offset)
        log(f"  {len(timed_openf1)} mapped to video timestamps\n")

        # ── Phase 2: Audio AI ─────────────────────────────────────────────
        audio_events = []
        if config.get("use_audio") and video_path and os.path.exists(video_path):
            log("━━━ Phase 2: Audio Excitement Detection (librosa) ━━━")
            audio_events = extract_audio_excitement(
                video_path, log=log,
                top_n=config.get("audio_top_n",25),
                threshold_percentile=config.get("audio_sensitivity",85),
            )
            log(f"  {len(audio_events)} audio peaks extracted\n")
        elif config.get("use_audio"):
            log("━━━ Phase 2: Audio ━━━\n  [skipped — no video file]\n")

        # ── Fuse signals ──────────────────────────────────────────────────
        log("━━━ Signal Fusion ━━━")
        all_events = fuse_signals(timed_openf1, audio_events, merge_window_sec=15.0)
        log(f"  {len(all_events)} moments after fusion")

        # Filters
        sel = config.get("event_types",[])
        if sel:
            all_events = [e for e in all_events if e.get("type") in sel]
            log(f"  Type filter → {len(all_events)}")
        ms = config.get("min_score",0.0)
        if ms>0:
            all_events = [e for e in all_events if e.get("score",0)>=ms]
            log(f"  Score filter ≥{ms:.2f} → {len(all_events)}")
        tn = config.get("top_n",0)
        if tn and tn>0:
            all_events = sorted(all_events,key=lambda e:e.get("score",0),reverse=True)[:tn]
            log(f"  Top {tn} selected")
        if not all_events:
            raise ValueError("No events remain after filtering. Relax the filters.")
        all_events.sort(key=lambda e:e.get("video_timestamp",0))
        log(f"  Final: {len(all_events)} events\n")

        # ── Phase 3: Vision AI ────────────────────────────────────────────
        if config.get("use_vision") and video_path and os.path.exists(video_path):
            log("━━━ Phase 3: Vision Detection (YOLOv8) ━━━")
            try:
                ffb = get_ffmpeg_binary()
                all_events = run_vision_detection(
                    all_events, video_path, ffb, log=log,
                    model_size=config.get("yolo_model","n"),
                )
            except Exception as ex:
                log(f"  [!] Vision error: {ex} — continuing without vision")
            log("")
        elif config.get("use_vision"):
            log("━━━ Phase 3: Vision ━━━\n  [skipped — no video file]\n")

        # ── Phase 4a: LLM titles ──────────────────────────────────────────
        if config.get("use_llm_titles"):
            log("━━━ Phase 4a: LLM Titles (Claude API) ━━━")
            all_events = generate_clip_titles(
                all_events, race_name=race_name, session_type=session_type,
                api_key=config.get("claude_api_key",""), log=log,
            )
            log("")
        else:
            for ev in all_events: ev["title"] = _fallback_title(ev)

        # ── Build windows ──────────────────────────────────────────────────
        bb = config["before_buffer"]; ab = config["after_buffer"]; mg = config["min_gap"]
        raw_windows = sorted([
            (ev["video_timestamp"]-bb, ev["video_timestamp"]+ab,
             ev.get("title") or ev.get("label","F1 Moment"), ev.get("score",0.5))
            for ev in all_events
        ], key=lambda w:w[0])
        windows = merge_overlapping_windows(raw_windows, mg)
        log(f"━━━ Clip Windows ━━━\n  {len(raw_windows)} → {len(windows)} clips (gap={mg}s)\n")

        # ── Dry run ────────────────────────────────────────────────────────
        if config.get("dry_run"):
            log("━━━ DRY RUN PREVIEW ━━━\n")
            for i,(s,e,lbl,sc) in enumerate(windows,1):
                ms_=int(s//60); ss_=s%60; me=int(e//60); se=e%60
                log(f"  Clip {i:02d}  [{ms_}:{ss_:05.2f} → {me}:{se:05.2f}]  ({e-s:.0f}s)  score={sc:.2f}")
                log(f"          {lbl}")
            log(f"\n✓ DRY RUN — {len(windows)} clips ready.")
            log_queue.put({"type":"done"}); return

        # ── Video engine ───────────────────────────────────────────────────
        log("━━━ Video Engine (FFmpeg) ━━━")
        ffb = get_ffmpeg_binary()
        if not video_path or not os.path.exists(video_path):
            raise ValueError("Video file not found.")
        log(f"  {os.path.basename(video_path)}")
        vdur = get_video_duration(video_path, ffb)
        log(f"  Duration: {vdur:.1f}s ({vdur/60:.1f} min)\n")

        out_dir = config["output_dir"]
        os.makedirs(out_dir, exist_ok=True)
        clips_dir = os.path.join(out_dir,"clips_raw"); os.makedirs(clips_dir,exist_ok=True)
        total = len(windows); t0 = time.time()
        raw_clip_paths = []

        for i,(s,e,lbl,sc) in enumerate(windows,1):
            cs=max(0.0,s); ce=min(vdur,e)
            if ce<=cs: log(f"  SKIPPED {i:02d}: outside video"); continue
            rp=os.path.join(clips_dir,f"clip_{i:02d}_raw.mp4")
            log(f"  [{i:02d}/{total}] {lbl[:65]}  ({ce-cs:.0f}s)")
            cut_clip_ffmpeg(ffb, video_path, cs, ce, rp)
            raw_clip_paths.append((i, rp, lbl, sc))
            prog(i, total, time.time()-t0)
        log("")

        # ── Phase 4b: Commentary ──────────────────────────────────────────
        if config.get("use_commentary"):
            log("━━━ Phase 4b: Voice Commentary (gTTS) ━━━")
            all_events = generate_commentary_audio(
                all_events, output_dir=out_dir,
                language=config.get("commentary_language","both"), log=log,
            )
            log("")

        # ── Phase 5: Enrich clips ─────────────────────────────────────────
        use_ti = config.get("burn_titles",False)
        use_cm = config.get("use_commentary",False) and config.get("mix_commentary",False)
        final_clip_paths = []

        if use_ti or use_cm:
            log("━━━ Phase 5: Title Overlay + Commentary Mix ━━━")
            en_dir = os.path.join(out_dir,"clips_enriched"); os.makedirs(en_dir,exist_ok=True)
            for idx,(clip_num,rp,lbl,sc) in enumerate(raw_clip_paths):
                cur = rp
                ev  = all_events[idx] if idx<len(all_events) else {}
                title = ev.get("title",lbl)
                if use_ti:
                    tp=os.path.join(en_dir,f"clip_{clip_num:02d}_titled.mp4")
                    log(f"  Title overlay clip {clip_num:02d}")
                    burn_title_into_clip(ffb,cur,tp,title,clip_num,log=log); cur=tp
                if use_cm:
                    lang=config.get("commentary_language","en")
                    if lang=="both": lang="en"
                    cp=ev.get(f"commentary_{lang}","")
                    if cp and os.path.exists(cp):
                        mp=os.path.join(en_dir,f"clip_{clip_num:02d}_final.mp4")
                        log(f"  Commentary mix clip {clip_num:02d}")
                        mix_commentary_into_clip(ffb,cur,cp,mp,log=log); cur=mp
                final_clip_paths.append(cur)
            log("")
        else:
            final_clip_paths = [rp for _,rp,_,_ in raw_clip_paths]

        # ── Final output ───────────────────────────────────────────────────
        if config.get("individual_clips"):
            log("━━━ Saving Individual Clips ━━━")
            import shutil
            for idx,(clip_num,_,lbl,_) in enumerate(raw_clip_paths):
                src = final_clip_paths[idx] if idx<len(final_clip_paths) else raw_clip_paths[idx][1]
                ev  = all_events[idx] if idx<len(all_events) else {}
                title=ev.get("title",lbl)
                safe="".join(c if c.isalnum() or c=="_" else "_" for c in title[:40])
                dst=os.path.join(out_dir,f"{clip_num:02d}_{safe}.mp4")
                shutil.copy2(src,dst); log(f"  Saved: {os.path.basename(dst)}")
            log(f"\n✓ {len(raw_clip_paths)} clips → {os.path.abspath(out_dir)}/")
        else:
            log("━━━ Assembling Highlight Reel ━━━")
            out_path=os.path.join(out_dir,config["output_filename"])
            log(f"  Concatenating {len(final_clip_paths)} clips...")
            concat_clips_with_crossfade(ffb, final_clip_paths, out_path, fade_duration=0.5, log=log)
            mb=os.path.getsize(out_path)/1024/1024
            log(f"\n✓ Reel saved: {out_path}  ({mb:.1f} MB)")

        if not config.get("keep_raw_clips",False):
            import shutil
            try: shutil.rmtree(clips_dir)
            except: pass

        log("\n══════════════════════════════════════════════")
        log("  All phases complete ✓")
        log("══════════════════════════════════════════════")
        log_queue.put({"type":"done"})

    except Exception as e:
        import traceback
        log(f"\n✗ ERROR: {e}\n{traceback.format_exc()}")
        log_queue.put({"type":"error"})

# =============================================================================
# SESSION STATE
# =============================================================================
for k,v in [("video_path",""),("output_dir",""),
            ("fetched_session_key",None),("fetched_events",None),
            ("session_key_str",""),("race_name",""),("session_type","Race")]:
    if k not in st.session_state: st.session_state[k]=v

# =============================================================================
# UI
# =============================================================================
st.title("🏎️ AI ClipMaker F1")
st.caption("Full AI pipeline — OpenF1 API · Audio AI · Vision AI · Claude titles · Arabic commentary")
st.divider()

tab1, tab2, tab3, tab4 = st.tabs(["🏁 Race & Video","🤖 AI Pipeline","📁 Output","📊 Heatmap"])

# ─── TAB 1: RACE & VIDEO ──────────────────────────────────────────────────────
with tab1:
    c1, c2 = st.columns([1,1], gap="large")
    with c1:
        st.markdown('<div class="phase-header">PHASE 1 — Race Selection & OpenF1 API</div>', unsafe_allow_html=True)
        selected_race = st.selectbox("Select Race", list(KNOWN_RACES.keys()), index=0)
        race_info = KNOWN_RACES[selected_race]
        st.session_state.race_name = selected_race

        session_type = st.selectbox("Session",
            ["Race","Sprint","Qualifying","Practice 1","Practice 2","Practice 3"])
        st.session_state.session_type = session_type

        b1,b2 = st.columns(2)
        with b1:
            if st.button("🔍 Fetch Race Data", use_container_width=True):
                with st.spinner("Connecting to OpenF1..."):
                    sk,_ = fetch_session_key(race_info["year"],race_info["round"],session_type)
                    if sk:
                        st.session_state.fetched_session_key = sk
                        st.session_state.session_key_str = str(sk)
                        st.session_state.fetched_events = None
                        st.success(f"✓ Session key: {sk}")
                    else:
                        st.warning("Not found. Enter manually below.")

        manual_key = st.text_input("Session key (manual)",
            value=st.session_state.session_key_str,
            placeholder="e.g. 9158  — find at openf1.org/v1/sessions")
        if manual_key.strip(): st.session_state.session_key_str = manual_key.strip()
        final_session_key = st.session_state.session_key_str

        if final_session_key:
            if st.button("📋 Preview Events", use_container_width=True):
                with st.spinner("Fetching events..."):
                    try:
                        evs = fetch_f1_events(int(final_session_key) if final_session_key.isdigit() else final_session_key)
                        st.session_state.fetched_events = evs
                    except Exception as ex: st.error(str(ex))

            if st.session_state.fetched_events is not None:
                evs = st.session_state.fetched_events
                if evs:
                    tc = {}
                    for e in evs: tc[e["type"]]=tc.get(e["type"],0)+1
                    badges = " ".join([
                        f'<span class="badge" style="background:{BADGE_COLORS.get(t,"#555")};color:#fff">'
                        f'{t.replace("_"," ").title()} ({c})</span>'
                        for t,c in sorted(tc.items(),key=lambda x:-x[1])
                    ])
                    st.caption(f"{len(evs)} events found")
                    st.markdown(badges, unsafe_allow_html=True)
                else:
                    st.warning("No events returned. Try a different session or enter key manually.")

        st.divider()
        st.subheader("Event Filters")
        ev_type_opts = ["pit_stop","safety_car","virtual_safety_car",
                        "position_change","fastest_lap","flag"]
        selected_event_types = st.multiselect("Event Types",
            options=ev_type_opts,
            format_func=lambda x: x.replace("_"," ").title(),
            placeholder="All types if blank")
        ef1,ef2 = st.columns(2)
        with ef1: min_score = st.number_input("Min excitement (0–1)",0.0,1.0,0.0,0.05)
        with ef2: top_n = st.number_input("Top N moments (0=all)",0,200,0,1)

    with c2:
        st.markdown('<div class="phase-header">VIDEO FILE</div>', unsafe_allow_html=True)
        vc1,vc2 = st.columns([4,1])
        with vc1:
            vid_input = st.text_input("Race Broadcast Video",
                value=st.session_state.video_path, placeholder="Browse or paste full path")
        with vc2:
            st.write(""); st.write("")
            if st.button("Browse",key="bv"):
                p=browse_file([("Video","*.mp4 *.mkv *.avi *.mov"),("All","*.*")])
                if p: st.session_state.video_path=p; st.rerun()

        st.subheader("Race Start Offset")
        st.caption("How far into your video does the race begin (formation lap / lights)?")
        oc1,oc2 = st.columns(2)
        with oc1: off_mm = st.number_input("Minutes",0,999,0,1)
        with oc2: off_ss = st.number_input("Seconds",0,59,0,1)
        offset_total = off_mm*60+off_ss
        st.caption(f"Offset = **{offset_total}s** — all OpenF1 events shifted by this amount")

        st.subheader("Clip Timing")
        sc1,sc2,sc3 = st.columns(3)
        with sc1: before_buf = st.number_input("Before (s)",0,30,5,1)
        with sc2: after_buf  = st.number_input("After (s)",0,60,10,1)
        with sc3: min_gap    = st.number_input("Merge gap (s)",0,60,8,1)

# ─── TAB 2: AI PIPELINE ───────────────────────────────────────────────────────
with tab2:
    ai1,ai2 = st.columns([1,1], gap="large")
    with ai1:
        st.markdown('<div class="phase-header">PHASE 2 — Audio Excitement (librosa)</div>', unsafe_allow_html=True)
        use_audio = st.checkbox("Enable audio AI", value=True,
            help="Finds crowd roars, crashes, engine spikes from the broadcast audio")
        audio_top_n = st.slider("Max audio peaks",5,60,25,disabled=not use_audio)
        audio_sensitivity = st.slider("Sensitivity (percentile)",70,98,85,disabled=not use_audio,
            help="Higher = only biggest spikes. Lower = more moments.")
        st.caption("Requires: pip install librosa scipy")

        st.divider()
        st.markdown('<div class="phase-header">PHASE 3 — Vision AI (YOLOv8)</div>', unsafe_allow_html=True)
        use_vision = st.checkbox("Enable vision AI", value=False,
            help="Runs YOLOv8 on frames to confirm and score each moment visually")
        yolo_size_label = st.selectbox("Model size",
            ["n — nano (fastest)","s — small","m — medium (best)"],
            disabled=not use_vision)
        yolo_model_code = yolo_size_label[0]
        if use_vision:
            st.caption("Requires: pip install ultralytics opencv-python")
            st.warning("Vision is slow (~1–3 min/event). Use with Top N filter.")

    with ai2:
        st.markdown('<div class="phase-header">PHASE 4a — LLM Titles (Claude API)</div>', unsafe_allow_html=True)
        use_llm = st.checkbox("Generate AI clip titles", value=True)
        claude_key = st.text_input("Claude API Key", type="password",
            placeholder="sk-ant-...",
            help="Your Anthropic/Claude API key. Get one at console.anthropic.com",
            disabled=not use_llm)
        burn_titles = st.checkbox("Burn titles into video (FFmpeg drawtext)",
            value=False, disabled=not use_llm,
            help="Overlays the title at the bottom of each clip frame")

        st.divider()
        st.markdown('<div class="phase-header">PHASE 4b — Voice Commentary (gTTS)</div>', unsafe_allow_html=True)
        use_commentary = st.checkbox("Generate voice commentary", value=False)
        comm_lang_label = st.selectbox("Language",
            ["Arabic + English","English only","Arabic only"], disabled=not use_commentary)
        comm_lang_code = {"Arabic + English":"both","English only":"en","Arabic only":"ar"}[comm_lang_label]
        mix_commentary = st.checkbox("Mix commentary into clips", value=False,
            disabled=not use_commentary,
            help="Mixes the generated .mp3 over each clip's audio track")
        if use_commentary: st.caption("Requires: pip install gTTS")

# ─── TAB 3: OUTPUT ────────────────────────────────────────────────────────────
with tab3:
    out1,out2 = st.columns([1,1], gap="large")
    with out1:
        st.subheader("Output Folder")
        fo1,fo2 = st.columns([4,1])
        with fo1:
            out_dir_input = st.text_input("Output Folder",
                value=st.session_state.output_dir, placeholder="Browse or paste path")
        with fo2:
            st.write(""); st.write("")
            if st.button("Browse",key="bo"):
                p=browse_folder()
                if p: st.session_state.output_dir=p; st.rerun()

        individual = st.checkbox("Save individual clips (not one combined reel)")
        out_filename = st.text_input("Reel Filename","F1_Highlights.mp4",
            disabled=individual)
        keep_raw = st.checkbox("Keep raw intermediate clips",
            help="Keeps the clips_raw folder after final assembly")

    with out2:
        st.subheader("Run Mode")
        dry_run = st.checkbox("Dry Run (preview clip list — no rendering)", value=True)
        st.info("**Always do a Dry Run first** to review which clips would be cut "
                "before committing to a full render.")
        st.subheader("Status")
        fsk = st.session_state.session_key_str
        fv  = st.session_state.video_path or vid_input or ""
        status_ok = bool(fsk)
        st.write(f"Session key: {'✅ `'+fsk+'`' if fsk else '❌ not set'}")
        st.write(f"Video file: {'✅ `'+os.path.basename(fv)+'`' if fv else '⚠️ not needed for Dry Run'}")
        st.write(f"Offset: `{offset_total}s`")

# ─── TAB 4: HEATMAP ───────────────────────────────────────────────────────────
with tab4:
    st.subheader("📊 Audio Excitement Heatmap")
    st.caption(
        "Analyse your video's audio track to see excitement levels across the race. "
        "OpenF1 events are overlaid as dotted lines so you can see where data and audio agree."
    )
    hmap_video = st.session_state.video_path or vid_input or ""
    if hmap_video and os.path.exists(hmap_video):
        if st.button("📈 Generate Heatmap (30–90s)"):
            with st.spinner("Analysing audio..."):
                try:
                    times_arr, energy_arr = get_audio_energy_curve(hmap_video)
                    if times_arr:
                        import plotly.graph_objects as go
                        times_min = [t/60 for t in times_arr]
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(
                            x=times_min, y=energy_arr, mode="lines",
                            line=dict(color="#e94560",width=1),
                            fill="tozeroy", fillcolor="rgba(233,69,96,0.15)",
                            name="Audio energy",
                        ))
                        if st.session_state.fetched_events and offset_total>=0:
                            timed_evs = events_to_timestamps(st.session_state.fetched_events, offset_total)
                            for ev in timed_evs:
                                t_min = ev["video_timestamp"]/60
                                color = BADGE_COLORS.get(ev["type"],"#aaa")
                                fig.add_vline(x=t_min, line_width=1.5, line_dash="dot",
                                    line_color=color,
                                    annotation_text=ev["type"].replace("_"," ")[:10],
                                    annotation_font_size=9)
                        fig.update_layout(
                            title="Audio Excitement — Full Race",
                            xaxis_title="Race time (minutes)",
                            yaxis_title="Excitement level (0–1)",
                            template="plotly_dark", height=420,
                            margin=dict(l=50,r=20,t=50,b=50), showlegend=False,
                        )
                        st.plotly_chart(fig, use_container_width=True)
                        st.caption(
                            "Red peaks = audio excitement spikes. "
                            "Dotted lines = OpenF1 events. "
                            "Where they overlap = confirmed exciting moments."
                        )
                    else:
                        st.warning("Heatmap failed. Install librosa: pip install librosa scipy")
                except Exception as ex:
                    st.error(f"Heatmap error: {ex}")
    else:
        st.info("Select a video in the Race & Video tab first, then come back here.")

# =============================================================================
# RUN BUTTON
# =============================================================================
st.divider()
rb1,rb2 = st.columns([1,3])
with rb1:
    run_btn = st.button("▶  Run AI ClipMaker F1", type="primary", use_container_width=True)
with rb2:
    fsk2=st.session_state.session_key_str
    fv2=st.session_state.video_path or vid_input or ""
    st.caption(
        f"Session: `{fsk2 or 'not set'}` | "
        f"Video: `{os.path.basename(fv2) if fv2 else 'not set'}` | "
        f"Offset: `{offset_total}s` | "
        f"Mode: {'DRY RUN' if dry_run else 'RENDER'}"
    )

progress_ph = st.empty()
log_ph = st.empty()

final_video = st.session_state.video_path or vid_input or ""
final_out   = st.session_state.output_dir or out_dir_input or "output"
fsk_final   = st.session_state.session_key_str

if run_btn:
    errors = []
    if not fsk_final: errors.append("No session key — fetch race data or enter manually.")
    if not dry_run and (not final_video or not os.path.exists(final_video)):
        errors.append("Video file not found. Select a valid file, or tick Dry Run.")
    for err in errors: st.error(err)

    if not errors:
        config = {
            "race_name":            selected_race,
            "session_type":         session_type,
            "session_key":          int(fsk_final) if fsk_final.isdigit() else fsk_final,
            "video_file":           final_video,
            "video_offset_seconds": offset_total,
            "event_types":          selected_event_types,
            "min_score":            min_score,
            "top_n":                int(top_n) if top_n>0 else 0,
            "before_buffer":        before_buf,
            "after_buffer":         after_buf,
            "min_gap":              min_gap,
            "use_audio":            use_audio,
            "audio_top_n":          audio_top_n,
            "audio_sensitivity":    audio_sensitivity,
            "use_vision":           use_vision,
            "yolo_model":           yolo_model_code,
            "use_llm_titles":       use_llm,
            "claude_api_key":       claude_key if use_llm else "",
            "burn_titles":          burn_titles,
            "use_commentary":       use_commentary,
            "commentary_language":  comm_lang_code,
            "mix_commentary":       mix_commentary,
            "output_dir":           final_out,
            "output_filename":      out_filename,
            "individual_clips":     individual,
            "keep_raw_clips":       keep_raw,
            "dry_run":              dry_run,
        }

        lq = queue.Queue(); pq = queue.Queue()
        log_lines = []; last_p = {"current":0,"total":1,"elapsed":0}

        thread = threading.Thread(target=run_clipmaker_f1, args=(config,lq,pq), daemon=True)
        thread.start()

        while thread.is_alive() or not lq.empty():
            while not pq.empty(): last_p = pq.get_nowait()
            updated = False
            while not lq.empty():
                msg = lq.get_nowait()
                if msg["type"]=="log": log_lines.append(msg["msg"]); updated=True

            c=last_p["current"]; t=last_p["total"]; el=last_p["elapsed"]
            frac = c/t if t>0 else 0
            if c>0 and el>0:
                rem=(t-c)/(c/el)
                eta=f"{int(rem//60)}m {int(rem%60):02d}s remaining"
            else: eta="Working..."
            lbl = f"Clip {c} of {t} — {eta}" if c>0 else "Processing pipeline..."

            with progress_ph.container():
                st.markdown(f'<div class="progress-label">{lbl}</div>',unsafe_allow_html=True)
                st.progress(frac)
            if updated:
                log_ph.markdown(
                    f'<div class="log-box">{"<br>".join(log_lines[-80:])}</div>',
                    unsafe_allow_html=True)
            time.sleep(0.3)

        thread.join()
        while not lq.empty():
            msg=lq.get_nowait()
            if msg["type"]=="log": log_lines.append(msg["msg"])
        log_ph.markdown(
            f'<div class="log-box">{"<br>".join(log_lines)}</div>',
            unsafe_allow_html=True)
        progress_ph.empty()

st.markdown(
    '<div class="footer">AI ClipMaker F1 — Full Pipeline | '
    'OpenF1 · librosa · YOLOv8 · Claude API · gTTS | Built on ClipMaker 1.1 by B4L1</div>',
    unsafe_allow_html=True)
