"""
AI ClipMaker F1 — Detection Pipeline
Phase 2: Audio excitement detection (librosa)
Phase 3: Vision action detection (YOLOv8)
"""

import os
import time
import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
# PHASE 2 — AUDIO EXCITEMENT DETECTION
# ─────────────────────────────────────────────────────────────────────────────

def extract_audio_excitement(video_path, log=None, sr=22050, hop_length=512,
                              smoothing_sec=2.0, min_gap_sec=15.0,
                              top_n=30, threshold_percentile=85):
    """
    Analyse the audio track of a race broadcast and return a list of
    candidate excitement moments with scores.

    Returns: list of dicts:
      {type, video_timestamp, label, score, source}
    """
    def _log(msg):
        if log:
            log(msg)

    try:
        import librosa
        import scipy.signal
    except ImportError:
        _log("      [!] librosa not installed — skipping audio detection")
        _log("          Install with: pip install librosa scipy")
        return []

    _log("      Extracting audio from video...")
    try:
        # Use librosa to load audio directly from video via ffmpeg backend
        y, sr_actual = librosa.load(video_path, sr=sr, mono=True)
    except Exception as e:
        _log(f"      [!] Could not load audio: {e}")
        _log("          Make sure FFmpeg is installed and in PATH")
        return []

    duration = len(y) / sr_actual
    _log(f"      Audio duration: {duration:.1f}s  |  Sample rate: {sr_actual}Hz")

    # ── Energy curve ──────────────────────────────────────────────────────────
    # RMS energy per frame
    frame_length = 2048
    rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
    times = librosa.frames_to_time(np.arange(len(rms)), sr=sr_actual, hop_length=hop_length)

    # ── Spectral features ─────────────────────────────────────────────────────
    # Spectral centroid: high centroid = crowd noise dominates (screaming, cheering)
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr_actual, hop_length=hop_length)[0]
    centroid_norm = (centroid - centroid.min()) / (centroid.max() - centroid.min() + 1e-9)

    # Combine: weighted energy + spectral feature
    combined = rms * 0.7 + (rms * centroid_norm) * 0.3

    # ── Smooth the signal ─────────────────────────────────────────────────────
    frames_per_sec = sr_actual / hop_length
    smooth_frames = max(1, int(smoothing_sec * frames_per_sec))
    window = np.ones(smooth_frames) / smooth_frames
    smoothed = np.convolve(combined, window, mode="same")

    # ── Normalise 0–1 ─────────────────────────────────────────────────────────
    smoothed_norm = (smoothed - smoothed.min()) / (smoothed.max() - smoothed.min() + 1e-9)

    # ── Peak detection ────────────────────────────────────────────────────────
    threshold = np.percentile(smoothed_norm, threshold_percentile)
    min_gap_frames = max(1, int(min_gap_sec * frames_per_sec))

    peaks, props = scipy.signal.find_peaks(
        smoothed_norm,
        height=threshold,
        distance=min_gap_frames,
        prominence=0.05,
    )

    _log(f"      Found {len(peaks)} audio excitement peaks (threshold={threshold:.3f})")

    if len(peaks) == 0:
        return []

    # Rank by height, take top N
    peak_scores = smoothed_norm[peaks]
    sorted_idx = np.argsort(peak_scores)[::-1]
    top_peaks = peaks[sorted_idx[:top_n]]
    top_scores = peak_scores[sorted_idx[:top_n]]

    events = []
    for peak, score in zip(top_peaks, top_scores):
        t = float(times[peak])
        # Classify based on signal character
        local_rms = float(rms[peak])
        local_centroid = float(centroid_norm[peak])

        if local_centroid > 0.7:
            label = "Crowd roar / commentary spike"
            ev_type = "audio_crowd"
        elif local_rms > np.percentile(rms, 95):
            label = "Engine / impact noise spike"
            ev_type = "audio_impact"
        else:
            label = "Audio excitement peak"
            ev_type = "audio_excitement"

        events.append({
            "type": ev_type,
            "video_timestamp": t,
            "label": label,
            "score": float(score) * 0.85,  # cap audio score at 0.85
            "source": "audio",
        })

    _log(f"      Top {len(events)} audio moments extracted")
    return events


def get_audio_energy_curve(video_path, sr=22050, hop_length=512, smoothing_sec=2.0):
    """
    Return (times_seconds, energy_values_0_to_1) for plotting the heatmap.
    Returns ([], []) on failure.
    """
    try:
        import librosa
        y, sr_actual = librosa.load(video_path, sr=sr, mono=True)
        frame_length = 2048
        rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
        times = librosa.frames_to_time(np.arange(len(rms)), sr=sr_actual, hop_length=hop_length)
        frames_per_sec = sr_actual / hop_length
        smooth_frames = max(1, int(smoothing_sec * frames_per_sec))
        window = np.ones(smooth_frames) / smooth_frames
        smoothed = np.convolve(rms, window, mode="same")
        smoothed_norm = (smoothed - smoothed.min()) / (smoothed.max() - smoothed.min() + 1e-9)
        # Downsample to ~1 point per second for plotting
        step = max(1, len(times) // 3000)
        return times[::step].tolist(), smoothed_norm[::step].tolist()
    except Exception:
        return [], []


# ─────────────────────────────────────────────────────────────────────────────
# PHASE 3 — VISION ACTION DETECTION (YOLOv8)
# ─────────────────────────────────────────────────────────────────────────────

# COCO class IDs relevant to F1
F1_RELEVANT_CLASSES = {
    2: "car",
    3: "motorcycle",   # can pick up F1 cars
    5: "bus",          # safety car (large vehicle)
    7: "truck",
    0: "person",       # marshals, pit crew
}

# How close two cars need to be (as fraction of frame width) to flag proximity
PROXIMITY_THRESHOLD = 0.25


def sample_frames_around_timestamp(video_path, timestamp_sec, ffmpeg_bin,
                                    n_frames=5, spread_sec=3):
    """
    Extract n_frames around a timestamp from the video.
    Returns list of numpy arrays (BGR frames).
    """
    import subprocess
    import tempfile

    frames = []
    offsets = np.linspace(-spread_sec, spread_sec, n_frames)

    with tempfile.TemporaryDirectory() as tmp_dir:
        for i, offset in enumerate(offsets):
            t = max(0, timestamp_sec + offset)
            out_path = os.path.join(tmp_dir, f"frame_{i:03d}.jpg")
            cmd = [
                ffmpeg_bin, "-y",
                "-ss", str(t),
                "-i", video_path,
                "-frames:v", "1",
                "-q:v", "3",
                out_path
            ]
            result = subprocess.run(cmd, capture_output=True)
            if result.returncode == 0 and os.path.exists(out_path):
                try:
                    import cv2
                    frame = cv2.imread(out_path)
                    if frame is not None:
                        frames.append(frame)
                except ImportError:
                    break
    return frames


def score_frame_for_f1_action(frame, model):
    """
    Run YOLOv8 on a frame. Return (score_boost, description).
    score_boost: float 0.0 – 0.4 added to base excitement score
    description: human-readable label of what was detected
    """
    try:
        import cv2
        h, w = frame.shape[:2]
        results = model(frame, verbose=False)[0]
        boxes = results.boxes

        if boxes is None or len(boxes) == 0:
            return 0.0, ""

        # Find car-like detections
        car_boxes = []
        person_boxes = []
        for box in boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            if conf < 0.3:
                continue
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2
            bw = x2 - x1
            bh = y2 - y1

            if cls_id in (2, 3, 7):  # car / motorcycle / truck
                car_boxes.append({"cx": cx, "cy": cy, "w": bw, "h": bh, "conf": conf})
            elif cls_id == 0:  # person (pit crew, marshals)
                person_boxes.append({"cx": cx, "cy": cy, "conf": conf})

        score_boost = 0.0
        descriptions = []

        # ── Car proximity (overtake / battle) ─────────────────────────────
        if len(car_boxes) >= 2:
            for i in range(len(car_boxes)):
                for j in range(i + 1, len(car_boxes)):
                    a, b = car_boxes[i], car_boxes[j]
                    dist = abs(a["cx"] - b["cx"]) / w
                    if dist < PROXIMITY_THRESHOLD:
                        score_boost = max(score_boost, 0.30)
                        descriptions.append("Cars side-by-side (overtake/battle)")

        # ── Many people on track (pit lane activity / incident) ────────────
        if len(person_boxes) >= 3:
            score_boost = max(score_boost, 0.20)
            descriptions.append(f"Pit crew / marshal activity ({len(person_boxes)} people)")

        # ── Single car — baseline confirmation ────────────────────────────
        if car_boxes and score_boost == 0.0:
            score_boost = 0.05
            descriptions.append(f"Car detected (conf={car_boxes[0]['conf']:.2f})")

        return score_boost, " | ".join(descriptions)

    except Exception:
        return 0.0, ""


def run_vision_detection(candidate_events, video_path, ffmpeg_bin, log=None,
                          model_size="n", confidence=0.3):
    """
    For each candidate event, sample frames and run YOLOv8.
    Updates each event's score and adds vision_description field.

    Returns updated events list.
    """
    def _log(msg):
        if log:
            log(msg)

    try:
        from ultralytics import YOLO
        import cv2
    except ImportError:
        _log("      [!] ultralytics / opencv not installed — skipping vision detection")
        _log("          Install with: pip install ultralytics opencv-python")
        return candidate_events

    _log(f"      Loading YOLOv8{model_size} model...")
    try:
        model = YOLO(f"yolov8{model_size}.pt")
        model.conf = confidence
    except Exception as e:
        _log(f"      [!] Could not load YOLO model: {e}")
        return candidate_events

    _log(f"      Analysing {len(candidate_events)} events with YOLOv8...")
    updated = []

    for i, ev in enumerate(candidate_events):
        ts = ev.get("video_timestamp", 0)
        frames = sample_frames_around_timestamp(video_path, ts, ffmpeg_bin)

        best_boost = 0.0
        best_desc = ""

        for frame in frames:
            boost, desc = score_frame_for_f1_action(frame, model)
            if boost > best_boost:
                best_boost = boost
                best_desc = desc

        new_score = min(1.0, ev.get("score", 0.5) + best_boost)
        ev = dict(ev)
        ev["score"] = new_score
        ev["vision_description"] = best_desc
        if best_desc:
            ev["label"] = ev.get("label", "") + f" [{best_desc}]"
        updated.append(ev)

        if (i + 1) % 5 == 0 or (i + 1) == len(candidate_events):
            _log(f"      Vision: {i+1}/{len(candidate_events)} events processed")

    _log(f"      Vision detection complete")
    return updated


# ─────────────────────────────────────────────────────────────────────────────
# SIGNAL FUSION
# ─────────────────────────────────────────────────────────────────────────────

def fuse_signals(openf1_events, audio_events, merge_window_sec=20.0):
    """
    Merge OpenF1 events and audio events into a unified list.
    Events within merge_window_sec of each other are considered the same moment;
    their scores are combined (with bonus for multi-signal agreement).

    Returns sorted list of merged events.
    """
    all_events = list(openf1_events) + list(audio_events)
    if not all_events:
        return []

    all_events.sort(key=lambda e: e.get("video_timestamp", 0))

    merged = []
    used = [False] * len(all_events)

    for i, ev in enumerate(all_events):
        if used[i]:
            continue
        group = [ev]
        used[i] = True
        t_i = ev.get("video_timestamp", 0)

        for j in range(i + 1, len(all_events)):
            if used[j]:
                continue
            t_j = all_events[j].get("video_timestamp", 0)
            if abs(t_j - t_i) <= merge_window_sec:
                group.append(all_events[j])
                used[j] = True

        # Combine group into one event
        if len(group) == 1:
            merged.append(group[0])
        else:
            # Multi-signal agreement bonus
            sources = set(e.get("source", "openf1") for e in group)
            base_score = max(e.get("score", 0) for e in group)
            agreement_bonus = 0.10 * (len(sources) - 1)
            combined_score = min(1.0, base_score + agreement_bonus)

            labels = list(dict.fromkeys(e.get("label", "") for e in group if e.get("label")))
            primary = max(group, key=lambda e: e.get("score", 0))

            merged_ev = dict(primary)
            merged_ev["score"] = combined_score
            merged_ev["label"] = " + ".join(labels[:2])  # keep it short
            merged_ev["sources"] = list(sources)
            merged.append(merged_ev)

    merged.sort(key=lambda e: e.get("video_timestamp", 0))
    return merged
