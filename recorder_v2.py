# recorder_v2.py
# Changes from v1:
#   - MIN_BLOB_AREA raised 80 -> 1200 (kills small false positives)
#   - DARK_THRESHOLD lowered 60 -> 45 (stricter darkness)
#   - added continuity tracking: thief cannot teleport
#   - MAX_JUMP_DY = 60: reject impossible liftoffs
#   - logs a "detector_lost" event when it can't find the thief for N frames
#   - writes a summary.txt at the end of the session

import dxcam, cv2, numpy as np, time, json, sys, os, threading, zipfile
import pygetwindow as gw
import keyboard
from pathlib import Path
from datetime import datetime
from collections import Counter

OUT_ROOT = Path(r"C:\Users\asad\runs")
OUT_ROOT.mkdir(exist_ok=True)

WINDOW_KEY = "King of Thieves"
FPS = 60

# ---- perception tuning ----
DARK_THRESHOLD   = 45      # was 60
MIN_BLOB_AREA    = 1200    # was 80
MAX_BLOB_AREA    = 6000
ASPECT_MIN, ASPECT_MAX = 0.5, 2.0

# ---- motion / continuity ----
MAX_JUMP_PX_PER_FRAME = 60      # thief can't move more than this in 16ms
MAX_LIFTOFF_DY        = 60      # reject "jumps" bigger than this
LOST_FRAMES_BEFORE_RESET = 30   # if we lose the thief for this long, drop continuity

def find_game_window():
    hits = [w for w in gw.getAllWindows()
            if WINDOW_KEY.lower() in (w.title or "").lower()
            and not w.isMinimized and w.width > 400]
    return max(hits, key=lambda w: w.width * w.height) if hits else None

def find_blobs(frame):
    h, w = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    roi = gray[int(h * 0.33):h, :]
    _, mask = cv2.threshold(roi, DARK_THRESHOLD, 255, cv2.THRESH_BINARY_INV)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    y_off = int(h * 0.33)
    out = []
    for c in contours:
        x, y, ww, hh = cv2.boundingRect(c)
        area = ww * hh
        if not (MIN_BLOB_AREA < area < MAX_BLOB_AREA):
            continue
        aspect = ww / max(hh, 1)
        if not (ASPECT_MIN < aspect < ASPECT_MAX):
            continue
        out.append({"x": x + ww // 2, "y": y + hh + y_off, "w": ww, "h": hh, "area": area})
    return out

def pick_thief(blobs, last):
    """Pick the blob most consistent with the last known thief position."""
    if not blobs:
        return None
    if last is None:
        return max(blobs, key=lambda b: b["area"])
    lx, ly = last["x"], last["y"]
    def score(b):
        d = ((b["x"] - lx) ** 2 + (b["y"] - ly) ** 2) ** 0.5
        # reject teleports
        if d > MAX_JUMP_PX_PER_FRAME * 2:
            return 1e9
        # prefer close AND large
        return d - b["area"] * 0.01
    return min(blobs, key=score)

def main():
    win = find_game_window()
    if not win:
        print("ERROR: King of Thieves window not found / minimized.")
        sys.exit(1)
    print(f"[window] {win.title}  ({win.left},{win.top}) {win.width}x{win.height}")

    cam = dxcam.create(output_color="BGR", device_idx=0)
    cam.start(region=(win.left, win.top, win.left + win.width, win.top + win.height),
              target_fps=FPS, video_mode=True)
    time.sleep(1.0)
    if cam.get_latest_frame() is None:
        print("ERROR: no frames from DXcam.")
        sys.exit(1)

    session_dir = OUT_ROOT / datetime.now().strftime("%Y%m%d_%H%M%S")
    session_dir.mkdir(parents=True, exist_ok=True)
    frames_dir = session_dir / "frames"; frames_dir.mkdir(exist_ok=True)
    log_file = open(session_dir / "events.jsonl", "w", buffering=1)

    # stats we accumulate and write to summary.txt at the end
    stats = {
        "frames_total": 0,
        "frames_with_thief": 0,
        "detector_lost_events": 0,
        "liftoffs_logged": 0,
        "liftoffs_rejected_teleport": 0,
        "runs": 0,
        "position_histogram": Counter(),   # x rounded to 50px bucket
        "distinct_x_positions": set(),
    }

    state = {"recording": False, "run_index": 0,
             "prev_y": None, "thief_state": "unknown",
             "last_thief": None, "lost_frames": 0}

    def log_event(kind, **kw):
        log_file.write(json.dumps({"t": round(time.time(), 4), "kind": kind, **kw}) + "\n")

    def toggle():
        if not state["recording"]:
            state["recording"] = True
            state["run_index"] += 1
            state["prev_y"] = None
            state["thief_state"] = "unknown"
            stats["runs"] += 1
            print(f"\n>>> REC ON  (run {state['run_index']})")
            log_event("run_start", run=state["run_index"])
        else:
            state["recording"] = False
            print(f"<<< REC OFF (run {state['run_index']})")
            log_event("run_end", run=state["run_index"])

    def dump_bundle():
        out = OUT_ROOT / f"bundle_{int(time.time())}.zip"
        with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
            for f in session_dir.rglob("*"):
                if f.is_file():
                    z.write(f, f.relative_to(OUT_ROOT))
        print(f"\n[bundle] {out}  ({out.stat().st_size // 1024} KB)")

    def write_summary():
        s = session_dir / "summary.txt"
        with open(s, "w") as f:
            f.write(f"session: {session_dir.name}\n")
            f.write(f"frames_total: {stats['frames_total']}\n")
            f.write(f"frames_with_thief: {stats['frames_with_thief']} "
                    f"({100*stats['frames_with_thief']/max(stats['frames_total'],1):.1f}%)\n")
            f.write(f"detector_lost_events: {stats['detector_lost_events']}\n")
            f.write(f"liftoffs_logged: {stats['liftoffs_logged']}\n")
            f.write(f"liftoffs_rejected_teleport: {stats['liftoffs_rejected_teleport']}\n")
            f.write(f"runs: {stats['runs']}\n")
            f.write(f"distinct_x_positions (50px buckets): {len(stats['position_histogram'])}\n")
            f.write("\ntop 10 x-positions (x_bucket: count):\n")
            for x, c in stats["position_histogram"].most_common(10):
                f.write(f"  x~{x*50:4d}: {c}\n")
        print(f"[summary] {s}")

    keyboard.add_hotkey("F9", toggle)
    keyboard.add_hotkey("F10", lambda: (write_summary(), dump_bundle()))
    keyboard.add_hotkey("esc", lambda: (write_summary(), os._exit(0)))

    print("\nHotkeys:  F9 = toggle record   F10 = summary+bundle   Esc = quit\n")

    frame_idx = 0
    while True:
        frame = cam.get_latest_frame()
        if frame is None:
            time.sleep(0.005); continue
        stats["frames_total"] += 1

        blobs = find_blobs(frame)
        thief = pick_thief(blobs, state["last_thief"])

        vis = frame.copy()
        for b in blobs:
            color = (0, 200, 0) if (thief and b is thief) else (0, 100, 100)
            cv2.rectangle(vis, (b["x"]-b["w"]//2, b["y"]-b["h"]),
                          (b["x"]+b["w"]//2, b["y"]), color, 1)
        if thief:
            stats["frames_with_thief"] += 1
            stats["position_histogram"][thief["x"] // 50] += 1
            stats["distinct_x_positions"].add(thief["x"] // 50)
            cv2.putText(vis, f"THIEF ({thief['x']},{thief['y']})", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            state["lost_frames"] = 0
        else:
            state["lost_frames"] += 1
            if state["lost_frames"] == LOST_FRAMES_BEFORE_RESET:
                stats["detector_lost_events"] += 1
                state["last_thief"] = None
                log_event("detector_lost", frame=frame_idx)
            cv2.putText(vis, "THIEF LOST", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        cv2.putText(vis, f"{'REC' if state['recording'] else 'idle'}  run={state['run_index']}",
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (0, 0, 255) if state["recording"] else (128, 128, 128), 2)

        if state["recording"]:
            log_event("frame", frame=frame_idx,
                      thief=thief if thief else None,
                      state=state["thief_state"])

            if thief and state["prev_y"] is not None:
                dy = state["prev_y"] - thief["y"]
                if abs(dy) > MAX_LIFTOFF_DY:
                    stats["liftoffs_rejected_teleport"] += 1
                else:
                    if dy < -3 and state["thief_state"] != "airborne":
                        state["thief_state"] = "airborne"
                        stats["liftoffs_logged"] += 1
                        log_event("liftoff", frame=frame_idx, thief=thief, dy=dy)
                    elif dy > 3 and state["thief_state"] == "airborne":
                        state["thief_state"] = "grounded"
                        log_event("landing", frame=frame_idx, thief=thief, dy=dy)
                    elif abs(dy) <= 3:
                        state["thief_state"] = "grounded"

            if thief:
                state["prev_y"] = thief["y"]
                state["last_thief"] = thief
                if frame_idx % 2 == 0:
                    cv2.imwrite(str(frames_dir / f"r{state['run_index']:02d}_{frame_idx:06d}.jpg"),
                                vis, [cv2.IMWRITE_JPEG_QUALITY, 70])

        if frame_idx % 3 == 0:
            small = cv2.resize(vis, (vis.shape[1] // 2, vis.shape[0] // 2))
            cv2.imshow("KOT recorder v2", small)
            if cv2.waitKey(1) & 0xFF == 27:
                break
        frame_idx += 1

    cam.stop(); log_file.close(); write_summary(); cv2.destroyAllWindows()

if __name__ == "__main__":
    main()