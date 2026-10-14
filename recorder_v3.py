# recorder_v3.py -- King of Thieves recorder
# Tracks thief by motion + position (not size), writes a summary on F10.

import dxcam, cv2, numpy as np, time, json, sys, os, zipfile
import pygetwindow as gw
import keyboard
from pathlib import Path
from datetime import datetime
from collections import Counter

OUT_ROOT = Path(r"C:\Users\asad\runs")
OUT_ROOT.mkdir(exist_ok=True)
WINDOW_KEY = "King of Thieves"
FPS = 60

# ---- perception ----
DARK_THRESHOLD = 50
MIN_BLOB_AREA  = 300
MAX_BLOB_AREA  = 6000
ASPECT_MIN, ASPECT_MAX = 0.4, 2.5

# ---- motion / continuity ----
MAX_JUMP_PX_PER_FRAME = 80
MAX_LIFTOFF_DY        = 60
LOST_FRAMES_BEFORE_RESET = 30

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
        out.append({"x": x + ww // 2, "y": y + hh + y_off,
                    "w": ww, "h": hh, "area": area})
    return out

class ThiefTracker:
    """Tracks the thief by preferring MOTION over size.
    Blobs that stay still are treated as decoration."""

    def __init__(self):
        self.static_positions = []   # list of (x, y, count)
        self.last = None
        self.lost = 0

    def _dist(self, a, b):
        return ((a[0]-b[0])**2 + (a[1]-b[1])**2) ** 0.5

    def _is_static_position(self, blob):
        for (sx, sy, count) in self.static_positions:
            if self._dist((blob["x"], blob["y"]), (sx, sy)) < 25 and count > 15:
                return True
        return False

    def update(self, blobs):
        # register positions
        for b in blobs:
            found = False
            for i, (sx, sy, count) in enumerate(self.static_positions):
                if self._dist((b["x"], b["y"]), (sx, sy)) < 25:
                    self.static_positions[i] = (sx, sy, count + 1)
                    found = True
                    break
            if not found:
                self.static_positions.append((b["x"], b["y"], 1))

        moving = [b for b in blobs if not self._is_static_position(b)]

        if self.last is not None:
            candidates = moving if moving else blobs
            if not candidates:
                self.lost += 1
                if self.lost > LOST_FRAMES_BEFORE_RESET:
                    self.last = None
                    self.lost = 0
                return None

            def score(b):
                d = self._dist((b["x"], b["y"]),
                               (self.last["x"], self.last["y"]))
                if d > MAX_JUMP_PX_PER_FRAME * 2:
                    return 1e9
                return d - b["y"] * 0.5

            scored = [(score(b), b) for b in candidates]
            best_score, best = min(scored, key=lambda p: p[0])
            if best_score > 1e8:
                best = None
            if best is not None:
                self.last = best
                self.lost = 0
                return best
            self.lost += 1
            if self.lost > LOST_FRAMES_BEFORE_RESET:
                self.last = None
                self.lost = 0
            return None
        else:
            if moving:
                best = max(moving, key=lambda b: b["y"])
                self.last = best
                return best
            return None

def main():
    win = find_game_window()
    if not win:
        print("ERROR: King of Thieves window not found / minimized.")
        sys.exit(1)
    print(f"[window] {win.title}  ({win.left},{win.top}) {win.width}x{win.height}")

    cap_l = win.left
    cap_t = win.top
    cap_r = win.left + int(win.width * 0.95)
    cap_b = win.top + int(win.height * 0.92)
    print(f"[dxcam]  region = ({cap_l},{cap_t}) -> ({cap_r},{cap_b})")

    cam = dxcam.create(output_color="BGR", device_idx=0)
    cam.start(region=(cap_l, cap_t, cap_r, cap_b),
              target_fps=FPS, video_mode=True)
    time.sleep(1.0)
    f = cam.get_latest_frame()
    if f is None:
        print("ERROR: no frames from DXcam.")
        sys.exit(1)
    print(f"[dxcam]  frame shape: {f.shape}")

    session_dir = OUT_ROOT / datetime.now().strftime("%Y%m%d_%H%M%S")
    session_dir.mkdir(parents=True, exist_ok=True)
    frames_dir = session_dir / "frames"
    frames_dir.mkdir(exist_ok=True)
    log_file = open(session_dir / "events.jsonl", "w", buffering=1)

    stats = {"frames_total": 0, "frames_with_thief": 0,
             "detector_lost_events": 0, "liftoffs_logged": 0,
             "liftoffs_rejected_teleport": 0, "runs": 0,
             "position_histogram": Counter()}

    tracker = ThiefTracker()
    state = {"recording": False, "run_index": 0,
             "prev_y": None, "thief_state": "unknown"}

    def log_event(kind, **kw):
        log_file.write(json.dumps({"t": round(time.time(), 4),
                                   "kind": kind, **kw}) + "\n")

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

    def write_summary():
        s = session_dir / "summary.txt"
        with open(s, "w") as fp:
            fp.write(f"session: {session_dir.name}\n")
            fp.write(f"frames_total: {stats['frames_total']}\n")
            fp.write(f"frames_with_thief: {stats['frames_with_thief']} "
                     f"({100*stats['frames_with_thief']/max(stats['frames_total'],1):.1f}%)\n")
            fp.write(f"detector_lost_events: {stats['detector_lost_events']}\n")
            fp.write(f"liftoffs_logged: {stats['liftoffs_logged']}\n")
            fp.write(f"liftoffs_rejected_teleport: {stats['liftoffs_rejected_teleport']}\n")
            fp.write(f"runs: {stats['runs']}\n")
            fp.write(f"distinct_x_positions (50px buckets): {len(stats['position_histogram'])}\n")
            fp.write("\ntop 10 x-positions (x_bucket: count):\n")
            for x, c in stats["position_histogram"].most_common(10):
                fp.write(f"  x~{x*50:4d}: {c}\n")
        print(f"[summary] {s}")

    def dump_bundle():
        out = OUT_ROOT / f"bundle_{int(time.time())}.zip"
        with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
            for x in session_dir.rglob("*"):
                if x.is_file():
                    z.write(x, x.relative_to(OUT_ROOT))
        print(f"[bundle] {out}  ({out.stat().st_size // 1024} KB)")

    keyboard.add_hotkey("F9", toggle)
    keyboard.add_hotkey("F10", lambda: (write_summary(), dump_bundle()))
    keyboard.add_hotkey("esc", lambda: (write_summary(), os._exit(0)))

    print("\nHotkeys:  F9 = toggle record   F10 = summary+bundle   Esc = quit")
    print("BOOTSTRAP: let it watch for ~2 seconds before pressing F9.\n")

    frame_idx = 0
    while True:
        frame = cam.get_latest_frame()
        if frame is None:
            time.sleep(0.005)
            continue
        stats["frames_total"] += 1

        blobs = find_blobs(frame)
        thief = tracker.update(blobs)

        vis = frame.copy()
        for b in blobs:
            cv2.rectangle(vis,
                          (b["x"]-b["w"]//2, b["y"]-b["h"]),
                          (b["x"]+b["w"]//2, b["y"]),
                          (0, 100, 100), 1)
        if thief:
            stats["frames_with_thief"] += 1
            stats["position_histogram"][thief["x"] // 50] += 1
            cv2.rectangle(vis,
                          (thief["x"]-thief["w"]//2, thief["y"]-thief["h"]),
                          (thief["x"]+thief["w"]//2, thief["y"]),
                          (0, 255, 0), 3)
            cv2.putText(vis, f"THIEF ({thief['x']},{thief['y']})",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (0, 255, 0), 2)
        else:
            cv2.putText(vis, "searching...", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 180, 255), 2)

        cv2.putText(vis,
                    f"{'REC' if state['recording'] else 'idle'}  run={state['run_index']}",
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (0, 0, 255) if state["recording"] else (128, 128, 128), 2)

        if state["recording"]:
            log_event("frame", frame=frame_idx, thief=thief,
                      state=state["thief_state"])
            if thief and state["prev_y"] is not None:
                dy = state["prev_y"] - thief["y"]
                if abs(dy) > MAX_LIFTOFF_DY:
                    stats["liftoffs_rejected_teleport"] += 1
                else:
                    if dy < -3 and state["thief_state"] != "airborne":
                        state["thief_state"] = "airborne"
                        stats["liftoffs_logged"] += 1
                        log_event("liftoff", frame=frame_idx,
                                  thief=thief, dy=dy)
                    elif dy > 3 and state["thief_state"] == "airborne":
                        state["thief_state"] = "grounded"
                        log_event("landing", frame=frame_idx,
                                  thief=thief, dy=dy)
                    elif abs(dy) <= 3:
                        state["thief_state"] = "grounded"
            if thief:
                state["prev_y"] = thief["y"]
                if frame_idx % 2 == 0:
                    cv2.imwrite(
                        str(frames_dir / f"r{state['run_index']:02d}_{frame_idx:06d}.jpg"),
                        vis, [cv2.IMWRITE_JPEG_QUALITY, 70])

        if frame_idx % 3 == 0:
            small = cv2.resize(vis, (vis.shape[1] // 2, vis.shape[0] // 2))
            cv2.imshow("KOT recorder v3", small)
            if cv2.waitKey(1) & 0xFF == 27:
                break
        frame_idx += 1

    cam.stop()
    log_file.close()
    write_summary()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()