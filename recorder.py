# recorder.py — Phase 1
# Records your manual play: thief position each frame, jump events, JPEG frames.
# RUN:    python recorder.py
# USE:    Bring the game to the foreground. Press F9 to start a run.
#         Play one full run (get to the chest OR die). Press F9 to stop.
#         Repeat for 5-10 runs. Press F10 to bundle everything.
#         Esc to quit.

import dxcam, cv2, numpy as np, time, json, sys, os, threading, zipfile
import pygetwindow as gw
import keyboard
from pathlib import Path
from datetime import datetime

# ---------------- config ----------------
OUT_ROOT = Path(r"C:\Users\asad\runs")
OUT_ROOT.mkdir(exist_ok=True)

WINDOW_KEY = "King of Thieves"      # substring match on window title
FPS = 60

# thief detection tuning (adjust after we see the footage)
DARK_THRESHOLD = 60          # grayscale < this = candidate dark pixel
MIN_BLOB_AREA = 80
MAX_BLOB_AREA = 4000
ASPECT_MIN, ASPECT_MAX = 0.6, 1.6

# jump detection
JUMP_DY_PX = 3               # px change in thief y to count as motion
STILL_FRAMES = 4             # frames of <JUMP_DY to be considered grounded

# ---------------- window finder ----------------
def find_game_window():
    hits = [w for w in gw.getAllWindows()
            if WINDOW_KEY.lower() in (w.title or "").lower()
            and not w.isMinimized
            and w.width > 400]
    if not hits:
        return None
    hits.sort(key=lambda w: w.width * w.height, reverse=True)
    return hits[0]

# ---------------- thief detector ----------------
def find_thief(frame):
    """Return (x, y, w, h) of the thief, or None.
       Thief = darkest plausible blob in lower 2/3 of the frame."""
    h, w = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    roi = gray[int(h * 0.33):h, :]
    _, mask = cv2.threshold(roi, DARK_THRESHOLD, 255, cv2.THRESH_BINARY_INV)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best = None
    best_area = 0
    y_offset = int(h * 0.33)
    for c in contours:
        x, y, ww, hh = cv2.boundingRect(c)
        area = ww * hh
        if not (MIN_BLOB_AREA < area < MAX_BLOB_AREA):
            continue
        aspect = ww / max(hh, 1)
        if not (ASPECT_MIN < aspect < ASPECT_MAX):
            continue
        if area > best_area:
            best_area = area
            best = (x + ww // 2, y + hh + y_offset, ww, hh)
    return best

# ---------------- main ----------------
def main():
    win = find_game_window()
    if not win:
        print("ERROR: King of Thieves window not found / minimized.")
        print("Bring the game to the foreground and re-run.")
        sys.exit(1)

    print(f"[window] {win.title}")
    print(f"         pos=({win.left},{win.top}) size={win.width}x{win.height}")

    region = (win.left, win.top, win.left + win.width, win.top + win.height)
    print(f"[dxcam]  region={region}")

    cam = dxcam.create(output_color="BGR", device_idx=0)
    cam.start(region=region, target_fps=FPS, video_mode=True)
    time.sleep(1.0)  # warm up

    if cam.get_latest_frame() is None:
        print("ERROR: no frames from DXcam region capture.")
        cam.stop()
        sys.exit(1)

    session_dir = OUT_ROOT / datetime.now().strftime("%Y%m%d_%H%M%S")
    session_dir.mkdir(parents=True, exist_ok=True)
    frames_dir = session_dir / "frames"
    frames_dir.mkdir(exist_ok=True)
    log_path = session_dir / "events.jsonl"
    log_file = open(log_path, "w", buffering=1)
    print(f"[log]    {session_dir}")

    state = {"recording": False, "run_index": 0,
             "prev_y": None, "motion_hist": [],
             "thief_state": "unknown"}

    def toggle():
        if not state["recording"]:
            state["recording"] = True
            state["run_index"] += 1
            state["prev_y"] = None
            state["motion_hist"] = []
            state["thief_state"] = "unknown"
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
        print(f"         attach THIS file in the chat.\n")

    def log_event(kind, **kw):
        rec = {"t": round(time.time(), 4), "kind": kind, **kw}
        log_file.write(json.dumps(rec) + "\n")

    keyboard.add_hotkey("F9", toggle)
    keyboard.add_hotkey("F10", dump_bundle)
    keyboard.add_hotkey("esc", lambda: os._exit(0))

    print("\nHotkeys:  F9 = toggle record   F10 = bundle   Esc = quit")
    print("Bring game to foreground. Press F9, play a run, press F9.\n")

    frame_idx = 0
    t_loop_start = time.time()

    while True:
        frame = cam.get_latest_frame()
        if frame is None:
            time.sleep(0.005)
            continue

        thief = find_thief(frame)
        t_now = time.time()

        vis = frame.copy()
        if thief:
            x, y, w, h = thief
            cv2.rectangle(vis, (x - w // 2, y - h), (x + w // 2, y), (0, 255, 0), 2)
            cv2.putText(vis, f"thief ({x},{y})", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # jump state machine
        if thief and state["recording"]:
            y = thief[1]
            if state["prev_y"] is not None:
                dy = y - state["prev_y"]
                state["motion_hist"].append(dy)
                if len(state["motion_hist"]) > 8:
                    state["motion_hist"].pop(0)

                moved = abs(dy) >= JUMP_DY_PX
                if moved:
                    if dy < 0 and state["thief_state"] != "airborne":
                        state["thief_state"] = "airborne"
                        log_event("liftoff", frame=frame_idx,
                                  thief=thief, dy=dy)
                    elif dy > 0 and state["thief_state"] == "airborne":
                        state["thief_state"] = "grounded"
                        log_event("landing", frame=frame_idx,
                                  thief=thief, dy=dy)
                else:
                    if (state["thief_state"] == "unknown"
                            or (len(state["motion_hist"]) >= STILL_FRAMES
                                and all(abs(d) < JUMP_DY_PX
                                        for d in state["motion_hist"][-STILL_FRAMES:]))):
                        if state["thief_state"] != "grounded":
                            state["thief_state"] = "grounded"

            log_event("frame", frame=frame_idx, t=round(t_now, 4),
                      thief=thief if thief else None,
                      state=state["thief_state"])

            if frame_idx % 2 == 0:  # save 30fps JPEGs (half rate to save disk)
                jpg_path = frames_dir / f"r{state['run_index']:02d}_{frame_idx:06d}.jpg"
                cv2.imwrite(str(jpg_path), vis,
                            [cv2.IMWRITE_JPEG_QUALITY, 70])

            state["prev_y"] = y

        if not state["recording"]:
            # idle view: still show something
            cv2.putText(vis, "IDLE - press F9 to record a run",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (0, 0, 255), 2)

        # live preview window (small)
        if frame_idx % 3 == 0:
            small = cv2.resize(vis, (vis.shape[1] // 2, vis.shape[0] // 2))
            cv2.imshow("KOT recorder", small)
            if cv2.waitKey(1) & 0xFF == 27:
                break

        frame_idx += 1
        # pace to ~60fps
        target = t_loop_start + frame_idx / FPS
        dt = target - time.time()
        if dt > 0:
            time.sleep(dt)

    cam.stop()
    log_file.close()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()