# stage_a.py — find BlueStacks window, grab a frame from its region, save it.
import dxcam, cv2, numpy as np, time, sys
import pygetwindow as gw

# ---------- find the BlueStacks window ----------
def find_game_window():
    hits = []
    for w in gw.getAllWindows():
        t = (w.title or "").strip()
        if "King of Thieves" in t or "BlueStacks" in t:
            hits.append(w)
    if not hits:
        return None
    # prefer the one with the most specific title
    hits.sort(key=lambda w: len(w.title or ""), reverse=True)
    return hits[0]

win = find_game_window()
if win is None:
    print("ERROR: no BlueStacks / King of Thieves window found.")
    print("Open BlueStacks with the game running, then re-run.")
    sys.exit(1)

print("Windows found:")
for w in gw.getAllWindows():
    t = (w.title or "").strip()
    if "King of Thieves" in t or "BlueStacks" in t:
        print(f"  title={t!r}  pos=({w.left},{w.top})  size={w.width}x{w.height}  "
              f"visible={w.visible}  minimized={w.isMinimized}")

print()
print(f"chosen: {win.title!r}")
print(f"  left={win.left}  top={win.top}  width={win.width}  height={win.height}")

# ---------- capture that region ----------
LEFT, TOP, WIDTH, HEIGHT = win.left, win.top, win.width, win.height

cam = dxcam.create(output_color="BGR", device_idx=0)
# DXcam region format: (left, top, right, bottom)
region = (LEFT, TOP, LEFT + WIDTH, TOP + HEIGHT)
print(f"\nDXcam region: {region}")

cam.start(region=region, target_fps=60, video_mode=True)
print("warming up...")
time.sleep(1.0)

frame = cam.get_latest_frame()
cam.stop()

if frame is None:
    print("ERROR: DXcam returned no frame. Falling back to full-screen grab.")
    cam = dxcam.create(output_color="BGR")
    cam.start(target_fps=60, video_mode=True)
    time.sleep(0.5)
    full = cam.get_latest_frame()
    cam.stop()
    if full is None:
        print("ERROR: even fullscreen grab failed.")
        sys.exit(1)
    # crop manually
    frame = full[TOP:TOP+HEIGHT, LEFT:LEFT+WIDTH]
    print(f"cropped fullscreen -> {frame.shape}")

print(f"frame shape: {frame.shape}")

# annotate with a crosshair at the center so we can confirm orientation
h, w = frame.shape[:2]
vis = frame.copy()
cv2.line(vis, (w//2, 0), (w//2, h), (0, 255, 0), 2)
cv2.line(vis, (0, h//2), (w, h//2), (0, 255, 0), 2)
cv2.putText(vis, f"{w}x{h}", (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

out = r"C:\Users\asad\stage_a_capture.png"
cv2.imwrite(out, vis)
print(f"\nsaved {out}")
print("Open that file and confirm it shows the King of Thieves game.")