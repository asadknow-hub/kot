# diagnose.py — figure out what we can actually capture
import dxcam, cv2, numpy as np, time
import pygetwindow as gw

print("=" * 60)
print("1. Windows with game-ish titles:")
print("=" * 60)
for w in gw.getAllWindows():
    t = (w.title or "").strip()
    if not t:
        continue
    if any(k.lower() in t.lower() for k in
           ["king", "thief", "google", "play", "bluestacks", "emulator"]):
        print(f"  {t!r}")
        print(f"      pos=({w.left},{w.top})  size={w.width}x{w.height}  "
              f"visible={w.visible}  min={w.isMinimized}")

print()
print("=" * 60)
print("2. What DXcam sees fullscreen:")
print("=" * 60)

cam = dxcam.create(output_color="BGR")
print(f"  device: {cam.device_info if hasattr(cam, 'device_info') else 'unknown'}")

cam.start(target_fps=60, video_mode=True)
time.sleep(0.8)
frames = []
for i in range(10):
    f = cam.get_latest_frame()
    if f is not None:
        frames.append(f)
    time.sleep(0.05)
cam.stop()

if not frames:
    print("  ERROR: no frames at all")
else:
    f = frames[0]
    print(f"  shape: {f.shape}")
    print(f"  dtype: {f.dtype}")
    print(f"  mean brightness: {f.mean():.1f}  (near 0 = black/blank)")
    print(f"  std: {f.std():.1f}  (near 0 = uniform color)")
    cv2.imwrite(r"C:\Users\asad\diag_fullscreen.png", f)
    print(f"  saved diag_fullscreen.png")

print()
print("=" * 60)
print("3. What mss sees fullscreen (for comparison):")
print("=" * 60)
import mss
from PIL import Image
with mss.mss() as sct:
    shot = sct.grab(sct.monitors[1])
    img = np.array(Image.frombytes("RGB", shot.size, shot.bgra, "raw", "BGRX"))
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    print(f"  shape: {img.shape}")
    print(f"  mean brightness: {img.mean():.1f}")
    cv2.imwrite(r"C:\Users\asad\diag_mss.png", img)
    print(f"  saved diag_mss.png")

print()
print("DONE.")