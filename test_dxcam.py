import dxcam, time, numpy as np, cv2

cam = dxcam.create(output_color="BGR")
print("DXcam created. Starting background capture...")

cam.start(target_fps=60, video_mode=True)
time.sleep(1.0)  # let it warm up

# measure get_latest_frame speed
t0 = time.perf_counter()
frames = 0
for _ in range(60):
    f = cam.get_latest_frame()
    if f is not None:
        frames += 1
dt = (time.perf_counter() - t0) / 60 * 1000
print(f"get_latest_frame: {dt:.2f} ms/call, got {frames}/60 frames")

# save a sample
f = cam.get_latest_frame()
if f is not None:
    print(f"frame shape: {f.shape}")
    cv2.imwrite(r"C:\Users\asad\dxcam_sample.png", f)
    print("saved dxcam_sample.png")
else:
    print("no frame returned — DXcam may need a game/DirectX window visible")

cam.stop()