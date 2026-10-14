import mss, time
from PIL import Image

with mss.mss() as sct:
    for i, m in enumerate(sct.monitors):
        print(f"monitor {i}: {m['width']}x{m['height']} at ({m['left']},{m['top']})")

    shot = sct.grab(sct.monitors[1])
    img = Image.frombytes("RGB", shot.size, shot.bgra, "raw", "BGRX")
    img.save(r"C:\Users\asad\screen.png")
    print(f"saved screen.png {img.size}")

    t0 = time.perf_counter()
    for _ in range(30):
        sct.grab(sct.monitors[1])
    print(f"capture speed: {(time.perf_counter()-t0)/30*1000:.1f} ms/frame")