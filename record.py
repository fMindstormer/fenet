"""The recording in the README and on the page: real mouse strokes on the real page in Chromium.

    python record.py            writes site/fenet.gif (run build_page.py again afterwards: it reads the size)

A first visit: the opening example, a 7 drawn straight over it, Clear, a 4, Clear, a 2. Frames are
screenshots of the two boxes after each mouse move, with a pointer drawn in (screenshots leave it out).
"""
import io
from pathlib import Path
from PIL import Image, ImageDraw
from playwright.sync_api import sync_playwright
from readout import STROKES, launch

ARROW = [(0, 0), (0, 17), (4, 13), (7, 20), (10, 19), (7, 12), (12, 12)]


def main():
    frames = []                                   # (image, milliseconds)
    with sync_playwright() as p:
        b = launch(p)
        pg = b.new_page(viewport={"width": 1000, "height": 760}, device_scale_factor=1, color_scheme="light")
        pg.goto(Path("site/test.html").resolve().as_uri())
        pg.wait_for_timeout(1500)
        clip = pg.locator(".sheet").bounding_box()
        g = pg.locator(".grid").bounding_box(); cell = g["width"] / 14
        c = pg.locator(".clear").bounding_box()
        pos, down = [clip["x"] + clip["width"] * 0.55, clip["y"] + clip["height"] + 30], [False]

        def snap(ms):
            im = Image.open(io.BytesIO(pg.screenshot(clip=clip))).convert("RGB")
            x, y = pos[0] - clip["x"], pos[1] - clip["y"]
            d = ImageDraw.Draw(im)
            if down[0]:
                d.ellipse([x - 7, y - 7, x + 7, y + 7], outline=(209, 54, 43), width=2)
            d.polygon([(x + ax, y + ay) for ax, ay in ARROW], fill=(0, 0, 0), outline=(255, 255, 255))
            frames.append((im, ms))

        def glide(x, y, n=8, ms=30):
            x0, y0 = pos
            for t in range(1, n + 1):
                pos[:] = x0 + (x - x0) * t / n, y0 + (y - y0) * t / n
                pg.mouse.move(*pos)
                snap(ms)

        def draw(paths):
            for path in paths:
                pts = [(g["x"] + (col + 0.5) * cell, g["y"] + (row + 0.5) * cell) for col, row in path]
                glide(*pts[0], n=6)
                pg.mouse.down(); down[0] = True
                for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
                    glide(x1, y1, n=int(max(abs(x1 - x0), abs(y1 - y0)) / cell * 2) + 1, ms=35)
                pg.mouse.up(); down[0] = False

        def clear():
            glide(c["x"] + c["width"] / 2, c["y"] + c["height"] / 2, n=10)
            pg.mouse.down(); down[0] = True; snap(160)
            pg.mouse.up(); down[0] = False; pg.wait_for_timeout(80); snap(500)

        pg.mouse.move(*pos); snap(1800)           # the opening example and its reading
        for k, digit in enumerate(("7", "4", "2")):
            if k:
                clear()
            draw(STROKES[digit])
            glide(pos[0] + 60, pos[1] + 50, n=6)  # out of the way of the reading
            pg.wait_for_timeout(150); snap(2000)
        b.close()

    first, rest = frames[0][0], [im for im, _ in frames[1:]]
    first.save("site/fenet.gif", save_all=True, append_images=rest, duration=[ms for _, ms in frames],
               loop=0, optimize=True, disposal=1)
    print(f"site/fenet.gif: {len(frames)} frames, {first.width} x {first.height}, "
          f"{sum(ms for _, ms in frames) / 1000:.1f} s, {Path('site/fenet.gif').stat().st_size / 1000:.0f} kB")


if __name__ == "__main__":
    main()
