import cv2
import numpy as np


def filtro_1(roi: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    out = np.zeros_like(roi)
    out[gray < 60] = (15, 8, 10)
    out[(gray >= 60) & (gray < 130)] = (118, 30, 214)
    out[(gray >= 130) & (gray < 195)] = (35, 140, 235)
    out[gray >= 195] = (235, 240, 240)
    return out


def filtro_2(roi: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    cell = 6
    yy, xx = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    cx = (xx % cell) - cell / 2
    cy = (yy % cell) - cell / 2
    dist_center = np.sqrt(cx ** 2 + cy ** 2)
    radius = (1 - gray / 255.0) * (cell / 1.4)
    dot_mask = dist_center < radius
    out = np.full_like(roi, 245)
    out[dot_mask] = (15, 15, 15)
    return out


def filtro_3(roi: np.ndarray) -> np.ndarray:
    shift = 6
    b, g, r = cv2.split(roi)
    r_shift = np.roll(r, -shift, axis=1)
    b_shift = np.roll(b, shift, axis=1)
    out = cv2.merge([b_shift, g, r_shift])
    out[::3, :, :] = (out[::3, :, :] * 0.72).astype(np.uint8)
    return out


def filtro_5(roi: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    return cv2.applyColorMap(gray, cv2.COLORMAP_JET)


def filtro_6(roi: np.ndarray) -> np.ndarray:
    h, w = roi.shape[:2]
    sepia_kernel = np.array(
        [
            [0.272, 0.534, 0.131],
            [0.349, 0.686, 0.168],
            [0.393, 0.769, 0.189],
        ]
    )
    sepia = cv2.transform(roi, sepia_kernel)
    sepia = np.clip(sepia, 0, 255).astype(np.uint8)

    yy, xx = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    cy, cx = h / 2, w / 2
    dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    max_dist = np.sqrt(cx ** 2 + cy ** 2) or 1.0
    vignette = np.clip(1 - 0.5 * (dist / max_dist), 0, 1)[..., None]

    out = (sepia * vignette).astype(np.uint8)
    noise = np.random.randint(0, 25, out.shape, dtype=np.uint8)
    out = cv2.add(out, noise)
    return out


def filtro_blanco(roi: np.ndarray) -> np.ndarray:
    blurred = cv2.GaussianBlur(roi, (35, 35), 0)
    white = np.full_like(roi, 255)
    out = cv2.addWeighted(blurred, 0.55, white, 0.45, 0)
    return out


def filtro_rosa(roi: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    cell = 5
    yy, xx = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    cx = (xx % cell) - cell / 2
    cy = (yy % cell) - cell / 2
    dist_center = np.sqrt(cx ** 2 + cy ** 2)
    radius = (1 - gray / 255.0) * (cell / 1.3)
    dot_mask = dist_center < radius

    out = np.full_like(roi, (215, 190, 245))
    out[dot_mask] = (55, 20, 130)
    return out


def filtro_grid(roi: np.ndarray) -> np.ndarray:
    out = roi.copy()
    h, w = out.shape[:2]
    step = 22
    color = (235, 235, 235)

    overlay = out.copy()
    for x in range(0, w, step):
        cv2.line(overlay, (x, 0), (x, h), color, 1)
    for y in range(0, h, step):
        cv2.line(overlay, (0, y), (w, y), color, 1)

    out = cv2.addWeighted(overlay, 0.75, out, 0.25, 0)
    return out


def filtro_pixelado(roi: np.ndarray) -> np.ndarray:
    h, w = roi.shape[:2]
    factor = 10
    small_w = max(1, w // factor)
    small_h = max(1, h // factor)
    small = cv2.resize(roi, (small_w, small_h), interpolation=cv2.INTER_LINEAR)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)


def filtro_neon(roi: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 60, 150)
    edges = cv2.dilate(edges, np.ones((2, 2), np.uint8), iterations=1)

    colored_edges = cv2.applyColorMap(edges, cv2.COLORMAP_SPRING)
    lines = np.zeros_like(roi)
    lines[edges > 0] = colored_edges[edges > 0]

    glow = cv2.GaussianBlur(lines, (9, 9), 0)
    out = cv2.add(lines, glow)
    return out


def filtro_invertido(roi: np.ndarray) -> np.ndarray:
    inverted = cv2.bitwise_not(roi)
    levels = 4
    step = 256 // levels
    quantized = (inverted // step) * step
    return quantized.astype(np.uint8)


def filtro_matrix(roi: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape

    out = np.zeros_like(roi)
    out[..., 1] = gray

    stripe = np.tile(np.linspace(0.55, 1.0, h).reshape(-1, 1), (1, w))
    out = (out * stripe[..., None]).astype(np.uint8)
    return out


def filtro_kaleidoscopio(roi: np.ndarray) -> np.ndarray:
    h, w = roi.shape[:2]
    half_w = max(1, w // 2)
    left = roi[:, :half_w]
    mirrored = cv2.flip(left, 1)
    combined = np.concatenate([left, mirrored], axis=1)
    return cv2.resize(combined, (w, h))


def filtro_miles_morales(roi: np.ndarray) -> np.ndarray:
    """Heavy print-glitch look: aggressive RGB tearing, corrupted blocks,
    signal-loss dropouts, comic halftone, and scan-line artifacts."""
    h, w = roi.shape[:2]
    b, g, r = cv2.split(roi)
    rng = np.random.default_rng()

    out_r = r.copy()
    out_b = b.copy()
    out_g = g.copy()
    band_h = max(1, h // 30)  # thinner bands = more tearing
    for y0 in range(0, h, band_h):
        y1 = min(h, y0 + band_h)
        shift_r = int(rng.integers(-18, 19))
        shift_b = int(rng.integers(-18, 19))
        out_r[y0:y1] = np.roll(r[y0:y1], shift_r, axis=1)
        out_b[y0:y1] = np.roll(b[y0:y1], shift_b, axis=1)
        # occasional "signal loss" channel dropout/blowout on a band
        if rng.random() < 0.12:
            out_g[y0:y1] = 0
        if rng.random() < 0.08:
            out_r[y0:y1] = 255

    glitched = cv2.merge([out_b, out_g, out_r]).astype(np.float32)
    glitched[..., 0] *= 1.3   # blue up
    glitched[..., 1] *= 0.8   # green down -> pushes magenta/cyan energy
    glitched[..., 2] *= 1.25  # red up
    glitched = np.clip(glitched, 0, 255).astype(np.uint8)

    # corrupted rectangular blocks: invert, tear, or flip at random
    num_blocks = int(rng.integers(4, 9))
    for _ in range(num_blocks):
        bw_ = int(rng.integers(w // 12, max(w // 12 + 1, w // 4)))
        bh_ = int(rng.integers(h // 30, max(h // 30 + 1, h // 10)))
        x0 = int(rng.integers(0, max(1, w - bw_)))
        y0 = int(rng.integers(0, max(1, h - bh_)))
        block = glitched[y0:y0 + bh_, x0:x0 + bw_]
        choice = rng.random()
        if choice < 0.4:
            glitched[y0:y0 + bh_, x0:x0 + bw_] = cv2.bitwise_not(block)
        elif choice < 0.7:
            shift_x = int(rng.integers(-20, 21))
            glitched[y0:y0 + bh_, x0:x0 + bw_] = np.roll(block, shift_x, axis=1)
        else:
            glitched[y0:y0 + bh_, x0:x0 + bw_] = block[::-1]

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    cell = 5
    yy, xx = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    cx = (xx % cell) - cell / 2
    cy = (yy % cell) - cell / 2
    dist_center = np.sqrt(cx ** 2 + cy ** 2)
    radius = (1 - gray / 255.0) * (cell / 1.6)
    dot_mask = dist_center < radius

    out = glitched.copy()
    out[dot_mask] = (out[dot_mask].astype(np.float32) * 0.5).astype(np.uint8)

    streak_rows = rng.choice(h, size=max(2, h // 15), replace=False)
    out[streak_rows] = np.clip(out[streak_rows].astype(np.int16) + 70, 0, 255).astype(np.uint8)

    # occasional whole-frame horizontal jitter, like a dropped tracking line
    if rng.random() < 0.3:
        jitter = int(rng.integers(-6, 7))
        out = np.roll(out, jitter, axis=1)

    return out


def filtro_spider_noir(roi: np.ndarray) -> np.ndarray:
    """High-contrast black & white with heavy vignette and film grain."""
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    contrast = np.clip((gray.astype(np.float32) - 128) * 1.6 + 128, 0, 255).astype(np.uint8)
    bw = cv2.cvtColor(contrast, cv2.COLOR_GRAY2BGR)

    h, w = gray.shape
    yy, xx = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    cy, cx = h / 2, w / 2
    dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    max_dist = np.sqrt(cx ** 2 + cy ** 2) or 1.0
    vignette = np.clip(1 - 0.75 * (dist / max_dist) ** 1.5, 0, 1)[..., None]

    out = (bw * vignette).astype(np.uint8)

    grain = np.random.randint(0, 35, out.shape, dtype=np.uint8)
    out = cv2.subtract(out, grain // 2)
    out = cv2.add(out, grain // 3)
    return out


def filtro_spiderman_2099(roi: np.ndarray) -> np.ndarray:
    """Blue/purple to red/orange duotone via LUT."""
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    lut_b = np.interp(np.arange(256), [0, 128, 255], [80, 40, 20]).astype(np.uint8)
    lut_g = np.interp(np.arange(256), [0, 128, 255], [0, 20, 60]).astype(np.uint8)
    lut_r = np.interp(np.arange(256), [0, 128, 255], [60, 120, 220]).astype(np.uint8)
    b = cv2.LUT(gray, lut_b)
    g = cv2.LUT(gray, lut_g)
    r = cv2.LUT(gray, lut_r)
    return cv2.merge([b, g, r])


def filtro_comic(roi: np.ndarray) -> np.ndarray:
    """Cartoon/cel-shaded look: smoothed color + inked edges."""
    color = cv2.bilateralFilter(roi, 9, 200, 200)
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray_blur = cv2.medianBlur(gray, 5)
    edges = cv2.adaptiveThreshold(
        gray_blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 5
    )
    edges_colored = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    return cv2.bitwise_and(color, edges_colored)


def filtro_fisheye(roi: np.ndarray) -> np.ndarray:
    """Barrel/bulge lens distortion."""
    h, w = roi.shape[:2]
    cx, cy = w / 2.0, h / 2.0
    y_idx, x_idx = np.indices((h, w), dtype=np.float32)
    nx = (x_idx - cx) / cx
    ny = (y_idx - cy) / cy
    r = np.sqrt(nx ** 2 + ny ** 2)
    factor = 1 + 0.4 * (r ** 2)
    src_x = np.clip(cx + nx * factor * cx, 0, w - 1).astype(np.float32)
    src_y = np.clip(cy + ny * factor * cy, 0, h - 1).astype(np.float32)
    return cv2.remap(roi, src_x, src_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)


def filtro_jarvis(roi: np.ndarray) -> np.ndarray:
    """Holographic HUD look: cyan/blue duotone, glowing edge lines, grid
    overlay, scan lines, targeting reticle, and corner brackets."""
    h, w = roi.shape[:2]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    # deep-navy shadows -> bright cyan highlights duotone base
    lut_b = np.interp(np.arange(256), [0, 128, 255], [40, 140, 255]).astype(np.uint8)
    lut_g = np.interp(np.arange(256), [0, 128, 255], [15, 110, 235]).astype(np.uint8)
    lut_r = np.interp(np.arange(256), [0, 128, 255], [5, 30, 90]).astype(np.uint8)
    base = cv2.merge([cv2.LUT(gray, lut_b), cv2.LUT(gray, lut_g), cv2.LUT(gray, lut_r)])

    # glowing cyan edge lines traced over the scene, like a hologram scan
    edges = cv2.Canny(gray, 60, 140)
    edges = cv2.dilate(edges, np.ones((1, 1), np.uint8))
    edge_layer = np.zeros_like(roi)
    edge_layer[edges > 0] = (255, 220, 80)  # BGR: bright cyan-gold
    glow = cv2.GaussianBlur(edge_layer, (7, 7), 0)
    out = cv2.addWeighted(base, 0.9, cv2.add(edge_layer, glow), 0.9, 0)

    # faint HUD grid
    spacing = max(14, min(w, h) // 12)
    for x in range(0, w, spacing):
        cv2.line(out, (x, 0), (x, h), (90, 60, 25), 1)
    for y in range(0, h, spacing):
        cv2.line(out, (0, y), (w, y), (90, 60, 25), 1)

    # subtle horizontal scan-line darkening
    scan = np.ones((h, 1), dtype=np.float32)
    scan[::4] = 0.75
    out = (out * scan[:, :, None]).astype(np.uint8)

    accent = (255, 220, 80)
    cx, cy = w // 2, h // 2
    ring_r = min(w, h) // 3
    cv2.circle(out, (cx, cy), ring_r, accent, 1)
    cv2.circle(out, (cx, cy), max(1, ring_r - spacing), accent, 1)
    cv2.line(out, (cx - 6, cy), (cx + 6, cy), accent, 1)
    cv2.line(out, (cx, cy - 6), (cx, cy + 6), accent, 1)

    # HUD-style corner brackets
    bl = max(10, min(w, h) // 10)
    thickness = 2
    corners = [(0, 0, 1, 1), (w, 0, -1, 1), (0, h, 1, -1), (w, h, -1, -1)]
    for x0, y0, sx, sy in corners:
        cv2.line(out, (x0, y0), (x0 + sx * bl, y0), accent, thickness)
        cv2.line(out, (x0, y0), (x0, y0 + sy * bl), accent, thickness)

    return out


FILTROS = [
    filtro_grid,
    filtro_1,
    filtro_2,
    filtro_3,
    filtro_5,
    filtro_6,
    filtro_blanco,
    filtro_rosa,
    filtro_pixelado,
    filtro_neon,
    filtro_invertido,
    filtro_matrix,
    filtro_kaleidoscopio,
    filtro_miles_morales,
    filtro_spider_noir,
    filtro_spiderman_2099,
    filtro_comic,
    filtro_fisheye,
    filtro_jarvis,
]