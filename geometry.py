import time

import cv2
import numpy as np


def portal_width(p1, p2, p3, p4):
    top_w = np.hypot(p3[0] - p1[0], p3[1] - p1[1])
    bottom_w = np.hypot(p4[0] - p2[0], p4[1] - p2[1])
    return (top_w + bottom_w) / 2.0


class ClosingGestureDetector:

    def __init__(self, close_ratio=0.16, open_ratio=0.30):
        self.close_ratio = close_ratio
        self.open_ratio = open_ratio
        self.is_closed = False

    def update(self, width, frame_w):
        close_threshold = self.close_ratio * frame_w
        open_threshold = self.open_ratio * frame_w

        triggered = False
        if not self.is_closed and width < close_threshold:
            self.is_closed = True
            triggered = True
        elif self.is_closed and width > open_threshold:
            self.is_closed = False

        return triggered


def paint_filter_in_polygon(frame, polygon_pts, filtro_func):
    h, w = frame.shape[:2]

    x, y, bw, bh = cv2.boundingRect(polygon_pts)
    x, y = max(x, 0), max(y, 0)
    bw = min(bw, w - x)
    bh = min(bh, h - y)
    if bw <= 1 or bh <= 1:
        return frame

    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, [polygon_pts], 255)
    mask_roi = mask[y : y + bh, x : x + bw]

    roi = frame[y : y + bh, x : x + bw]
    filtered_roi = filtro_func(roi)

    mask_3ch = cv2.cvtColor(mask_roi, cv2.COLOR_GRAY2BGR).astype(np.float32) / 255.0
    blended = (filtered_roi * mask_3ch + roi * (1 - mask_3ch)).astype(np.uint8)
    frame[y : y + bh, x : x + bw] = blended
    return frame


def render_portal(frame, p1, p2, p3, p4, filtro_func):
    full_polygon = np.array([p1, p3, p4, p2], dtype=np.int32)

    paint_filter_in_polygon(frame, full_polygon, filtro_func)

    cv2.polylines(frame, [full_polygon], isClosed=True, color=(255, 255, 255), thickness=1)
    return frame


# ---------------------------------------------------------------------------
# Two-portal mode: one independent circular portal per hand.
# ---------------------------------------------------------------------------


def hand_scale(hand_landmarks, w, h):
    """Rough on-screen hand size (wrist to middle-finger knuckle), used to
    keep the portal a stable size regardless of the pinch gesture itself."""
    lm = hand_landmarks.landmark
    wrist = np.array([lm[0].x * w, lm[0].y * h])
    middle_mcp = np.array([lm[9].x * w, lm[9].y * h])
    return float(np.linalg.norm(wrist - middle_mcp))


def pinch_point_and_radius(hand_landmarks, w, h, index_tip_id, thumb_tip_id, radius_scale=1.15):
    """Returns (center, radius, pinch_dist, scale) for a single hand's portal.

    center: midpoint between thumb tip and index tip (where the portal floats)
    radius: portal size, derived from overall hand size (not the pinch gap,
            so the portal doesn't shrink to nothing as you pinch to switch filters)
    pinch_dist: current distance between thumb tip and index tip (gesture signal)
    scale: the underlying hand_scale value, for normalizing the gesture threshold
    """
    lm = hand_landmarks.landmark
    idx = np.array([lm[index_tip_id].x * w, lm[index_tip_id].y * h])
    thb = np.array([lm[thumb_tip_id].x * w, lm[thumb_tip_id].y * h])
    center = (idx + thb) / 2.0
    pinch_dist = float(np.linalg.norm(idx - thb))
    scale = hand_scale(hand_landmarks, w, h)
    radius = scale * radius_scale
    return center, radius, pinch_dist, scale


class FistGestureDetector:
    """Detects a closed-fist gesture (few/no extended fingers) with hysteresis.
    Used as an on/off toggle for locking a portal in place."""

    def __init__(self, close_count=1, open_count=3):
        self.close_count = close_count
        self.open_count = open_count
        self.is_closed = False

    def update(self, extended_count):
        triggered = False
        if not self.is_closed and extended_count <= self.close_count:
            self.is_closed = True
            triggered = True
        elif self.is_closed and extended_count >= self.open_count:
            self.is_closed = False
        return triggered


class MergeGestureDetector:
    """Hysteresis for merging two portals into one blended region once
    they're brought close enough together (nearly touching/overlapping)."""

    def __init__(self, close_ratio=0.95, open_ratio=1.35):
        self.close_ratio = close_ratio
        self.open_ratio = open_ratio
        self.is_merged = False

    def update(self, distance, combined_radius):
        if combined_radius <= 0:
            return self.is_merged
        ratio = distance / combined_radius

        if not self.is_merged and ratio < self.close_ratio:
            self.is_merged = True
        elif self.is_merged and ratio > self.open_ratio:
            self.is_merged = False
        return self.is_merged


class PinchGestureDetector:
    """Same open/close hysteresis idea as ClosingGestureDetector, but scaled
    per-hand (relative to that hand's own size) instead of frame width, and
    driven by the thumb-index pinch gap instead of the two-hand portal width."""

    def __init__(self, close_ratio=0.35, open_ratio=0.65):
        self.close_ratio = close_ratio
        self.open_ratio = open_ratio
        self.is_closed = False

    def update(self, pinch_dist, scale):
        if scale <= 0:
            return False
        ratio = pinch_dist / scale

        triggered = False
        if not self.is_closed and ratio < self.close_ratio:
            self.is_closed = True
            triggered = True
        elif self.is_closed and ratio > self.open_ratio:
            self.is_closed = False

        return triggered


def paint_filter_in_circle(frame, center, radius, filtro_func):
    h, w = frame.shape[:2]
    cx, cy = int(center[0]), int(center[1])
    r = int(radius)
    if r < 4:
        return frame

    x0, y0 = max(cx - r, 0), max(cy - r, 0)
    x1, y1 = min(cx + r, w), min(cy + r, h)
    bw, bh = x1 - x0, y1 - y0
    if bw <= 1 or bh <= 1:
        return frame

    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(mask, (cx, cy), r, 255, -1)
    mask_roi = mask[y0:y1, x0:x1]

    roi = frame[y0:y1, x0:x1]
    filtered_roi = filtro_func(roi)

    mask_3ch = cv2.cvtColor(mask_roi, cv2.COLOR_GRAY2BGR).astype(np.float32) / 255.0
    blended = (filtered_roi * mask_3ch + roi * (1 - mask_3ch)).astype(np.uint8)
    frame[y0:y1, x0:x1] = blended
    return frame


def draw_lock_icon(frame, center, radius, color):
    """Small padlock glyph drawn at the upper-right of a locked portal."""
    cx, cy = int(center[0]), int(center[1])
    r = int(radius)
    icon_x = cx + int(r * 0.7)
    icon_y = cy - int(r * 0.7)

    body_w, body_h = 14, 11
    top_left = (icon_x - body_w // 2, icon_y)
    bottom_right = (icon_x + body_w // 2, icon_y + body_h)
    cv2.rectangle(frame, top_left, bottom_right, color, -1)
    cv2.ellipse(frame, (icon_x, icon_y - 2), (6, 7), 0, 180, 360, color, 2)
    return frame


def draw_glow_ring(frame, center, radius, base_color=(255, 255, 255), t=None, locked=False):
    """Animated, pulsing multi-layer glow border around a circular portal.
    When locked, the ring goes static (no pulse) and shows a lock icon, so
    it's visually obvious the portal isn't following the hand anymore."""
    if t is None:
        t = time.time()
    cx, cy = int(center[0]), int(center[1])
    r = int(radius)
    if r < 4:
        return frame

    if locked:
        for spread, alpha in ((10, 0.08), (6, 0.14), (3, 0.22)):
            overlay = frame.copy()
            cv2.circle(overlay, (cx, cy), r + spread, base_color, 2)
            frame[:] = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)
        cv2.circle(frame, (cx, cy), r, base_color, 2)
        draw_lock_icon(frame, center, radius, base_color)
        return frame

    pulse = 0.5 + 0.5 * np.sin(t * 4.0)  # smooth 0..1 breathing oscillation

    for spread, alpha in ((10, 0.10), (6, 0.18), (3, 0.30)):
        glow_r = r + int(spread * (0.5 + pulse))
        overlay = frame.copy()
        cv2.circle(overlay, (cx, cy), glow_r, base_color, 2)
        frame[:] = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

    core_thickness = 2 + int(pulse * 2)
    cv2.circle(frame, (cx, cy), r, base_color, core_thickness)
    return frame


def render_circular_portal(frame, center, radius, filtro_func, border_color=(255, 255, 255), t=None, locked=False):
    paint_filter_in_circle(frame, center, radius, filtro_func)
    frame = draw_glow_ring(frame, center, radius, border_color, t, locked)
    return frame


def render_merged_portal(
    frame,
    center_a, radius_a, filtro_a,
    center_b, radius_b, filtro_b,
    color_a=(255, 255, 255), color_b=(255, 255, 255),
    t=None,
):
    """Blend two portals into one region once they're brought together.
    Both filters are applied across the union of the two circles, mixed with
    a gradient that runs along the line connecting the two centers, so the
    effect visually flows from one filter into the other."""
    h, w = frame.shape[:2]
    if t is None:
        t = time.time()

    cx_a, cy_a = float(center_a[0]), float(center_a[1])
    cx_b, cy_b = float(center_b[0]), float(center_b[1])

    x0 = int(max(min(cx_a - radius_a, cx_b - radius_b), 0))
    y0 = int(max(min(cy_a - radius_a, cy_b - radius_b), 0))
    x1 = int(min(max(cx_a + radius_a, cx_b + radius_b), w))
    y1 = int(min(max(cy_a + radius_a, cy_b + radius_b), h))
    bw, bh = x1 - x0, y1 - y0
    if bw <= 1 or bh <= 1:
        return frame

    mask_a = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(mask_a, (int(cx_a), int(cy_a)), int(radius_a), 255, -1)
    mask_b = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(mask_b, (int(cx_b), int(cy_b)), int(radius_b), 255, -1)
    mask_union = cv2.bitwise_or(mask_a, mask_b)
    mask_roi = mask_union[y0:y1, x0:x1]

    roi = frame[y0:y1, x0:x1]
    filtered_a = filtro_a(roi)
    filtered_b = filtro_b(roi)

    yy, xx = np.meshgrid(np.arange(y0, y1), np.arange(x0, x1), indexing="ij")
    dx, dy = (cx_b - cx_a), (cy_b - cy_a)
    length_sq = dx * dx + dy * dy
    if length_sq < 1e-3:
        weight = np.full((bh, bw), 0.5, dtype=np.float32)
    else:
        proj = ((xx - cx_a) * dx + (yy - cy_a) * dy) / length_sq
        weight = np.clip(proj, 0, 1).astype(np.float32)

    weight_3ch = weight[..., None]
    blended_filter = (
        filtered_a.astype(np.float32) * (1 - weight_3ch)
        + filtered_b.astype(np.float32) * weight_3ch
    ).astype(np.uint8)

    mask_3ch = cv2.cvtColor(mask_roi, cv2.COLOR_GRAY2BGR).astype(np.float32) / 255.0
    blended = (blended_filter * mask_3ch + roi * (1 - mask_3ch)).astype(np.uint8)
    frame[y0:y1, x0:x1] = blended

    # pulsing blended-color glow around both circles, plus a connecting line
    pulse = 0.5 + 0.5 * np.sin(t * 5.0)
    blend_color = tuple(int((ca + cb) / 2) for ca, cb in zip(color_a, color_b))
    for spread, alpha in ((10, 0.10), (6, 0.20), (3, 0.32)):
        overlay = frame.copy()
        glow_r = int(spread * (0.5 + pulse))
        cv2.circle(overlay, (int(cx_a), int(cy_a)), int(radius_a) + glow_r, blend_color, 2)
        cv2.circle(overlay, (int(cx_b), int(cy_b)), int(radius_b) + glow_r, blend_color, 2)
        frame[:] = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

    core_thickness = 2 + int(pulse * 2)
    cv2.circle(frame, (int(cx_a), int(cy_a)), int(radius_a), blend_color, core_thickness)
    cv2.circle(frame, (int(cx_b), int(cy_b)), int(radius_b), blend_color, core_thickness)
    cv2.line(frame, (int(cx_a), int(cy_a)), (int(cx_b), int(cy_b)), blend_color, 1)

    return frame