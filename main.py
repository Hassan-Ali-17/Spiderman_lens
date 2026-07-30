import time

import cv2
import mediapipe as mp

from hand_tracking import INDEX_TIP, THUMB_TIP, count_extended_fingers
from geometry import (
    # rectangle mode (original design): one portal from both hands' tips
    portal_width,
    ClosingGestureDetector,
    render_portal,
    # circle mode: one independent portal per hand, with lock + merge
    PinchGestureDetector,
    FistGestureDetector,
    MergeGestureDetector,
    pinch_point_and_radius,
    render_circular_portal,
    render_merged_portal,
)
from filters import FILTROS

# One glow color per hand so it's obvious which portal belongs to which hand
# (circle mode only).
BORDER_COLORS = {
    "Left": (255, 90, 220),   # magenta/pink glow
    "Right": (255, 210, 60),  # cyan/gold glow
}


def choose_mode():
    print("Choose the portal mode:")
    print("  1) Rectangle - original design: one portal formed between both hands")
    print("  2) Circles   - one independent portal per hand (fist to lock, bring together to merge)")
    while True:
        choice = input("Choice (1/2): ").strip()
        if choice in ("1", "2"):
            return choice
        print("Invalid choice, type 1 or 2.")


def run_rectangle_mode(cap, hands):
    """Original design: a single portal stretched between both hands'
    thumb+index tips. Bringing your hands together (closing the rectangle)
    cycles the filter."""
    filtro_index = 0
    closing_detector = ClosingGestureDetector()

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        left_hand = None
        right_hand = None

        if results.multi_hand_landmarks and results.multi_handedness:
            for hand_landmarks, handedness in zip(
                results.multi_hand_landmarks, results.multi_handedness
            ):
                raw_label = handedness.classification[0].label
                label = "Right" if raw_label == "Left" else "Left"

                if label == "Left":
                    left_hand = hand_landmarks
                else:
                    right_hand = hand_landmarks

        if left_hand is not None and right_hand is not None:
            lm_left = left_hand.landmark
            lm_right = right_hand.landmark

            p1 = (lm_left[INDEX_TIP].x * w, lm_left[INDEX_TIP].y * h)
            p2 = (lm_left[THUMB_TIP].x * w, lm_left[THUMB_TIP].y * h)
            p3 = (lm_right[INDEX_TIP].x * w, lm_right[INDEX_TIP].y * h)
            p4 = (lm_right[THUMB_TIP].x * w, lm_right[THUMB_TIP].y * h)

            width = portal_width(p1, p2, p3, p4)

            if closing_detector.update(width, w):
                filtro_index = (filtro_index + 1) % len(FILTROS)

            frame = render_portal(frame, p1, p2, p3, p4, FILTROS[filtro_index])

        cv2.imshow(" ", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break


def run_circle_mode(cap, hands):
    """One independent circular portal per hand. Pinch to cycle that hand's
    filter, make a fist to lock the portal in place, bring both portals
    together to merge/blend their filters."""
    filtro_index = {"Left": 0, "Right": 1 % len(FILTROS)}
    pinch_detectors = {"Left": PinchGestureDetector(), "Right": PinchGestureDetector()}

    fist_detectors = {"Left": FistGestureDetector(), "Right": FistGestureDetector()}
    locked = {"Left": False, "Right": False}
    frozen_portal = {"Left": None, "Right": None}
    last_open_portal = {"Left": None, "Right": None}

    merge_detector = MergeGestureDetector()

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]
        t = time.time()

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        detected = {}
        if results.multi_hand_landmarks and results.multi_handedness:
            for hand_landmarks, handedness in zip(
                results.multi_hand_landmarks, results.multi_handedness
            ):
                raw_label = handedness.classification[0].label
                label = "Right" if raw_label == "Left" else "Left"
                detected[label] = hand_landmarks

        active_portals = {}

        for label in ("Left", "Right"):
            hand_landmarks = detected.get(label)

            if hand_landmarks is not None:
                center, radius, pinch_dist, scale = pinch_point_and_radius(
                    hand_landmarks, w, h, INDEX_TIP, THUMB_TIP
                )
                extended_count = count_extended_fingers(hand_landmarks, w, h)

                if extended_count > 1:
                    last_open_portal[label] = (center, radius)

                if fist_detectors[label].update(extended_count):
                    locked[label] = not locked[label]
                    if locked[label]:
                        frozen_portal[label] = last_open_portal[label] or (center, radius)

                if not locked[label] and pinch_detectors[label].update(pinch_dist, scale):
                    filtro_index[label] = (filtro_index[label] + 1) % len(FILTROS)

                if not locked[label]:
                    active_portals[label] = (center, radius)

            if locked[label] and frozen_portal[label] is not None:
                active_portals[label] = frozen_portal[label]

        merged_this_frame = False
        if "Left" in active_portals and "Right" in active_portals:
            center_a, radius_a = active_portals["Left"]
            center_b, radius_b = active_portals["Right"]
            dist = ((center_a[0] - center_b[0]) ** 2 + (center_a[1] - center_b[1]) ** 2) ** 0.5
            if merge_detector.update(dist, radius_a + radius_b):
                merged_this_frame = True
                frame = render_merged_portal(
                    frame,
                    center_a, radius_a, FILTROS[filtro_index["Left"]],
                    center_b, radius_b, FILTROS[filtro_index["Right"]],
                    BORDER_COLORS["Left"], BORDER_COLORS["Right"], t,
                )

        if not merged_this_frame:
            for label, (center, radius) in active_portals.items():
                frame = render_circular_portal(
                    frame,
                    center,
                    radius,
                    FILTROS[filtro_index[label]],
                    BORDER_COLORS.get(label, (255, 255, 255)),
                    t,
                    locked=locked[label],
                )

        cv2.imshow(" ", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break


def main():
    mode = choose_mode()

    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        max_num_hands=2,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.6,
    )

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError(
            "Could not open the camera. Check the camera index or permissions."
        )

    try:
        if mode == "1":
            run_rectangle_mode(cap, hands)
        else:
            run_circle_mode(cap, hands)
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()