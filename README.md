# Spiderman Lens

An augmented reality project that generates an interactive portal in your camera feed using real-time hand tracking. Through natural hand gestures, you can switch between visual filters — including several Spider-Man-inspired lenses — rendered live inside the portal.

---

## Features

- Real-time hand tracking via MediaPipe Hands.
- Two selectable portal modes:
  - **Rectangle mode** (original design) — a single perspective portal built dynamically between the index and thumb fingertips of both hands.
  - **Circle mode** — one independent circular portal per hand, each cycling through filters on its own.
- 19 visual filters applied exclusively inside the portal area.
- Filter switching by gesture:
  - **Rectangle mode:** bringing your hands together triggers the transition to the next filter in the sequence.
  - **Circle mode:** pinching thumb and index together on a hand cycles that hand's filter.
- **Lock gesture (circle mode):** make a fist to freeze a portal in place at its last position, so you don't have to keep your hand raised. Make a fist again to unlock it.
- **Merge gesture (circle mode):** bring both portals close together to blend both filters into one region, with the effect flowing from one filter into the other.
- Hysteresis system to prevent accidental switching from tracking jitter or imprecision.

## Included filters

| Filter | Description |
|--------|-------------|
| `filtro_grid` | Grid overlay on the original image |
| `filtro_1` | Duotone segmented by luminosity thresholds |
| `filtro_2` | Black & white halftone (dot pattern) |
| `filtro_3` | Chromatic aberration with RGB channel separation |
| `filtro_5` | Thermal-camera simulation via colormap |
| `filtro_6` | Vintage sepia style with vignette and grain |
| `filtro_blanco` | Frosted-glass effect over the image |
| `filtro_rosa` | Pink-magenta duotone halftone |
| `filtro_pixelado` | Blocky mosaic/pixelation effect |
| `filtro_neon` | Neon edge-glow (Canny edges + colormap + blur) |
| `filtro_invertido` | Inverted colors with posterization |
| `filtro_matrix` | Green Matrix-style tint with vertical fade |
| `filtro_kaleidoscopio` | Mirrored kaleidoscope split |
| `filtro_miles_morales` | Heavy print-glitch look: RGB tearing, corrupted blocks, signal-loss dropouts, comic halftone |
| `filtro_spider_noir` | High-contrast black & white with vignette and film grain |
| `filtro_spiderman_2099` | Blue/purple to red/orange duotone |
| `filtro_comic` | Cartoon/cel-shaded look with inked edges |
| `filtro_fisheye` | Barrel/bulge lens distortion |
| `filtro_jarvis` | Holographic HUD look: cyan/blue duotone, glowing edge tracing, grid overlay, scan lines, targeting reticle, and corner brackets |

## Installation

Clone the repository:

```bash
git clone https://github.com/Hassan-Ali-17/Spiderman_lens.git
cd Spiderman_lens
```

Create a virtual environment:

```bash
python -m venv venv
```

Activate it:

```bash
venv\Scripts\activate      # Windows
source venv/bin/activate   # macOS / Linux
```

Install the dependencies:

```bash
pip install -r requirements.txt
```

## How to run this

```bash
python main.py
```

On startup you'll be asked to choose a portal mode:

```
Choose the portal mode:
  1) Rectangle - original design: one portal formed between both hands
  2) Circles   - one independent portal per hand (fist to lock, bring together to merge)
Choice (1/2):
```

**Rectangle mode (1):** with the camera active, raise both hands with your index and thumb extended — the portal forms automatically between them. Bring your hands together to "close" it and advance to the next filter in the list.

**Circle mode (2):** raise either hand with index and thumb extended — a circular portal forms between the fingertips, and each hand cycles its filter independently.
- **Pinch** thumb and index together to cycle that hand's filter.
- **Make a fist** to lock that portal in place so it stays put even if you lower your hand; make a fist again to unlock it.
- **Bring both portals close together** to merge them into one region that blends both filters.

In either mode, press **`q`** with the window focused to quit.

## Project structure

```
Spiderman_lens/
├── main.py            Entry point: mode selection, capture loop, and filter cycling
├── hand_tracking.py    Extended-finger and fist detection from the hand landmarks
├── geometry.py          Portal geometry, gesture detection, lock/merge rendering
├── filters.py            Definition of all available filters
├── requirements.txt
└── README.md
```

## Extending the project

To add a new filter, just define a function in `filters.py` that takes a BGR crop (`numpy.ndarray`) and returns a crop of the same size:

```python
def filtro_nuevo(roi: np.ndarray) -> np.ndarray:
    return roi
```

Then add it to the `FILTROS` list at the end of the file. The filter cycle automatically adjusts to however many items that list contains.

## Tech stack

- Python 3.10
- OpenCV
- MediaPipe
- NumPy

## License

This project is distributed under the MIT license.