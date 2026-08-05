# Video Pixel Snapper (ComfyUI)

A ComfyUI custom node that turns a *batch* of AI-generated video frames
into stable pixel art. Unlike single-image pixel-art nodes (which process
every frame independently and end up with a jumping grid and flickering
colors), this node estimates the pixel grid and color palette **once**
from the whole clip, then applies those fixed parameters to every frame.

Includes an in-graph live palette editor: preview the raw input, the
node's auto-quantized output, and a re-colored live preview side by
side, edit the palette (pick colors straight from any of the three
previews, or type them in), and save the result back into ComfyUI as a
reusable `custom_palette` image — no need to re-run the graph to see
palette edits.

## Features

- **Grid size + phase auto-detection**, averaged (or taken from frame 1)
  across a sample of frames instead of guessed per frame — this is what
  removes the jumping grid you get from single-image pixel-art nodes run
  on video.
- **Temporally stable color reduction**: `majority` / `center_weighted`
  cell methods classify pixels against a palette that's fixed for the
  whole clip, with a confidence-margin fallback to a blended color on
  ambiguous (usually anti-aliased edge) cells instead of a noisy snap.
- **Accent-color reservation** (`accent_slots`): reserves palette slots
  for rare-but-visually-important colors (eyes, thin outlines, small
  highlights) that plain frequency-based quantization tends to drop.
- **`custom_palette` input**: feed in your own fixed palette (as an
  image — e.g. this node's own `palette_preview` output, edited).
- **`despeckle`**: removes isolated single-cell noise.
- **Flexible output sizing**: `manual` scale, or auto-match the
  original video's width / height / total pixel count.
- **In-graph live palette editor** (see below) — no need to leave
  ComfyUI to tweak the palette.

## Installation

```
ComfyUI/custom_nodes/ComfyUI-VideoPixelSnapper/
├── __init__.py
├── video_pixel_snapper.py
└── web/
    └── video_pixel_snapper.js
```

Clone or copy this folder (keeping the structure above — the `web/`
folder must stay a subfolder, not be flattened) into
`ComfyUI/custom_nodes/`, then restart ComfyUI. No extra Python
dependencies beyond `torch`, which ComfyUI already requires.

## Node: `Video Pixel Snapper`

**Input:** `image` (a batch of same-size frames). Optional
`custom_palette` (IMAGE) — if connected, the palette is read from its
unique colors and `k_colors`/`accent_slots` are ignored.

**Grid parameters:**
- `pixel_size` — `0.0` = auto-detect using `grid_detection_mode` below;
  `>0` = manual override. Grid *phase* (alignment) is always
  auto-detected separately from size.
- `grid_detection_mode` — `average_across_frames` (median/mode over a
  sample of frames, robust to one bad frame) or `first_frame` (faster,
  more fragile).

**Cell-reduction parameters:**
- `cell_method`:
  - `majority` (default) — classify every pixel against the palette,
    keep the most frequent index; falls back to a blended color when
    the vote is ambiguous (<5% margin).
  - `center_weighted` — same, but pixels near the cell center count more.
  - `median` — per-channel median of the cell.
  - `center` — center pixel only. Sharpest, but noticeably more
    sensitive to per-frame AI noise — prefer `majority` for video.
- `despeckle` — removes single cells that disagree with all 4
  neighbors. Off by default (can also erase an intentional 1px highlight).

**Palette parameters:**
- `k_colors` — palette size (ignored when `custom_palette` is connected).
- `accent_slots` — how many palette slots to reserve for rare-but-distinct
  colors instead of plain frequency-based clustering.

**Other:** `sample_frames`, `dither` (`bayer2/4/8` — deterministic, does
not flicker across frames, unlike Floyd–Steinberg), `output_scale_mode`
(`manual` / `match_width` / `match_height` / `match_pixel_count`),
`output_scale` (for `manual`), `seed`.

**Outputs:** `image`, `palette_preview` (a swatch strip — feed back in
as `custom_palette` after editing), `info` (a summary string: detected
grid size/phase, cell count, palette source, resolved scale).

## Two nodes: core processing vs. live editor

As of this version the editor UI is a **separate, optional node**,
`Video Pixel Snapper (Live Editor)` (`VideoPixelSnapperEditor`), instead
of being built into the main node. Reasoning: the editor widget is
fairly large and only some workflows need it — splitting it out keeps
`Video Pixel Snapper` itself small and fast, and you only pay for the
editor's UI weight when you actually add it to the graph.

Wire it up like this:

```
[Video Pixel Snapper]
   image (input)  ──────────────┐
   image (output) ───┐          │
   palette_preview ──┤          │
                      ▼          ▼
        [Video Pixel Snapper (Live Editor)]
           original_image  <- (same image feeding the core node)
           snapped_image   <- core node's `image` output
           palette_preview <- core node's `palette_preview` output
```

i.e. three connections: the same frames you feed into the core node
also go into the editor's `original_image`, and the core node's two
outputs go into the editor's other two inputs. The editor node passes
`snapped_image` through as its own `image` output, so it can also sit
inline in the middle of a chain if that's more convenient.

`max_preview_frames` (on the editor node) caps how many frames the
frame-paging controls can page through — default 24. Each one is
written to disk as a PNG for the browser to fetch, so pushing this very
high on a big batch has a real disk/time cost; raise it if you actually
need to page through more frames.

## In-graph live palette editor

The widget on the editor node:

- **Three previews side by side**: the original input frame, the core
  node's output with whatever palette it actually used, and a live
  re-color preview that updates instantly as you edit the palette —
  computed client-side, no need to re-run the graph. For large
  images/palettes this uses an exact pixel-for-pixel color match; above
  a size threshold (~15M pixel×color operations) it switches to a
  quantized lookup table for speed (roughly 7x faster at 800x1000px/96
  colors in testing), at the cost of ~10-15% of pixels landing on an
  adjacent rather than the exact-nearest palette color — visually close
  in practice, but worth knowing it's not pixel-exact at that scale.
- **Frame paging** (`←`/`→` + counter) when the input is a batch of
  more than one frame, up to `max_preview_frames`.
- **Zoom & pan**: scroll to zoom (centered on the cursor), middle-click
  drag to pan, double-click to reset the view. Shared across all three
  previews so you can compare the same region.

  **Known issue on ComfyUI's "Nodes 2.0" (Vue) rendering mode**: pan/zoom,
  and possibly the live-preview update timing, may not work correctly
  there. This is very new (beta) and changes how DOM widgets like this
  one are hosted — there's an open upstream issue specifically about DOM
  widget layout in Nodes 2.0
  (`Comfy-Org/ComfyUI_frontend#7942`). I can't develop against or verify
  behavior on Nodes 2.0 from here, so if you hit this, toggling Nodes
  2.0 off (ComfyUI logo menu) is the practical fix for now — that's the
  rendering mode this was actually built and tested against.
- **Pick / Delete / Replace tools**: turn one on, then click any of the
  three previews — Pick adds the color under the cursor to the palette,
  Delete removes the palette color closest to it, Replace lets you
  click a color to select it (or click a palette swatch directly) and
  then click another color — or use manual entry + "Replace selected"
  — to swap it in place. Hovering (without clicking) shows a live color
  preview swatch.
- **Measure tool**: for matching a physical size (e.g. "every character
  should be 192 pixel-art cells tall") when auto-detection doesn't fit
  the shot and manual `pixel_size` guessing is inconvenient (padding
  around the subject, etc). Click two points on the **Original** preview
  — top/bottom or left/right of whatever you're sizing against — type
  the target size in cells, and it computes `pixel_size = measured
  distance / target cells`. "Apply to pixel_size" walks this node's
  input connections looking for the upstream `Video Pixel Snapper` node
  and writes the value into its widget directly — this only works when
  the editor's inputs trace back to it directly (per the wiring above);
  otherwise it tells you the value to set manually. Honest caveat: the
  core node rounds `pixel_size` to a whole number when it runs (a
  pixel-art grid can't use a fractional block size), so the panel also
  shows the actual cell count you'll get after rounding — it may be off
  by a fraction from your exact target, which is inherent to integer
  grids, not a bug in the tool.
- **Manual color entry** via the browser/OS native color picker + hex
  field.
- **Save as custom_palette**: uploads the edited palette into ComfyUI's
  `input/` folder (via the standard `/upload/image` endpoint, with an
  editable subfolder + filename), ready to pick up in a `LoadImage`
  node feeding the core node's `custom_palette` input. Type is fixed to
  `input` — `LoadImage` only lists files from `input/`, so anything else
  would be invisible right where you need it. There's no way to reach an
  arbitrary filesystem path from browser JS — this is a genuine platform
  limit, not a corner we cut.
- **+/− buttons** to resize the preview height (60–400px) — the node
  grows to fit, since this is a normal DOM widget.

## Companion: `palette_editor.html`

A standalone, single-file HTML tool (open it directly in a browser, no
ComfyUI required) with the same palette-editing idea plus manual pixel
painting, a before/after/recolored comparison slider, and one-click
folder loading. Useful if you'd rather work outside the graph, or want
manual touch-up beyond what the in-graph widget offers (it intentionally
doesn't do pixel painting — no room for a usable canvas at that size).

## Troubleshooting

- **"Apply to pixel_size" (Measure tool) doesn't seem to update the
  widget on screen**: it sets `node.widgets.find(w => w.name ===
  "pixel_size").value` directly, which is what execution actually reads
  — so the computed value should take effect on the next run even if
  the number shown on the node doesn't visibly refresh right away.
  Clicking the node/canvas usually forces a redraw. If it's genuinely
  not applying (check by re-running and reading the `info` output,
  which reports the grid size actually used), open an issue.

- **Widget doesn't appear / preview stays empty**: open
  `http://127.0.0.1:8188/extensions/ComfyUI-VideoPixelSnapper/video_pixel_snapper.js`
  directly in a browser tab while ComfyUI is running (adjust host/port
  if different). If you see the JS source, the file is being served
  correctly and the issue is elsewhere — open an issue with your
  ComfyUI/frontend version. If you get a 404, double check the folder
  structure above (the `web/` subfolder is required, and the top-level
  folder name must not contain spaces).
- **No browser devtools in ComfyUI Desktop**: check the Python-side
  Logs panel instead (bottom of the window, or `Ctrl+\``) — it shows
  the ComfyUI server's console output in real time, including this
  node's own diagnostic prints if something fails server-side.
- Diagnostics from the JS extension go to the browser console
  (`[VideoPixelSnapper] ...`) if devtools are available to you.

## Notes on technique sources

Grid-phase search (brute-force offset scan maximizing edge energy),
`majority`/`center_weighted` cell voting with a confidence-margin
blend fallback, and the general idea of decoupling a video pixel-art
pipeline into a one-time estimation pass + a fixed-parameter apply pass
were cross-checked against jenissimo/unfake.js and
mediapixelkr/ComfyUI-SpriteFusion-PixelSnapper's public approach
descriptions, and adapted for batched video rather than single images.
`pixel_size` manual-override convention follows
x0x0b/ComfyUI-spritefusion-pixel-snapper.

## Known limitations

- Grid size/phase assume a single, uniform block size across X and Y.
- `accent_slots` is a real trade-off, not a free bonus: each reserved
  slot is taken from the regular (bulk) palette.
- The in-graph widget's palette editor operates on a *quantized* preview
  (nearest-color remap of the already-snapped frame) — it doesn't
  re-run grid/phase detection live. For a different grid, re-run the node.
