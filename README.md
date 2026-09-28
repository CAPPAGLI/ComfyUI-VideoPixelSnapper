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
- **Mask-aware palette isolation**: connect a BiRefNet/RMBG foreground
  mask to exclude the removed background and soft compositing fringe from
  foreground palette estimation. Every masked-out cell is locked to one
  background color, optionally sampled from an `Empty Image` input.
- **`custom_palette` input**: feed in your own fixed palette (as an
  image — e.g. this node's own `palette_preview` output, edited).
- **`despeckle`**: removes isolated single-cell noise.
- **Flexible output sizing**: `manual` scale, or auto-match the
  original video's width / height / total pixel count.
- **Hard-alpha PNG output**: transparent mode keeps an invisible internal
  background key for temporal topology while exporting exact 0/1 alpha—no
  second background-removal pass and no soft fringe.
- **RGBA sprite sheets**: a focused row-major assembler preserves every frame
  and alpha value without resizing or blending.
- **Palette coverage analysis**: measures Oklab error against original masked
  character cells, selects a master-only subpalette, reports missing observed
  ramps and near-duplicate master slots without modifying the master.
- **Palette-locked selective outlining for stills**: replaces black outer and
  optionally internal linework with darker material-related colors already in
  a supplied palette. Manual or per-image automatic light direction is
  supported; RGB, RGBA, and optional masks preserve hard pixel edges.
- **True RGBA still loader**: bypasses standard Load Image's RGB/MASK split,
  optionally zeroes invisible RGB under alpha=0, and lets Core/Analyzer consume
  embedded PNG alpha without a separate mask wire.
- **Motion-aware discrete cleanup**: estimates bidirectional motion from
  the original video, rejects occlusions/flow errors, and stabilizes palette
  labels along trajectories without RGB blending—including external outlines,
  internal palette-region borders, and thin drawn features.
- **In-graph live palette editor** (see below) — no need to leave
  ComfyUI to tweak the palette.

## Installation

```
ComfyUI/custom_nodes/ComfyUI-Video-Pixel-Snapper/
├── __init__.py
├── video_pixel_snapper.py
├── frame_retimer.py
├── temporal_denoise.py
├── palette_analyzer.py
├── selective_outline.py
├── rgba_loader.py
├── sprite_sheet.py
└── web/
    ├── video_pixel_snapper.js
    └── frame_retimer.js
```

Clone or copy this folder (keeping the structure above — the `web/`
folder must stay a subfolder, not be flattened) into
`ComfyUI/custom_nodes/`, then restart ComfyUI. The model-free path needs only
`torch`. The RAFT backend uses `torchvision.models.optical_flow` (normally
already bundled with ComfyUI) and downloads its ~4 MB pretrained weights on
first use.

An importable VHS example is included at
`examples/Video Pixel Snapper.json`. It includes the recommended `info`
connection, `output_scale=1`, and the new raw-video `guide_image` connection
through Motion-Aware Cleanup. Choose your own video and palette after loading
it; upscale with nearest-neighbor only after cleanup/retiming if needed.

## Node: `Video Pixel Snapper`

**Input:** `image` (a batch of same-size RGB or RGBA frames). Nontrivial
embedded RGBA alpha is used automatically when no explicit foreground mask is
connected. Standard ComfyUI Load Image emits RGB plus a separate MASK, so use
that MASK or the focused `Load RGBA Image` node. Optional `custom_palette`
(IMAGE) — if connected, the palette is read from its unique colors and
`k_colors`/`accent_slots` are ignored.

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
- `color_distance`:
  - `oklab` (recommended) selects the palette entry that looks perceptually
    closest and strongly reduces wrong-hue snaps;
  - `rgb_legacy` preserves the old Euclidean encoded-RGB behavior.

**Palette parameters:**
- `k_colors` — palette size (ignored when `custom_palette` is connected).
  With `foreground_mask`, one of these slots is reserved for the background.
- `accent_slots` — how many palette slots to reserve for rare-but-distinct
  colors instead of plain frequency-based clustering.

### Background removal / foreground-mask wiring

For transparent BiRefNet/RMBG workflows, use:

```text
original video IMAGE ───────────> Video Pixel Snapper.image
BiRefNet/RMBG foreground MASK ──> Video Pixel Snapper.foreground_mask
background_mode = transparent
background_image = disconnected
```

**Feed the original RGB video—not the RMBG cutout—to `image`.** This prevents
white/partially composited RMBG fringe from entering foreground color
reduction. Transparent mode requires the mask and produces hard cell alpha:
each output pixel is exactly opaque or transparent.

Motion Cleanup still needs a discrete background label for silhouette voting.
Transparent mode therefore chooses an internal 8-bit RGB key absent from the
foreground palette, writes it under transparent pixels, and reports it in
`info`. The key is invisible in `transparent_image` and prevents a real black
or other foreground color from being mistaken for background.

- `background_mode=solid` preserves the old behavior and optional
  `background_image` flat color.
- `background_mode=transparent` ignores visual background appearance and uses
  the invisible unique key. `background_image` is not needed.
- `mask_threshold` (default `0.5`) controls which source pixels may influence
  foreground colors. Raise it toward `0.7–0.9` if soft RMBG fringe colors
  still leak in.
- `mask_cell_threshold` (default `0.25`) controls hard silhouette coverage at
  cell level. Lower values preserve thin details; higher values remove more
  edge contamination.
- `invert_mask` handles nodes where white means background. BiRefNet/RMBG
  normally outputs white foreground, so leave it off first.

The original `image` output remains RGB for compatibility. Use the appended
RGBA `transparent_image` for Cleanup and export. `transparency_mask` follows
ComfyUI's Load Image convention: `1` is transparent and `0` is opaque.

**Transparent mode always forces `scale=1`.** A batch of 32 float32 RGBA frames
at 5184×2736 alone requires 7.26 GB, before the RGB source, mask, previews, and
ComfyUI cache. Cleanup also requires cell scale 1. Build and retime at native
pixel-art resolution; if a large presentation image is needed, use
nearest-neighbor scaling only after Sprite Sheet, where just one sheet tensor
must be enlarged. The `info` output reports
`transparent_scale=forced_1x(requested=...)` when a larger setting was ignored.

**Other:** `sample_frames`, `dither` (`bayer2/4/8` — deterministic, does
not flicker across frames, unlike Floyd–Steinberg), `output_scale_mode`
(`manual` / `match_width` / `match_height` / `match_pixel_count`),
`output_scale` (for `manual`), `seed`.

**Outputs:** existing `image`, `palette_preview`, and `info`, followed by
`transparent_image` (RGBA) and `transparency_mask`. Feed `palette_preview` back
as `custom_palette` after editing; use RGBA only on the alpha-aware branch.
Custom strips retain their first-occurrence swatch order in `palette_preview`
and Live Editor.

### Why an available palette color may not be selected

The core classifies all source samples in a cell and then applies the chosen
cell method; it does not search by color name or by visual intent. Legacy RGB
distance can also cross hue families. For example, source orange `#E4954B` is
numerically closer in encoded RGB to salmon `#EF7D57` than to the orange bridge
`#EA8A2E`, even though the latter looks more faithful. `color_distance=oklab`
selects `#EA8A2E` for that case. An exactly equal source pixel still maps to its
exact palette entry in either mode.

Start custom-palette evaluation with `oklab`, `majority`, and `dither=none`.
Only enable Bayer dithering afterward if additional texture is desired; dither
cannot repair a wrong distance metric and makes one-to-one comparisons harder.

## Node: `Palette Coverage Analyzer`

This separate diagnostic node answers two different questions without silently
changing your global palette:

1. Which existing master colors form the best subpalette for this character?
2. Which observed foreground color families are genuinely missing from master,
   and which current master slots are near-duplicates that could be replaced?

Standalone wiring (no Pixel Snapper required):

```text
16-render IMAGE batch ───────────────> Palette Coverage Analyzer.image
matching foreground MASK batch ─────> Palette Coverage Analyzer.foreground_mask
Load Image: master palette PNG ──────> Palette Coverage Analyzer.master_palette
analysis_pixel_size = measured block size (for example 5)
```

`analysis_pixel_size=0` auto-detects one shared block size from the sampled
batch; a measured manual value is more reliable. Shared phase is still
estimated across the renders. Optional `snapper_info` is only a compatibility
shortcut: when connected and source dimensions match, its exact block, phase,
crop, and mask thresholds take precedence. It is not required.

Controls:

- `subpalette_size`: existing master entries to retain for this character;
- `analysis_pixel_size`: manual source block size, or `0` for auto-detection;
- `mask_threshold` / `mask_cell_threshold`: standalone mask sampling and cell
  coverage thresholds; ignored when compatible `snapper_info` is used;
- `suggestion_count`: maximum observed gap colors to show for review;
- `sample_frames`: evenly spaced frames, each with equal total weight;
- `missing_threshold`: Oklab distance counted as a visible coverage gap (`0.06`
  is a practical starting point);
- `duplicate_threshold`: pairs closer than this are reported as possible slot
  replacements (`0.03` default);
- `heatmap_limit`: error shown as full red (`0.12` default).

Outputs:

- `error_heatmap`: black background; blue means well covered, green is
  intermediate, red reaches/exceeds the heatmap limit;
- `suggested_subpalette`: only colors already present in master, selected with
  a weighted coverage objective and returned in master-strip order;
- `missing_color_suggestions`: observed 8-bit cell medoids representing
  uncovered clusters. Near-match resize residues and sub-24 near-black values
  (when master contains black) are suppressed. Suggestions are evidence for a missing *ramp*,
  not instructions to append every swatch;
- `info`: mean/median/p95/max error, gap-cell percentage, suggestion coverage
  counted only within threshold-qualified gap cells, nearest master colors, and
  near-duplicate master pairs.

The analyzer never edits master automatically. Save a reviewed subpalette or
suggestion strip and load it on a later run; connecting it back into the same
core while also consuming that core's `info` would create a graph cycle.

## Node: `Load RGBA Image`

ComfyUI's standard `Load Image` separates a transparent PNG into RGB `IMAGE`
and a separate `MASK`; the IMAGE socket alone contains no alpha, so no
downstream node can recover it automatically. `VideoPixelSnapperLoadRGBA` is a
focused still loader that emits both forms:

- `image`: compatibility RGB;
- `transparent_image`: true four-channel RGBA IMAGE;
- `foreground_mask`: alpha (`1=opaque`), ready for Core/Analyzer;
- `transparency_mask`: Comfy convention (`1=transparent`);
- `info`: alpha statistics and hidden-RGB handling.

Recommended one-wire alpha path:

```text
Load RGBA Image.transparent_image -> Video Pixel Snapper.image
Video Pixel Snapper.transparent_image -> Selective Outline.image
Selective Outline.transparent_image -> Live Editor.snapped_image
```

Core and Palette Coverage Analyzer automatically use nontrivial embedded RGBA
alpha when no explicit `foreground_mask` is connected. Fully opaque RGBA does
not activate masking. Standard Load Image still requires its MASK to be wired
(or inverted for a white-foreground input), because its IMAGE output is RGB by
design.

`sanitize_hidden_rgb=true` changes RGB only where alpha is exactly zero. This
is visually lossless but prevents Photoshop's hidden white/magenta mattes from
reappearing if any later RGB-only preview or node drops alpha. Partial-alpha and
opaque pixels remain byte-identical. Core continues to emit deliberately hard
cell-level alpha; the loader itself preserves the source's soft alpha.

## Node: `Selective Outline / Sel-Out`

`VideoPixelSnapperSelectiveOutline` is a focused, model-free still-image node.
It detects black line pixels and replaces them with darker colors related to
the adjacent body material. Every replacement is an exact entry from the
connected `palette`; the node never averages, interpolates, or invents an RGB
shade. A batch is accepted for ComfyUI convenience, but each image is processed
independently—there is no temporal state or cross-image palette decision.

Recommended wiring after snapping a single sprite:

```text
Video Pixel Snapper.image/transparent_image ──> Selective Outline.image
Video Pixel Snapper.info ─────────────────────> Selective Outline.snapper_info
16/24-color character subpalette strip ───────> Selective Outline.palette
foreground mask (when needed) ────────────────> Selective Outline.foreground_mask
```

With `input_pixel_scale=0`, compatible `snapper_info` supplies the exact core
`output_scale`. For scale greater than one the node first collapses every exact
nearest-neighbor block to one logical pixel, performs Sel-Out on that logical
grid, then restores the original dimensions by exact nearest-neighbor repeat.
This makes the complete N×N outline cell change together instead of recoloring
only its outermost raster row. If `info` is unavailable, set
`input_pixel_scale` manually to the core output scale; set it to `1` for native
pixel resolution.

The scaled path intentionally rejects images that are not an exact
nearest-neighbor enlargement (apart from a one-level PNG tolerance). It will not
silently collapse bilinear, antialiased, or incorrectly declared input. Run
Sel-Out before such presentation scaling.

Controls:

- `outline_scope=outer_only` changes eligible dark pixels only on the external
  one-pixel silhouette. `outer_and_internal` also changes thin internal black
  linework; use it carefully around pupils, mouths, and intentionally black
  filled details.
- `lighting_mode=manual` uses `light_direction` and is deterministic.
  `auto` estimates an independent eight-way direction for each still from the
  centroid of its brightest non-outline material pixels. This is a heuristic:
  a large white costume can dominate it, so the `info` output reports the
  inferred direction and confidence and manual mode remains the recommended
  production setting.
- `style=subtle/balanced/strong` controls how light the lit-side replacement may
  become. Start with `balanced`; `subtle` stays closer to a conventional dark
  outline, while `strong` produces a more visible sel-out.
- `black_threshold` is the maximum value allowed in every 8-bit RGB channel.
  Use `0` for exact `#000000`, the default `12` for tiny near-black residue, or
  roughly `37–40` when deliberately targeting `#181425` (its largest channel
  is hex `25` = decimal `37`).
- `mask_meaning` supports both white-foreground RMBG/BiRefNet masks and
  ComfyUI Load Image masks where white means transparency.
- `input_pixel_scale=0` reads scale from `snapper_info`; positive values are a
  manual override. The `info` output reports `scale_source`, logical dimensions,
  and the restored output dimensions.

Foreground detection priority is: connected mask, then incoming RGBA alpha,
then a conservative flood fill of the dominant near-flat RGB border color.
A mask/alpha is strongly recommended when the sprite touches the canvas edge,
the background is textured, or the background and outline have similar dark
colors. Plain RGB fallback is only intended for one still on a flat backdrop.
For a scaled masked core result, prefer its output-resolution
`transparency_mask` with `mask_meaning=white_is_transparent`; this keeps the
logical silhouette aligned with the scaled snapped cells.

Outputs:

- `image`: RGB result, retaining the original background;
- `transparent_image`: RGBA result. Existing input alpha is preserved; a
  connected mask supplies thresholded hard 0/1 alpha so a soft removal fringe
  is not reintroduced; plain RGB fallback remains fully opaque and does not
  silently remove its inferred background;
- `changed_outline`: mask of exactly the pixels recolored;
- `info`: mask source, scope, light direction/confidence, candidate/change
  counts, and unresolved dark pixels.

The technique is selective outlining (`sel-out`), not literal RGB-negative
inversion. Lit-facing silhouette sections select a lighter dark from the local
material family; shadow-facing sections select a deeper tinted dark. The node
only recolors existing line pixels—it does not grow, smooth, antialias, or break
the silhouette.

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
   info ─────────────┤          │
                      ▼          ▼
        [Video Pixel Snapper (Live Editor)]
           original_image  <- (same image feeding the core node)
           snapped_image   <- core node's `image` output
           palette_preview <- core node's `palette_preview` output
           info             <- core node's `info` output (recommended)
```

When Selective Outline sits before Live Editor, use:

```text
original/pre-snap IMAGE ─────────────────────> Live Editor.original_image
Load Image MASK (white=transparent) ─────────> Live Editor.original_transparency_mask
Selective Outline.transparent_image ────────> Live Editor.snapped_image
active character palette strip ─────────────> Live Editor.palette_preview
Video Pixel Snapper.info ───────────────────> Live Editor.info
Selective Outline.changed_outline ──────────> Live Editor.postprocess_mask
```

The optional `postprocess_mask` fixes edited-preview fallback through Sel-Out.
With no palette change, Live still displays the authoritative `snapped_image`
exactly. After Add/Delete/Replace, ordinary body cells are reclassified from
RAW so new colors can appear, while mask-selected outline cells are
reclassified from the authoritative Sel-Out result. Thus the edited Live panel
keeps the selective-outline geometry instead of returning those cells to raw
black. The browser preview still does not duplicate the full Python Sel-Out
material/light solver; save the palette, feed it to both Pixel Snapper and
Selective Outline, then re-run for the authoritative result.

ComfyUI's standard `Load Image` separates an alpha PNG into an RGB `IMAGE` and a
`MASK` where white means transparent. Alpha does not erase the RGB stored under
transparent pixels. Connect that MASK to the optional
`original_transparency_mask` if the Original browser panel should display true
transparency. Feed `Selective Outline.transparent_image`, not its compatibility
RGB `image`, to downstream alpha-aware nodes. Live Editor now saves RGBA preview
PNGs, preserves authoritative alpha after palette edits, and draws its three
canvases over a checkerboard; transparent hidden RGB is no longer shown as a
mysteriously restored background.

The first three image connections are required. Connecting `info` is
optional but recommended: it gives the browser widget the exact detected
block size, phase, cell dimensions, and output scale. Without it, Live
uses a centered best-effort reduction and may not line up with Snapped.
Normally the editor passes `snapped_image` through unchanged. For a single
still, **Commit Live → output** serializes the exact current Live canvas into a
hidden workflow value; after one Queue, the existing RGB/RGBA/mask outputs
materialize that committed image instead. The appended `commit_info` output
states whether output is pass-through or an exact committed browser PNG.

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
  computed client-side, no need to re-run the graph. Live first reduces
  the raw frame to the detected cell grid (using the real phase/cell size
  when `info` is connected), then remaps those representatives to the
  edited palette. While the working palette is unchanged, Live directly
  reuses the authoritative Snapped frame, so those two panels are
  pixel-identical. Palette comparison is based on the color set, not swatch
  order. After an actual color edit, Live switches to RAW-based cell
  reclassification so newly added colors can appear immediately. If the
  optional `postprocess_mask` is connected, those selected cells instead use
  palette-remapped authoritative Snapped pixels, preserving Sel-Out or another
  discrete post-process in the edited preview. RAW median reduction excludes
  samples below alpha 0.5, so white/magenta RGB hidden under transparent
  Photoshop pixels cannot influence visible palette assignments.
  Nearest-color lookup is exact below ~15M cell×color operations; above
  that it switches to a quantized lookup table for speed. The edited Live
  path is still an approximation of the core `majority`/dither logic; the
  re-run Python output is authoritative.
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
- **Commit Live → output** (single-image input): captures the exact visible
  Live canvas—including Sel-Out structure and alpha—at the Snapped output
  dimensions. Queue Prompt once afterward, then take this node's `image` or
  `transparent_image` output. The committed PNG is stored in a hidden serialized
  widget. Original/output dimensions guard against accidental stale reuse;
  the byte fingerprint is diagnostic only because browser premultiplied-alpha
  round-trips can legitimately differ from Torch at transparent edges. Any
  later palette edit clears the commit; commit again when the Live look is
  final. `Clear commit` restores
  ordinary Snapped pass-through.
- **Save as custom_palette**: uploads only the edited *color list* into ComfyUI's
  `input/` folder (via the standard `/upload/image` endpoint, with an
  editable subfolder + filename), ready to pick up in a `LoadImage`
  node feeding the core node's `custom_palette` input. Type is fixed to
  `input` — `LoadImage` only lists files from `input/`, so anything else
  would be invisible right where you need it. A palette PNG cannot encode
  which color each source cell selected; reloading it into Core legitimately
  re-runs Core's majority/median logic and may not reproduce Live's RAW-based
  approximation. Use Commit Live when exact cell assignments matter. There's
  no way to reach an arbitrary filesystem path from browser JS — this is a
  genuine platform limit, not a corner we cut.
- **+/− buttons** to resize the preview height (60–400px) — the node
  grows to fit, since this is a normal DOM widget.

## Node: `Frame Retimer`

A third, standalone node (`VideoPixelSnapperFrameRetimer`) for editing
the timing and *order* of frames in any IMAGE batch — drop unwanted
frames, hold others longer, reorder them, loop a short section, or
drive broad pacing from a keyframed duration curve. Not pixel-art
specific; it's generic frame retiming, included here because none of
the existing ComfyUI video nodes do quite this (see below).

**Why a new node instead of an existing one**: checked
`ComfyUI-VideoHelperSuite` (only whole-batch operations — force a frame
rate, take every Nth frame, duplicate/split/merge whole batches — no
per-frame editing) and `LiquidTime-Interpolation` (has a similar "time
curve" concept, but it works by generating new **AI-interpolated**
in-between frames via the FILM model — the opposite of what a
pixel-art pipeline wants, since that reintroduces blending/anti-
aliasing). This node only ever duplicates, drops, or reorders
**existing** frames — nothing is ever blended or regenerated, so it's
safe to use after `Video Pixel Snapper` without undoing its work.

**Data model**: `sequence_json` is a flat array of source-frame
indices, one per *output* frame, in output order — the single source
of truth. This is deliberately more general than a plain "repeat count
per source frame" array (which is all the first version had): that
model can only ever hold each source frame's repeats together at its
original position, so it literally cannot express reordering or
looping a section. An explicit index sequence can.

**Two layers, composed**:
1. **Pacing** — the duration curve, or a single frame's `Duration`
   field — regenerate the *entire* sequence from scratch (each source
   frame repeated N times, in original order). Good for broad
   timing/speed-ramping.
2. **Timeline** — further edits whatever sequence pacing produced:
   drag a block to reorder it, click (shift-click for a range) then
   **Duplicate selection** to create a loop (repeat again for more
   repeats) or **Delete selection** to cut. Touching pacing again
   *regenerates* the sequence and discards timeline edits —
   intentional (block out timing first, then reorder/loop), not a bug;
   the widget's status line says so.

**Widget**:
- **Frame strip aligned directly above the duration curve**, each
  thumbnail at the exact x-position its corresponding curve point sits
  at, with an `×N` badge showing how many times it currently occurs in
  the output. Click one to select it for the `Duration` field below.
- **Duration curve**: click empty space to add a keyframe, drag points
  to reposition, double-click a point to remove it (keeps a minimum of
  2) — applies **live**, on every drag/add/remove, no separate "Apply"
  needed. `smooth` checkbox switches interpolation between keys from
  linear to eased (smoothstep — a soft S-curve, not true sine, but the
  same idea of easing in/out rather than a sharp linear ramp). `max`
  field sets the curve's Y-axis ceiling (default 4); lowering it clamps
  any existing keyframes above the new max down to it, so nothing goes
  invisibly off-chart.
- **Timeline**: see "two layers" above.
- **+/− buttons** resize the play/pause preview (60–400px) — it was
  fixed-size before, unlike the Live Editor node's previews, which was
  an oversight, not intentional; now matches.
- **Output count + play/pause preview**, with real seconds shown next
  to frame counts when `source_fps` is set (see below) — scrubs the
  actual resulting sequence using already-loaded thumbnails, entirely
  client-side, no need to re-run the graph just to check the timing.

`sequence_json` (on the node itself, previously named `durations_json`
— renamed because it now holds a different, more general kind of data)
is the field the widget writes to, hidden from the node body
(best-effort, same as the Live Editor node's trick — worst case it
just shows as a small field, not a giant box). Left untouched, the
node passes every frame through once in original order. Editing
survives a re-run as long as it still only references valid frame
indices for the current input; re-running is only needed to
materialize the real reordered tensor for downstream nodes.

`source_fps` is optional and purely informational — lets the widget
show real seconds next to frame counts. Doesn't affect the retiming
math at all. Right-click → "Convert to Input" (standard ComfyUI, works
on any FLOAT/INT widget) to feed it from wherever your pipeline already
knows the source video's fps — same trick works for `max_preview_frames`
if you'd rather drive that from a connected frame-count value than set
it by hand.

**`max_preview_frames` vs. the actual sequence**: this setting only
caps how many thumbnails get *loaded for editing* (disk/time tradeoff
on very long batches) — it does not limit how many frames you can
actually retime. The sequence always covers the true total frame
count; frames beyond the thumbnail cap just don't get a clickable
strip/timeline cell (no image to show), but pacing still reaches and
edits them correctly, and the status line says when this is happening.
An earlier version had a real bug here — editing past the cap silently
got discarded on the next run because the duration array was sized to
the thumbnail count instead of the true total. Fixed by sending the
true count separately; verified with a 300-frame / 30-thumbnail test
where an edit at frame 250 (beyond the cap) correctly survived a re-run.

For RGBA input, Retimer reorders alpha with exactly the same index sequence.
The original `image` output remains RGB for compatibility with video encoders;
use appended `transparent_image` for PNG/sprite export and
`transparency_mask` when a separate Comfy MASK is needed.

## Node: `Sprite Sheet`

`VideoPixelSnapperSpriteSheet` is a focused, model-free row-major assembler.
Connect Frame Retimer's `transparent_image`, choose `columns`, and optionally
add transparent `padding`. Frames are copied bit-exactly into the sheet—there
is no scaling, interpolation, palette conversion, or RGB/alpha blending.
Unused cells in the final row and all padding remain transparent.

Outputs are `sprite_sheet`, a summary string, and integer frame width, frame
height, columns, and rows. Connect `sprite_sheet` directly to ComfyUI's
standard `Save Image` for a native-resolution transparent PNG. For a larger
presentation sheet, insert nearest-neighbor Image Scale between these two
nodes; do not upscale every frame before assembling the sheet.

Recommended transparent chain:

```text
Video Pixel Snapper.transparent_image
  → Motion-Aware Cleanup.image
Motion-Aware Cleanup.transparent_image
  → Live Editor.snapped_image
Live Editor.transparent_image
  → Frame Retimer.image
Frame Retimer.transparent_image
  → Sprite Sheet.image
Sprite Sheet.sprite_sheet
  → Save Image
```

The parallel RGB outputs remain available for VHS/MP4 encoding, whose common
formats do not preserve alpha.

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
  `http://127.0.0.1:8188/extensions/ComfyUI-Video-Pixel-Snapper/video_pixel_snapper.js`
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

The Motion-Aware Cleanup redesign follows the recurring structure in temporal
video abstraction/stylization research: propagate correspondences with optical
flow, use bidirectional or error-aware validation, and refuse propagation in
occluded/unreliable regions. Relevant references include Zhang et al.'s
[flow-guided coherent segmentation](https://cg.cs.tsinghua.edu.cn/papers/TMM_2011_videostream.pdf),
[Interactive Control over Temporal Consistency](https://doi.org/10.1111/cgf.14891),
and work on [flow-error/occlusion reduction](https://www.mdpi.com/2076-3417/14/6/2630).
The implementation differs at the last step: instead of blending stylized RGB,
it votes over already-quantized palette labels and copies an existing color.

## Node: `Motion-Aware Cleanup`

The node ID remains `VideoPixelSnapperTemporalCleanup`, so existing graphs
recognize it, but the old fixed-coordinate majority filter has been replaced.
That filter assumed cell `(x,y)` represented the same material in every frame;
on motion it therefore copied stale colors into new positions and caused the
smearing/trailing it was intended to remove.

New required wiring:

```text
original/pre-snap video IMAGE ─────────────────────> Motion-Aware Cleanup.guide_image
Video Pixel Snapper transparent_image (scale=1) ──> Motion-Aware Cleanup.image
Video Pixel Snapper info ─────────────────────────> Motion-Aware Cleanup.snapper_info
Motion-Aware Cleanup.transparent_image ───────────> Live Editor / Retimer / PNG
```

The old RGB `image` connection remains valid, but RGBA input lets Cleanup
preserve alpha exactly and update it only for accepted silhouette changes.
Cleanup appends `transparent_image` and `transparency_mask` after all existing
diagnostics.

Motion is estimated from `guide_image`, where texture and edges still exist.
A 1920p guide and ~144p cell image are expected: flow vectors are resized to
the snapped dimensions and `dx/dy` are converted from guide pixels to cell
units. Connecting `snapper_info` additionally applies the core node's exact
grid-phase crop before resizing, including proportional coordinates when the
guide resolution differs from the core source. Without it the node can only
resize the whole guide and reports `resize_only` in `info`.

Neighboring snapped frames are then sampled along forward/backward motion
trajectories. A palette color is replaced only when:

- motion-compensated neighbors have a **unique** discrete consensus;
- forward/backward flow agrees;
- the warped guide remains photometrically consistent;
- the candidate still fits the current guide color;
- the palette-color jump is below `max_color_distance`.

Foreground/background topology has a separate pass. When the core `info`
contains the locked mask background color, `silhouette_stabilization` can
remove one-frame teeth/holes and restore one-frame missing edge cells using
stricter geometry consensus (`silhouette_agreement=0.75`,
`silhouette_min_support=3`). It intentionally ignores the ordinary
`max_color_distance` gate, because foreground↔background is usually a large
RGB jump, and uses cycle-consistent geometry rather than photometric agreement
at the changing edge.

Retained foreground colors have two separate geometry-aligned regions. The
external edge pass covers foreground cells near the locked background. The
internal feature pass detects strong four-neighbor palette-color transitions
inside the current snapped foreground, then dilates them narrowly to cover
one- or two-cell lines and small structures. It excludes both background cells
and the external background-adjacent region, so a large `edge_radius` is no
longer needed to reach belly/skin borders, muscles, nose marks, folds, eyes,
mouth, or ears. The internal pass requires locked-background metadata from the
connected core `snapper_info`.

Both regions try exact discrete geometry consensus first. If exact palette
labels do not repeat—such as `A→B→C→D` shade shimmer—the fallback chooses the
observed aligned temporal medoid from a compact color cluster. The medoid is an
existing snapped color, never an RGB mean.

For stronger internal stability, `stability_lock` adds bounded trajectory
hysteresis. A previous stabilized palette label is warped into the current
frame with cycle-consistent motion and may be held for up to three frames while
the aligned candidate colors remain a compact cluster around it. The accepted
feature region is also propagated through valid motion, so a tiny eye or line
that vanishes completely—and therefore has no current-frame boundary—can still
be restored briefly. `maximum_lock` extends the window to 9 frames, region
radius to 3 cells, and hold cap to 6 frames. Holds and propagated regions still
expire; neither mode votes at fixed screen coordinates or creates new colors.

Occlusion, disocclusion, scene cuts, out-of-bounds motion, and unreliable flow
all fall back to the current snapped frame. Final RGB values are selected from
existing snapped frames only—there is no averaging, bilinear RGB blend, or new
off-palette color.

### Simple presets vs. Advanced node

The normal `Motion-Aware Cleanup` exposes only five controls:

- `cleanup_preset`: `balanced`, `strong`, `very_strong`, `outline_lock`,
  `detail_lock`, `stability_lock`, or `maximum_lock`;
- `flow_quality`: `fast`, `balanced`, or `quality`;
- `flow_backend`;
- `compute_device`;
- `edge_radius`.

Use `outline_lock` for crawling spine teeth/outlines: it keeps interior color
cleanup conservative while making external-edge consensus and temporal-medoid
fallback most aggressive. Use `detail_lock` when the remaining ripple is on
internal palette boundaries, thin lines, facial features, eyes, mouth, ears,
or similar structure. Use `stability_lock` when those internal details still
buzz after the other presets and suppressing brief micro-animation is an
acceptable trade-off. Use `maximum_lock` only when `stability_lock` still
buzzes: it uses a wider internal region, a 9-frame window, and up to six held
frames, so it can suppress more genuine detail animation. Use `very_strong` for
broad flicker not limited to recognized boundaries.

All individual thresholds remain available in the separate
`Motion-Aware Cleanup Advanced` node. This keeps ordinary workflows readable
without removing expert control.

If every frame chooses a different nearby palette shade, exact mode has no
winner regardless of threshold. The dedicated edge/feature passes can fall
back to a **temporal palette medoid**: one color that actually occurred in the
aligned window and minimizes distance to the other observed shades. It never
averages RGB or creates an off-palette color.

### Flow backends

- `raft_small` (default): torchvision's pretrained RAFT Small. It handles
  deformation and larger motion better; weights (~4 MB) download once on first
  use. Start with `flow_scale=0.5`, `raft_updates=8`, and reduce
  `flow_batch_size` if VRAM is tight.
- `integer_block_matching`: pure Torch, no model/download. It searches integer
  cell translations and is fast at sprite resolution, but is less reliable on
  rotation, deformation, textureless areas, and displacement beyond
  `search_radius`.

Set `compute_device=auto` (or `gpu` to fail loudly instead of falling back).
The `info` output must report `device=cuda...`. Full-resolution guide frames
are streamed to RAFT in `flow_batch_size` chunks. Forward and backward RAFT
passes run sequentially instead of doubling the effective model batch, and
each full-resolution flow is reduced to cell resolution immediately instead
of being retained until the whole clip finishes. The `quality` preset uses a
one-pair RAFT batch; `info` reports `flow_batch=1 raft_pass=sequential`.

These changes bound the Cleanup node's avoidable RAFT allocations, but ComfyUI
may still retain the original video, RMBG output, mask, other node outputs, and
previously loaded models. Those tensors are outside Cleanup's ownership, so the
total process RAM/VRAM can remain higher than Cleanup's own working set.

Start with `cleanup_preset=strong`, `flow_quality=balanced`, and
`edge_radius=2`. For crocodile-spine/outer-outline shimmer, switch to
`cleanup_preset=outline_lock`. For beige/green body borders, muscles, folds,
nose lines, eyes, mouth, ears, and other internal detail ripple, use
`cleanup_preset=detail_lock`. If that is still insufficient and stability is
more important than brief detail motion, use `cleanup_preset=stability_lock`,
then `maximum_lock` as the final aggressive step. If the detail's motion itself
is mistracked, also switch `flow_quality=quality`; otherwise `balanced` is
usually preferable.

In the Advanced node the roughly equivalent balanced edge controls are
`edge_agreement=0.6`, `edge_min_support=3`, `edge_max_color_distance=0.8`,
`edge_medoid_fallback=true`, and `edge_cluster_radius=0.35`. Internal locking
also requires `feature_edge_stabilization=true`; start with
`feature_radius=1` and `feature_contrast_threshold=0.08`. Bounded inertia is
controlled by `feature_hysteresis`, `feature_hold_frames`, and
`feature_hold_radius`; the `stability_lock` equivalents are `true`, `3`, and
`0.75`, with radius 2 and contrast threshold 0.05. `maximum_lock` uses window
9, radius 3, contrast threshold 0.04, hold 6, and hold radius 0.90. Advanced
windows up to 11 are available, but longer is not automatically safer.

Diagnostics:

- `motion_confidence` is a numeric MASK in `[0,1]`. Black means no reliable
  motion-aligned support; white means strong support. A fully black mask is a
  valid result, not a missing image.
- `confidence_preview` is the same information as a visible blue→green→yellow
  IMAGE. Connect this to ordinary `Preview Image` when Preview Mask is blank or
  visually ambiguous.
- `changed_cells` is a MASK showing all cells the node actually replaced. If
  this is black and `info` says `replaced=0`, the output is intentionally identical.
- `silhouette_changes` isolates only foreground↔background corrections—the
  useful diagnostic for crawling spine teeth and outline holes.
- `edge_color_changes` shows palette-index corrections in the retained
  foreground immediately beside the external background/silhouette.
- `feature_color_changes` shows actual output corrections on internal palette
  borders and thin drawn details.
- `feature_hysteresis_actions` shows where the previous motion-aligned stable
  label overrode the independently selected current label (output 9).
- `transparent_image` and `transparency_mask` are appended as outputs 10–11;
  outputs 1–9 retain their previous order.
- `info` reports `reliable_links`, `confidence_mean`, replacement counts,
  hysteresis state and `feature_held`, scene cuts, and whether exact
  `grid_crop` or fallback `resize_only` alignment ran.

If pixels still flicker, first inspect confidence: low confidence means the
node is deliberately refusing to invent correspondence. Increase RAFT quality
before lowering agreement thresholds. On the simple node, choose by region:
`strong` for general cleanup, `outline_lock` for the outside contour,
`detail_lock` for internal boundaries/lines, `stability_lock` when internal
medoid chatter remains, and `maximum_lock` only if that still buzzes; reserve
`very_strong` for broad flicker. Use
`flow_quality=quality` when confidence is low rather than merely increasing
cleanup strength. In the Advanced node, all-unique edge shades are
controlled mainly by `edge_medoid_fallback` and `edge_cluster_radius`; raise
the radius from `0.35` to `0.5–0.7` only if those shades genuinely belong to
one compact color family.

For continuous noise in the raw source, `ComfyUI-FlowDenoise` can still be used
**before** Video Pixel Snapper; it aligns frames with MEMFOF/RAFT and averages
continuous RGB, which is appropriate pre-quantization but not as a post-snap
pixel-art operation.

### Comparing before/after video

A dedicated comparer already exists, so this pack does not duplicate it. In
ComfyUI Manager search for **Deno Custom Nodes** and add `(Deno) Video Compare`;
it supports synchronized slider, side-by-side, difference, toggle, playback,
and a lossless IMAGE comparison output suitable for VHS Video Combine:
https://github.com/Deno2026/comfyui-deno-custom-nodes

Lighter alternatives are `Compare Frames`:
https://github.com/sidmehraajm/ssd_frame_compare
and `ComfyUI-compare-videos`:
https://github.com/surinder83singh/ComfyUI-compare-videos

Compare `Video Pixel Snapper.image` (A) against `Motion-Aware Cleanup.image`
(B), both before any later upscale. Toggle and Difference modes reveal sparse
single-cell changes much better than side-by-side playback.

## Recent fixes

- **v2.6.2 makes Load RGBA Image visible on ComfyUI 0.33.x.** Input-file
  enumeration now uses the stable input-directory API instead of the unavailable
  `folder_paths.get_input_files()` call.
- **v2.6.1 fixes release-folder ambiguity.** The ZIP now installs as
  `ComfyUI-Video-Pixel-Snapper` and startup logs explicitly confirm nine nodes
  plus `RGBA loader=yes`.
- **v2.6.0 adds a true RGBA still loader and embedded-alpha processing.** Hidden
  Photoshop RGB under alpha=0 is ignored/sanitized instead of becoming white
  versus magenta Live regions; Core/Analyzer accept RGBA without a mask wire.
- **v2.5.2 accepts valid Commit canvases despite browser alpha-rounding.** Exact
  browser/Torch source-hash mismatch is diagnostic rather than fatal, and every
  invalid hidden payload falls back to Snapped instead of crashing the graph.
- **v2.5.1 restores the Live Editor frontend.** A stray invalid trailer made
  browsers reject the v2.5.0 ES module before widget registration; validation
  now forces `.mjs` grammar and includes a DOM lifecycle smoke test.
- **Commit Live → output materializes the exact visible single-image edit.** A
  palette saves colors but not cell assignments; commit stores the actual Live
  RGBA canvas and emits it after the next Queue.
- **Live Editor no longer makes alpha-zero RGB look like a restored background.**
  Snapped/Live preview PNGs retain RGBA, edited Live preserves authoritative
  alpha, and `original_transparency_mask` restores alpha in the Original panel.
- **Sel-Out now processes `output_scale > 1` on the logical cell grid.** Connect
  core `info` and leave `input_pixel_scale=0`, or set the known scale manually;
  whole nearest-neighbor blocks are recolored together and restored exactly.
- **Live Editor now preserves Sel-Out after Add/Delete/Replace.** Connect
  `Selective Outline.changed_outline` to the new optional
  `Live Editor.postprocess_mask`; edited body cells still use RAW so new colors
  can appear, while outline cells retain the post-process structure.
- **Selective Outline / Sel-Out now recolors still-image linework from a fixed
  palette.** It supports outer-only or outer+internal scope, manual/auto light,
  RGB/RGBA/masks, exact changed-pixel diagnostics, and no off-palette output
  replacements.
- **Palette Coverage Analyzer now runs standalone.** Set the measured
  `analysis_pixel_size` and feed the render/mask batches directly; core
  `snapper_info` is optional.
- **Palette Coverage Analyzer replaces ad-hoc palette expansion.** It reports
  exact cell-level Oklab gaps, master-only character subpalettes, and redundant
  master slots without modifying the palette automatically.
- **Oklab perceptual matching prevents wrong-hue palette snaps.** The supplied
  fox orange now selects the orange ramp instead of a numerically close salmon;
  `rgb_legacy` remains available for comparison.
- **Transparent mode no longer builds multi-gigabyte duplicate batches.** It
  forces cell scale 1, writes chunks into one RGBA allocation, and exposes RGB
  as a zero-copy view; Live Editor/Retimer also reuse incoming RGBA storage.
- **Hard-alpha PNG now survives the complete graph.** Core, Cleanup, Live
  Editor, and Retimer append RGBA/mask outputs while preserving their existing
  RGB outputs for compatibility.
- **A focused Sprite Sheet node was added.** It copies reordered RGBA frames
  row-major without interpolation and keeps padding/unused cells transparent.
- **`maximum_lock` covers disappearing internal details.** It propagates the
  accepted feature region through valid motion and applies a wider, longer but
  still bounded discrete-label hold.
- **`stability_lock` suppresses internal medoid chatter.** Bounded,
  motion-aligned label hysteresis holds a discrete observed color for at most
  three frames while geometry and the local candidate cluster remain valid.
- **RAFT peak memory is bounded more tightly.** Directions run sequentially,
  full-resolution flow is discarded per chunk after cell-grid resizing, and
  the `quality` preset uses a one-pair model batch.
- **RMBG white fringe wiring is explicit.** Send original RGB to Pixel Snapper
  `image`; send only RMBG's mask to `foreground_mask` rather than using the
  already cut-out white-fringed IMAGE as both inputs.
- **Internal palette boundaries and thin drawn features now have a dedicated
  pass.** `detail_lock` targets belly/skin borders, muscles, folds, nose lines,
  eyes, mouth, ears, and similar foreground structure; `feature_color_changes`
  isolates its edits.
- **The normal cleanup node still has only five controls.** Seven tested cleanup
  presets and three flow-quality presets cover ordinary tuning; the full
  parameter surface remains in `Motion-Aware Cleanup Advanced`.
- **All-unique edge shades now use temporal palette medoid fallback.** This
  handles `A→B→C→D` shimmer where exact mode can never reach support 2.
- **Motion-Aware Cleanup now explicitly uses ComfyUI's GPU device.** Raw
  Comfy IMAGE batches commonly reside on CPU; guide frames are now streamed in
  chunks while flow/consensus run on CUDA.
- **Thin edge details have a separate palette pass.** Retained teeth/outlines
  can stabilize their internal palette indices without weakening ordinary
  photometric safeguards; `edge_color_changes` isolates these edits.
- **Silhouette stabilization now has a separate topology pass.** Transient
  one-cell teeth/holes can cross the large foreground/background color gap
  without weakening ordinary color safeguards; `silhouette_changes` shows
  exactly those edits.
- **Fixed-coordinate Temporal Cleanup was replaced, not tuned.** The same node
  ID now performs bidirectional motion estimation, occlusion/error rejection,
  and discrete palette-label consensus along trajectories. It requires the
  original video as `guide_image`.
- **BiRefNet/RMBG masks can now isolate foreground palette estimation.**
  Soft compositing fringes no longer consume several background-like palette
  slots, and masked-out cells share one locked background color.
- **Snapped and Live now match exactly before palette edits.** The Live
  panel uses the authoritative Snapped frame while its palette is unchanged,
  rather than independently approximating Python's cell-reduction method.
- **Core and edited Live preview now share Oklab/RGB matching.** Edited Live is
  still a fast median-cell approximation, but no longer uses a different
  nearest-color metric from the Python core.
- **All nodes use their own category** (`Video Pixel
  Snapper`) in the Add Node menu, instead of sitting in the generic
  `image/transform` folder.
- **Live Editor: Pick/Delete/Replace edits no longer get silently
  wiped by a re-run.** `onExecuted` used to unconditionally reset the
  working palette to the freshly auto-detected one on *every*
  execution — including re-runs triggered by ComfyUI itself (cache
  invalidation, re-queued runs) that have nothing to do with wanting
  your edits discarded. This looked exactly like "Pick doesn't add
  colors" or "Delete works but Add doesn't" from the outside, depending
  on timing. Untouched palettes now refresh when the core result changes;
  manually edited palettes survive incidental re-runs, and `Reset`
  explicitly accepts the latest auto-detected palette.
- **Grid-phase alignment fixed by one pixel.** Edge samples represent the
  transition between pixels `i` and `i+1`; phase now starts at `i+1`
  instead of on the boundary pixel itself. A perfectly aligned 8 px grid
  therefore reports phase 0 rather than 7.
- **Live Editor now honors output upscaling.** Its backend metadata also
  carries the true cell dimensions, so a 5x nearest-neighbor output is
  reduced/recolored as cells rather than being mistaken for five times
  as many tiny cells.
- **Palette previews support the full 256 colors.** The old sparse row
  sampler could skip wide-strip swatches and silently capped the editor
  at 64 colors.
- **Long clips use bounded processing chunks** in the core snapper,
  reducing peak temporary-memory use without changing results.
- **Frame Retimer restores `sequence_json` after reopening a workflow**
  and shows the correct output count immediately; adjacent drag-reorder
  is no longer a silent no-op.
- **Live Editor: widget now explicitly fills the node's width**
  (`width:100%; box-sizing:border-box`) instead of relying on implicit
  block-fill layout. On at least one other machine, widening the node
  left the widget's internal content a fixed size instead of growing
  with it — this is the fix, on the theory that a different frontend
  version/layout context didn't stretch the container the way ours did
  by default. Combined with the existing `ResizeObserver` on the
  widget, which now has an actual width change to react to.

## Roadmap / not built yet

Requested and deliberately deferred rather than built all at once
(quality over quantity — everything above already went through several
rounds of "build it, test it for real, fix what's actually wrong"; five
more substantial features in one pass isn't a good way to keep that up):

- Per-frame transform tools (scale, rotate, x/y position)
- A "shake" preset (x/y shake + rotation, frequency + amplitude per
  axis, keyframeable, applicable to a range or the whole clip)
- A wave-distortion effect (configurable frequency/amplitude)
- Single-frame interpolation for filling 1-2 gap frames (likely needs a
  small model, or a call out to an existing interpolation node)
- A dedicated node for drawing directly on frames — `palette_editor.html`
  already has basic manual pixel painting, just not as an in-ComfyUI node

## Known limitations

- Coverage suggestions describe the analyzed clips only. A production global
  palette must be calibrated on representative original frames/masks from many
  character hue families; do not freeze master from screenshots or one sprite.
- Transparent export is hard cell-level alpha intended for PNG and sprite
  sheets. H.264/MP4 does not preserve it; use the parallel RGB output for VHS.
- Not every third-party ComfyUI image node accepts four-channel IMAGE tensors.
  Keep processing on RGB where needed and use appended `transparent_image`
  outputs only through the documented alpha-aware branch.
- Grid size/phase assume a single, uniform block size across X and Y.
- Motion-Aware Cleanup is deliberately conservative: flow failure or
  disocclusion leaves the current frame unchanged, so some flicker can remain.
  `stability_lock` deliberately trades up to three frames of internal
  micro-animation for stronger stability, while `maximum_lock` trades up to six;
  use `detail_lock` when that trade-off is undesirable. The node cannot
  reconstruct a detail that the source video
  semantically changes or hallucinates in every frame; keyframe
  propagation/manual cleanup is still the production answer for those regions.
- Integer block matching only models local cell translations. Use RAFT for
  deformation, rotation, and larger motion.
- Mask output is intentionally cell-level and hard-edged. Tune
  `mask_cell_threshold` for thin details; it is not an alpha-matting output.
- `accent_slots` is a real trade-off, not a free bonus: each reserved
  slot is taken from the regular (bulk) palette.
- The in-graph Live panel uses a fast per-channel-median representative
  of each raw grid cell. It honors the core grid/phase when `info` is
  connected, but it does not reproduce palette-dependent majority voting,
  dithering, or despeckle exactly. Re-run the core node for the final result.
