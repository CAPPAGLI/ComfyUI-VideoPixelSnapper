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
- **Motion-aware discrete cleanup**: estimates bidirectional motion from
  the original video, rejects occlusions/flow errors, and stabilizes palette
  labels along trajectories without RGB blending—including external outlines,
  internal palette-region borders, and thin drawn features.
- **In-graph live palette editor** (see below) — no need to leave
  ComfyUI to tweak the palette.

## Installation

```
ComfyUI/custom_nodes/ComfyUI-VideoPixelSnapper/
├── __init__.py
├── video_pixel_snapper.py
├── frame_retimer.py
├── temporal_denoise.py
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
  With `foreground_mask`, one of these slots is reserved for the background.
- `accent_slots` — how many palette slots to reserve for rare-but-distinct
  colors instead of plain frequency-based clustering.

### Background removal / foreground-mask wiring

For BiRefNet/RMBG workflows, the cleanest setup is:

```text
original video IMAGE ─────────────────────> Video Pixel Snapper.image
BiRefNet/RMBG foreground MASK ────────────> Video Pixel Snapper.foreground_mask
Empty Image (your flat background color) ─> Video Pixel Snapper.background_image
```

**Feed the original RGB video—not the RMBG cutout—to `image`.** Connect only
BiRefNet/RMBG's MASK output to `foreground_mask`, and connect the desired flat
`Empty Image` to `background_image`. This avoids feeding white/transparent
RMBG fringe pixels into foreground palette reduction. If random white cells
are already visible immediately after RMBG and no Cleanup diagnostic marks
them, Cleanup did not create them and cannot reliably infer their original
color.

A pre-composited/cutout IMAGE can still work when its edge colors are clean,
but it is not the recommended wiring. The node uses confident mask pixels for
palette estimation and cell voting, then writes one exact background color to
every masked-out grid cell.

- `mask_threshold` (default `0.5`) controls which source pixels may influence
  foreground colors. Raise it toward `0.7–0.9` if soft RMBG fringe colors
  still leak in.
- `mask_cell_threshold` (default `0.25`) controls silhouette coverage at cell
  level. Lower values preserve thin details; higher values remove more edge
  contamination.
- `invert_mask` handles nodes where white means background. BiRefNet/RMBG
  normally outputs white foreground, so leave it off first.
- If `background_image` is omitted, the node samples one background color
  from high-confidence masked-out pixels in `image`.

The output remains RGB with a solid background; this feature does not create
an alpha-channel output. The Live Editor preserves locked background cells
when it produces an edited approximation.

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
   info ─────────────┤          │
                      ▼          ▼
        [Video Pixel Snapper (Live Editor)]
           original_image  <- (same image feeding the core node)
           snapped_image   <- core node's `image` output
           palette_preview <- core node's `palette_preview` output
           info             <- core node's `info` output (recommended)
```

The first three image connections are required. Connecting `info` is
optional but recommended: it gives the browser widget the exact detected
block size, phase, cell dimensions, and output scale. Without it, Live
uses a centered best-effort reduction and may not line up with Snapped.
The editor node passes `snapped_image` through as its own `image` output,
so it can also sit inline in the middle of a chain.

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
  reclassification so newly added colors can appear immediately.
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
original/pre-snap video IMAGE ───────────────> Motion-Aware Cleanup.guide_image
Video Pixel Snapper image (output_scale=1) ──> Motion-Aware Cleanup.image
Video Pixel Snapper info ────────────────────> Motion-Aware Cleanup.snapper_info
Motion-Aware Cleanup.image ──────────────────> Live Editor / Retimer / output
```

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

Occlusion, disocclusion, scene cuts, out-of-bounds motion, and unreliable flow
all fall back to the current snapped frame. Final RGB values are selected from
existing snapped frames only—there is no averaging, bilinear RGB blend, or new
off-palette color.

### Simple presets vs. Advanced node

The normal `Motion-Aware Cleanup` exposes only five controls:

- `cleanup_preset`: `balanced`, `strong`, `very_strong`, `outline_lock`, or
  `detail_lock`;
- `flow_quality`: `fast`, `balanced`, or `quality`;
- `flow_backend`;
- `compute_device`;
- `edge_radius`.

Use `outline_lock` for crawling spine teeth/outlines: it keeps interior color
cleanup conservative while making external-edge consensus and temporal-medoid
fallback most aggressive. Use `detail_lock` when the remaining ripple is on
internal palette boundaries, thin lines, facial features, eyes, mouth, ears,
or similar structure. Use `very_strong` for broad flicker not limited to
recognized boundaries.

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
`cleanup_preset=detail_lock`. If the detail's motion itself is mistracked, also
switch `flow_quality=quality`.

In the Advanced node the roughly equivalent balanced edge controls are
`edge_agreement=0.6`, `edge_min_support=3`, `edge_max_color_distance=0.8`,
`edge_medoid_fallback=true`, and `edge_cluster_radius=0.35`. Internal locking
also requires `feature_edge_stabilization=true`; start with
`feature_radius=1` and `feature_contrast_threshold=0.08`.

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
- `feature_color_changes` shows only corrections on internal palette borders
  and thin drawn details. It is appended as output 8; outputs 1–7 retain their
  previous order.
- `info` reports `reliable_links`, `confidence_mean`, replacement counts
  (including separate `edge_replaced` and `feature_replaced`), scene cuts, and
  whether exact `grid_crop` or fallback `resize_only` alignment ran.

If pixels still flicker, first inspect confidence: low confidence means the
node is deliberately refusing to invent correspondence. Increase RAFT quality
before lowering agreement thresholds. On the simple node, choose by region:
`strong` for general cleanup, `outline_lock` for the outside contour, and
`detail_lock` for internal boundaries/lines; reserve `very_strong` for broad
flicker. Use `flow_quality=quality` when confidence is low rather than merely
increasing cleanup strength. In the Advanced node, all-unique edge shades are
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
- **The normal cleanup node still has only five controls.** Five tested cleanup
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

- Grid size/phase assume a single, uniform block size across X and Y.
- Motion-Aware Cleanup is deliberately conservative: flow failure or
  disocclusion leaves the current frame unchanged, so some flicker can remain.
  It cannot reconstruct a detail that the source video semantically changes or
  hallucinates in every frame; keyframe propagation/manual cleanup is still the
  production answer for those regions.
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
