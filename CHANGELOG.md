# Changelog

## 1.7.1 — 2026-08-16

### RAFT memory optimization

- Forward and backward RAFT directions now execute sequentially. Previously
  they were concatenated into one effective batch twice the requested
  `flow_batch_size`, which made full-resolution `quality` correlation volumes
  unnecessarily large.
- Each RAFT flow is resized to the snapped cell grid immediately per chunk.
  Full-resolution flows are no longer retained for every adjacent pair until
  the entire clip finishes.
- The simple `quality` preset now uses `flow_batch_size=1`; Advanced users can
  still choose their own batch size. `info` reports the actual flow batch and
  `raft_pass=sequential`.
- A CPU RAFT Small comparison produced bit-identical low-resolution flows
  between v1.7.0 and the streamed implementation on the smoke clip.
- Added a regression confirming sequential direction calls and per-chunk
  reduction to cell resolution. 28 automated tests pass.

### RMBG edge-fringe clarification

- Documented the recommended wiring: original RGB video to Pixel Snapper
  `image`, BiRefNet/RMBG MASK only to `foreground_mask`, and the desired flat
  background to `background_image`.
- Feeding the already cut-out RMBG IMAGE can preserve upstream white fringe
  pixels. Existing `mask_threshold` and `mask_cell_threshold` remain available
  for stricter rejection when needed.

## 1.7.0 — 2026-08-16

### Internal feature-edge stabilization

- Added a dedicated motion-aligned pass for internal foreground structure:
  major palette-region borders, thin muscle/nose/fold lines, and small eyes,
  mouth, ears, and similar details.
- Internal regions are detected from strong four-neighbor palette-color
  transitions in the current snapped frame, then narrowly dilated to cover
  one- or two-cell lines and small features.
- Locked-background cells and the existing background-adjacent outline region
  are excluded, so internal diagnostics are distinct from external silhouette
  and outline corrections. Core `snapper_info` background metadata is required.
- Only in-bounds, forward/backward cycle-consistent, non-scene-cut candidates
  participate. Exact discrete consensus is attempted first; the observed-color
  temporal medoid is used only when exact consensus cannot replace the cell.
- Every replacement copies an existing snapped RGB value. No averaging,
  interpolation, or off-palette color synthesis was added.

### Interface and diagnostics

- Added the `detail_lock` cleanup preset to the normal node without increasing
  its five visible controls. Existing presets retain their previous behavior.
- Added `feature_edge_stabilization`, `feature_radius`, and
  `feature_contrast_threshold` to `Motion-Aware Cleanup Advanced`.
- Appended `feature_color_changes` as the eighth output; the first seven outputs
  keep their prior order. `edge_color_changes` now remains strictly external.
- Added `features` and `feature_replaced` fields to the `info` diagnostic.
- Added regressions for an internal green/beige boundary, a missing cell in a
  thin line far from the background, and `detail_lock` preset activation.
- 27 automated tests pass.

## 1.6.0 — 2026-08-15

### Simplified controls

- The normal `Motion-Aware Cleanup` now exposes five controls instead of the
  full 24-parameter surface: cleanup preset, flow quality, backend, device,
  and edge radius.
- Added `balanced`, `strong`, `very_strong`, and `outline_lock` cleanup presets.
- Added `fast`, `balanced`, and `quality` flow presets.
- The complete controls remain available in a separate
  `Motion-Aware Cleanup Advanced` node.

### Temporal palette medoid

- Exact palette mode cannot stabilize `A→B→C→D` edge shimmer when every frame
  uses a different index. Added a temporal medoid fallback that selects the
  existing aligned color minimizing distance to the other observed colors.
- Added `edge_medoid_fallback` and `edge_cluster_radius` to the Advanced node.
- The medoid is accepted only for a compact color cluster and never creates an
  averaged/off-palette RGB value.
- Added regressions for all-unique palette shades and the preset-driven node.
- 23 automated tests pass.

## 1.5.0 — 2026-08-15

### GPU execution

- Added `compute_device=auto/gpu/cpu`. Auto uses
  `comfy.model_management.get_torch_device()` instead of inheriting the CPU
  device of ComfyUI IMAGE tensors.
- Motion flow, confidence tests, warping, and label consensus now execute on
  the selected GPU and return results to the original Comfy tensor device.
- RAFT now streams full-resolution guide pairs in `flow_batch_size` chunks;
  the entire 1920p guide batch is no longer duplicated onto VRAM at once.
- `info` reports the actual compute device.

### Edge palette stabilization

- Added a dedicated geometry-aligned color pass for foreground cells within
  `edge_radius` cells of the locked background. It targets palette shimmer
  inside retained spine teeth, horns, outline pixels, fingers, and other thin
  edge details.
- Added `edge_color_stabilization`, `edge_radius`, `edge_agreement`,
  `edge_min_support`, and `edge_max_color_distance` controls.
- Added `edge_color_changes` MASK output and `edge_replaced` diagnostics.
- Added a regression where a one-frame photometrically different palette color
  inside a persistent edge detail is corrected without changing its shape.
- 21 automated tests pass.

## 1.4.0 — 2026-08-15

### Added

- Separate motion-compensated silhouette/topology stabilization for transient
  one-cell teeth, holes, missing outline cells, and crawling sprite edges.
- `silhouette_stabilization`, `silhouette_agreement`, and
  `silhouette_min_support` controls.
- `silhouette_changes` MASK output isolating only foreground/background edits.

### Why it is separate

- Ordinary color cleanup intentionally limits RGB palette jumps. That safety
  gate prevented spine-color↔background corrections and left silhouette teeth
  untouched.
- Silhouette changes now require stricter temporal support but use
  forward/backward cycle-consistent geometry rather than photometric agreement
  at the changing edge. This lets a one-frame tooth be removed without globally
  raising `max_color_distance` and reopening color trails.

### Validation

- Added a regression where a one-frame green spine tooth differs
  photometrically from all neighboring black-background frames. The new
  topology pass removes only that cell while preserving the persistent body.
- 20 automated tests pass.

## 1.3.1 — 2026-08-15

### Added

- Optional `snapper_info` input on Motion-Aware Cleanup. Connect the core
  `info` output to transfer source resolution, block size, phase, and cell
  bounds.
- Exact guide crop alignment before flow estimation, including proportional
  crop coordinates when guide and core-source resolutions differ.
- Visible blue→green→yellow `confidence_preview` IMAGE for ordinary Preview
  Image nodes.
- `changed_cells` MASK showing only cells actually replaced by cleanup.
- `confidence_mean` and guide-alignment mode (`grid_crop`/`resize_only`) in the
  diagnostic info string.

### Clarified

- A 1920p guide with a ~144p snapped image is supported: flow is resized and
  vector magnitudes are converted from guide pixels to output-cell units.
  `snapper_info` fixes the additional phase/crop alignment that a whole-frame
  resize cannot infer.
- Documented existing video comparison nodes instead of adding another
  incompatible duplicate.

## 1.3.0 — 2026-08-15

### Breaking change: Temporal Cleanup replaced

- `VideoPixelSnapperTemporalCleanup` keeps the same node ID, but now requires
  `guide_image` containing the matching original/pre-snap frames. Existing
  workflows must reconnect/recreate this node.
- Removed fixed-screen-coordinate temporal mode voting. It inherently confused
  motion with flicker and caused frozen colors, smearing, and trails.

### Added

- Bidirectional motion estimation with two backends:
  - pretrained torchvision RAFT Small (default, one-time ~4 MB download);
  - pure-Torch integer block matching as a no-model fallback.
- Forward/backward cycle-consistency, photometric, bounds, patch-cost, and
  scene-cut rejection.
- Motion-compensated discrete palette-label consensus over 3–7 frames.
- Conservative data-fidelity and maximum-color-jump gates.
- `motion_confidence` MASK and detailed diagnostic `info` output.

### Guarantees

- Final colors are copied from existing snapped frames only. No RGB temporal
  averaging, bilinear RGB output, or off-palette color generation.
- Unreliable motion, occlusion, disocclusion, and scene cuts fall back to the
  current snapped frame instead of propagating stale history.

### Validation

- Added moving-sprite regression coverage: a palette-label flicker moving one
  cell per frame is corrected while the silhouette remains trail-free.
- RAFT Small was smoke-tested end-to-end in addition to the model-free backend.

## 1.2.0 — 2026-08-15

### Added

- Optional `foreground_mask` input for BiRefNet/RMBG workflows.
- Optional `background_image` input; connect the flat `Empty Image` to lock
  its median RGB as the single background palette color.
- `mask_threshold`, `mask_cell_threshold`, and `invert_mask` controls.

### Changed

- With a mask connected, palette estimation and majority/median/center cell
  reduction use only confident foreground pixels.
- Masked-out output cells are forced to one dedicated background palette
  index, preventing soft compositing fringe shades from consuming multiple
  palette slots.
- Auto palettes reserve one of `k_colors` for that background; custom
  palettes gain or replace one entry when the exact background is absent.
- The Live Editor receives the locked background color through the existing
  `info` connection and preserves those background cells after palette edits.
- Grid edge detection now uses perceptual luma instead of a plain RGB mean,
  so equal-average colors such as pure red, green, and blue remain
  distinguishable.
- Duplicate k-means centers are removed from auto palettes.

## 1.1.2 — 2026-08-15

### Fixed

- Exact-baseline detection is now independent of palette order. Two palettes
  containing the same colors no longer fall into the approximate RAW
  reclassification path just because their swatches are ordered differently.
- An untouched working palette always uses the exact Snapped canvas, even if
  frontend state makes an ordered palette comparison unreliable.
- The Live panel now visibly reports either `exactly matches Snapped` or
  `edited preview`, making the active path unambiguous while testing.

## 1.1.1 — 2026-08-15

### Fixed

- When the working palette is unchanged, **Live (your edit)** now reuses
  the authoritative Python-generated **Snapped (auto palette)** frame.
  The two panels are therefore pixel-identical at baseline. Previously,
  Live independently applied a median-cell browser approximation, so it
  could differ despite having exactly the same colors whenever the core
  used majority voting, center weighting, dithering, or despeckle.
- After the first real palette edit, Live still switches to RAW-based
  reclassification so newly added colors can appear immediately. That path
  remains a fast preview approximation; re-running the core is authoritative.

## 1.1.0 — 2026-08-15

### Fixed

- Corrected grid phase by one source pixel: edge index `i` now maps to the
  cell start at `i + 1`, so an aligned 8 px grid reports phase 0 rather than 7.
- Live Editor now consumes block, phase, true cell dimensions, and output
  scale from the optional `info` connection. Upscaled snapped outputs are no
  longer mistaken for denser cell grids.
- Live palette loading scans the complete swatch strip and supports all 256
  colors instead of sparsely sampling and silently capping at 64.
- Untouched Live Editor palettes refresh when the core result changes, while
  manually edited palettes remain protected from incidental re-executions.
- Invalid manual hex colors are rejected instead of becoming unintended black.
- Frame Retimer restores `sequence_json` when reopening a workflow, updates the
  output counter immediately, and correctly supports adjacent drag-reordering.
- `despeckle` no longer resolves four-way neighbor ties by arbitrary palette
  index order.
- Block-size fallback 1 no longer produces NaN saliency values.
- RGBA inputs are normalized to RGB in all processing nodes.

### Performance

- Core processing automatically chunks long clips to bound majority-histogram
  and source-pixel temporaries.
- Temporal Cleanup votes in spatial chunks and keeps packed color keys in
  integer space.

### Behavior

- Temporal Cleanup now requires a unique temporal winner; equal-frequency
  color ties are left unchanged.
- Corrected temporal cells copy an exact RGB value from a contributing
  source frame. Untouched and corrected colors are no longer reconstructed
  through an 8-bit key, so arbitrary floating-point palette values remain
  bit-exact.

## 1.0.0

- Initial four-node release: Video Pixel Snapper, Live Editor, Frame Retimer,
  and Temporal Cleanup, plus the standalone palette editor.
