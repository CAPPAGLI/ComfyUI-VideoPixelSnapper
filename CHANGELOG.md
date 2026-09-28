# Changelog

## 2.6.2 — 2026-08-30

### RGBA Loader appears on ComfyUI 0.33.x

- Fixed `Load RGBA Image` being absent from the Add Node menu even though the
  package registered nine class mappings. Its `INPUT_TYPES` called
  `folder_paths.get_input_files()`, which is not available in the user's
  ComfyUI 0.33.1 build; object-info generation therefore skipped that one node.
- Replaced the version-dependent call with recursive enumeration rooted at the
  stable `folder_paths.get_input_directory()` API. PNG, JPEG, WebP, BMP, GIF,
  and TIFF files are listed with input-relative forward-slash paths.
- Added a recursive-listing regression. All 69 tests pass; canonical-folder
  extracted import, nine-node registration, ES-module validation, and Live
  Editor DOM lifecycle smoke pass.

## 2.6.1 — 2026-08-30

### Canonical install folder and registration diagnostic

- Release ZIP now uses the canonical top-level folder
  `ComfyUI-Video-Pixel-Snapper`, matching the user's existing installation and
  preventing an old eight-node folder from being loaded instead of—or after—a
  separately extracted `ComfyUI-VideoPixelSnapper` folder.
- Startup now prints
  `[VideoPixelSnapper] v2.6.1 loaded: 9 nodes (RGBA loader=yes)`, providing an
  immediate authoritative registration check in ComfyUI logs.
- No processing behavior changed from v2.6.0. All 68 tests, nine-node extracted
  import, ES-module validation, and Live Editor DOM lifecycle smoke pass.

## 2.6.0 — 2026-08-30

### Alpha-aware still-image workflow

- Added the ninth focused node, `Load RGBA Image`, because standard ComfyUI
  `Load Image` deliberately separates PNG alpha into a MASK and returns RGB on
  its IMAGE socket. The new loader emits true RGBA, foreground/transparency
  masks, and RGB compatibility output.
- `sanitize_hidden_rgb=true` canonicalizes RGB to black only where alpha is
  exactly zero. The supplied `captain.png` contains 220,787 alpha-zero pixels
  with 13,202 hidden RGB variants—including large white and magenta regions—so
  Photoshop selection/mask and eraser operations looked different whenever
  alpha was dropped. Visible and partial-alpha pixels are untouched.
- Video Pixel Snapper Core now automatically uses nontrivial embedded RGBA
  alpha when no explicit `foreground_mask` is connected. Hidden transparent RGB
  is excluded from palette/cell statistics; `background_mode=transparent`
  accepts embedded alpha and reports `mask_source=embedded_alpha`.
- Palette Coverage Analyzer likewise uses embedded alpha automatically.
- Live Editor's RAW median reducer now excludes source samples with alpha below
  0.5, preventing invisible white/magenta Photoshop mattes from receiving or
  influencing visible palette colors.
- Fully opaque RGBA remains on the unmasked path, preserving prior behavior.
  Standard Load Image still requires its MASK connection because alpha is no
  longer present in its RGB IMAGE tensor; this cannot be inferred downstream.
- Added embedded-alpha Core/Analyzer, alpha-aware browser median, and RGBA-loader
  sanitization/preservation regressions. All 68 tests pass; ES-module and DOM
  lifecycle validation pass.

## 2.5.2 — 2026-08-30

### Commit fingerprint no longer crashes or rejects a valid browser canvas

- Fixed `Committed Live belongs to a different Original image` immediately
  after Commit. Browser Canvas can round RGB at soft/transparent edges through
  premultiplied-alpha conversion, so its exact byte FNV is not guaranteed to
  match the original Torch tensor even when the image is the same.
- Fingerprint mismatch is now diagnostic (`mismatch_accepted`) rather than a
  hard rejection. Output dimensions, single-image batch, Original dimensions
  for new commits, PNG structure, and decoded output dimensions remain hard
  integrity checks.
- Any malformed, stale-size, or otherwise invalid hidden commit now fails soft:
  Live Editor passes through the authoritative Snapped input and reports
  `commit rejected` through `commit_info`/the browser status instead of crashing
  the complete workflow.
- New commit payloads include Original width/height as a stable source guard.
  Existing v2.5.1 payloads remain accepted.
- Added regressions for the reported browser/Torch fingerprint mismatch and for
  corrupt-commit fail-soft behavior. All 63 tests pass; ES-module syntax and
  jsdom widget lifecycle validation pass.

## 2.5.1 — 2026-08-30

### Live Editor frontend load repair

- Fixed the actual reason the v2.5.0 Live Editor UI disappeared: four stray
  closing lines remained at the end of `web/video_pixel_snapper.js`, so the
  browser rejected the entire ES module before `registerExtension` ran. Python
  still loaded and displayed `committed_live_png`, which made the failure look
  like a widget/layout problem.
- Corrected the validation gap: `node --check file.js` used the local CommonJS
  package default and returned success for this file, while the browser parses
  Comfy extensions as ES modules. The automated suite now copies the frontend
  to a `.mjs` path and checks the actual ES-module grammar.
- Converted remaining UI punctuation to ASCII to avoid misleading mojibake when
  Windows/Comfy serves the JavaScript source without an explicit UTF-8 charset.
- Added an external jsdom lifecycle smoke test during release validation: the
  module registers, `onNodeCreated` completes, the DOM widget attaches, the
  Commit button exists, and the hidden commit-state widget is hidden.
- All 62 automated tests pass, including the new ES-module syntax regression.

## 2.5.0 — 2026-08-30

### Exact Live image commit for single stills

- Added **Commit Live → output** to the Live Editor. A palette PNG stores only a
  list of colors, not the per-cell assignments visible in the browser. Reloading
  that palette into Core therefore legitimately re-ran Core's majority logic
  and could not reproduce Live's separate RAW-median approximation.
- Commit captures the exact visible Live RGBA canvas at the authoritative
  Snapped output dimensions with nearest-neighbor scaling, stores it in the
  hidden serialized `committed_live_png` widget, and materializes that PNG on
  Live Editor's existing `image`, `transparent_image`, and `transparency_mask`
  outputs after the next Queue.
- Added `Clear commit`; any subsequent Add/Delete/Replace/Reset/substitution
  action invalidates the old commit so stale pixels cannot silently survive a
  later palette edit.
- Commits are restricted to one-image batches, matching the requested still
  workflow. A source fingerprint lets a committed look survive upstream palette
  reloads while rejecting reuse on a different Original image; dimensions and
  payload integrity are validated server-side.
- Appended `commit_info` after the three existing outputs. Existing RGB/RGBA/mask
  output positions remain unchanged.
- Palette-save messaging now explains that saving colors alone cannot preserve
  Live pixel assignments and points to Commit Live for exact output.
- Added exact PNG materialization and stale-source rejection regressions. All 61
  Python tests pass; frontend JavaScript syntax validation passes.

## 2.4.3 — 2026-08-30

### Honest alpha in Live Editor previews

- Fixed a misleading Live Editor preview path that saved only RGB even when
  `snapped_image` was RGBA. Transparent pixels retain hidden RGB by design;
  dropping alpha therefore made the removed background/key look restored in
  Snapped and edited Live panels.
- Snapped browser previews now use the actual `transparent_image` RGBA tensor.
  Edited Live previews copy authoritative cell alpha after RAW palette
  reclassification and Sel-Out post-process-mask overlay.
- Added optional `original_transparency_mask` to Live Editor. Connect standard
  Load Image's MASK (white means transparent) to reconstruct honest RGBA for
  the Original panel; ComfyUI normally separates alpha from its RGB IMAGE.
- Added checkerboards behind all three canvases and renamed the middle panel to
  `Snapped / processed`, making both real transparency and a downstream Sel-Out
  stage explicit.
- Pick/Delete/Replace now ignore alpha-zero preview pixels instead of treating
  their invisible RGB payload as a visible palette color.
- Existing append-only output contract is unchanged: `image` remains RGB for
  compatibility and cannot contain transparency; use `transparent_image` for
  alpha-aware saving/downstream nodes.
- Added regressions for RGBA preview payload preservation, reconstruction of
  Original alpha from a Comfy transparency mask, and RGB Snapped alpha recovery
  from the core background metadata used by the reported workflow. All 59
  Python tests pass; frontend JavaScript syntax validation passes.

## 2.4.2 — 2026-08-30

### Scale-aware Sel-Out

- Fixed Selective Outline on `Video Pixel Snapper.output_scale > 1`. The old
  raster-space detector saw an enlarged logical outline cell as a thick line
  and recolored only its outermost one-pixel row.
- Added `input_pixel_scale`: `0` reads the exact scale and cell/output
  dimensions from optional `snapper_info`; positive values provide a manual
  override when info is unavailable.
- Scaled input is collapsed to the logical pixel grid, processed once there,
  and restored by exact nearest-neighbor repetition. RGB, RGBA, alpha, and
  `changed_outline` keep the original output dimensions, with complete N×N
  logical cells changed together.
- Added strict validation that a declared scaled input is an exact
  nearest-neighbor enlargement, with only a one-level encoded PNG tolerance.
  Bilinear/antialiased or incorrectly declared inputs fail clearly instead of
  being silently damaged.
- Added four regressions proving scaled RGBA and RGB+mask output are
  bit-identical to native-Sel-Out-then-nearest, `snapper_info` scale parsing
  works, and invalid non-nearest input is rejected. All 57 tests pass.

## 2.4.1 — 2026-08-30

### Live Editor preserves Sel-Out after palette edits

- Added the optional Live Editor `postprocess_mask` input. Connect Selective
  Outline / Sel-Out's `changed_outline` while feeding the Sel-Out RGB/RGBA
  result to `snapped_image`.
- The unedited Live baseline still reuses the authoritative Snapped image
  exactly. After Add/Delete/Replace, ordinary cells continue to reclassify
  from pre-snap RAW so newly added colors can appear; mask-selected cells now
  reclassify from the authoritative post-processed image, preserving Sel-Out
  geometry instead of reverting those pixels to the raw black outline.
- Post-processed pixels are remapped through the edited palette rather than
  copied verbatim, so removed/replaced colors cannot remain hidden in the Live
  preview. The final Python result after graph re-execution remains
  authoritative because the browser does not duplicate Sel-Out's full local
  material/light solver.
- Hard-thresholded post-process masks are saved with the preview payload and
  reduced to cell resolution with nearest sampling only.
- Added a regression for mask normalization, preview payload registration, and
  unchanged RGBA pass-through. All 53 Python tests pass; frontend JavaScript
  syntax validation passes.

## 2.4.0 — 2026-08-30

### Palette-locked selective outlining for still images

- Added the eighth focused node, `Selective Outline / Sel-Out`, for replacing
  black sprite linework with darker material-related entries from a connected
  palette. Replacements are exact palette colors; no RGB interpolation or new
  shade synthesis is performed.
- Added `outer_only` and `outer_and_internal` scope modes. The conservative
  default changes only one-pixel silhouette boundary cells, while the opt-in
  mode also handles thin internal black lines.
- Added deterministic eight-way manual lighting and an optional per-still auto
  heuristic based on the bright-material centroid. Auto direction and
  confidence are reported in `info`; manual remains the predictable production
  choice.
- Added `subtle`, `balanced`, and `strong` style presets instead of exposing
  low-level Oklab target parameters.
- Added exact/near-black threshold control, including support for deliberate
  dark master entries such as `#181425`.
- Added RGB, RGBA, white-foreground mask, and Comfy white-transparency mask
  handling. Detection prefers a connected mask, then RGBA alpha, then a
  conservative flat-border RGB flood fill. Input alpha is preserved and RGB
  fallback never silently removes the solid background.
- Added RGB, RGBA, exact changed-outline MASK, and diagnostic outputs.
- Added seven regressions for external/internal scope, palette-only replacement,
  manual/automatic lighting, both mask conventions, hard mask-derived alpha,
  solid-background fallback, RGBA preservation, and dark non-outline
  protection. All 52 tests pass.

## 2.3.1 — 2026-08-30

### Missing-cluster percentage correction

- Fixed `(% cells)` coverage in missing-color suggestions. It previously counted
  every observed cell that a suggestion represented better than master,
  including already-covered cells below `missing_threshold`; totals could
  therefore exceed the reported `gap_cells` share.
- Suggestion coverage now counts only the original threshold-qualified gap
  cells after near-match and near-black suppression. Cluster percentages are
  mutually assigned and their total cannot exceed gap share.
- Extended the lavender/background regression to enforce that invariant. All
  45 automated tests pass.

## 2.3.0 — 2026-08-30

### Standalone palette analysis

- Palette Coverage Analyzer no longer requires Video Pixel Snapper. New
  `analysis_pixel_size` uses a manually measured source block size, or `0` to
  auto-detect size across sampled renders; shared grid phase is estimated from
  the same batch.
- Added standalone `mask_threshold` and `mask_cell_threshold` controls. When a
  compatible optional `snapper_info` is connected, its exact grid and mask
  thresholds still take precedence for pipeline comparison.
- Reports now identify `grid_source=snapper_info`, `standalone_manual`,
  `standalone_auto`, or `standalone_auto_fallback_1` explicitly.
- Added a regression proving a manual two-pixel grid produces exact standalone
  cell analysis with no Pixel Snapper node. 45 automated tests pass.

## 2.2.1 — 2026-08-30

### Dark-residue coverage correction

- Near-black resize/compression residues such as `#040307` were sometimes
  reported as major missing colors even when exact `#000000` existed. Oklab is
  intentionally sensitive near black, while these few-level encoded-RGB
  differences are not useful new palette slots.
- Coverage error and missing suggestions now treat a cell as already covered
  when it lies within 16/255 Euclidean RGB distance of *any* master entry,
  regardless of which dark entry Oklab ranks first. If exact black exists,
  sub-24/255 near-black residues are also suppressed.
- Added a regression proving near-black residue produces zero gap share and no
  missing-color suggestion. 44 automated tests pass.

## 2.2.0 — 2026-08-30

### Palette Coverage Analyzer

- Added a seventh focused node, `Palette Coverage Analyzer`, for measuring a
  fixed game palette against original/pre-snap character frames in Oklab.
- When core `snapper_info` is connected, analysis uses the exact block size,
  phase, crop, mask threshold, and cell threshold rather than raw source pixels.
- `error_heatmap` visualizes cell coverage from blue (close) through green to
  red (at/above `heatmap_limit`); masked background is black.
- `suggested_subpalette` is selected only from existing master colors with a
  weighted greedy coverage objective. It never expands or silently changes the
  master palette and retains the master's semantic swatch order.
- `missing_color_suggestions` contains observed 8-bit cell colors representing
  genuine Oklab gaps. Suggestions are clustered by residual coverage, reported
  for human review, and never fed back automatically.
- The report includes mean/median/p95/max error, foreground gap-cell share,
  suggested color coverage and nearest master entries, plus near-duplicate
  master pairs below a configurable threshold.
- Each sampled frame receives equal total statistical weight, preventing a
  larger silhouette or one long frame sequence from dominating suggestions.
- Full error-map nearest-color measurement is spatially chunked to avoid large
  source-cell × palette distance allocations.
- Added regressions for the supplied light-lavender gap, background exclusion,
  master-only subpalette selection, and near-duplicate reporting. 43 automated
  tests pass.

## 2.1.0 — 2026-08-23

### Perceptual custom-palette matching

- Added `color_distance=oklab/rgb_legacy` to Video Pixel Snapper, appended after
  all existing controls. Oklab is the new recommended default.
- Legacy nearest-color classification used Euclidean encoded-sRGB distance.
  That metric can prefer a numerically nearby but visibly wrong hue. In the
  supplied fox screenshot, dominant source orange around `#E4954B` maps to
  salmon `#EF7D57` in RGB, while Oklab selects the intended orange bridge
  `#EA8A2E` from the same 59-color palette.
- Oklab matching is applied consistently to majority/center-weighted source
  votes, center/median representatives, and ambiguous-cell fallbacks. It still
  selects an existing palette entry and never synthesizes a color.
- Live Editor now receives the core distance mode through `info` and uses the
  same RGB/Oklab metric for both exact and lookup-table edited previews.
- Custom palette strips preserve first-occurrence swatch/ramp order instead of
  being lexicographically scrambled by `torch.unique` before Live Editor.
- Added orange hue-protection, end-to-end metric-selection, metadata, and
  swatch-order regressions. 41 automated tests pass.

## 2.0.1 — 2026-08-23

### Transparent-output CPU memory fix

- Fixed a multi-gigabyte allocation introduced by v2.0.0 when a large frame
  batch combined `background_mode=transparent` with an upscaled core output.
  A reported 7,261,913,088-byte request corresponds exactly to one float32 RGBA
  tensor shaped `32×2736×5184×4`.
- Transparent mode now intentionally forces the core output to native cell
  scale (`scale=1`). Temporal cleanup already requires scale 1, and sprite
  sheets should be assembled before any final nearest-neighbor presentation
  upscale. `info` reports the ignored requested dimensions.
- Core processing now preallocates one final RGBA tensor and exposes the legacy
  RGB output as a zero-copy view of its first three channels. The previous path
  first retained a complete RGB batch and then allocated another complete RGBA
  copy.
- Long-batch processing writes chunks directly into that final tensor instead
  of retaining chunk outputs and concatenating them at the end.
- Live Editor and Frame Retimer reuse incoming RGBA storage rather than creating
  an unnecessary second four-channel input batch.
- Added scale-forcing/storage-sharing regressions. 40 automated tests pass.

## 2.0.0 — 2026-08-23

### Hard-alpha PNG pipeline

- Added `background_mode=solid/transparent` to Video Pixel Snapper. Transparent
  mode requires the connected BiRefNet/RMBG foreground mask and creates hard
  cell-level alpha with no semi-transparent fringe.
- Transparent mode uses a deterministic internal 8-bit background key that is
  absent from the foreground palette. The key remains available to Motion-Aware
  Cleanup for topology consensus but is invisible in RGBA export.
- Appended `transparent_image` (RGBA IMAGE) and `transparency_mask` to Video
  Pixel Snapper. Existing RGB `image`, `palette_preview`, and `info` outputs keep
  their names, types, and order.
- Appended the same two export outputs to both Cleanup nodes. Input alpha is
  preserved exactly; silhouette corrections update alpha only where topology
  actually changed. Cleanup can also derive hard alpha from background metadata
  when RGB input is used.
- Live Editor now preserves incoming hard alpha and appends RGBA/mask outputs.
- Frame Retimer no longer discards alpha. Its original RGB `image` and `info`
  outputs remain first, with reordered RGBA/mask outputs appended.
- `transparency_mask` follows ComfyUI/Load Image convention: 1 means transparent,
  0 means opaque. All alpha is nearest-neighbor/hard; no matting or blending is
  introduced.

### Sprite Sheet node

- Added a sixth focused node, `Sprite Sheet (Video Pixel Snapper)`, for row-major
  assembly of an RGB/RGBA frame batch.
- The node exposes only `columns` and `padding`, never resizes frames, preserves
  RGBA values bit-exactly, and leaves unused cells and padding transparent.
- Outputs include the sheet plus frame width/height and resolved columns/rows
  metadata for game-engine import.

### Validation

- Added hard-alpha core, unique hidden-key, missing-mask error, Cleanup alpha,
  RGBA Retimer, transparent sheet, and complete
  Core→Cleanup→Live Editor→Retimer→Sprite Sheet integration regressions.
- 39 automated tests pass. Existing v1.9.0 output prefixes and serialized input
  controls remain compatible; only new outputs/controls are appended.

## 1.9.0 — 2026-08-16

### Maximum internal stability

- Added the opt-in `maximum_lock` preset for cases where `stability_lock` still
  leaves visible internal buzzing. It uses a 9-frame motion-aligned window,
  three-cell internal-region dilation, a six-frame bounded hold, and wider but
  still finite observed-color cluster gates.
- Hysteresis now propagates the previously accepted internal feature region
  through cycle-consistent motion. A one-cell eye, fold, or line that vanishes
  completely has no boundary in the current snapped frame; the propagated
  region lets the previous observed label restore it temporarily.
- Propagated regions survive only where a hysteresis hold is actually accepted.
  They are rejected by background/external-edge classification, invalid motion,
  scene cuts, color spread, color jump, and the finite hold age.
- Increased the Advanced temporal-window ceiling from 7 to 11 for deliberate
  maximum-stability work. Existing presets and serialized controls are unchanged.
- Added regressions for a fully missing internal dot restored from the prior
  motion-aligned feature region and for `maximum_lock`'s wider/longer settings.
  33 automated tests pass.

## 1.8.0 — 2026-08-16

### Bounded motion-aligned feature hysteresis

- Added `stability_lock`, an opt-in maximum-stability preset for internal
  palette regions, eyes, mouth, folds, muscle lines, and other small details.
- Independent sliding-window medoids can choose a different nearby palette
  shade on consecutive frames. The new pass motion-warps the previous
  stabilized discrete label into the current frame and briefly prefers it
  while the cycle-consistent candidate cloud remains compact.
- Hysteresis is restricted to detected internal feature regions, requires
  forward/backward-valid geometry, rejects background labels, observes the
  existing color-distance gate, and expires after three held frames in the
  preset. It is not a fixed-screen-coordinate filter.
- Every held label is copied from a previously stabilized snapped frame. No RGB
  averaging, interpolation, or synthesized palette colors are used.
- Added Advanced controls: `feature_hysteresis`, `feature_hold_frames`, and
  `feature_hold_radius`. They are appended after all v1.7.1 inputs.
- Appended `feature_hysteresis_actions` as output 9; outputs 1–8 retain their
  prior names, types, and order. `info` reports hysteresis state and
  `feature_held` action count.
- Added regressions showing five internal medoid transitions reduced to one,
  bounded release after the hold cap, exact observed-color preservation,
  scene-cut isolation, and `stability_lock` preset activation. 31 automated
  tests pass.

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
