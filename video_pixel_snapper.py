"""
Video Pixel Snapper
--------------------
A single ComfyUI node that turns a *batch* of AI-hallucinated video frames
into stable pixel art: it estimates the pixel grid (size AND phase/offset)
and the color palette ONCE from the whole clip (not per frame), then
applies those fixed parameters to every frame with a temporally-stable
cell-reduction method. This is what removes the jumping grid / flickering
colors / speckled outlines you get from single-image pixel-art nodes run
frame by frame.

Adapted techniques and where they come from:
- pixel_size manual override convention: x0x0b/ComfyUI-spritefusion-pixel-snapper
- grid PHASE search (brute-force offset scan maximizing edge energy):
  independently confirmed by both this file's own approach and
  jenissimo/unfake.js's `findOptimalCrop` (same idea: loop offsets
  0..scale-1, sum the edge profile at that stride, keep the best).
- cell_method majority / center_weighted / center vocabulary:
  mediapixelkr/ComfyUI-SpriteFusion-PixelSnapper and HexaDucket's wrapper
  around the same upstream engine (majority-color voting as the stable
  default).
- confidence-margin + mean-blend fallback for ambiguous cells:
  jenissimo/unfake.js's "dominant color wins only if it leads by >5%,
  otherwise blend" rule (their Step 4).
- accent/rare color reservation for the palette: adapted from the
  "supplemented histogram of important colors" idea used in some
  production color-quantizers (e.g. skin/sky-tone reservation), and from
  the general practice of decoupling clustering weight from raw pixel
  frequency (as in weighted k-means / Wu quantization) so small-but-
  distinct regions (eyes, thin outlines, highlights) aren't absorbed by
  the nearest large cluster.
- despeckle (orphan-pixel cleanup): adapted from the general idea behind
  community tools like ClusterSweep's "orphan destruction" pass.

Install: drop this folder into ComfyUI/custom_nodes/ and restart ComfyUI.
"""

import os
import re
import torch
import torch.nn.functional as F


def _drop_alpha(image: torch.Tensor) -> torch.Tensor:
    """Normalize a ComfyUI IMAGE tensor to RGB."""
    if image.shape[-1] > 3:
        return image[..., :3]
    return image


def _prepare_mask(mask: torch.Tensor, batch: int, height: int, width: int,
                  device, invert: bool = False) -> torch.Tensor:
    """Normalize common ComfyUI MASK layouts to ``(B,H,W)`` float32.

    A single mask is broadcast across a video batch; spatial mismatches are
    resized with bilinear interpolation so masks from RMBG nodes remain usable
    even when their preview/output size differs from the IMAGE input.
    """
    m = mask.to(device=device, dtype=torch.float32)
    if m.ndim == 2:
        m = m.unsqueeze(0)
    elif m.ndim == 4:
        if m.shape[-1] == 1:
            m = m[..., 0]
        elif m.shape[1] == 1:
            m = m[:, 0]
        else:
            raise ValueError(f"foreground_mask must have one channel, got shape {tuple(m.shape)}")
    if m.ndim != 3:
        raise ValueError(f"foreground_mask must be (H,W), (B,H,W), or one-channel 4D; got {tuple(m.shape)}")

    if m.shape[0] == 1 and batch > 1:
        m = m.expand(batch, -1, -1)
    elif m.shape[0] != batch:
        raise ValueError(
            f"foreground_mask batch ({m.shape[0]}) must be 1 or match image batch ({batch})"
        )

    if m.shape[1:] != (height, width):
        m = F.interpolate(
            m.unsqueeze(1), size=(height, width), mode="bilinear", align_corners=False
        ).squeeze(1)
    m = m.clamp(0.0, 1.0)
    return 1.0 - m if invert else m


def _sample_rows(flat: torch.Tensor, cap: int = 200_000) -> torch.Tensor:
    """Deterministically cap rows before a robust median operation."""
    if flat.shape[0] <= cap:
        return flat
    idx = torch.linspace(0, flat.shape[0] - 1, cap, device=flat.device).long()
    return flat[idx]


def _background_color(images: torch.Tensor, mask: torch.Tensor,
                      sample_idx: torch.Tensor, mask_threshold: float,
                      background_image: torch.Tensor = None) -> torch.Tensor:
    """Return one robust RGB background representative.

    ``background_image`` is preferred (wire the same Empty Image used for
    compositing). Otherwise sample high-confidence masked-out pixels from the
    input frames, which works well when the input is already composited onto a
    flat color.
    """
    if background_image is not None:
        bg = _drop_alpha(background_image.to(images.device)).reshape(-1, 3)
        if bg.numel():
            return _sample_rows(bg).median(dim=0).values.clamp(0.0, 1.0)

    samples = images[sample_idx]
    masks = mask[sample_idx]
    high_confidence_bg = masks <= max(0.0, 1.0 - float(mask_threshold))
    pixels = samples[high_confidence_bg]
    if not pixels.numel():
        pixels = samples[masks < 0.5]
    if not pixels.numel():
        return torch.zeros(3, device=images.device, dtype=images.dtype)
    return _sample_rows(pixels.reshape(-1, 3)).median(dim=0).values.clamp(0.0, 1.0)


def _ensure_background_color(palette: torch.Tensor, color: torch.Tensor,
                             cap: int = 256):
    """Ensure an exact, dedicated background entry and return its index."""
    color = color.to(device=palette.device, dtype=palette.dtype).reshape(1, 3)
    if palette.numel():
        d = torch.cdist(color, palette).squeeze(0)
        exact = torch.nonzero(d < 1e-7, as_tuple=False)
        if exact.numel():
            return palette, int(exact[0, 0].item())
        if palette.shape[0] >= cap:
            idx = int(torch.argmin(d).item())
            palette = palette.clone()
            palette[idx] = color[0]
            return palette, idx
    palette = torch.cat([palette, color], dim=0)
    return palette, palette.shape[0] - 1


def _unique_transparent_background_color(palette: torch.Tensor) -> torch.Tensor:
    """Choose an invisible 8-bit RGB key absent from the foreground palette.

    Motion Cleanup still needs a discrete working label for background cells.
    In transparent mode that label is hidden by alpha, so it should be chosen
    for identity rather than appearance and must not collide with foreground.
    """
    if palette.numel():
        q = (palette[:, :3].clamp(0.0, 1.0) * 255.0).round().to(torch.int64)
        used = set((q[:, 0] * 65536 + q[:, 1] * 256 + q[:, 2]).tolist())
    else:
        used = set()
    preferred = [
        0xFF00FF, 0x00FFFF, 0xFFFF00, 0xFF0000,
        0x00FF00, 0x0000FF, 0x010101, 0xFEFEFE,
    ]
    key = next((value for value in preferred if value not in used), None)
    if key is None:
        # At most 256 palette entries are in use, so a short deterministic
        # scan of the 24-bit space is guaranteed to find a free key quickly.
        key = 0
        while key in used:
            key += 1
    rgb = [(key >> 16) & 255, (key >> 8) & 255, key & 255]
    return torch.tensor(rgb, device=palette.device, dtype=palette.dtype) / 255.0


# folder_paths / PIL only exist inside a real ComfyUI process. Import
# defensively so the node still works (just without the live-editor
# preview data) if this file is ever loaded outside ComfyUI, e.g. for
# unit testing the math in isolation.
try:
    import folder_paths
    from PIL import Image
    import numpy as np
    _COMFY_PREVIEW_AVAILABLE = True
except Exception:
    _COMFY_PREVIEW_AVAILABLE = False


def _save_preview_images(tensor_batch: torch.Tensor, prefix: str, max_frames: int = 24):
    """Save up to `max_frames` frames of an IMAGE batch to ComfyUI's temp
    folder and return [{filename, subfolder, type}, ...] refs, the same
    shape ComfyUI's own PreviewImage/SaveImage nodes use — so the frontend
    can fetch them back via GET /view?filename=...&subfolder=...&type=...

    NOT independently verified against a running ComfyUI instance — if the
    companion JS widget shows nothing, check the browser console first;
    this is the most likely place a version mismatch would show up (see
    README, "Про встраивание в ComfyUI").
    """
    if not _COMFY_PREVIEW_AVAILABLE:
        return []
    try:
        output_dir = folder_paths.get_temp_directory()
        n = min(tensor_batch.shape[0], max_frames)
        full_output_folder, filename, counter, subfolder, _ = folder_paths.get_save_image_path(
            prefix, output_dir, tensor_batch.shape[2], tensor_batch.shape[1]
        )
        results = []
        for i in range(n):
            arr = (tensor_batch[i].clamp(0.0, 1.0).detach().cpu().numpy() * 255).astype(np.uint8)
            img = Image.fromarray(arr)
            fname = f"{filename}_{counter + i:05}_.png"
            img.save(os.path.join(full_output_folder, fname))
            results.append({"filename": fname, "subfolder": subfolder, "type": "temp"})
        return results
    except Exception as e:
        print(f"[VideoPixelSnapper] preview save failed (node output is unaffected): {e}")
        return []

# ----------------------------------------------------------------------
# Bayer ordered-dither matrices (normalized to 0..1)
# ----------------------------------------------------------------------

_BAYER = {
    2: torch.tensor([[0, 2],
                      [3, 1]], dtype=torch.float32) / 4.0,
    4: torch.tensor([[0, 8, 2, 10],
                      [12, 4, 14, 6],
                      [3, 11, 1, 9],
                      [15, 7, 13, 5]], dtype=torch.float32) / 16.0,
    8: torch.tensor([[0, 32, 8, 40, 2, 34, 10, 42],
                      [48, 16, 56, 24, 50, 18, 58, 26],
                      [12, 44, 4, 36, 14, 46, 6, 38],
                      [60, 28, 52, 20, 62, 30, 54, 22],
                      [3, 35, 11, 43, 1, 33, 9, 41],
                      [51, 19, 59, 27, 49, 17, 57, 25],
                      [15, 47, 7, 39, 13, 45, 5, 37],
                      [63, 31, 55, 23, 61, 29, 53, 21]], dtype=torch.float32) / 64.0,
}


def _bayer_tile(dither: str, h: int, w: int, device) -> torch.Tensor:
    """Tiled, zero-centered Bayer matrix of shape (1, h, w, 1)."""
    bsize = int(dither.replace("bayer", ""))
    mat = (_BAYER[bsize].to(device) - 0.5)
    reps_h, reps_w = (h // bsize) + 1, (w // bsize) + 1
    tiled = mat.repeat(reps_h, reps_w)[:h, :w]
    return tiled.unsqueeze(0).unsqueeze(-1)


# ----------------------------------------------------------------------
# 1. Pixel-grid SIZE + PHASE estimation
# ----------------------------------------------------------------------

def _dominant_period(signal: torch.Tensor, min_period: int = 2, max_period: int = 64):
    n = signal.shape[0]
    max_period = min(max_period, n // 2 - 1)
    if max_period < min_period or n < 4:
        return None
    signal = signal - signal.mean()
    nfft = 1
    while nfft < 2 * n:
        nfft *= 2
    spec = torch.fft.rfft(signal, n=nfft)
    acf = torch.fft.irfft(spec * spec.conj(), n=nfft)[:max_period + 1]
    candidates = acf[min_period:max_period + 1]
    if candidates.numel() == 0 or float(candidates.max()) <= 0:
        return None
    return int(torch.argmax(candidates).item()) + min_period


def _edge_signals(frame_hw3: torch.Tensor):
    # Perceptual luma keeps equal-average RGB colors (pure red/green/blue
    # all average to 1/3) distinguishable for grid detection.
    gray = (
        frame_hw3[..., 0] * 0.299
        + frame_hw3[..., 1] * 0.587
        + frame_hw3[..., 2] * 0.114
    )
    col_edges = gray.diff(dim=1).abs().sum(dim=0)
    row_edges = gray.diff(dim=0).abs().sum(dim=1)
    return col_edges, row_edges


def estimate_pixel_size(frame_hw3: torch.Tensor, min_period: int = 2, max_period: int = 64):
    col_edges, row_edges = _edge_signals(frame_hw3)
    px = _dominant_period(col_edges, min_period, max_period)
    py = _dominant_period(row_edges, min_period, max_period)
    vals = [v for v in (px, py) if v]
    if not vals:
        return None
    return sum(vals) / len(vals)


def _estimate_phase(edge_signal: torch.Tensor, period: int) -> int:
    """Estimate the first cell's start coordinate for a fixed period.

    ``edge_signal[i]`` represents the transition *between* source pixels
    ``i`` and ``i + 1``. The strongest folded edge phase is therefore a
    boundary index, while ``crop_bounds`` needs the first pixel *after*
    that boundary. The ``+1`` is important: without it a perfectly aligned
    8 px grid is reported as phase 7 and every reduction cell straddles two
    real cells.
    """
    period = max(1, int(period))
    n = edge_signal.shape[0]
    if period <= 1 or n < period:
        return 0
    trimmed_len = (n // period) * period
    folded = edge_signal[:trimmed_len].reshape(-1, period).sum(dim=0)
    boundary_phase = int(torch.argmax(folded).item())
    return (boundary_phase + 1) % period


def estimate_phase_for_frame(frame_hw3: torch.Tensor, block: int):
    col_edges, row_edges = _edge_signals(frame_hw3)
    return _estimate_phase(col_edges, block), _estimate_phase(row_edges, block)


def crop_bounds(h: int, w: int, block: int, phase_y: int, phase_x: int):
    phase_y %= block
    phase_x %= block
    oh = (h - phase_y) // block
    ow = (w - phase_x) // block
    return phase_y, phase_x, oh * block, ow * block, oh, ow


# ----------------------------------------------------------------------
# 2. Nearest-palette classification (chunked to bound memory)
# ----------------------------------------------------------------------

def nearest_palette_index(flat: torch.Tensor, palette: torch.Tensor, chunk: int = 200_000) -> torch.Tensor:
    n = flat.shape[0]
    if n <= chunk:
        return torch.cdist(flat, palette).argmin(dim=1)
    out = torch.empty(n, dtype=torch.long, device=flat.device)
    for i in range(0, n, chunk):
        j = min(i + chunk, n)
        out[i:j] = torch.cdist(flat[i:j], palette).argmin(dim=1)
    return out


# ----------------------------------------------------------------------
# 3. Per-cell stats used to build the palette pool: median color (robust
#    representative) + saliency (std within the cell — high on edges,
#    outlines, eyes, highlights; near-zero on flat regions).
# ----------------------------------------------------------------------

def _masked_median(values: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    """Lower median over the penultimate dimension, ignoring invalid rows.

    ``values`` is ``(..., samples, channels)`` and ``valid`` is
    ``(..., samples)``. All-invalid groups return zero and are expected to be
    rejected/overridden by their caller.
    """
    count = valid.sum(dim=-1)
    masked = values.masked_fill(~valid.unsqueeze(-1), float("inf"))
    ordered = masked.sort(dim=-2).values
    median_pos = ((count - 1).clamp(min=0) // 2)
    gather_idx = median_pos.unsqueeze(-1).unsqueeze(-1).expand(
        *median_pos.shape, 1, values.shape[-1]
    )
    result = torch.gather(ordered, -2, gather_idx).squeeze(-2)
    return torch.where((count > 0).unsqueeze(-1), result, torch.zeros_like(result))


def cell_stats(frame_hw3: torch.Tensor, block: int, phase_y: int = 0, phase_x: int = 0):
    h, w, c = frame_hw3.shape
    y0, x0, ch, cw, oh, ow = crop_bounds(h, w, block, phase_y, phase_x)
    cropped = frame_hw3[y0:y0 + ch, x0:x0 + cw, :]
    blocks = cropped.reshape(oh, block, ow, block, c).permute(0, 2, 1, 3, 4).reshape(oh, ow, block * block, c)
    colors = blocks.median(dim=2).values
    # unbiased=False keeps block=1 valid (the unbiased estimator returns
    # NaN for a single sample, which previously poisoned palette weights).
    saliency = blocks.std(dim=2, unbiased=False).mean(dim=-1)  # oh, ow
    return colors, saliency


def masked_cell_stats(frame_hw3: torch.Tensor, mask_hw: torch.Tensor, block: int,
                      phase_y: int, phase_x: int, mask_threshold: float):
    """Cell colors/saliency computed only from confident foreground pixels."""
    h, w, c = frame_hw3.shape
    y0, x0, ch, cw, oh, ow = crop_bounds(h, w, block, phase_y, phase_x)
    pixels = frame_hw3[y0:y0 + ch, x0:x0 + cw, :]
    masks = mask_hw[y0:y0 + ch, x0:x0 + cw]
    pixels = pixels.reshape(oh, block, ow, block, c).permute(0, 2, 1, 3, 4)
    pixels = pixels.reshape(oh, ow, block * block, c)
    mask_blocks = masks.reshape(oh, block, ow, block).permute(0, 2, 1, 3)
    mask_blocks = mask_blocks.reshape(oh, ow, block * block)
    valid = mask_blocks >= mask_threshold
    count = valid.sum(dim=-1)
    colors = _masked_median(pixels, valid)

    weights = valid.to(pixels.dtype).unsqueeze(-1)
    denom = weights.sum(dim=-2).clamp(min=1.0)
    mean = (pixels * weights).sum(dim=-2) / denom
    var = ((pixels - mean.unsqueeze(-2)) ** 2 * weights).sum(dim=-2) / denom
    saliency = var.clamp(min=0.0).sqrt().mean(dim=-1)
    coverage = mask_blocks.mean(dim=-1)
    return colors, saliency, coverage, count


# ----------------------------------------------------------------------
# 4. Palette: bulk (k-means) + reserved accent slots (saliency-weighted
#    outlier reservation) — see module docstring for the reasoning.
# ----------------------------------------------------------------------

def kmeans(pixels: torch.Tensor, k: int, iters: int = 25, seed: int = 42) -> torch.Tensor:
    n = pixels.shape[0]
    k = max(1, min(k, n))
    g = torch.Generator(device="cpu").manual_seed(seed)
    idx = torch.randperm(n, generator=g)[:k].to(pixels.device)
    centers = pixels[idx].clone()
    for _ in range(iters):
        dists = torch.cdist(pixels, centers)
        assign = dists.argmin(dim=1)
        new_centers = centers.clone()
        for ci in range(k):
            mask = assign == ci
            if mask.any():
                new_centers[ci] = pixels[mask].mean(dim=0)
        if torch.allclose(new_centers, centers, atol=1e-4):
            centers = new_centers
            break
        centers = new_centers
    return centers


def deduplicate_palette(palette: torch.Tensor, tolerance: float = 1e-5) -> torch.Tensor:
    """Preserve order while dropping duplicate/near-identical centers."""
    if palette.shape[0] < 2:
        return palette
    chosen = [0]
    for i in range(1, palette.shape[0]):
        if float(torch.cdist(palette[i:i + 1], palette[chosen]).min()) > tolerance:
            chosen.append(i)
    return palette[chosen]


def farthest_point_select(candidates: torch.Tensor, scores: torch.Tensor, k: int) -> torch.Tensor:
    """Greedily pick k rows of `candidates`: start from the highest-scoring
    one, then repeatedly take whichever remaining point is farthest (in
    color space) from everything already picked. Keeps reserved accent
    slots diverse instead of collapsing onto near-duplicates."""
    n = candidates.shape[0]
    k = max(0, min(k, n))
    if k == 0:
        return candidates[:0]
    chosen = [int(torch.argmax(scores).item())]
    for _ in range(1, k):
        d = torch.cdist(candidates, candidates[chosen]).min(dim=1).values
        d[chosen] = -1.0
        chosen.append(int(torch.argmax(d).item()))
    return candidates[chosen]


def build_palette(pool_colors: torch.Tensor, pool_saliency: torch.Tensor, k_colors: int,
                   accent_slots: int, seed: int):
    accent_slots = max(0, min(accent_slots, k_colors - 1))
    k_bulk = k_colors - accent_slots

    # mild saliency weighting on the bulk pool: duplicate high-saliency rows
    # a little so they carry slightly more weight in k-means without fully
    # overriding frequency (defense in depth on top of the explicit
    # reservation below, which is what actually guarantees accent colors
    # survive).
    weight = 1.0 + (pool_saliency / (pool_saliency.mean() + 1e-6)).clamp(0, 3)
    rep = weight.round().long().clamp(min=1)
    weighted_pool = pool_colors.repeat_interleave(rep, dim=0)
    if weighted_pool.shape[0] > 20000:
        g = torch.Generator(device="cpu").manual_seed(seed)
        keep = torch.randperm(weighted_pool.shape[0], generator=g)[:20000].to(pool_colors.device)
        weighted_pool = weighted_pool[keep]

    bulk = kmeans(weighted_pool, k_bulk, seed=seed)

    if accent_slots == 0:
        return deduplicate_palette(bulk)

    # candidates for reservation: cells whose color the bulk palette does
    # NOT already represent well, ranked by (distance from bulk) x (saliency)
    dist_to_bulk = torch.cdist(pool_colors, bulk).min(dim=1).values
    outlier_score = dist_to_bulk * (pool_saliency + 1e-3)
    reserved = farthest_point_select(pool_colors, outlier_score, accent_slots)

    palette = torch.cat([bulk, reserved], dim=0)
    # drop accidental near-duplicates (reserved color landed on a bulk one)
    keep_mask = torch.ones(palette.shape[0], dtype=torch.bool, device=palette.device)
    for i in range(k_bulk, palette.shape[0]):
        prior = palette[:i][keep_mask[:i]]
        if prior.shape[0] and float(torch.cdist(palette[i:i+1], prior).min()) < 1e-3:
            keep_mask[i] = False
    return deduplicate_palette(palette[keep_mask])


def palette_from_image(custom_palette: torch.Tensor, cap: int, seed: int) -> torch.Tensor:
    flat = custom_palette.reshape(-1, custom_palette.shape[-1])[:, :3]
    uniq = torch.unique(flat, dim=0)
    if uniq.shape[0] > cap:
        uniq = kmeans(uniq, cap, seed=seed)
    return uniq


def make_palette_preview(palette: torch.Tensor, swatch: int = 32) -> torch.Tensor:
    tiles = [col.view(1, 1, 1, 3).expand(1, swatch, swatch, 3) for col in palette]
    return torch.cat(tiles, dim=2).clamp(0.0, 1.0)


def dither_strength(palette: torch.Tensor) -> float:
    if palette.shape[0] < 2:
        return 0.0
    d = torch.cdist(palette, palette)
    d.fill_diagonal_(float("inf"))
    return float(d.min(dim=1).values.mean().item())


# ----------------------------------------------------------------------
# 5. Weighted-mode cell reduction (majority / center_weighted), with an
#    unfake.js-style confidence-margin fallback to a mean-color blend for
#    ambiguous cells (mostly anti-aliased edge/outline cells) — this is
#    what stops soft AI edges from being force-snapped to a random nearby
#    palette color ("digital noise" on outlines).
# ----------------------------------------------------------------------

def _cell_kernel(block: int, kind: str, device) -> torch.Tensor:
    if kind == "majority":
        return torch.ones(block * block, device=device)
    yy, xx = torch.meshgrid(torch.arange(block, device=device), torch.arange(block, device=device), indexing="ij")
    center = (block - 1) / 2.0
    dist2 = (yy - center) ** 2 + (xx - center) ** 2
    sigma2 = max(1.0, (block / 2.0) ** 2)
    return torch.exp(-dist2 / (2 * sigma2)).reshape(-1)


def weighted_vote_reduce(cropped: torch.Tensor, block: int, palette: torch.Tensor, kind: str,
                         margin: float = 0.05, pixel_mask: torch.Tensor = None):
    b, ch, cw, c = cropped.shape
    oh, ow = ch // block, cw // block

    # A 1x1 cell has nothing to vote over.
    if block == 1:
        return nearest_palette_index(cropped.reshape(-1, c), palette).reshape(b, oh, ow)

    n_cells = b * oh * ow
    pixels = cropped.reshape(b, oh, block, ow, block, c).permute(0, 1, 3, 2, 4, 5)
    pixels = pixels.reshape(n_cells, block * block, c)
    idx_px = nearest_palette_index(cropped.reshape(-1, c), palette)
    idx_px = idx_px.reshape(b, oh, block, ow, block).permute(0, 1, 3, 2, 4)
    idx_px = idx_px.reshape(n_cells, block * block)

    kernel = _cell_kernel(block, kind, cropped.device).unsqueeze(0)
    valid = None
    if pixel_mask is not None:
        valid = pixel_mask.reshape(b, oh, block, ow, block).permute(0, 1, 3, 2, 4)
        valid = valid.reshape(n_cells, block * block)
        vote_weights = kernel * valid.to(cropped.dtype)
    else:
        vote_weights = kernel.expand(n_cells, -1)

    k = palette.shape[0]
    hist = torch.zeros(n_cells, k, device=cropped.device)
    hist.scatter_add_(1, idx_px, vote_weights)

    top2 = hist.topk(min(2, k), dim=1)
    raw_total = hist.sum(dim=1)
    total = raw_total.clamp(min=1e-8)
    winner_idx = top2.indices[:, 0]
    share_top = top2.values[:, 0] / total
    share_second = (top2.values[:, 1] / total) if k > 1 else torch.zeros_like(share_top)
    ambiguous = ((share_top - share_second) < margin) | (raw_total <= 0)

    if valid is None:
        mean_rgb = pixels.mean(dim=1)
    else:
        mask_f = valid.to(cropped.dtype).unsqueeze(-1)
        mean_rgb = (pixels * mask_f).sum(dim=1) / mask_f.sum(dim=1).clamp(min=1.0)
    fallback_idx = nearest_palette_index(mean_rgb, palette)

    final_idx = torch.where(ambiguous, fallback_idx, winner_idx)
    return final_idx.reshape(b, oh, ow)


# ----------------------------------------------------------------------
# 6. Despeckle: 4-neighbor mode filter on the palette-index grid. Removes
#    truly isolated single cells that disagree with every neighbor
#    (adapts the "orphan pixel" cleanup idea from community pixel-art
#    sanitizer tools). Off by default: it can also erase a genuine
#    intentional single-pixel highlight, so it's an explicit opt-in.
# ----------------------------------------------------------------------

def despeckle_indices(idx_grid: torch.Tensor, k_colors: int) -> torch.Tensor:
    b, oh, ow = idx_grid.shape
    if oh < 3 or ow < 3:
        return idx_grid
    padded = F.pad(idx_grid.unsqueeze(1).float(), (1, 1, 1, 1), mode="replicate").squeeze(1)
    up = padded[:, :-2, 1:-1]
    down = padded[:, 2:, 1:-1]
    left = padded[:, 1:-1, :-2]
    right = padded[:, 1:-1, 2:]
    center = idx_grid.float()
    disagree = (up != center) & (down != center) & (left != center) & (right != center)
    # replacement = mode of the 4 neighbors
    stacked = torch.stack([up, down, left, right], dim=0)  # 4,b,oh,ow
    stacked_l = stacked.long().permute(1, 2, 3, 0).reshape(-1, 4)
    mode_flat = torch.mode(stacked_l, dim=1).values
    mode_count = (stacked_l == mode_flat.unsqueeze(1)).sum(dim=1).reshape(b, oh, ow)
    mode_val = mode_flat.reshape(b, oh, ow).float()

    # If all four neighbors are different, torch.mode resolves the tie by
    # palette-index order; replacing with that arbitrary color creates a
    # new artifact. Require at least two neighbors to agree.
    should_replace = disagree & (mode_count >= 2)
    out = torch.where(should_replace, mode_val, center)
    return out.long().clamp(0, k_colors - 1)


# ----------------------------------------------------------------------
# 7. Apply the fixed grid + fixed palette to the whole batch
# ----------------------------------------------------------------------

def process_batch(images: torch.Tensor, block: int, phase_y: int, phase_x: int, palette: torch.Tensor,
                  cell_method: str, dither: str, despeckle: bool,
                  out_h: int, out_w: int, foreground_mask: torch.Tensor = None,
                  mask_threshold: float = 0.5, mask_cell_threshold: float = 0.25,
                  background_index: int = None,
                  return_foreground_mask: bool = False) -> torch.Tensor:
    """Apply one fixed grid/palette, automatically chunking long clips.

    ``nearest_palette_index`` already chunks its distance matrix, but the
    majority methods also allocate a per-cell palette histogram. Processing
    a whole long video at once made that tensor scale with frame count and
    could OOM even though each frame was individually modest. This keeps the
    exact same result while bounding those per-frame temporaries.
    """
    b, h, w, c = images.shape
    y0, x0, ch, cw, oh, ow = crop_bounds(h, w, block, phase_y, phase_x)
    k = palette.shape[0]

    if cell_method in ("majority", "center_weighted") and block > 1:
        # Roughly cap the largest histogram at ~8M float elements (~32 MB).
        per_frame_hist = max(1, oh * ow * k)
        frames_per_chunk = max(1, min(16, 8_000_000 // per_frame_hist))
    else:
        # Other methods have no cell x palette histogram; source pixels are
        # the dominant temporary, so a looser pixel budget is sufficient.
        per_frame_pixels = max(1, ch * cw)
        frames_per_chunk = max(1, min(32, 8_000_000 // per_frame_pixels))

    dither_tile = None
    if dither != "none":
        dither_tile = _bayer_tile(dither, ch, cw, images.device) * dither_strength(palette)

    outputs = []
    foreground_outputs = []
    for start in range(0, b, frames_per_chunk):
        end = min(start + frames_per_chunk, b)
        cropped = images[start:end, y0:y0 + ch, x0:x0 + cw, :]
        cb = end - start

        pixel_mask = None
        foreground_cells = None
        if foreground_mask is not None:
            mask_crop = foreground_mask[start:end, y0:y0 + ch, x0:x0 + cw]
            mask_blocks = mask_crop.reshape(cb, oh, block, ow, block).permute(0, 1, 3, 2, 4)
            mask_blocks = mask_blocks.reshape(cb, oh, ow, block * block)
            pixel_mask_cells = mask_blocks >= mask_threshold
            foreground_cells = (
                (mask_blocks.mean(dim=-1) >= mask_cell_threshold)
                & pixel_mask_cells.any(dim=-1)
            )
            # weighted_vote_reduce expects source-resolution mask layout.
            pixel_mask = mask_crop >= mask_threshold

        if dither_tile is not None:
            cropped = cropped + dither_tile

        if cell_method in ("majority", "center_weighted"):
            idx_grid = weighted_vote_reduce(
                cropped, block, palette, cell_method, pixel_mask=pixel_mask
            )
        elif cell_method == "center":
            cy, cx = block // 2, block // 2
            sampled = cropped[:, cy::block, cx::block, :][:, :oh, :ow, :]
            idx_grid = nearest_palette_index(sampled.reshape(-1, c), palette).reshape(cb, oh, ow)
            if pixel_mask is not None:
                center_valid = pixel_mask[:, cy::block, cx::block][:, :oh, :ow]
                pixels = cropped.reshape(cb, oh, block, ow, block, c).permute(0, 1, 3, 2, 4, 5)
                pixels = pixels.reshape(cb, oh, ow, block * block, c)
                valid = pixel_mask_cells
                mask_f = valid.to(cropped.dtype).unsqueeze(-1)
                mean = (pixels * mask_f).sum(dim=3) / mask_f.sum(dim=3).clamp(min=1.0)
                fallback = nearest_palette_index(mean.reshape(-1, c), palette).reshape(cb, oh, ow)
                idx_grid = torch.where(center_valid, idx_grid, fallback)
        else:  # median
            blocks = cropped.reshape(cb, oh, block, ow, block, c).permute(0, 1, 3, 2, 4, 5)
            blocks = blocks.reshape(cb, oh, ow, block * block, c)
            if pixel_mask is None:
                down = blocks.median(dim=3).values
            else:
                down = _masked_median(blocks, pixel_mask_cells)
            idx_grid = nearest_palette_index(down.reshape(-1, c), palette).reshape(cb, oh, ow)

        if despeckle:
            idx_grid = despeckle_indices(idx_grid, k)

        if foreground_cells is not None:
            if background_index is None:
                raise ValueError("background_index is required when foreground_mask is used")
            idx_grid = torch.where(
                foreground_cells,
                idx_grid,
                torch.full_like(idx_grid, int(background_index)),
            )

        mapped = palette[idx_grid].clamp(0.0, 1.0)  # cb, oh, ow, c
        mapped = F.interpolate(
            mapped.permute(0, 3, 1, 2), size=(out_h, out_w), mode="nearest"
        ).permute(0, 2, 3, 1)
        outputs.append(mapped)
        if return_foreground_mask:
            if foreground_cells is None:
                opaque = torch.ones(
                    (cb, oh, ow), device=images.device, dtype=images.dtype
                )
            else:
                opaque = foreground_cells.to(images.dtype)
            opaque = F.interpolate(
                opaque.unsqueeze(1), size=(out_h, out_w), mode="nearest"
            ).squeeze(1)
            foreground_outputs.append(opaque)

    result = torch.cat(outputs, dim=0)
    if return_foreground_mask:
        return result, torch.cat(foreground_outputs, dim=0)
    return result


def resolve_output_size(oh: int, ow: int, orig_h: int, orig_w: int, mode: str, manual_scale: int):
    if mode == "match_width":
        s = max(1, round(orig_w / ow))
    elif mode == "match_height":
        s = max(1, round(orig_h / oh))
    elif mode == "match_pixel_count":
        s = max(1, round(((orig_h * orig_w) / (oh * ow)) ** 0.5))
    else:  # manual
        s = max(1, int(manual_scale))
    return oh * s, ow * s, s


# ----------------------------------------------------------------------
# ComfyUI node
# ----------------------------------------------------------------------

class VideoPixelSnapper:
    CATEGORY = "Video Pixel Snapper"
    RETURN_TYPES = ("IMAGE", "IMAGE", "STRING", "IMAGE", "MASK")
    RETURN_NAMES = (
        "image", "palette_preview", "info", "transparent_image",
        "transparency_mask",
    )
    FUNCTION = "run"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "pixel_size": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 128.0, "step": 0.5,
                                          "tooltip": "0 = auto-detect size using grid_detection_mode below. "
                                                      ">0 = manual override. Grid PHASE is always auto-detected."}),
                "grid_detection_mode": (["average_across_frames", "first_frame"],
                                         {"default": "average_across_frames"}),
                "cell_method": (["majority", "center_weighted", "median", "center"],
                                 {"default": "majority",
                                  "tooltip": "majority: classify every pixel against the palette, keep the most "
                                              "frequent index. center_weighted: same, but pixels near the cell "
                                              "center count more. Both fall back to a mean-color blend on "
                                              "ambiguous (likely anti-aliased edge) cells instead of forcing a "
                                              "noisy snap. median: per-channel median. center: center pixel only."}),
                "k_colors": ("INT", {"default": 16, "min": 2, "max": 256}),
                "accent_slots": ("INT", {"default": 2, "min": 0, "max": 64,
                                          "tooltip": "Reserve this many palette slots for rare-but-distinct "
                                                      "colors (eyes, thin outlines, highlights) that plain "
                                                      "frequency-based clustering tends to drop. 0 disables. "
                                                      "Ignored when custom_palette is connected."}),
                "sample_frames": ("INT", {"default": 8, "min": 1, "max": 64}),
                "dither": (["none", "bayer2", "bayer4", "bayer8"], {"default": "none"}),
                "despeckle": ("BOOLEAN", {"default": False,
                                           "tooltip": "Remove single grid cells that disagree with all 4 "
                                                       "neighbors. Can also erase an intentional 1px highlight — "
                                                       "off by default."}),
                "output_scale_mode": (["manual", "match_width", "match_height", "match_pixel_count"],
                                       {"default": "manual",
                                        "tooltip": "manual: use output_scale directly. match_*: auto-compute the "
                                                    "integer nearest-neighbor scale so the output best matches "
                                                    "the ORIGINAL video's width / height / total pixel count."}),
                "output_scale": ("INT", {"default": 1, "min": 1, "max": 16}),
                "seed": ("INT", {"default": 42, "min": 0, "max": 2 ** 31 - 1}),
                "mask_threshold": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.05,
                                              "tooltip": "Used only when foreground_mask is connected. Pixels "
                                                          "at or above this value may contribute to the foreground "
                                                          "palette/cell vote; softer fringe pixels are excluded."}),
                "mask_cell_threshold": ("FLOAT", {"default": 0.25, "min": 0.0, "max": 1.0, "step": 0.05,
                                                   "tooltip": "Minimum average foreground coverage for an output "
                                                               "grid cell. Lower preserves thinner silhouette details; "
                                                               "higher removes more edge/background contamination."}),
                "invert_mask": ("BOOLEAN", {"default": False,
                                              "tooltip": "Enable when white means background in your mask. "
                                                          "BiRefNet/RMBG normally uses white as foreground."}),
                "background_mode": (["solid", "transparent"], {
                    "default": "solid",
                    "tooltip": "solid preserves the existing RGB working background. transparent "
                               "uses a unique invisible RGB key plus hard cell alpha; connect "
                               "transparent_image to the alpha-aware pipeline. Requires foreground_mask."
                }),
            },
            "optional": {
                "custom_palette": ("IMAGE", {"tooltip": "Optional fixed palette, fed in as an image (e.g. a "
                                                          "swatch strip, or this node's own palette_preview "
                                                          "output after you edit it). Unique colors are read "
                                                          "from it directly, k_colors/accent_slots are ignored."}),
                "foreground_mask": ("MASK", {"tooltip": "Optional foreground mask from BiRefNet/RMBG. Masked-out "
                                                        "pixels are excluded from foreground palette estimation "
                                                        "and collapsed to one locked background color."}),
                "background_image": ("IMAGE", {"tooltip": "Optional flat Empty Image used as the background. Its "
                                                          "median color becomes the single locked background entry. "
                                                          "If omitted, background color is sampled from masked-out "
                                                          "pixels of image."}),
            },
        }

    def run(self, image, pixel_size, grid_detection_mode, cell_method, k_colors, accent_slots,
            sample_frames, dither, despeckle, output_scale_mode, output_scale, seed,
            mask_threshold=0.5, mask_cell_threshold=0.25, invert_mask=False,
            background_mode="solid", custom_palette=None, foreground_mask=None,
            background_image=None):
        image = _drop_alpha(image)
        if custom_palette is not None:
            custom_palette = _drop_alpha(custom_palette)
        if background_image is not None:
            background_image = _drop_alpha(background_image)
        b, h, w, c = image.shape
        device = image.device
        mask_threshold = float(max(0.0, min(1.0, mask_threshold)))
        mask_cell_threshold = float(max(0.0, min(1.0, mask_cell_threshold)))
        prepared_mask = None
        if foreground_mask is not None:
            prepared_mask = _prepare_mask(
                foreground_mask, b, h, w, device, invert=bool(invert_mask)
            )
        transparent_mode = background_mode == "transparent"
        if transparent_mode and prepared_mask is None:
            raise ValueError(
                "background_mode='transparent' requires foreground_mask from "
                "BiRefNet/RMBG so hard alpha can be defined"
            )

        n_samples = min(sample_frames, b)
        sample_idx = torch.linspace(
            0, b - 1, n_samples, device=device
        ).round().long().unique()
        samples = image[sample_idx]

        # --- grid SIZE ---
        if pixel_size and pixel_size > 0:
            block = max(1, int(round(pixel_size)))
        elif grid_detection_mode == "first_frame":
            est = estimate_pixel_size(image[0])
            block = max(1, int(round(est))) if est else 1
        else:
            estimates = []
            for i in range(samples.shape[0]):
                est = estimate_pixel_size(samples[i])
                if est:
                    estimates.append(est)
            estimates.sort()
            block = int(round(estimates[len(estimates) // 2])) if estimates else 1
        # Keep at least one complete cell even for tiny synthetic/test
        # images; the previous h//2,w//2 clamp could turn block into 0
        # when either dimension was 1.
        max_block = max(1, min(h, w) // 2)
        block = max(1, min(block, max_block))

        # --- grid PHASE ---
        if grid_detection_mode == "first_frame":
            phase_x, phase_y = estimate_phase_for_frame(image[0], block)
        else:
            phase_xs, phase_ys = [], []
            for i in range(samples.shape[0]):
                px_, py_ = estimate_phase_for_frame(samples[i], block)
                phase_xs.append(px_)
                phase_ys.append(py_)
            phase_x = int(torch.mode(torch.tensor(phase_xs)).values.item())
            phase_y = int(torch.mode(torch.tensor(phase_ys)).values.item())

        # --- palette ---
        background_index = None
        background_rgb = None
        if prepared_mask is not None and not transparent_mode:
            background_rgb = _background_color(
                image, prepared_mask, sample_idx, mask_threshold, background_image
            )

        if custom_palette is not None:
            palette = palette_from_image(custom_palette.to(device), cap=256, seed=seed)
            if prepared_mask is not None:
                if transparent_mode:
                    background_rgb = _unique_transparent_background_color(palette)
                palette, background_index = _ensure_background_color(palette, background_rgb)
                mode_note = "transparent" if transparent_mode else "masked-bg"
                palette_source = f"custom+{mode_note} ({palette.shape[0]} colors)"
            else:
                palette_source = f"custom ({palette.shape[0]} colors)"
        else:
            pools_c, pools_s = [], []
            if prepared_mask is None:
                for i in range(samples.shape[0]):
                    col, sal = cell_stats(samples[i], block, phase_y, phase_x)
                    pools_c.append(col.reshape(-1, c))
                    pools_s.append(sal.reshape(-1))
            else:
                sample_masks = prepared_mask[sample_idx]
                for i in range(samples.shape[0]):
                    col, sal, _coverage, count = masked_cell_stats(
                        samples[i], sample_masks[i], block, phase_y, phase_x, mask_threshold
                    )
                    valid_cells = count > 0
                    if valid_cells.any():
                        pools_c.append(col[valid_cells])
                        pools_s.append(sal[valid_cells])

            if pools_c:
                pool_colors = torch.cat(pools_c, dim=0)
                pool_saliency = torch.cat(pools_s, dim=0)
                if pool_colors.shape[0] > 20000:
                    g = torch.Generator(device="cpu").manual_seed(seed)
                    keep = torch.randperm(pool_colors.shape[0], generator=g)[:20000].to(device)
                    pool_colors, pool_saliency = pool_colors[keep], pool_saliency[keep]

                foreground_k = max(1, k_colors - 1) if prepared_mask is not None else k_colors
                foreground_accents = min(accent_slots, max(0, foreground_k - 1))
                palette = build_palette(
                    pool_colors, pool_saliency, foreground_k, foreground_accents, seed
                )
                n_accent = max(0, palette.shape[0] - (foreground_k - foreground_accents))
            else:
                palette = image.new_empty((0, 3))
                n_accent = 0

            if prepared_mask is not None:
                if transparent_mode:
                    background_rgb = _unique_transparent_background_color(palette)
                palette, background_index = _ensure_background_color(palette, background_rgb)
                mode_note = "transparent" if transparent_mode else "masked"
                palette_source = (
                    f"auto {mode_note} ({palette.shape[0]} colors, "
                    f"{n_accent} accent, 1 bg key)"
                )
            else:
                palette_source = f"auto ({palette.shape[0]} colors, {n_accent} accent)"

        # --- output size ---
        oh = (h - phase_y % block) // block
        ow = (w - phase_x % block) // block
        out_h, out_w, resolved_scale = resolve_output_size(oh, ow, h, w, output_scale_mode, output_scale)

        # --- apply ---
        out, foreground_alpha = process_batch(
            image, block, phase_y, phase_x, palette, cell_method, dither,
            despeckle, out_h, out_w, foreground_mask=prepared_mask,
            mask_threshold=mask_threshold, mask_cell_threshold=mask_cell_threshold,
            background_index=background_index, return_foreground_mask=True,
        )
        transparent_image = torch.cat(
            [out, foreground_alpha.unsqueeze(-1)], dim=-1
        )
        transparency_mask = 1.0 - foreground_alpha
        preview = make_palette_preview(palette)
        mask_info = ""
        if prepared_mask is not None:
            bg8 = (background_rgb.clamp(0.0, 1.0) * 255).round().long().tolist()
            mask_info = (
                f" mask=on bg=#{bg8[0]:02x}{bg8[1]:02x}{bg8[2]:02x} "
                f"background_mode={background_mode} mask_threshold={mask_threshold:g} "
                f"cell_threshold={mask_cell_threshold:g}"
            )
        info = (f"grid={block}px phase=({phase_x},{phase_y}) cells={ow}x{oh} "
                f"source={w}x{h} palette={palette_source} "
                f"scale={resolved_scale}x -> {out_w}x{out_h}{mask_info}")

        return (out, preview, info, transparent_image, transparency_mask)


class VideoPixelSnapperEditor:
    """
    Optional companion node: hosts the in-graph live palette editor
    widget (web/video_pixel_snapper.js). Kept separate from the core
    VideoPixelSnapper node on purpose — it does no image processing of
    its own, only saves preview frames for the widget to fetch, so
    adding the (fairly large) editor UI to your graph is opt-in and
    doesn't bloat or slow down the core node when you don't need it.

    Wire it up from the core node: `image` -> original_image,
    the core node's `image` output -> snapped_image, and its
    `palette_preview` output -> palette_preview.
    """

    CATEGORY = "Video Pixel Snapper"
    RETURN_TYPES = ("IMAGE", "IMAGE", "MASK")
    RETURN_NAMES = ("image", "transparent_image", "transparency_mask")
    FUNCTION = "run"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "original_image": ("IMAGE", {"tooltip": "The raw frames fed into Video Pixel Snapper "
                                                          "(i.e. the same tensor as its own `image` input)."}),
                "snapped_image": ("IMAGE", {"tooltip": "Video Pixel Snapper's `image` output."}),
                "palette_preview": ("IMAGE", {"tooltip": "Video Pixel Snapper's `palette_preview` output."}),
                "max_preview_frames": ("INT", {"default": 24, "min": 1, "max": 300,
                                                "tooltip": "How many frames the frame-paging controls in the "
                                                            "widget can page through. Each one is written as a "
                                                            "PNG file for the browser to fetch, so pushing this "
                                                            "very high on a big batch costs real disk/time."}),
            },
            "optional": {
                "info": ("STRING", {"forceInput": True,
                                     "tooltip": "Optional but recommended: connect Video Pixel Snapper's `info` "
                                                 "output here. Lets Live use the exact grid size, phase, cell "
                                                 "dimensions, and output scale. Without it, Live falls back to "
                                                 "a centered best-effort reduction."}),
            },
        }

    def run(self, original_image, snapped_image, palette_preview, max_preview_frames, info=""):
        snapped_alpha = (
            snapped_image[..., 3].float().clamp(0.0, 1.0)
            if snapped_image.shape[-1] > 3 else None
        )
        original_image = _drop_alpha(original_image)
        snapped_image = _drop_alpha(snapped_image)
        palette_preview = _drop_alpha(palette_preview)
        if snapped_alpha is None:
            bg_match = re.search(r"mask=on bg=#([0-9a-fA-F]{6})", info or "")
            if bg_match:
                bg_key = int(bg_match.group(1), 16)
                q = (snapped_image.clamp(0.0, 1.0) * 255.0).round().to(torch.int64)
                keys = q[..., 0] * 65536 + q[..., 1] * 256 + q[..., 2]
                snapped_alpha = (keys != bg_key).to(snapped_image.dtype)
            else:
                snapped_alpha = torch.ones(
                    snapped_image.shape[:3], device=snapped_image.device,
                    dtype=snapped_image.dtype,
                )
        transparent_image = torch.cat(
            [snapped_image, snapped_alpha.unsqueeze(-1)], dim=-1
        )
        transparency_mask = 1.0 - snapped_alpha

        raw_refs = _save_preview_images(original_image, "VPS_raw", max_frames=max_preview_frames)
        frame_refs = _save_preview_images(snapped_image, "VPS_frame", max_frames=max_preview_frames)
        palette_refs = _save_preview_images(palette_preview, "VPS_palette", max_frames=1)

        grid = None
        if info:
            # Keep this as a compact numeric array because ComfyUI passes
            # values in the ``ui`` payload straight to the frontend. The
            # first three entries retain compatibility with the original
            # format; cell dimensions and scale let the editor reduce RAW
            # frames at the true cell resolution even when the core output
            # has been nearest-neighbor upscaled.
            m = re.search(
                r"grid=(\d+)px phase=\((\d+),(\d+)\) cells=(\d+)x(\d+).*?scale=(\d+)x",
                info,
            )
            if m:
                grid = [int(v) for v in m.groups()]
                # [block, phase_x, phase_y, cells_w, cells_h, output_scale]
            else:
                # Backward-compatible fallback for info strings produced
                # by older versions that did not report cells/scale.
                m = re.search(r"grid=(\d+)px phase=\((\d+),(\d+)\)", info)
                if m:
                    grid = [int(v) for v in m.groups()]

        ui_data = {"vps_raw": raw_refs, "vps_frames": frame_refs, "vps_palette": palette_refs}
        if grid:
            ui_data["vps_grid"] = grid
        if info:
            bg_match = re.search(r"mask=on bg=(#[0-9a-fA-F]{6})", info)
            if bg_match:
                ui_data["vps_background"] = [bg_match.group(1).lower()]
        return {
            "ui": ui_data,
            "result": (snapped_image, transparent_image, transparency_mask),
        }


NODE_CLASS_MAPPINGS = {
    "VideoPixelSnapper": VideoPixelSnapper,
    "VideoPixelSnapperEditor": VideoPixelSnapperEditor,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "VideoPixelSnapper": "Video Pixel Snapper",
    "VideoPixelSnapperEditor": "Video Pixel Snapper (Live Editor)",
}

