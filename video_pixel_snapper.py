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
import torch
import torch.nn.functional as F

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
    gray = frame_hw3.mean(dim=-1)
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
    """Brute-force offset scan: same idea as unfake.js's findOptimalCrop —
    try every offset 0..period-1, score = sum of edge energy at that stride,
    keep the offset with the highest score."""
    period = max(1, int(period))
    n = edge_signal.shape[0]
    if period <= 1 or n < period:
        return 0
    trimmed_len = (n // period) * period
    folded = edge_signal[:trimmed_len].reshape(-1, period).sum(dim=0)
    return int(torch.argmax(folded).item())


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

def cell_stats(frame_hw3: torch.Tensor, block: int, phase_y: int = 0, phase_x: int = 0):
    h, w, c = frame_hw3.shape
    y0, x0, ch, cw, oh, ow = crop_bounds(h, w, block, phase_y, phase_x)
    cropped = frame_hw3[y0:y0 + ch, x0:x0 + cw, :]
    blocks = cropped.reshape(oh, block, ow, block, c).permute(0, 2, 1, 3, 4).reshape(oh, ow, block * block, c)
    colors = blocks.median(dim=2).values
    saliency = blocks.std(dim=2).mean(dim=-1)  # oh, ow
    return colors, saliency


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
        return bulk

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
    return palette[keep_mask]


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
                          margin: float = 0.05):
    b, ch, cw, c = cropped.shape
    oh, ow = ch // block, cw // block
    n_cells = b * oh * ow

    flat_px = cropped.reshape(-1, c)
    idx_px = nearest_palette_index(flat_px, palette)
    idx_px = idx_px.reshape(b, oh, block, ow, block).permute(0, 1, 3, 2, 4).reshape(n_cells, block * block)

    weights = _cell_kernel(block, kind, cropped.device)
    k = palette.shape[0]
    hist = torch.zeros(n_cells, k, device=cropped.device)
    hist.scatter_add_(1, idx_px, weights.unsqueeze(0).expand(n_cells, -1))

    top2 = hist.topk(min(2, k), dim=1)
    total = hist.sum(dim=1).clamp(min=1e-8)
    winner_idx = top2.indices[:, 0]
    share_top = top2.values[:, 0] / total
    share_second = (top2.values[:, 1] / total) if k > 1 else torch.zeros_like(share_top)
    ambiguous = (share_top - share_second) < margin

    mean_rgb = cropped.reshape(b, oh, block, ow, block, c).permute(0, 1, 3, 2, 4, 5)
    mean_rgb = mean_rgb.reshape(n_cells, block * block, c).mean(dim=1)
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
    mode_val = torch.mode(stacked_l, dim=1).values.reshape(b, oh, ow).float()
    out = torch.where(disagree, mode_val, center)
    return out.long().clamp(0, k_colors - 1)


# ----------------------------------------------------------------------
# 7. Apply the fixed grid + fixed palette to the whole batch
# ----------------------------------------------------------------------

def process_batch(images: torch.Tensor, block: int, phase_y: int, phase_x: int, palette: torch.Tensor,
                   cell_method: str, dither: str, despeckle: bool,
                   out_h: int, out_w: int) -> torch.Tensor:
    b, h, w, c = images.shape
    y0, x0, ch, cw, oh, ow = crop_bounds(h, w, block, phase_y, phase_x)
    cropped = images[:, y0:y0 + ch, x0:x0 + cw, :]
    k = palette.shape[0]

    if dither != "none":
        cropped = cropped + _bayer_tile(dither, ch, cw, images.device) * dither_strength(palette)

    if cell_method in ("majority", "center_weighted"):
        idx_grid = weighted_vote_reduce(cropped, block, palette, cell_method)
    elif cell_method == "center":
        cy, cx = block // 2, block // 2
        sampled = cropped[:, cy::block, cx::block, :][:, :oh, :ow, :]
        idx_grid = nearest_palette_index(sampled.reshape(-1, c), palette).reshape(b, oh, ow)
    else:  # median
        blocks = cropped.reshape(b, oh, block, ow, block, c).permute(0, 1, 3, 2, 4, 5)
        down = blocks.reshape(b, oh, ow, block * block, c).median(dim=3).values
        idx_grid = nearest_palette_index(down.reshape(-1, c), palette).reshape(b, oh, ow)

    if despeckle:
        idx_grid = despeckle_indices(idx_grid, k)

    mapped = palette[idx_grid].clamp(0.0, 1.0)  # b, oh, ow, c

    mapped = mapped.permute(0, 3, 1, 2)
    mapped = F.interpolate(mapped, size=(out_h, out_w), mode="nearest")
    mapped = mapped.permute(0, 2, 3, 1)
    return mapped


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
    CATEGORY = "image/transform"
    RETURN_TYPES = ("IMAGE", "IMAGE", "STRING")
    RETURN_NAMES = ("image", "palette_preview", "info")
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
            },
            "optional": {
                "custom_palette": ("IMAGE", {"tooltip": "Optional fixed palette, fed in as an image (e.g. a "
                                                          "swatch strip, or this node's own palette_preview "
                                                          "output after you edit it). Unique colors are read "
                                                          "from it directly, k_colors/accent_slots are ignored."}),
            },
        }

    def run(self, image, pixel_size, grid_detection_mode, cell_method, k_colors, accent_slots,
            sample_frames, dither, despeckle, output_scale_mode, output_scale, seed, custom_palette=None):
        b, h, w, c = image.shape
        device = image.device

        n_samples = min(sample_frames, b)
        sample_idx = torch.linspace(0, b - 1, n_samples).round().long().unique()
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
        block = max(1, min(block, h // 2, w // 2))

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
        if custom_palette is not None:
            palette = palette_from_image(custom_palette.to(device), cap=256, seed=seed)
            palette_source = f"custom ({palette.shape[0]} colors)"
        else:
            pools_c, pools_s = [], []
            for i in range(samples.shape[0]):
                col, sal = cell_stats(samples[i], block, phase_y, phase_x)
                pools_c.append(col.reshape(-1, c))
                pools_s.append(sal.reshape(-1))
            pool_colors = torch.cat(pools_c, dim=0)
            pool_saliency = torch.cat(pools_s, dim=0)
            if pool_colors.shape[0] > 20000:
                g = torch.Generator(device="cpu").manual_seed(seed)
                keep = torch.randperm(pool_colors.shape[0], generator=g)[:20000].to(device)
                pool_colors, pool_saliency = pool_colors[keep], pool_saliency[keep]
            palette = build_palette(pool_colors, pool_saliency, k_colors, accent_slots, seed)
            n_accent = max(0, palette.shape[0] - (k_colors - accent_slots))
            palette_source = f"auto ({palette.shape[0]} colors, {n_accent} accent)"

        # --- output size ---
        oh = (h - phase_y % block) // block
        ow = (w - phase_x % block) // block
        out_h, out_w, resolved_scale = resolve_output_size(oh, ow, h, w, output_scale_mode, output_scale)

        # --- apply ---
        out = process_batch(image, block, phase_y, phase_x, palette, cell_method, dither, despeckle, out_h, out_w)
        preview = make_palette_preview(palette)
        info = (f"grid={block}px phase=({phase_x},{phase_y}) cells={ow}x{oh} "
                f"palette={palette_source} scale={resolved_scale}x -> {out_w}x{out_h}")

        return (out, preview, info)


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

    CATEGORY = "image/transform"
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
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
            }
        }

    def run(self, original_image, snapped_image, palette_preview, max_preview_frames):
        raw_refs = _save_preview_images(original_image, "VPS_raw", max_frames=max_preview_frames)
        frame_refs = _save_preview_images(snapped_image, "VPS_frame", max_frames=max_preview_frames)
        palette_refs = _save_preview_images(palette_preview, "VPS_palette", max_frames=1)
        ui_data = {"vps_raw": raw_refs, "vps_frames": frame_refs, "vps_palette": palette_refs}
        return {"ui": ui_data, "result": (snapped_image,)}


NODE_CLASS_MAPPINGS = {
    "VideoPixelSnapper": VideoPixelSnapper,
    "VideoPixelSnapperEditor": VideoPixelSnapperEditor,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "VideoPixelSnapper": "Video Pixel Snapper",
    "VideoPixelSnapperEditor": "Video Pixel Snapper (Live Editor)",
}

