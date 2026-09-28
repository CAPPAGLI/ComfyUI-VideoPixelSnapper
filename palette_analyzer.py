"""Perceptual coverage diagnostics for fixed game palettes.

This node never modifies the master palette. It measures how well that palette
covers observed foreground cell colors, suggests a master-only character
subpalette, and reports observed colors that occupy genuine Oklab gaps.
"""

import re
from typing import Optional

import torch
import torch.nn.functional as F

try:
    from .video_pixel_snapper import (
        _drop_alpha,
        _masked_median,
        _prepare_mask,
        _rgb_to_oklab,
        estimate_phase_for_frame,
        estimate_pixel_size,
        make_palette_preview,
        palette_from_image,
    )
except Exception:
    from video_pixel_snapper import (
        _drop_alpha,
        _masked_median,
        _prepare_mask,
        _rgb_to_oklab,
        estimate_phase_for_frame,
        estimate_pixel_size,
        make_palette_preview,
        palette_from_image,
    )


def _parse_snapper_grid(info: str):
    match = re.search(
        r"grid=(\d+)px phase=\((\d+),(\d+)\) cells=(\d+)x(\d+)"
        r"(?: source=(\d+)x(\d+))?",
        info or "",
    )
    if not match:
        return None
    values = [int(value) if value is not None else None for value in match.groups()]
    block, phase_x, phase_y, cells_w, cells_h, source_w, source_h = values
    mask_match = re.search(r"mask_threshold=([0-9.]+)", info or "")
    cell_match = re.search(r"cell_threshold=([0-9.]+)", info or "")
    return {
        "block": block,
        "phase_x": phase_x,
        "phase_y": phase_y,
        "cells_w": cells_w,
        "cells_h": cells_h,
        "source_w": source_w,
        "source_h": source_h,
        "mask_threshold": float(mask_match.group(1)) if mask_match else 0.5,
        "cell_threshold": float(cell_match.group(1)) if cell_match else 0.25,
    }


def _standalone_grid_info(
    sampled: torch.Tensor,
    analysis_pixel_size: float,
    mask_threshold: float,
    mask_cell_threshold: float,
):
    """Build core-compatible grid metadata without running Pixel Snapper."""
    _batch, height, width, _channels = sampled.shape
    if float(analysis_pixel_size) > 0:
        block = max(1, int(round(float(analysis_pixel_size))))
        source_note = "manual"
    else:
        estimates = []
        for frame in sampled:
            estimate = estimate_pixel_size(frame)
            if estimate:
                estimates.append(float(estimate))
        estimates.sort()
        block = (
            max(1, int(round(estimates[len(estimates) // 2])))
            if estimates else 1
        )
        source_note = "auto" if estimates else "auto_fallback_1"
    block = min(block, max(1, min(height, width) // 2))

    phase_xs, phase_ys = [], []
    for frame in sampled:
        phase_x, phase_y = estimate_phase_for_frame(frame, block)
        phase_xs.append(phase_x)
        phase_ys.append(phase_y)
    phase_x = int(torch.mode(torch.tensor(phase_xs)).values.item())
    phase_y = int(torch.mode(torch.tensor(phase_ys)).values.item())
    cells_w = max(1, (width - phase_x) // block)
    cells_h = max(1, (height - phase_y) // block)
    info = (
        f"grid={block}px phase=({phase_x},{phase_y}) "
        f"cells={cells_w}x{cells_h} source={width}x{height} "
        f"mask_threshold={float(mask_threshold):g} "
        f"cell_threshold={float(mask_cell_threshold):g}"
    )
    return info, f"standalone_{source_note}"


def _cell_representatives(
    image: torch.Tensor,
    mask: Optional[torch.Tensor],
    snapper_info: str,
):
    """Return BxHxWx3 representatives, BxHxW valid mask, and grid note."""
    batch, height, width, channels = image.shape
    grid = _parse_snapper_grid(snapper_info)
    if grid and (
        (grid["source_w"] in (None, width))
        and (grid["source_h"] in (None, height))
    ):
        block = max(1, grid["block"])
        x0, y0 = grid["phase_x"] % block, grid["phase_y"] % block
        cells_w = min(grid["cells_w"], (width - x0) // block)
        cells_h = min(grid["cells_h"], (height - y0) // block)
        crop_w, crop_h = cells_w * block, cells_h * block
        crop = image[:, y0:y0 + crop_h, x0:x0 + crop_w]
        pixels = crop.reshape(
            batch, cells_h, block, cells_w, block, channels
        ).permute(0, 1, 3, 2, 4, 5).reshape(
            batch, cells_h, cells_w, block * block, channels
        )
        if mask is None:
            representatives = pixels.median(dim=3).values
            valid_cells = torch.ones(
                (batch, cells_h, cells_w), device=image.device, dtype=torch.bool
            )
        else:
            mask_crop = mask[:, y0:y0 + crop_h, x0:x0 + crop_w]
            mask_blocks = mask_crop.reshape(
                batch, cells_h, block, cells_w, block
            ).permute(0, 1, 3, 2, 4).reshape(
                batch, cells_h, cells_w, block * block
            )
            valid_pixels = mask_blocks >= grid["mask_threshold"]
            representatives = _masked_median(pixels, valid_pixels)
            valid_cells = (
                (mask_blocks.mean(dim=3) >= grid["cell_threshold"])
                & valid_pixels.any(dim=3)
            )
        return representatives, valid_cells, (
            f"exact_grid block={block} phase=({x0},{y0}) "
            f"cells={cells_w}x{cells_h}"
        )

    # Conservative fallback for an unconnected/foreign info string. Area
    # reduction bounds memory, but exact Pixel Snapper cells require info.
    max_axis = max(height, width)
    scale = min(1.0, 256.0 / max(1, max_axis))
    out_h = max(1, int(round(height * scale)))
    out_w = max(1, int(round(width * scale)))
    representatives = F.interpolate(
        image.permute(0, 3, 1, 2), size=(out_h, out_w), mode="area"
    ).permute(0, 2, 3, 1)
    if mask is None:
        valid_cells = torch.ones(
            (batch, out_h, out_w), device=image.device, dtype=torch.bool
        )
    else:
        coverage = F.interpolate(
            mask.unsqueeze(1), size=(out_h, out_w), mode="area"
        ).squeeze(1)
        valid_cells = coverage >= 0.25
    return representatives, valid_cells, (
        f"fallback_area={out_w}x{out_h} (connect snapper_info for exact cells)"
    )


def _pack_8bit(rgb: torch.Tensor):
    q = (rgb.clamp(0.0, 1.0) * 255.0).round().to(torch.int64)
    return q[..., 0] * 65536 + q[..., 1] * 256 + q[..., 2]


def _unpack_8bit(keys: torch.Tensor, dtype, device):
    keys = keys.to(device=device, dtype=torch.int64)
    r = (keys // 65536) % 256
    g = (keys // 256) % 256
    b = keys % 256
    return torch.stack([r, g, b], dim=-1).to(dtype) / 255.0


def _hex(rgb: torch.Tensor):
    values = (rgb.detach().cpu().clamp(0.0, 1.0) * 255.0).round().long().tolist()
    return f"#{values[0]:02X}{values[1]:02X}{values[2]:02X}"


def _error_heatmap(distance: torch.Tensor, valid: torch.Tensor, limit: float):
    t = (distance / max(1e-6, float(limit))).clamp(0.0, 1.0)
    red = t
    green = (1.0 - (2.0 * t - 1.0).abs()).clamp(0.0, 1.0)
    blue = 1.0 - t
    heat = torch.stack([red, green, blue], dim=-1)
    return torch.where(valid.unsqueeze(-1), heat, torch.zeros_like(heat))


def _nearest_oklab_distance(
    values: torch.Tensor,
    palette: torch.Tensor,
    palette_lab: torch.Tensor,
    chunk: int = 200_000,
):
    flat = values.reshape(-1, 3)
    output = torch.empty(flat.shape[0], device=flat.device, dtype=flat.dtype)
    master_has_black = bool((palette.max(dim=1).values <= 2.0 / 255.0).any())
    for start in range(0, flat.shape[0], chunk):
        end = min(start + chunk, flat.shape[0])
        current = flat[start:end]
        perceptual = torch.cdist(
            _rgb_to_oklab(current), palette_lab
        ).min(dim=1).values
        rgb_near = torch.cdist(current.float(), palette.float()).min(dim=1).values
        # Tiny encoded-RGB deviations around black and other exact swatches
        # are usually resize/compression residue, not evidence for a new ramp.
        covered_residue = rgb_near < 16.0 / 255.0
        if master_has_black:
            covered_residue |= current.max(dim=1).values < 24.0 / 255.0
        output[start:end] = torch.where(
            covered_residue, torch.zeros_like(perceptual), perceptual
        )
    return output.reshape(values.shape[:-1])


def _master_subpalette(
    source: torch.Tensor,
    weights: torch.Tensor,
    master: torch.Tensor,
    size: int,
):
    size = max(1, min(int(size), master.shape[0]))
    source_lab = _rgb_to_oklab(source)
    master_lab = _rgb_to_oklab(master)
    distances = torch.cdist(source_lab, master_lab)
    current = torch.ones(source.shape[0], device=source.device, dtype=source.dtype)
    selected = []
    for _ in range(size):
        candidate_distance = torch.minimum(
            current.unsqueeze(1), distances
        )
        costs = (
            candidate_distance.square() * weights.unsqueeze(1)
        ).sum(dim=0)
        if selected:
            costs[torch.tensor(selected, device=costs.device)] = float("inf")
        best = int(costs.argmin().item())
        selected.append(best)
        current = torch.minimum(current, distances[:, best])
    selected.sort()  # preserve the master strip's semantic/ramp order
    return master[torch.tensor(selected, device=master.device)]


def _missing_suggestions(
    source: torch.Tensor,
    weights: torch.Tensor,
    master: torch.Tensor,
    count: int,
    threshold: float,
):
    if count <= 0 or source.numel() == 0:
        return source.new_empty((0, 3)), []
    source_lab = _rgb_to_oklab(source)
    master_lab = _rgb_to_oklab(master)
    to_master = torch.cdist(source_lab, master_lab)
    nearest_distance, nearest_index = to_master.min(dim=1)
    min_rgb_gap = torch.cdist(source.float(), master.float()).min(dim=1).values
    residual = nearest_distance.clone()
    # Suppress near-black/compression variants only a few 8-bit levels from
    # any master entry, even if Oklab points at a different dark swatch.
    eligible = (
        (nearest_distance >= float(threshold))
        & (min_rgb_gap >= 16.0 / 255.0)
    )
    if bool((master.max(dim=1).values <= 2.0 / 255.0).any()):
        eligible &= source.max(dim=1).values >= 24.0 / 255.0
    gap_eligible = eligible.clone()
    selected = []
    for _ in range(min(int(count), source.shape[0])):
        score = weights * residual.square()
        score = torch.where(eligible, score, torch.full_like(score, -1.0))
        best = int(score.argmax().item())
        if float(score[best]) <= 0.0:
            break
        selected.append(best)
        distance_to_new = torch.cdist(
            source_lab, source_lab[best:best + 1]
        ).squeeze(1)
        residual = torch.minimum(residual, distance_to_new)
        eligible[best] = False
    if not selected:
        return source.new_empty((0, 3)), []

    selected_index = torch.tensor(selected, device=source.device)
    suggestions = source[selected_index]
    to_suggestion = torch.cdist(source_lab, source_lab[selected_index])
    suggestion_distance, assignment = to_suggestion.min(dim=1)
    improved = suggestion_distance < nearest_distance
    details = []
    for slot, source_index in enumerate(selected):
        covered_weight = weights[
            (assignment == slot) & improved & gap_eligible
        ].sum()
        details.append((
            source[source_index],
            float(covered_weight.item()),
            master[int(nearest_index[source_index].item())],
            float(nearest_distance[source_index].item()),
        ))
    return suggestions, details


def _empty_preview(reference: torch.Tensor):
    return torch.zeros(
        (1, 32, 32, 3), device=reference.device, dtype=reference.dtype
    )


class VideoPixelSnapperPaletteCoverage:
    CATEGORY = "Video Pixel Snapper"
    RETURN_TYPES = ("IMAGE", "IMAGE", "IMAGE", "STRING")
    RETURN_NAMES = (
        "error_heatmap", "suggested_subpalette",
        "missing_color_suggestions", "info",
    )
    FUNCTION = "run"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE", {
                    "tooltip": "Original/pre-snap character frames. Analyze the RGB source, "
                               "not the already quantized result."
                }),
                "master_palette": ("IMAGE", {
                    "tooltip": "Exact master palette strip loaded directly from PNG."
                }),
                "subpalette_size": ("INT", {
                    "default": 24, "min": 2, "max": 128,
                    "tooltip": "Number of existing master colors to select for this character. "
                               "No new colors are introduced."
                }),
                "suggestion_count": ("INT", {
                    "default": 8, "min": 0, "max": 32,
                    "tooltip": "Maximum observed gap colors to report for human review. "
                               "They are never merged into the master automatically."
                }),
                "sample_frames": ("INT", {
                    "default": 8, "min": 1, "max": 64,
                    "tooltip": "Evenly spaced frames used for coverage statistics."
                }),
                "missing_threshold": ("FLOAT", {
                    "default": 0.06, "min": 0.0, "max": 0.30, "step": 0.005,
                    "tooltip": "Minimum Oklab distance considered a visible palette gap."
                }),
                "duplicate_threshold": ("FLOAT", {
                    "default": 0.03, "min": 0.0, "max": 0.15, "step": 0.005,
                    "tooltip": "Master colors closer than this are listed as possible "
                               "duplicate-slot candidates, not removed automatically."
                }),
                "heatmap_limit": ("FLOAT", {
                    "default": 0.12, "min": 0.01, "max": 0.40, "step": 0.01,
                    "tooltip": "Oklab error displayed as red in error_heatmap."
                }),
                "analysis_pixel_size": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 128.0, "step": 0.5,
                    "tooltip": "Standalone grid size: 0 auto-detects from the image batch; "
                               "set the measured value (for example 5) for reliable analysis. "
                               "Ignored when compatible snapper_info is connected."
                }),
                "mask_threshold": ("FLOAT", {
                    "default": 0.5, "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": "Standalone mode: minimum foreground-mask value for a source pixel."
                }),
                "mask_cell_threshold": ("FLOAT", {
                    "default": 0.25, "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": "Standalone mode: minimum average foreground coverage per cell."
                }),
            },
            "optional": {
                "foreground_mask": ("MASK", {
                    "tooltip": "Strongly recommended. Excludes background from coverage, "
                               "subpalette selection, and missing-color suggestions."
                }),
                "snapper_info": ("STRING", {
                    "forceInput": True,
                    "tooltip": "Optional compatibility mode: when connected and dimensions "
                               "match, reuse the exact Pixel Snapper grid. Otherwise the analyzer "
                               "works independently from analysis_pixel_size."
                }),
            },
        }

    def run(
        self, image, master_palette, subpalette_size, suggestion_count,
        sample_frames, missing_threshold, duplicate_threshold, heatmap_limit,
        analysis_pixel_size=0.0, mask_threshold=0.5,
        mask_cell_threshold=0.25, foreground_mask=None, snapper_info="",
    ):
        embedded_alpha = (
            image[..., 3].float().clamp(0.0, 1.0)
            if image.shape[-1] > 3 else None
        )
        image = _drop_alpha(image).float()
        master_palette = _drop_alpha(master_palette).float()
        batch, height, width, _ = image.shape
        device = image.device
        frame_count = min(max(1, int(sample_frames)), batch)
        frame_indices = torch.linspace(
            0, batch - 1, frame_count, device=device
        ).round().long().unique()
        sampled = image[frame_indices]

        prepared_mask = None
        if foreground_mask is not None:
            prepared_mask = _prepare_mask(
                foreground_mask, batch, height, width, device
            )[frame_indices]
        elif embedded_alpha is not None and bool((embedded_alpha < 1.0 - 1e-6).any()):
            prepared_mask = embedded_alpha.to(
                device=device, dtype=torch.float32
            )[frame_indices]

        master = palette_from_image(
            master_palette.to(device), cap=256, seed=0
        ).to(device=device, dtype=sampled.dtype)
        if master.shape[0] < 1:
            raise ValueError("master_palette contains no RGB colors")

        parsed_grid = _parse_snapper_grid(snapper_info)
        compatible_info = bool(parsed_grid) and (
            parsed_grid["source_w"] in (None, width)
            and parsed_grid["source_h"] in (None, height)
        )
        if compatible_info:
            analysis_info = snapper_info
            grid_source = "snapper_info"
        else:
            analysis_info, grid_source = _standalone_grid_info(
                sampled, float(analysis_pixel_size),
                float(mask_threshold), float(mask_cell_threshold),
            )
        representatives, valid, grid_note = _cell_representatives(
            sampled, prepared_mask, analysis_info
        )
        grid_note = f"grid_source={grid_source} {grid_note}"
        flat = representatives.reshape(-1, 3)
        flat_valid = valid.reshape(-1)
        if not bool(flat_valid.any()):
            raise ValueError(
                "No foreground analysis cells remain. Check foreground_mask, "
                "mask thresholds, and snapper_info alignment."
            )

        # Equalize each sampled frame so a larger silhouette/frame cannot own
        # the global suggestion score merely by contributing more cells.
        frame_weights = valid.float()
        denominators = frame_weights.sum(dim=(1, 2), keepdim=True).clamp(min=1.0)
        frame_weights = (frame_weights / denominators).reshape(-1)[flat_valid]
        valid_rgb = flat[flat_valid]
        keys = _pack_8bit(valid_rgb)
        unique_keys, inverse = torch.unique(keys, return_inverse=True)
        source = _unpack_8bit(unique_keys, valid_rgb.dtype, device)
        weights = torch.zeros(
            unique_keys.shape[0], device=device, dtype=valid_rgb.dtype
        )
        weights.scatter_add_(0, inverse, frame_weights)
        weights /= weights.sum().clamp(min=1e-8)

        # Bound worst-case per-pixel fallback analysis without biasing toward
        # arbitrary key order: retain the highest-weight observed colors.
        if source.shape[0] > 50_000:
            keep = weights.topk(50_000).indices
            source, weights = source[keep], weights[keep]
            weights /= weights.sum().clamp(min=1e-8)

        master_lab = _rgb_to_oklab(master)
        all_distance = _nearest_oklab_distance(
            representatives, master, master_lab
        )
        valid_distance = all_distance[valid]
        heatmap = _error_heatmap(all_distance, valid, heatmap_limit)

        subpalette = _master_subpalette(
            source, weights, master, int(subpalette_size)
        )
        missing, missing_details = _missing_suggestions(
            source, weights, master, int(suggestion_count),
            float(missing_threshold),
        )

        pair_distance = torch.cdist(master_lab, master_lab)
        pair_distance.fill_diagonal_(float("inf"))
        duplicate_pairs = []
        for i in range(master.shape[0]):
            for j in range(i + 1, master.shape[0]):
                distance = float(pair_distance[i, j].item())
                if distance < float(duplicate_threshold):
                    duplicate_pairs.append((distance, i, j))
        duplicate_pairs.sort()

        mean_error = float(valid_distance.mean().item())
        median_error = float(valid_distance.median().item())
        p95_error = float(torch.quantile(valid_distance, 0.95).item())
        max_error = float(valid_distance.max().item())
        missing_share = float(
            (valid_distance >= float(missing_threshold)).float().mean().item()
        )

        suggestion_lines = []
        for color, weight, nearest, distance in missing_details:
            suggestion_lines.append(
                f"{_hex(color)} ({weight * 100:.2f}% cells; "
                f"nearest {_hex(nearest)} d={distance:.4f})"
            )
        duplicate_lines = [
            f"{_hex(master[i])}/{_hex(master[j])} d={distance:.4f}"
            for distance, i, j in duplicate_pairs[:16]
        ]
        subpalette_hex = ",".join(_hex(color) for color in subpalette)
        info = (
            f"Palette Coverage Analyzer: sampled={len(frame_indices)}/{batch} "
            f"master={master.shape[0]} subpalette={subpalette.shape[0]} "
            f"[{subpalette_hex}]; {grid_note}; Oklab error mean={mean_error:.4f} "
            f"median={median_error:.4f} p95={p95_error:.4f} "
            f"max={max_error:.4f}; gap_cells@{float(missing_threshold):.3f}="
            f"{missing_share * 100:.2f}%; suggestions="
            f"{' | '.join(suggestion_lines) if suggestion_lines else 'none'}; "
            f"duplicate_pairs<{float(duplicate_threshold):.3f}="
            f"{' | '.join(duplicate_lines) if duplicate_lines else 'none'}"
        )

        subpalette_preview = make_palette_preview(subpalette)
        missing_preview = (
            make_palette_preview(missing)
            if missing.shape[0] else _empty_preview(master)
        )
        return (heatmap, subpalette_preview, missing_preview, info)


NODE_CLASS_MAPPINGS = {
    "VideoPixelSnapperPaletteCoverage": VideoPixelSnapperPaletteCoverage,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "VideoPixelSnapperPaletteCoverage": "Palette Coverage Analyzer (Video Pixel Snapper)",
}
