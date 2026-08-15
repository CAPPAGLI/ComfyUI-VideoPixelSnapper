"""
Motion-Aware Temporal Cleanup for Video Pixel Snapper
=====================================================

This replaces the old fixed-coordinate temporal mode filter. That method was
fundamentally wrong for moving pixel art: it compared (x,y) across frames as if
the same material stayed under that screen coordinate, so it both missed real
flicker and copied stale colors across moving edges.

The new node uses the original pre-snap video as a motion guide:

1. estimate adjacent-frame motion in both directions (RAFT Small, or a pure
   Torch integer block matcher with no model dependency);
2. reject unreliable correspondences using forward/backward cycle error,
   photometric error, bounds, and scene-cut detection;
3. gather palette colors from neighboring frames *along those trajectories*;
4. choose a discrete existing palette color only when motion-compensated
   neighbors reach a unique consensus and that color remains plausible under
   the current guide frame.

No RGB averaging, interpolation, or off-palette color synthesis is used in the
final image. On uncertain motion, occlusion, disocclusion, or a scene cut, the
current snapped frame wins. This makes the failure mode conservative (residual
flicker) rather than destructive (smear/trails).

Expected wiring:
    raw/original video ------------> guide_image
    Video Pixel Snapper scale=1 ---> image

The RAFT backend uses torchvision's pretrained raft_small weights. ComfyUI
normally already ships torchvision; the weights are downloaded by torchvision
on first use. Integer block matching remains available as an offline/no-model
fallback and is particularly natural at low sprite resolution.
"""

import re
from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F

try:
    from .video_pixel_snapper import _drop_alpha
except Exception:
    def _drop_alpha(image):
        return image[..., :3] if image.shape[-1] > 3 else image


_RAFT_SMALL_MODEL = None
_RAFT_SMALL_TRANSFORMS = None


def _resolve_compute_device(mode: str):
    if mode == "cpu":
        return torch.device("cpu")
    try:
        import comfy.model_management as model_management
        comfy_device = model_management.get_torch_device()
    except Exception:
        comfy_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if mode == "gpu":
        if comfy_device.type != "cpu":
            return comfy_device
        if torch.cuda.is_available():
            return torch.device("cuda")
        raise RuntimeError("compute_device='gpu' requested, but no GPU device is available")
    return comfy_device


def _pack_keys(image: torch.Tensor) -> torch.Tensor:
    q = (image.clamp(0.0, 1.0) * 255.0).round().to(torch.int64)
    return q[..., 0] * 65536 + q[..., 1] * 256 + q[..., 2]


def _resize_guide(guide: torch.Tensor, height: int, width: int) -> torch.Tensor:
    """BHWC guide -> B3HW at the snapped cell resolution."""
    x = _drop_alpha(guide).permute(0, 3, 1, 2).float()
    if x.shape[-2:] == (height, width):
        return x
    return F.interpolate(x, size=(height, width), mode="area")


def _resize_guide_streamed(
    guide: torch.Tensor, height: int, width: int, device, batch_size: int
):
    """Downsample guide chunks on the compute device without a full-res copy."""
    parts = []
    chunk = max(1, int(batch_size))
    for start in range(0, guide.shape[0], chunk):
        x = _drop_alpha(guide[start:start + chunk]).permute(0, 3, 1, 2)
        x = x.to(device=device, dtype=torch.float32, non_blocking=True)
        if x.shape[-2:] != (height, width):
            x = F.interpolate(x, size=(height, width), mode="area")
        parts.append(x)
    return torch.cat(parts, dim=0)


def _align_guide_to_snapper(
    guide: torch.Tensor, snapper_info: str, snapped_h: int, snapped_w: int
):
    """Crop a guide to the exact grid bounds reported by the core node.

    Pixel Snapper starts at (phase_x, phase_y) and discards incomplete cells
    on the far edges. A plain whole-frame resize ignores that crop and causes
    a fractional systematic offset when, for example, a 1920p guide is mapped
    to a ~144p cell grid. New info strings include the core source resolution,
    allowing the crop to be scaled even if guide_image has another resolution.
    """
    text = snapper_info or ""
    m = re.search(
        r"grid=(\d+)px phase=\((\d+),(\d+)\) cells=(\d+)x(\d+)"
        r"(?: source=(\d+)x(\d+))?",
        text,
    )
    if not m:
        return guide, "resize_only(no snapper_info)"

    block, phase_x, phase_y, cells_w, cells_h = map(int, m.groups()[:5])
    source_w = int(m.group(6)) if m.group(6) else None
    source_h = int(m.group(7)) if m.group(7) else None
    guide_h, guide_w = guide.shape[1:3]

    if source_w and source_h:
        scale_x = guide_w / float(source_w)
        scale_y = guide_h / float(source_h)
    elif guide_w >= phase_x + cells_w * block and guide_h >= phase_y + cells_h * block:
        scale_x = scale_y = 1.0
    else:
        return guide, (
            f"resize_only(old info; cells={cells_w}x{cells_h}, "
            f"snapped={snapped_w}x{snapped_h})"
        )

    x0 = max(0, min(guide_w - 1, int(round(phase_x * scale_x))))
    y0 = max(0, min(guide_h - 1, int(round(phase_y * scale_y))))
    x1 = max(x0 + 1, min(guide_w, int(round((phase_x + cells_w * block) * scale_x))))
    y1 = max(y0 + 1, min(guide_h, int(round((phase_y + cells_h * block) * scale_y))))
    cropped = guide[:, y0:y1, x0:x1, :]
    scale_note = "scale1" if (snapped_w, snapped_h) == (cells_w, cells_h) else (
        f"snapped_scale={snapped_w / max(1, cells_w):.3g}x"
    )
    return cropped, (
        f"grid_crop=({x0},{y0})-({x1},{y1}) guide={guide_w}x{guide_h} "
        f"cells={cells_w}x{cells_h} {scale_note}"
    )


def _confidence_heatmap(confidence: torch.Tensor) -> torch.Tensor:
    """Visible blue->green->yellow diagnostic IMAGE for ordinary Preview Image."""
    c = confidence.clamp(0.0, 1.0)
    red = (2.0 * c - 0.5).clamp(0.0, 1.0)
    green = (2.0 * c).clamp(0.0, 1.0)
    blue = (1.0 - 1.5 * c).clamp(0.0, 1.0)
    return torch.stack([red, green, blue], dim=-1)


def _background_key_from_info(snapper_info: str):
    m = re.search(r"mask=on bg=#([0-9a-fA-F]{6})", snapper_info or "")
    if not m:
        return None
    value = int(m.group(1), 16)
    r, g, b = (value >> 16) & 255, (value >> 8) & 255, value & 255
    return r * 65536 + g * 256 + b


def _flow_grid(flow: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Create grid_sample coordinates for target->source flow.

    flow is Bx2xHxW in pixel/cell units. At target coordinate (x,y), source
    is sampled at (x+dx,y+dy).
    """
    b, _, h, w = flow.shape
    yy, xx = torch.meshgrid(
        torch.arange(h, device=flow.device, dtype=flow.dtype),
        torch.arange(w, device=flow.device, dtype=flow.dtype),
        indexing="ij",
    )
    sx = xx.unsqueeze(0) + flow[:, 0]
    sy = yy.unsqueeze(0) + flow[:, 1]
    in_bounds = (sx >= 0) & (sx <= w - 1) & (sy >= 0) & (sy <= h - 1)
    if w > 1:
        gx = sx * (2.0 / (w - 1)) - 1.0
    else:
        gx = torch.zeros_like(sx)
    if h > 1:
        gy = sy * (2.0 / (h - 1)) - 1.0
    else:
        gy = torch.zeros_like(sy)
    return torch.stack([gx, gy], dim=-1), in_bounds


def _warp_bchw(source: torch.Tensor, flow: torch.Tensor, mode: str = "nearest"):
    grid, in_bounds = _flow_grid(flow)
    warped = F.grid_sample(
        source, grid, mode=mode, padding_mode="zeros", align_corners=True
    )
    return warped, in_bounds


def _warp_hwc(source: torch.Tensor, flow_2hw: torch.Tensor, mode: str = "nearest"):
    warped, in_bounds = _warp_bchw(
        source.permute(2, 0, 1).unsqueeze(0), flow_2hw.unsqueeze(0), mode
    )
    return warped[0].permute(1, 2, 0), in_bounds[0]


def _warp_mask(source: torch.Tensor, flow_2hw: torch.Tensor):
    warped, in_bounds = _warp_bchw(
        source.float().unsqueeze(0).unsqueeze(0), flow_2hw.unsqueeze(0), "nearest"
    )
    return (warped[0, 0] > 0.5) & in_bounds[0]


def _integer_block_flow(
    source: torch.Tensor,
    target: torch.Tensor,
    search_radius: int,
    patch_radius: int,
    motion_bias: float = 5e-4,
):
    """Estimate target->source integer flow with patch SSD.

    source/target are Bx3xHxW. Returns flow Bx2xHxW, best matching cost,
    and the gap to the second-best hypothesis (a reliability cue).
    """
    b, _c, h, w = target.shape
    dtype, device = target.dtype, target.device
    best = torch.full((b, h, w), float("inf"), device=device, dtype=dtype)
    second = torch.full_like(best, float("inf"))
    best_dx = torch.zeros((b, h, w), device=device, dtype=dtype)
    best_dy = torch.zeros_like(best_dx)
    radius = max(0, int(search_radius))
    patch = max(0, int(patch_radius))

    yy, xx = torch.meshgrid(
        torch.arange(h, device=device), torch.arange(w, device=device), indexing="ij"
    )
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            # roll(-d) gives candidate[y,x] = source[y+d, x+d]
            candidate = torch.roll(source, shifts=(-dy, -dx), dims=(-2, -1))
            valid = (
                (xx + dx >= 0) & (xx + dx < w)
                & (yy + dy >= 0) & (yy + dy < h)
            )
            point_cost = (candidate - target).square().mean(dim=1)
            point_cost = torch.where(valid.unsqueeze(0), point_cost, 1000.0)
            if patch:
                point_cost = F.avg_pool2d(
                    F.pad(
                        point_cost.unsqueeze(1),
                        (patch, patch, patch, patch),
                        mode="replicate",
                    ),
                    kernel_size=2 * patch + 1,
                    stride=1,
                ).squeeze(1)
            cost = point_cost + motion_bias * float(dx * dx + dy * dy)
            better = cost < best
            second = torch.where(better, best, torch.minimum(second, cost))
            best = torch.where(better, cost, best)
            best_dx = torch.where(better, float(dx), best_dx)
            best_dy = torch.where(better, float(dy), best_dy)

    flow = torch.stack([best_dx, best_dy], dim=1)
    return flow, best, (second - best).clamp(min=0.0)


def _load_raft_small(device):
    global _RAFT_SMALL_MODEL, _RAFT_SMALL_TRANSFORMS
    try:
        from torchvision.models.optical_flow import (
            Raft_Small_Weights,
            raft_small,
        )
    except Exception as exc:
        raise RuntimeError(
            "RAFT Small requires torchvision optical_flow support. Use "
            "flow_backend='integer_block_matching', or install a ComfyUI build "
            "with torchvision. Original error: " + str(exc)
        ) from exc

    if _RAFT_SMALL_MODEL is None:
        try:
            weights = Raft_Small_Weights.DEFAULT
            _RAFT_SMALL_MODEL = raft_small(
                weights=weights, progress=True
            ).eval().cpu()
            _RAFT_SMALL_TRANSFORMS = weights.transforms()
        except Exception as exc:
            raise RuntimeError(
                "Could not load/download torchvision RAFT Small weights. "
                "Check network/cache permissions or switch to "
                "flow_backend='integer_block_matching'. Original error: "
                + str(exc)
            ) from exc
    _RAFT_SMALL_MODEL = _RAFT_SMALL_MODEL.to(device).eval()
    return _RAFT_SMALL_MODEL, _RAFT_SMALL_TRANSFORMS


def _raft_bidirectional_flow(
    guide_bhwc: torch.Tensor,
    out_h: int,
    out_w: int,
    flow_scale: float,
    updates: int,
    batch_size: int,
    keep_model_loaded: bool,
    compute_device,
):
    """Return adjacent flows in cell units, streaming guide pairs to GPU."""
    n, src_h, src_w, _ = guide_bhwc.shape
    device = torch.device(compute_device)
    scale = max(0.1, min(1.0, float(flow_scale)))
    # torchvision RAFT builds a four-level correlation pyramid after an
    # 8x encoder downsample, so each input axis must be at least 128.
    fh = max(128, int(round(src_h * scale / 8.0)) * 8)
    fw = max(128, int(round(src_w * scale / 8.0)) * 8)
    model, transforms = _load_raft_small(device)
    forward_parts, backward_parts = [], []
    pair_batch = max(1, int(batch_size))

    def to_cells(flow):
        """Immediately discard full-resolution RAFT flow after cell resize."""
        resized = F.interpolate(
            flow, size=(out_h, out_w), mode="bilinear", align_corners=False
        )
        resized[:, 0] *= out_w / float(fw)
        resized[:, 1] *= out_h / float(fh)
        return resized

    with torch.inference_mode():
        for start in range(0, n - 1, pair_batch):
            end = min(start + pair_batch, n - 1)
            prev = _drop_alpha(guide_bhwc[start:end]).permute(0, 3, 1, 2)
            curr = _drop_alpha(guide_bhwc[start + 1:end + 1]).permute(0, 3, 1, 2)
            prev = F.interpolate(
                prev.to(device=device, dtype=torch.float32, non_blocking=True),
                size=(fh, fw), mode="bilinear", align_corners=False,
            )
            curr = F.interpolate(
                curr.to(device=device, dtype=torch.float32, non_blocking=True),
                size=(fh, fw), mode="bilinear", align_corners=False,
            )

            # Run directions sequentially. The previous implementation doubled
            # the effective RAFT batch by concatenating forward and backward
            # pairs, then retained every full-resolution flow until the whole
            # clip finished. On long 1080p clips that alone can occupy many GB.
            a, b = transforms(prev, curr)
            flow = model(a, b, num_flow_updates=max(1, int(updates)))[-1]
            forward_parts.append(to_cells(flow))    # prev -> current, at prev
            del a, b, flow

            a, b = transforms(curr, prev)
            flow = model(a, b, num_flow_updates=max(1, int(updates)))[-1]
            backward_parts.append(to_cells(flow))   # current -> prev, at current
            del a, b, flow, prev, curr

    # Only cell-resolution trajectories survive between chunks.
    forward = torch.cat(forward_parts, dim=0)
    backward = torch.cat(backward_parts, dim=0)
    if not keep_model_loaded:
        _RAFT_SMALL_MODEL.to("cpu")
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return forward, backward, None, None


def _estimate_flows(
    guide_bhwc: torch.Tensor,
    guide_low: torch.Tensor,
    backend: str,
    search_radius: int,
    patch_radius: int,
    flow_scale: float,
    raft_updates: int,
    flow_batch_size: int,
    keep_model_loaded: bool,
    compute_device,
):
    n, _c, h, w = guide_low.shape
    if backend == "raft_small":
        return _raft_bidirectional_flow(
            guide_bhwc, h, w, flow_scale, raft_updates,
            flow_batch_size, keep_model_loaded, compute_device,
        )

    # target prev samples source current -> conventional forward flow.
    forward, f_cost, _f_gap = _integer_block_flow(
        guide_low[1:], guide_low[:-1], search_radius, patch_radius
    )
    # target current samples source prev -> conventional backward flow.
    backward, b_cost, _b_gap = _integer_block_flow(
        guide_low[:-1], guide_low[1:], search_radius, patch_radius
    )
    return forward, backward, f_cost, b_cost


def _pair_reliability(
    source: torch.Tensor,
    target: torch.Tensor,
    flow_target_to_source: torch.Tensor,
    reverse_flow: torch.Tensor,
    photometric_threshold: float,
    cycle_threshold: float,
    match_cost: Optional[torch.Tensor] = None,
    match_cost_threshold: float = 0.05,
):
    """Reliability at target coordinates for one adjacent pair batch."""
    warped_source, in_bounds = _warp_bchw(source, flow_target_to_source, "bilinear")
    warped_reverse, reverse_bounds = _warp_bchw(
        reverse_flow, flow_target_to_source, "bilinear"
    )
    photo = (warped_source - target).abs().mean(dim=1)
    cycle = (flow_target_to_source + warped_reverse).square().sum(dim=1).sqrt()
    geometry_reliable = in_bounds & reverse_bounds & (cycle <= cycle_threshold)
    reliable = geometry_reliable & (photo <= photometric_threshold)
    if match_cost is not None:
        reliable &= match_cost <= match_cost_threshold
    return reliable, geometry_reliable, photo, cycle


def _build_pair_confidence(
    guide_low: torch.Tensor,
    forward: torch.Tensor,
    backward: torch.Tensor,
    forward_cost: Optional[torch.Tensor],
    backward_cost: Optional[torch.Tensor],
    photometric_threshold: float,
    cycle_threshold: float,
    match_cost_threshold: float,
    scene_threshold: float,
):
    # At frame t coordinates, forward[t] samples frame t+1.
    f_conf, f_geometry, f_photo, _ = _pair_reliability(
        guide_low[1:], guide_low[:-1], forward, backward,
        photometric_threshold, cycle_threshold, forward_cost,
        match_cost_threshold,
    )
    # At frame t+1 coordinates, backward[t] samples frame t.
    b_conf, b_geometry, b_photo, _ = _pair_reliability(
        guide_low[:-1], guide_low[1:], backward, forward,
        photometric_threshold, cycle_threshold, backward_cost,
        match_cost_threshold,
    )
    scene_diff = (guide_low[1:] - guide_low[:-1]).abs().mean(dim=(1, 2, 3))
    cuts = scene_diff > scene_threshold
    if cuts.any():
        f_conf[cuts] = False
        b_conf[cuts] = False
        f_geometry[cuts] = False
        b_geometry[cuts] = False
    return (
        f_conf, b_conf, f_geometry, b_geometry,
        f_photo, b_photo, cuts,
    )


def _past_candidate(
    frames: torch.Tensor,
    target_t: int,
    source_t: int,
    backward: torch.Tensor,
    backward_conf: torch.Tensor,
):
    candidate = frames[source_t]
    valid = torch.ones(candidate.shape[:2], device=frames.device, dtype=torch.bool)
    for k in range(source_t + 1, target_t + 1):
        candidate, bounds = _warp_hwc(candidate, backward[k - 1], "nearest")
        valid = _warp_mask(valid, backward[k - 1]) & backward_conf[k - 1] & bounds
    return candidate, valid


def _future_candidate(
    frames: torch.Tensor,
    target_t: int,
    source_t: int,
    forward: torch.Tensor,
    forward_conf: torch.Tensor,
):
    candidate = frames[source_t]
    valid = torch.ones(candidate.shape[:2], device=frames.device, dtype=torch.bool)
    for k in range(source_t - 1, target_t - 1, -1):
        candidate, bounds = _warp_hwc(candidate, forward[k], "nearest")
        valid = _warp_mask(valid, forward[k]) & forward_conf[k] & bounds
    return candidate, valid


def motion_compensated_label_consensus(
    frames: torch.Tensor,
    guide_low: torch.Tensor,
    forward: torch.Tensor,
    backward: torch.Tensor,
    forward_conf: torch.Tensor,
    backward_conf: torch.Tensor,
    forward_geometry_conf: torch.Tensor,
    backward_geometry_conf: torch.Tensor,
    window: int,
    agreement_threshold: float,
    data_tolerance: float,
    max_color_distance: float,
    background_key: Optional[int] = None,
    silhouette_stabilization: bool = True,
    silhouette_agreement: float = 0.75,
    silhouette_min_support: int = 3,
    edge_color_stabilization: bool = True,
    edge_radius: int = 2,
    edge_agreement: float = 0.6,
    edge_min_support: int = 3,
    edge_max_color_distance: float = 0.8,
    edge_medoid_fallback: bool = True,
    edge_cluster_radius: float = 0.35,
    feature_edge_stabilization: bool = False,
    feature_radius: int = 1,
    feature_contrast_threshold: float = 0.08,
):
    """Discrete label consensus with a separate silhouette/topology pass.

    Ordinary color changes use strict photometric confidence and palette-color
    distance. Foreground/background changes use stricter temporal agreement but
    geometry confidence (bounds + forward/backward cycle) so a one-frame tooth
    is not automatically rejected merely because its appearance differs.
    """
    n, h, w, _ = frames.shape
    half = max(1, int(window) // 2)
    output = frames.clone()
    debug_conf = torch.zeros((n, h, w), device=frames.device)
    silhouette_changes = torch.zeros((n, h, w), device=frames.device)
    edge_color_changes = torch.zeros((n, h, w), device=frames.device)
    feature_color_changes = torch.zeros((n, h, w), device=frames.device)
    replaced = 0
    silhouette_replaced = 0
    edge_replaced = 0
    feature_replaced = 0

    def select_consensus(colors, keys, valid, current_key, current_rgb):
        current_count = ((keys == current_key.unsqueeze(0)) & valid).sum(dim=0)
        best_count = current_count
        best_key = current_key
        best_rgb = current_rgb
        for i in range(1, colors.shape[0]):
            count = ((keys == keys[i].unsqueeze(0)) & valid).sum(dim=0)
            better = valid[i] & (count > best_count)
            best_count = torch.where(better, count, best_count)
            best_key = torch.where(better, keys[i], best_key)
            best_rgb = torch.where(better.unsqueeze(-1), colors[i], best_rgb)

        tied = torch.zeros((h, w), device=frames.device, dtype=torch.bool)
        for i in range(colors.shape[0]):
            count = ((keys == keys[i].unsqueeze(0)) & valid).sum(dim=0)
            tied |= valid[i] & (keys[i] != best_key) & (count == best_count)
        valid_count = valid.sum(dim=0).clamp(min=1)
        agreement = best_count.float() / valid_count.float()
        return best_rgb, best_key, best_count, ~tied, valid_count, agreement

    def select_medoid(colors, keys, valid, current_rgb, current_key):
        """Pick an existing candidate minimizing distance to valid colors."""
        valid_count = valid.sum(dim=0).clamp(min=1)
        best_rgb = current_rgb
        best_key = current_key
        best_score = torch.full(
            (h, w), float("inf"), device=frames.device, dtype=frames.dtype
        )
        for i in range(colors.shape[0]):
            score = torch.zeros((h, w), device=frames.device, dtype=frames.dtype)
            for j in range(colors.shape[0]):
                distance = (colors[i] - colors[j]).square().sum(dim=-1).sqrt()
                score += distance * valid[j].to(frames.dtype)
            score = score / valid_count.to(frames.dtype)
            usable = valid[i]
            better = usable & (score < best_score)
            best_score = torch.where(better, score, best_score)
            best_key = torch.where(better, keys[i], best_key)
            best_rgb = torch.where(better.unsqueeze(-1), colors[i], best_rgb)
        return best_rgb, best_key, best_score, valid_count

    for t in range(n):
        candidates: List[torch.Tensor] = [frames[t]]
        color_valids: List[torch.Tensor] = [
            torch.ones((h, w), device=frames.device, dtype=torch.bool)
        ]
        geometry_valids: List[torch.Tensor] = [color_valids[0]]

        for delta in range(1, half + 1):
            s = t - delta
            if s >= 0:
                cand, color_valid = _past_candidate(
                    frames, t, s, backward, backward_conf
                )
                _same, geometry_valid = _past_candidate(
                    frames, t, s, backward, backward_geometry_conf
                )
                candidates.append(cand)
                color_valids.append(color_valid)
                geometry_valids.append(geometry_valid)
            s = t + delta
            if s < n:
                cand, color_valid = _future_candidate(
                    frames, t, s, forward, forward_conf
                )
                _same, geometry_valid = _future_candidate(
                    frames, t, s, forward, forward_geometry_conf
                )
                candidates.append(cand)
                color_valids.append(color_valid)
                geometry_valids.append(geometry_valid)

        colors = torch.stack(candidates, dim=0)
        keys = _pack_keys(colors)
        current_key = keys[0]
        color_valid = torch.stack(color_valids, dim=0)
        geometry_valid = torch.stack(geometry_valids, dim=0)

        (
            best_rgb, best_key, best_count, unique_winner,
            valid_count, agreement,
        ) = select_consensus(colors, keys, color_valid, current_key, frames[t])
        (
            geom_rgb, geom_key, geom_count, geom_unique,
            geom_valid_count, geom_agreement,
        ) = select_consensus(colors, keys, geometry_valid, current_key, frames[t])
        medoid_rgb, medoid_key, medoid_spread, _ = select_medoid(
            colors, keys, geometry_valid, frames[t], current_key
        )

        guide = guide_low[t].permute(1, 2, 0)
        current_data = (guide - frames[t]).square().mean(dim=-1)
        candidate_data = (guide - best_rgb).square().mean(dim=-1)
        color_jump = (best_rgb - frames[t]).square().sum(dim=-1).sqrt()

        normal_replace = (
            unique_winner
            & (best_key != current_key)
            & (best_count >= 2)
            & (agreement >= agreement_threshold)
            & (candidate_data <= current_data + data_tolerance)
            & (color_jump <= max_color_distance)
        )

        silhouette_replace = torch.zeros_like(normal_replace)
        silhouette_candidate = torch.zeros_like(normal_replace)
        edge_replace = torch.zeros_like(normal_replace)
        edge_medoid_replace = torch.zeros_like(normal_replace)
        edge_candidate = torch.zeros_like(normal_replace)
        external_edge_region = torch.zeros_like(normal_replace)
        external_proximity_region = torch.zeros_like(normal_replace)
        feature_region = torch.zeros_like(normal_replace)

        if background_key is not None:
            current_is_bg = current_key == int(background_key)
            normal_is_bg = best_key == int(background_key)
            geom_is_bg = geom_key == int(background_key)
            medoid_is_bg = medoid_key == int(background_key)
            if silhouette_stabilization:
                # Never let the normal color pass perform topology changes.
                normal_replace &= normal_is_bg == current_is_bg
                silhouette_candidate = geom_is_bg != current_is_bg
                silhouette_replace = (
                    geom_unique
                    & silhouette_candidate
                    & (geom_count >= max(2, int(silhouette_min_support)))
                    & (geom_agreement >= silhouette_agreement)
                )
            if edge_color_stabilization or feature_edge_stabilization:
                radius = max(1, int(edge_radius))
                nearby_bg = F.max_pool2d(
                    current_is_bg.float().unsqueeze(0).unsqueeze(0),
                    kernel_size=2 * radius + 1,
                    stride=1,
                    padding=radius,
                )[0, 0] > 0.5
                external_proximity_region = (~current_is_bg) & nearby_bg
                if edge_color_stabilization:
                    external_edge_region = external_proximity_region
        else:
            current_is_bg = torch.zeros_like(normal_replace)
            geom_is_bg = torch.zeros_like(normal_replace)
            medoid_is_bg = torch.zeros_like(normal_replace)

        if feature_edge_stabilization and background_key is not None:
            rgb = frames[t]
            rgb_chw = rgb.permute(2, 0, 1).unsqueeze(0)
            padded = F.pad(rgb_chw, (1, 1, 1, 1), mode="replicate")
            center = padded[:, :, 1:-1, 1:-1]
            neighbors = [
                padded[:, :, :-2, 1:-1], padded[:, :, 2:, 1:-1],
                padded[:, :, 1:-1, :-2], padded[:, :, 1:-1, 2:],
            ]
            local_contrast = torch.zeros((h, w), device=frames.device, dtype=frames.dtype)
            for neighbor in neighbors:
                difference = (center - neighbor).square().sum(dim=1).sqrt()[0]
                local_contrast = torch.maximum(local_contrast, difference)
            boundary = local_contrast >= feature_contrast_threshold
            radius = max(0, int(feature_radius))
            if radius:
                boundary = F.max_pool2d(
                    boundary.float().unsqueeze(0).unsqueeze(0),
                    kernel_size=2 * radius + 1,
                    stride=1,
                    padding=radius,
                )[0, 0] > 0.5
            feature_region = boundary & (~current_is_bg) & (~external_proximity_region)

        color_lock_region = external_edge_region | feature_region
        if edge_color_stabilization or feature_edge_stabilization:
            same_foreground_side = (~current_is_bg) & (~geom_is_bg)
            edge_candidate = (
                color_lock_region & same_foreground_side & (geom_key != current_key)
            )
            geom_color_jump = (
                geom_rgb - frames[t]
            ).square().sum(dim=-1).sqrt()
            edge_replace = (
                geom_unique
                & edge_candidate
                & (geom_count >= max(2, int(edge_min_support)))
                & (geom_agreement >= edge_agreement)
                & (geom_color_jump <= edge_max_color_distance)
            )

            # If every frame picked a different nearby palette shade,
            # exact mode has no winner. Select the existing temporal medoid.
            medoid_candidate = (
                color_lock_region & (~current_is_bg) & (~medoid_is_bg)
                & (medoid_key != current_key)
            )
            medoid_jump = (
                medoid_rgb - frames[t]
            ).square().sum(dim=-1).sqrt()
            if edge_medoid_fallback:
                edge_medoid_replace = (
                    (~edge_replace)
                    & medoid_candidate
                    & (geom_valid_count >= max(2, int(edge_min_support)))
                    & (medoid_spread <= edge_cluster_radius)
                    & (medoid_jump <= edge_max_color_distance)
                )
                edge_candidate |= medoid_candidate
            # Dedicated geometry passes own these pixels.
            normal_replace &= ~color_lock_region

        effective_edge_replace = edge_replace | edge_medoid_replace
        replace = normal_replace | effective_edge_replace | silhouette_replace
        replacement_rgb = torch.where(
            edge_replace.unsqueeze(-1), geom_rgb, best_rgb
        )
        replacement_rgb = torch.where(
            edge_medoid_replace.unsqueeze(-1), medoid_rgb, replacement_rgb
        )
        replacement_rgb = torch.where(
            silhouette_replace.unsqueeze(-1), geom_rgb, replacement_rgb
        )
        output[t] = torch.where(replace.unsqueeze(-1), replacement_rgb, frames[t])

        color_neighbors = (valid_count.float() - 1.0).clamp(min=0.0)
        color_neighbors /= max(1.0, float(colors.shape[0] - 1))
        geom_neighbors = (geom_valid_count.float() - 1.0).clamp(min=0.0)
        geom_neighbors /= max(1.0, float(colors.shape[0] - 1))
        color_debug = agreement * color_neighbors
        geom_debug = geom_agreement * geom_neighbors
        geometry_candidate = silhouette_candidate | edge_candidate
        debug_conf[t] = torch.where(geometry_candidate, geom_debug, color_debug)
        silhouette_changes[t] = silhouette_replace.float()
        external_changes = effective_edge_replace & external_edge_region
        internal_changes = effective_edge_replace & feature_region
        edge_color_changes[t] = external_changes.float()
        feature_color_changes[t] = internal_changes.float()
        replaced += int(replace.sum().item())
        silhouette_replaced += int(silhouette_replace.sum().item())
        edge_replaced += int(external_changes.sum().item())
        feature_replaced += int(internal_changes.sum().item())

    return (
        output,
        debug_conf.clamp(0.0, 1.0),
        replaced,
        silhouette_changes,
        silhouette_replaced,
        edge_color_changes,
        edge_replaced,
        feature_color_changes,
        feature_replaced,
    )


class VideoPixelSnapperTemporalCleanupAdvanced:
    CATEGORY = "Video Pixel Snapper"
    RETURN_TYPES = (
        "IMAGE", "MASK", "STRING", "IMAGE", "MASK", "MASK", "MASK", "MASK"
    )
    RETURN_NAMES = (
        "image", "motion_confidence", "info", "confidence_preview",
        "changed_cells", "silhouette_changes", "edge_color_changes",
        "feature_color_changes",
    )
    FUNCTION = "run"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE", {
                    "tooltip": "Video Pixel Snapper output with output_scale=1. "
                               "Colors must already be palette-quantized."
                }),
                "guide_image": ("IMAGE", {
                    "tooltip": "The corresponding original/pre-snap video batch. "
                               "Motion is estimated here, where texture and edges still exist."
                }),
                "flow_backend": (["raft_small", "integer_block_matching"], {
                    "default": "raft_small",
                    "tooltip": "raft_small: higher-quality deformable motion via torchvision "
                               "(downloads weights once). integer_block_matching: model-free, "
                               "fast at sprite resolution, but limited to local translations."
                }),
                "compute_device": (["auto", "gpu", "cpu"], {
                    "default": "auto",
                    "tooltip": "auto uses ComfyUI's active torch device (normally CUDA). "
                               "gpu forces acceleration; cpu is a diagnostic fallback."
                }),
                "window": ("INT", {
                    "default": 5, "min": 3, "max": 7, "step": 2,
                    "tooltip": "Motion-aligned temporal support. 5 is a conservative default."
                }),
                "agreement_threshold": ("FLOAT", {
                    "default": 0.6, "min": 0.34, "max": 1.0, "step": 0.02,
                    "tooltip": "Required fraction of valid motion-aligned candidates that "
                               "must agree on one palette color."
                }),
                "data_tolerance": ("FLOAT", {
                    "default": 0.015, "min": 0.0, "max": 0.25, "step": 0.005,
                    "tooltip": "How much worse than the current label a temporal candidate "
                               "may fit the current guide color. Lower protects motion/edges."
                }),
                "max_color_distance": ("FLOAT", {
                    "default": 0.35, "min": 0.0, "max": 1.732, "step": 0.025,
                    "tooltip": "Never replace an ordinary foreground color with a temporal "
                               "candidate farther away than this RGB distance. Silhouette "
                               "foreground/background changes use separate controls below."
                }),
                "silhouette_stabilization": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Stabilize transient foreground/background cells (one-frame "
                               "teeth, holes, and crawling silhouette pixels) with a separate "
                               "strict motion consensus. Requires mask background metadata in "
                               "the connected snapper_info."
                }),
                "silhouette_agreement": ("FLOAT", {
                    "default": 0.75, "min": 0.5, "max": 1.0, "step": 0.05,
                    "tooltip": "Required geometry-consensus fraction before changing a cell "
                               "between foreground and the locked background color."
                }),
                "silhouette_min_support": ("INT", {
                    "default": 3, "min": 2, "max": 7,
                    "tooltip": "Minimum number of motion-aligned frames supporting a "
                               "foreground/background topology change."
                }),
                "edge_color_stabilization": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Stabilize palette indices inside thin foreground regions "
                               "near the locked background: outline pixels, spine teeth, "
                               "horns, fingers, and other narrow edge details."
                }),
                "edge_radius": ("INT", {
                    "default": 2, "min": 1, "max": 6,
                    "tooltip": "Foreground cells within this many cells of background use "
                               "the dedicated edge-color consensus."
                }),
                "edge_agreement": ("FLOAT", {
                    "default": 0.6, "min": 0.34, "max": 1.0, "step": 0.05,
                    "tooltip": "Required geometry-aligned agreement for edge palette colors."
                }),
                "edge_min_support": ("INT", {
                    "default": 3, "min": 2, "max": 7,
                    "tooltip": "Minimum aligned frames supporting an edge color."
                }),
                "edge_max_color_distance": ("FLOAT", {
                    "default": 0.8, "min": 0.0, "max": 1.75, "step": 0.05,
                    "tooltip": "Maximum palette-color jump inside edge regions. This is "
                               "separate from ordinary max_color_distance."
                }),
                "edge_medoid_fallback": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "When no exact palette index repeats enough, choose an existing "
                               "temporal medoid from a compact cluster of edge shades."
                }),
                "edge_cluster_radius": ("FLOAT", {
                    "default": 0.35, "min": 0.0, "max": 1.75, "step": 0.05,
                    "tooltip": "Maximum mean RGB spread for temporal-medoid fallback. Lower "
                               "accepts only closely related shades."
                }),
                "photometric_threshold": ("FLOAT", {
                    "default": 0.12, "min": 0.0, "max": 1.0, "step": 0.01,
                    "tooltip": "Reject flow where the warped guide disagrees with the current "
                               "guide. Lower is safer against trails."
                }),
                "flow_consistency": ("FLOAT", {
                    "default": 1.25, "min": 0.0, "max": 10.0, "step": 0.25,
                    "tooltip": "Maximum forward/backward cycle error in output cell units."
                }),
                "scene_threshold": ("FLOAT", {
                    "default": 0.30, "min": 0.01, "max": 1.0, "step": 0.01,
                    "tooltip": "Mean guide-frame difference above which temporal links are "
                               "disabled as a scene cut."
                }),
                "search_radius": ("INT", {
                    "default": 4, "min": 0, "max": 16,
                    "tooltip": "Integer block matching only: maximum cell displacement per frame."
                }),
                "patch_radius": ("INT", {
                    "default": 2, "min": 0, "max": 6,
                    "tooltip": "Integer block matching only: local patch radius."
                }),
                "match_cost_threshold": ("FLOAT", {
                    "default": 0.05, "min": 0.0, "max": 1.0, "step": 0.01,
                    "tooltip": "Integer block matching only: reject high patch-MSE matches."
                }),
                "flow_scale": ("FLOAT", {
                    "default": 0.5, "min": 0.1, "max": 1.0, "step": 0.1,
                    "tooltip": "RAFT only: guide resolution scale. 0.5 is much faster; 1.0 "
                               "is best for thin/fast details."
                }),
                "raft_updates": ("INT", {
                    "default": 8, "min": 1, "max": 24,
                    "tooltip": "RAFT refinement iterations. 8-12 is usually sufficient."
                }),
                "flow_batch_size": ("INT", {
                    "default": 4, "min": 1, "max": 32,
                    "tooltip": "RAFT pair batch size; lower if VRAM is insufficient."
                }),
                "keep_flow_model_loaded": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Keep RAFT on GPU after the run for faster repeated tests, at "
                               "the cost of persistent VRAM use."
                }),
                "feature_edge_stabilization": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Extend geometry/medoid color stabilization to internal "
                               "palette boundaries: belly-vs-skin borders, muscles, nose, "
                               "mouth, eyes, folds, and other thin drawn features."
                }),
                "feature_radius": ("INT", {
                    "default": 1, "min": 0, "max": 4,
                    "tooltip": "Dilate detected internal palette boundaries by this many cells."
                }),
                "feature_contrast_threshold": ("FLOAT", {
                    "default": 0.08, "min": 0.0, "max": 1.75, "step": 0.02,
                    "tooltip": "Minimum local RGB palette difference considered an internal "
                               "feature edge."
                }),
            },
            "optional": {
                "snapper_info": ("STRING", {
                    "forceInput": True,
                    "tooltip": "Strongly recommended: connect Video Pixel Snapper's info "
                               "output. It provides source size, grid phase, and crop bounds "
                               "so a 1920p guide aligns exactly with the low-res cell image."
                }),
            },
        }

    def run(
        self, image, guide_image, flow_backend, compute_device, window,
        agreement_threshold, data_tolerance, max_color_distance,
        silhouette_stabilization, silhouette_agreement, silhouette_min_support,
        edge_color_stabilization, edge_radius, edge_agreement,
        edge_min_support, edge_max_color_distance, edge_medoid_fallback,
        edge_cluster_radius, photometric_threshold, flow_consistency,
        scene_threshold, search_radius, patch_radius,
        match_cost_threshold, flow_scale, raft_updates, flow_batch_size,
        keep_flow_model_loaded, feature_edge_stabilization, feature_radius,
        feature_contrast_threshold, snapper_info="",
    ):
        image = _drop_alpha(image)
        guide_image = _drop_alpha(guide_image)
        n, h, w, _ = image.shape
        if guide_image.shape[0] != n:
            raise ValueError(
                f"guide_image has {guide_image.shape[0]} frames but image has {n}; "
                "they must describe the same clip and frame order"
            )
        output_device = image.device
        aligned_guide, alignment_info = _align_guide_to_snapper(
            guide_image, snapper_info, h, w
        )
        if n <= 1:
            confidence = torch.zeros((n, h, w), device=output_device)
            changed = torch.zeros_like(confidence)
            return (
                image, confidence, "1 frame: no temporal processing; " + alignment_info,
                _confidence_heatmap(confidence), changed, changed, changed, changed,
            )

        work_device = _resolve_compute_device(compute_device)
        image = image.to(work_device, non_blocking=True)
        guide_low = _resize_guide_streamed(
            aligned_guide, h, w, work_device, flow_batch_size
        )
        forward, backward, f_cost, b_cost = _estimate_flows(
            aligned_guide, guide_low, flow_backend, search_radius, patch_radius,
            flow_scale, raft_updates, flow_batch_size, keep_flow_model_loaded,
            work_device,
        )
        (
            f_conf, b_conf, f_geometry, b_geometry,
            _f_photo, _b_photo, cuts,
        ) = _build_pair_confidence(
            guide_low, forward, backward, f_cost, b_cost,
            float(photometric_threshold), float(flow_consistency),
            float(match_cost_threshold), float(scene_threshold),
        )
        background_key = _background_key_from_info(snapper_info)
        (
            out, confidence, replaced,
            silhouette_changes, silhouette_replaced,
            edge_color_changes, edge_replaced,
            feature_color_changes, feature_replaced,
        ) = motion_compensated_label_consensus(
            image, guide_low, forward, backward,
            f_conf, b_conf, f_geometry, b_geometry,
            int(window), float(agreement_threshold), float(data_tolerance),
            float(max_color_distance), background_key,
            bool(silhouette_stabilization), float(silhouette_agreement),
            int(silhouette_min_support), bool(edge_color_stabilization),
            int(edge_radius), float(edge_agreement), int(edge_min_support),
            float(edge_max_color_distance), bool(edge_medoid_fallback),
            float(edge_cluster_radius), bool(feature_edge_stabilization),
            int(feature_radius), float(feature_contrast_threshold),
        )
        out = out.clamp(0.0, 1.0)
        changed = (_pack_keys(out) != _pack_keys(image)).float()
        preview = _confidence_heatmap(confidence)
        total = max(1, n * h * w)
        reliable = (f_conf.float().mean() + b_conf.float().mean()) * 0.5
        if not bool(silhouette_stabilization):
            silhouette_state = "off(user)"
        elif background_key is None:
            silhouette_state = "off(no background metadata)"
        else:
            silhouette_state = "on"
        if not bool(edge_color_stabilization):
            edge_state = "off(user)"
        elif background_key is None:
            edge_state = "off(no background metadata)"
        else:
            edge_state = "on"
        if not bool(feature_edge_stabilization):
            feature_state = "off(user)"
        elif background_key is None:
            feature_state = "off(no background metadata)"
        else:
            feature_state = "on"
        raft_memory_note = (
            f" flow_batch={int(flow_batch_size)} raft_pass=sequential"
            if flow_backend == "raft_small" else ""
        )
        info = (
            f"backend={flow_backend} device={work_device}{raft_memory_note} frames={n} "
            f"cells={w}x{h} window={window} "
            f"reliable_links={float(reliable) * 100:.1f}% "
            f"confidence_mean={float(confidence.mean()) * 100:.1f}% "
            f"replaced={replaced}/{total} ({replaced / total * 100:.2f}%) "
            f"silhouette={silhouette_state} silhouette_replaced={silhouette_replaced} "
            f"edge_colors={edge_state} edge_replaced={edge_replaced} "
            f"features={feature_state} feature_replaced={feature_replaced} "
            f"scene_cuts={int(cuts.sum().item())}; {alignment_info}"
        )
        return (
            out.to(output_device),
            confidence.to(output_device),
            info,
            preview.to(output_device),
            changed.to(output_device),
            silhouette_changes.to(output_device),
            edge_color_changes.to(output_device),
            feature_color_changes.to(output_device),
        )


class VideoPixelSnapperTemporalCleanup:
    """Preset-driven front end for the advanced motion cleanup engine."""

    CATEGORY = "Video Pixel Snapper"
    RETURN_TYPES = VideoPixelSnapperTemporalCleanupAdvanced.RETURN_TYPES
    RETURN_NAMES = VideoPixelSnapperTemporalCleanupAdvanced.RETURN_NAMES
    FUNCTION = "run"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE", {
                    "tooltip": "Video Pixel Snapper output with output_scale=1."
                }),
                "guide_image": ("IMAGE", {
                    "tooltip": "Matching original/pre-snap video frames."
                }),
                "cleanup_preset": ([
                    "balanced", "strong", "very_strong", "outline_lock",
                    "detail_lock"
                ], {
                    "default": "strong",
                    "tooltip": "outline_lock targets the outside contour. detail_lock also "
                               "stabilizes internal palette borders and thin drawn details."
                }),
                "flow_quality": (["fast", "balanced", "quality"], {
                    "default": "balanced",
                    "tooltip": "RAFT resolution/iterations and batch-size preset."
                }),
                "flow_backend": (["raft_small", "integer_block_matching"], {
                    "default": "raft_small"
                }),
                "compute_device": (["auto", "gpu", "cpu"], {
                    "default": "auto"
                }),
                "edge_radius": ("INT", {
                    "default": 2, "min": 1, "max": 6,
                    "tooltip": "How many foreground cells inward from background count as "
                               "outline/thin-detail region."
                }),
            },
            "optional": {
                "snapper_info": ("STRING", {
                    "forceInput": True,
                    "tooltip": "Connect Video Pixel Snapper's info output."
                }),
            },
        }

    def run(
        self, image, guide_image, cleanup_preset, flow_quality,
        flow_backend, compute_device, edge_radius, snapper_info="",
    ):
        presets = {
            "balanced": dict(
                window=5, agreement=0.60, data=0.015, max_color=0.35,
                sil_agree=0.75, sil_support=3,
                edge_agree=0.60, edge_support=3, edge_max=0.80,
                cluster=0.30, photo=0.12, cycle=1.25,
            ),
            "strong": dict(
                window=7, agreement=0.50, data=0.030, max_color=0.60,
                sil_agree=0.65, sil_support=2,
                edge_agree=0.50, edge_support=2, edge_max=1.00,
                cluster=0.45, photo=0.15, cycle=1.50,
            ),
            "very_strong": dict(
                window=7, agreement=0.40, data=0.055, max_color=0.90,
                sil_agree=0.55, sil_support=2,
                edge_agree=0.40, edge_support=2, edge_max=1.30,
                cluster=0.60, photo=0.20, cycle=2.00,
            ),
            "outline_lock": dict(
                window=7, agreement=0.60, data=0.015, max_color=0.35,
                sil_agree=0.60, sil_support=2,
                edge_agree=0.34, edge_support=2, edge_max=1.50,
                cluster=0.70, photo=0.12, cycle=1.50,
            ),
            "detail_lock": dict(
                window=7, agreement=0.60, data=0.015, max_color=0.35,
                sil_agree=0.60, sil_support=2,
                edge_agree=0.40, edge_support=2, edge_max=1.30,
                cluster=0.60, photo=0.14, cycle=1.50,
            ),
        }
        flow_presets = {
            "fast": dict(scale=0.35, updates=6, batch=6, search=3),
            "balanced": dict(scale=0.50, updates=8, batch=4, search=4),
            "quality": dict(scale=1.00, updates=12, batch=1, search=6),
        }
        p = presets[cleanup_preset]
        q = flow_presets[flow_quality]
        result = VideoPixelSnapperTemporalCleanupAdvanced().run(
            image, guide_image, flow_backend, compute_device,
            p["window"], p["agreement"], p["data"], p["max_color"],
            True, p["sil_agree"], p["sil_support"],
            True, int(edge_radius), p["edge_agree"], p["edge_support"],
            p["edge_max"], True, p["cluster"],
            p["photo"], p["cycle"], 0.30,
            q["search"], 2, 0.05, q["scale"], q["updates"], q["batch"],
            False, cleanup_preset == "detail_lock", 1, 0.08, snapper_info,
        )
        # Prefix preset diagnostics without changing output ordering.
        values = list(result)
        values[2] = (
            f"preset={cleanup_preset} flow_quality={flow_quality}; " + values[2]
        )
        return tuple(values)


NODE_CLASS_MAPPINGS = {
    "VideoPixelSnapperTemporalCleanup": VideoPixelSnapperTemporalCleanup,
    "VideoPixelSnapperTemporalCleanupAdvanced": VideoPixelSnapperTemporalCleanupAdvanced,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "VideoPixelSnapperTemporalCleanup": "Motion-Aware Cleanup (Video Pixel Snapper)",
    "VideoPixelSnapperTemporalCleanupAdvanced": "Motion-Aware Cleanup Advanced (Video Pixel Snapper)",
}
