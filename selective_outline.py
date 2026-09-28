"""Palette-locked selective outlining (sel-out) for still pixel art.

The node replaces detected black line pixels with darker, material-related
colors that already exist in a supplied palette.  It never interpolates the
output and never invents a new RGB value.  Each IMAGE batch member is treated
as an independent still; there is deliberately no temporal behavior here.
"""

from collections import deque
import math
import re

import torch
import torch.nn.functional as F

try:  # Package import in ComfyUI; direct import in the test suite.
    from .video_pixel_snapper import _prepare_mask, _rgb_to_oklab, palette_from_image
except ImportError:  # pragma: no cover - exercised by direct test imports.
    from video_pixel_snapper import _prepare_mask, _rgb_to_oklab, palette_from_image


_DIRECTIONS = {
    "top_left": (-1.0, -1.0),
    "top": (0.0, -1.0),
    "top_right": (1.0, -1.0),
    "left": (-1.0, 0.0),
    "right": (1.0, 0.0),
    "bottom_left": (-1.0, 1.0),
    "bottom": (0.0, 1.0),
    "bottom_right": (1.0, 1.0),
}

# (shadow-side L reduction, light-side L reduction, target chroma retention)
# "subtle" keeps the replacement dark and close to a conventional outline;
# "strong" allows the lit edge to approach the local body midtone.
_STYLE = {
    "subtle": (0.30, 0.18, 0.72),
    "balanced": (0.27, 0.11, 0.84),
    "strong": (0.23, 0.06, 0.96),
}


def _unit_direction(name):
    x, y = _DIRECTIONS.get(name, _DIRECTIONS["top_left"])
    length = math.hypot(x, y) or 1.0
    return x / length, y / length


def _shift(values, dy, dx, fill=0.0):
    """Translate BxHxW or BxHxWxC without wraparound."""
    result = torch.full_like(values, fill)
    h, w = values.shape[1:3]
    src_y0, src_y1 = max(0, -dy), min(h, h - dy)
    src_x0, src_x1 = max(0, -dx), min(w, w - dx)
    dst_y0, dst_y1 = max(0, dy), min(h, h + dy)
    dst_x0, dst_x1 = max(0, dx), min(w, w + dx)
    if src_y1 > src_y0 and src_x1 > src_x0:
        result[:, dst_y0:dst_y1, dst_x0:dst_x1] = values[
            :, src_y0:src_y1, src_x0:src_x1
        ]
    return result


def _neighbor_count(mask, radius=1):
    kernel = 2 * int(radius) + 1
    return F.conv2d(
        mask.float().unsqueeze(1),
        torch.ones((1, 1, kernel, kernel), device=mask.device),
        padding=radius,
    )[:, 0]


def _solid_background_foreground(rgb):
    """Best-effort flood fill for an exact/near-flat border background.

    It is intentionally conservative and CPU based: this fallback is for one
    still RGB sprite on a flat backdrop.  A supplied mask or RGBA alpha is both
    more reliable and avoids this path.
    """
    batch, height, width, _ = rgb.shape
    result = torch.ones((batch, height, width), device=rgb.device, dtype=rgb.dtype)
    quantized = (rgb.detach().clamp(0.0, 1.0) * 255.0).round().to(torch.uint8).cpu()

    for b in range(batch):
        frame = quantized[b]
        border = torch.cat((
            frame[0], frame[-1], frame[1:-1, 0], frame[1:-1, -1]
        ), dim=0)
        keys = (
            border[:, 0].to(torch.int64) << 16
            | border[:, 1].to(torch.int64) << 8
            | border[:, 2].to(torch.int64)
        )
        unique, counts = torch.unique(keys, return_counts=True)
        key = int(unique[counts.argmax()].item())
        bg = torch.tensor(
            [(key >> 16) & 255, (key >> 8) & 255, key & 255],
            dtype=torch.int16,
        )
        # Eight encoded levels tolerate a tiny flat-background residue without
        # swallowing antialiased foreground edges of visibly different color.
        close = (frame.to(torch.int16) - bg).abs().amax(dim=-1) <= 8
        outside = torch.zeros((height, width), dtype=torch.bool)
        queue = deque()
        for x in range(width):
            if bool(close[0, x]):
                queue.append((0, x))
            if height > 1 and bool(close[height - 1, x]):
                queue.append((height - 1, x))
        for y in range(1, max(1, height - 1)):
            if bool(close[y, 0]):
                queue.append((y, 0))
            if width > 1 and bool(close[y, width - 1]):
                queue.append((y, width - 1))
        while queue:
            y, x = queue.popleft()
            if outside[y, x] or not bool(close[y, x]):
                continue
            outside[y, x] = True
            if y: queue.append((y - 1, x))
            if y + 1 < height: queue.append((y + 1, x))
            if x: queue.append((y, x - 1))
            if x + 1 < width: queue.append((y, x + 1))
        result[b] = (~outside).to(device=rgb.device, dtype=rgb.dtype)
    return result


def _surface_normals(foreground):
    """Outward normal estimated from neighboring non-foreground pixels."""
    outside = (~foreground).float().unsqueeze(1)
    kx = torch.tensor(
        [[-1.0, 0.0, 1.0], [-1.0, 0.0, 1.0], [-1.0, 0.0, 1.0]],
        device=foreground.device,
    ).view(1, 1, 3, 3)
    ky = torch.tensor(
        [[-1.0, -1.0, -1.0], [0.0, 0.0, 0.0], [1.0, 1.0, 1.0]],
        device=foreground.device,
    ).view(1, 1, 3, 3)
    nx = F.conv2d(outside, kx, padding=1)[:, 0]
    ny = F.conv2d(outside, ky, padding=1)[:, 0]
    length = torch.sqrt(nx.square() + ny.square()).clamp(min=1e-6)
    return nx / length, ny / length


def _radial_normals(foreground):
    """Fallback normal for internal lines, pointing away from sprite center."""
    batch, height, width = foreground.shape
    yy, xx = torch.meshgrid(
        torch.arange(height, device=foreground.device, dtype=torch.float32),
        torch.arange(width, device=foreground.device, dtype=torch.float32),
        indexing="ij",
    )
    nx = torch.zeros_like(foreground, dtype=torch.float32)
    ny = torch.zeros_like(nx)
    for b in range(batch):
        valid = foreground[b]
        if bool(valid.any()):
            cx = xx[valid].mean()
            cy = yy[valid].mean()
        else:
            cx = torch.tensor((width - 1) / 2, device=foreground.device)
            cy = torch.tensor((height - 1) / 2, device=foreground.device)
        dx, dy = xx - cx, yy - cy
        length = torch.sqrt(dx.square() + dy.square()).clamp(min=1e-6)
        nx[b], ny[b] = dx / length, dy / length
    return nx, ny


def _local_material(rgb, material, radius=2):
    """Spatially weighted local body color used only as a lookup reference."""
    total = torch.zeros_like(rgb)
    weight_sum = torch.zeros_like(material, dtype=rgb.dtype)
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            if dx == 0 and dy == 0:
                continue
            weight = 1.0 / math.sqrt(float(dx * dx + dy * dy))
            valid = _shift(material, dy, dx, fill=False)
            colors = _shift(rgb, dy, dx, fill=0.0)
            total += colors * valid.unsqueeze(-1) * weight
            weight_sum += valid * weight
    return total / weight_sum.clamp(min=1e-8).unsqueeze(-1), weight_sum > 0


def _infer_light(rgb, foreground, black):
    """Infer one 8-way light direction per still from bright-color centroid.

    This is deliberately reported as a heuristic.  It works well when the
    brightest material samples describe lighting, but a large white costume can
    dominate it; manual mode is the deterministic fallback.
    """
    batch, height, width, _ = rgb.shape
    yy, xx = torch.meshgrid(
        torch.arange(height, device=rgb.device, dtype=torch.float32),
        torch.arange(width, device=rgb.device, dtype=torch.float32),
        indexing="ij",
    )
    luma = 0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]
    vectors, names, confidences = [], [], []
    direction_names = list(_DIRECTIONS)
    direction_vectors = torch.tensor(
        [_unit_direction(name) for name in direction_names],
        device=rgb.device,
    )
    for b in range(batch):
        valid = foreground[b] & ~black[b]
        if int(valid.sum()) < 3:
            vectors.append(_unit_direction("top_left"))
            names.append("top_left(fallback)")
            confidences.append(0.0)
            continue
        values = luma[b][valid]
        threshold = torch.quantile(values, 0.75)
        bright_weight = (luma[b] - threshold).clamp(min=0.0) * valid
        if float(bright_weight.sum()) <= 1e-6:
            bright_weight = valid.float()
        fg_weight = valid.float()
        cx = (xx * fg_weight).sum() / fg_weight.sum().clamp(min=1.0)
        cy = (yy * fg_weight).sum() / fg_weight.sum().clamp(min=1.0)
        bx = (xx * bright_weight).sum() / bright_weight.sum().clamp(min=1e-6)
        by = (yy * bright_weight).sum() / bright_weight.sum().clamp(min=1e-6)
        vx, vy = bx - cx, by - cy
        raw_length = torch.sqrt(vx.square() + vy.square())
        confidence = float(raw_length / max(1.0, math.hypot(width, height) / 2.0))
        if float(raw_length) < 0.5:
            vector = torch.tensor(_unit_direction("top_left"), device=rgb.device)
            name = "top_left(fallback)"
        else:
            vector = torch.stack((vx, vy)) / raw_length
            best = int((direction_vectors @ vector).argmax().item())
            name = direction_names[best]
        vectors.append((float(vector[0]), float(vector[1])))
        names.append(name)
        confidences.append(confidence)
    return vectors, names, confidences


def _repeat_pixels(values, scale):
    if int(scale) <= 1:
        return values
    return values.repeat_interleave(int(scale), dim=1).repeat_interleave(int(scale), dim=2)


def _reduce_exact_nearest_image(image, scale):
    """Collapse an exact nearest-neighbor enlargement to logical pixels."""
    scale = int(scale)
    if scale <= 1:
        return image
    height, width = image.shape[1:3]
    if height % scale or width % scale:
        raise ValueError(
            f"input_pixel_scale={scale} requires image dimensions divisible "
            f"by {scale}, got {width}x{height}"
        )
    offset = scale // 2
    reduced = image[:, offset::scale, offset::scale]
    reconstructed = _repeat_pixels(reduced, scale)
    # Core nearest-neighbor output is bit-exact. A one-level allowance also
    # accepts a PNG round trip without treating antialiased scaling as cells.
    max_error = float((reconstructed.float() - image.float()).abs().max())
    if max_error > 1.01 / 255.0:
        raise ValueError(
            f"input_pixel_scale={scale} does not match an exact nearest-neighbor "
            f"enlargement (max channel error {max_error:.5f}). Run Sel-Out "
            f"before interpolation/upscaling or set the correct scale."
        )
    return reduced


def _scale_from_snapper_info(info, width, height):
    match = re.search(
        r"cells=(\d+)x(\d+).*?scale=(\d+)x\s*->\s*(\d+)x(\d+)",
        info or "",
    )
    if not match:
        return 1, "fallback_1(no compatible snapper_info)"
    cells_w, cells_h, scale, out_w, out_h = map(int, match.groups())
    if out_w != width or out_h != height:
        return 1, (
            f"fallback_1(info output {out_w}x{out_h} != input {width}x{height})"
        )
    if cells_w * scale != width or cells_h * scale != height:
        return 1, "fallback_1(info cell/scale mismatch)"
    return max(1, scale), "snapper_info"


def apply_selective_outline(
    image, palette_image, outline_scope="outer_only", lighting_mode="manual",
    light_direction="top_left", style="balanced", black_threshold=12,
    foreground_mask=None, mask_meaning="white_is_foreground", input_scale=1,
):
    """Functional implementation used by the Comfy node and tests."""
    if image.ndim != 4 or image.shape[-1] not in (3, 4):
        raise ValueError("image must have shape BxHxWx3 or BxHxWx4")
    if palette_image.ndim != 4 or palette_image.shape[-1] < 3:
        raise ValueError("palette must be an IMAGE containing RGB swatches")

    scale = max(1, int(input_scale))
    if scale > 1:
        full_batch, full_h, full_w = image.shape[:3]
        native_image = _reduce_exact_nearest_image(image, scale)
        native_mask = None
        if foreground_mask is not None:
            invert = mask_meaning == "white_is_transparent"
            full_foreground = _prepare_mask(
                foreground_mask, full_batch, full_h, full_w,
                image.device, invert=invert,
            )
            offset = scale // 2
            native_mask = full_foreground[:, offset::scale, offset::scale]
        native_rgb, native_rgba, native_changed, native_info = apply_selective_outline(
            native_image, palette_image, outline_scope, lighting_mode,
            light_direction, style, black_threshold, native_mask,
            "white_is_foreground", input_scale=1,
        )
        rgb = _repeat_pixels(native_rgb, scale)
        rgba = _repeat_pixels(native_rgba, scale)
        changed = _repeat_pixels(native_changed, scale)
        logical_h, logical_w = native_rgb.shape[1:3]
        info = (
            f"{native_info}; input_scale={scale}x logical={logical_w}x{logical_h} "
            f"output={full_w}x{full_h} scale_processing=logical_then_nearest"
        )
        return rgb, rgba, changed, info

    batch, height, width, channels = image.shape
    rgb = image[..., :3].float().clamp(0.0, 1.0)
    input_alpha = image[..., 3].float().clamp(0.0, 1.0) if channels == 4 else None

    if foreground_mask is not None:
        invert = mask_meaning == "white_is_transparent"
        alpha = _prepare_mask(
            foreground_mask, batch, height, width, image.device, invert=invert
        ).to(dtype=rgb.dtype)
        foreground = alpha >= 0.5
        mask_source = "mask(transparency)" if invert else "mask(foreground)"
        # Post-snap pixel art uses cell-level hard alpha. Do not reintroduce a
        # soft RMBG/compositing fringe through this still-image utility.
        output_alpha = foreground.to(dtype=rgb.dtype)
        alpha_note = "hard_from_mask"
    elif input_alpha is not None:
        alpha = input_alpha
        foreground = alpha >= 0.5
        mask_source = "rgba_alpha"
        output_alpha = input_alpha
        alpha_note = "preserved_rgba"
    else:
        inferred = _solid_background_foreground(rgb)
        foreground = inferred >= 0.5
        mask_source = "auto_flat_border"
        # An RGB node should not silently remove its solid background merely
        # because the outline detector inferred it.
        output_alpha = torch.ones((batch, height, width), device=image.device, dtype=rgb.dtype)
        alpha_note = "opaque_rgb"

    palette = palette_from_image(
        palette_image[..., :3].to(image.device), cap=256, seed=0
    ).to(device=image.device, dtype=rgb.dtype)
    if palette.shape[0] < 1:
        raise ValueError("palette contains no RGB colors")

    threshold = float(max(0, min(96, int(black_threshold)))) / 255.0
    black = rgb.amax(dim=-1) <= threshold
    colored_material = foreground & ~black
    material_nearby = _neighbor_count(colored_material, radius=2) > 0

    # A one-cell silhouette boundary is safest: it cannot eat a filled black
    # feature merely because that feature happens to lie near the sprite edge.
    eroded = _neighbor_count(foreground, radius=1) >= 9.0
    outer_boundary = foreground & ~eroded
    outer_candidates = black & outer_boundary & material_nearby
    if outline_scope == "outer_and_internal":
        candidates = black & foreground & material_nearby
    else:
        candidates = outer_candidates

    local_rgb, has_material = _local_material(rgb, colored_material, radius=2)
    candidates &= has_material

    if lighting_mode == "auto":
        light_vectors, light_names, auto_confidence = _infer_light(rgb, foreground, black)
    else:
        vector = _unit_direction(light_direction)
        light_vectors = [vector] * batch
        light_names = [light_direction] * batch
        auto_confidence = [1.0] * batch

    outer_nx, outer_ny = _surface_normals(foreground)
    radial_nx, radial_ny = _radial_normals(foreground)
    use_radial = ~outer_boundary
    nx = torch.where(use_radial, radial_nx, outer_nx)
    ny = torch.where(use_radial, radial_ny, outer_ny)

    shadow_delta, light_delta, chroma_scale = _STYLE.get(style, _STYLE["balanced"])
    output = rgb.clone()
    changed = torch.zeros((batch, height, width), device=image.device, dtype=torch.bool)
    unresolved = 0
    palette_lab = _rgb_to_oklab(palette)
    palette_nonblack = palette.amax(dim=-1) > threshold

    for b in range(batch):
        ys, xs = torch.nonzero(candidates[b], as_tuple=True)
        if ys.numel() == 0:
            continue
        lx, ly = light_vectors[b]
        facing = (nx[b, ys, xs] * lx + ny[b, ys, xs] * ly).clamp(-1.0, 1.0)
        exposure = (facing + 1.0) * 0.5
        delta = shadow_delta + exposure * (light_delta - shadow_delta)
        reference_lab = _rgb_to_oklab(local_rgb[b, ys, xs])
        target_lab = reference_lab.clone()
        target_lab[:, 0] = (reference_lab[:, 0] - delta).clamp(min=0.015)
        target_lab[:, 1:] = reference_lab[:, 1:] * chroma_scale

        # MxK is small for outline pixels and typical <=72-entry palettes.
        score = torch.cdist(target_lab, palette_lab)
        darker = palette_lab[:, 0].unsqueeze(0) <= (
            reference_lab[:, 0].unsqueeze(1) - 0.02
        )
        allowed = darker & palette_nonblack.unsqueeze(0)
        score = torch.where(allowed, score, torch.full_like(score, float("inf")))

        missing = ~allowed.any(dim=1)
        if bool(missing.any()):
            # A very dark material may have no still-darker nonblack slot. Use
            # the closest tinted dark if possible, but never synthesize one.
            fallback = torch.cdist(target_lab[missing], palette_lab)
            fallback[:, ~palette_nonblack] = float("inf")
            score[missing] = fallback
        index = score.argmin(dim=1)
        valid_pick = torch.isfinite(score[torch.arange(score.shape[0], device=score.device), index])
        if bool(valid_pick.any()):
            vy, vx = ys[valid_pick], xs[valid_pick]
            replacement = palette[index[valid_pick]]
            output[b, vy, vx] = replacement
            changed[b, vy, vx] = (replacement != rgb[b, vy, vx]).any(dim=-1)
        unresolved += int((~valid_pick).sum().item())

    rgba = torch.cat((output, output_alpha.unsqueeze(-1)), dim=-1)
    light_text = ",".join(
        f"{name}:{confidence:.2f}" if lighting_mode == "auto" else name
        for name, confidence in zip(light_names, auto_confidence)
    )
    candidate_count = int(candidates.sum().item())
    changed_count = int(changed.sum().item())
    info = (
        f"Selective Outline: stills={batch} palette={palette.shape[0]} "
        f"scope={outline_scope} lighting={lighting_mode}[{light_text}] "
        f"style={style} black_threshold={int(black_threshold)}/255 "
        f"foreground={mask_source} candidates={candidate_count} "
        f"changed={changed_count} unresolved={unresolved}; "
        f"replacement_colors=palette_only alpha={alpha_note}"
    )
    return output, rgba, changed.float(), info


class VideoPixelSnapperSelectiveOutline:
    """Replace black sprite linework with palette-locked selective outlines."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE", {
                    "tooltip": "A snapped/pixel-perfect still image. IMAGE batches are accepted, but every item is processed independently with no temporal logic."
                }),
                "palette": ("IMAGE", {
                    "tooltip": "Allowed palette strip. Every replacement is selected from these exact RGB entries; no new shade is generated."
                }),
                "outline_scope": (["outer_only", "outer_and_internal"], {
                    "default": "outer_only",
                    "tooltip": "outer_only changes only black silhouette pixels touching background/alpha. outer_and_internal also changes black linework inside the sprite."
                }),
                "lighting_mode": (["manual", "auto"], {
                    "default": "manual",
                    "tooltip": "manual uses the chosen light direction. auto estimates it independently for each still from the bright-color centroid and reports confidence."
                }),
                "light_direction": (list(_DIRECTIONS), {
                    "default": "top_left",
                    "tooltip": "Used by manual mode; ignored by auto. Direction points toward the light source."
                }),
                "style": (["subtle", "balanced", "strong"], {
                    "default": "balanced",
                    "tooltip": "subtle keeps a dark conventional outline; strong permits a much lighter lit-side sel-out; balanced is the recommended start."
                }),
                "black_threshold": ("INT", {
                    "default": 12, "min": 0, "max": 96, "step": 1,
                    "tooltip": "A line pixel is eligible only when all RGB channels are at or below this 8-bit value. Use 0 for exact #000000."
                }),
                "mask_meaning": (["white_is_foreground", "white_is_transparent"], {
                    "default": "white_is_foreground",
                    "tooltip": "How to interpret the optional mask. RMBG/BiRefNet normally uses white foreground; ComfyUI Load Image MASK normally uses white transparency."
                }),
                "input_pixel_scale": ("INT", {
                    "default": 0, "min": 0, "max": 32, "step": 1,
                    "tooltip": "0 reads the exact nearest-neighbor output scale from optional snapper_info (falls back to 1). Set 1 for native pixels or enter the known Video Pixel Snapper output_scale manually."
                }),
            },
            "optional": {
                "foreground_mask": ("MASK", {
                    "tooltip": "Recommended for RGB sprites. If absent, RGBA alpha is used; plain RGB falls back to a conservative flat-border flood fill."
                }),
                "snapper_info": ("STRING", {
                    "forceInput": True,
                    "tooltip": "Connect Video Pixel Snapper.info so input_pixel_scale=0 can process an upscaled result on its logical cell grid and restore the original size with exact nearest-neighbor blocks."
                }),
            },
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "MASK", "STRING")
    RETURN_NAMES = ("image", "transparent_image", "changed_outline", "info")
    FUNCTION = "run"
    CATEGORY = "Video Pixel Snapper"
    DESCRIPTION = (
        "Palette-locked selective outlining for still pixel art. Replaces black "
        "silhouette and optionally internal line pixels with darker related "
        "colors already present in the supplied palette."
    )

    def run(
        self, image, palette, outline_scope, lighting_mode, light_direction,
        style, black_threshold, mask_meaning, foreground_mask=None,
        snapper_info="", input_pixel_scale=0,
    ):
        requested_scale = int(input_pixel_scale)
        if requested_scale > 0:
            scale = requested_scale
            scale_source = "manual"
        else:
            scale, scale_source = _scale_from_snapper_info(
                snapper_info, image.shape[2], image.shape[1]
            )
        result = apply_selective_outline(
            image, palette, outline_scope, lighting_mode, light_direction,
            style, black_threshold, foreground_mask, mask_meaning,
            input_scale=scale,
        )
        info = f"{result[3]}; scale_source={scale_source}"
        return result[0], result[1], result[2], info


NODE_CLASS_MAPPINGS = {
    "VideoPixelSnapperSelectiveOutline": VideoPixelSnapperSelectiveOutline,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "VideoPixelSnapperSelectiveOutline": "Selective Outline / Sel-Out (Video Pixel Snapper)",
}
