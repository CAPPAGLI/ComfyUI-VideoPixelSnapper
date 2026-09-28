"""
Frame Retimer — edit the timing and ORDER of frames in an IMAGE batch:
drop unwanted frames, hold others longer, reorder them, or loop a
short section — plus a keyframed duration curve for broad pacing.

Generic — works on any image batch, not specific to pixel art — but
ships in this pack since that's where it was asked for.

Data model: `sequence_json` is a flat JSON array of SOURCE-frame
indices, one per OUTPUT frame, in output order. This is the single
source of truth — it's what makes reordering and loops possible at
all: a plain "repeat count per source frame" array (the previous
version's model) can only ever hold each source frame's occurrences
together at its original position, so it literally cannot represent
"play frames 5-9, then play them again" or "swap frame 3 and frame 7".
An explicit index sequence can.

The actual editing happens in the DOM widget (web/frame_retimer.js):
- the duration curve / per-frame duration control REGENERATE the whole
  sequence from scratch (repeat each source frame N times, in original
  order) — good for broad pacing/speed-ramping.
- the timeline further edits whatever sequence resulted from that:
  drag entries to reorder, select a range and duplicate it to create a
  loop, delete a range to cut.
Touching the curve/duration controls again after timeline edits
regenerates the sequence and discards those timeline edits — this is
intentional (do pacing first, then reordering), not a bug, and the
widget says so.

All of this runs client-side against already-loaded thumbnails, so you
get an instant preview of the new order/loops/count without re-running
the graph. Re-running is only needed to get the real reordered IMAGE
tensor for downstream nodes.
"""
import json
import torch

try:
    from .video_pixel_snapper import _save_preview_images, _drop_alpha
except Exception:
    _save_preview_images = None
    def _drop_alpha(image):
        return image[..., :3] if image.shape[-1] > 3 else image


class VideoPixelSnapperFrameRetimer:
    CATEGORY = "Video Pixel Snapper"
    RETURN_TYPES = ("IMAGE", "STRING", "IMAGE", "MASK")
    RETURN_NAMES = (
        "image", "info", "transparent_image", "transparency_mask"
    )
    FUNCTION = "run"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "sequence_json": ("STRING", {
                    "default": "[]",
                    "tooltip": "Written by the widget below — not meant to be typed by hand. JSON array of "
                                "source-frame indices, one per output frame, in output order: repeats hold a "
                                "frame longer, reordered or duplicated runs reorder/loop. Left as [] (or "
                                "containing an out-of-range index for the current input), every frame passes "
                                "through once in original order."
                }),
                "source_fps": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 1000.0, "step": 0.01,
                    "tooltip": "Optional: the original video's fps. Purely informational — lets the widget show "
                                "real seconds next to frame counts. Doesn't change the retiming math. Right-"
                                "click -> Convert to Input to feed this from wherever your pipeline already "
                                "knows the source fps; leave at 0 to just work in frame counts."
                }),
                "max_preview_frames": ("INT", {"default": 100, "min": 1, "max": 500,
                                                "tooltip": "How many input-frame thumbnails the widget loads "
                                                            "for editing. Each is written as a PNG file, so very "
                                                            "high values on a long batch cost real disk/time. "
                                                            "Right-click -> Convert to Input to drive this from "
                                                            "a frame-count output elsewhere in your graph."}),
            },
        }

    def run(self, image, sequence_json, source_fps, max_preview_frames):
        rgba_input = image[..., :4] if image.shape[-1] > 3 else None
        alpha = (
            rgba_input[..., 3] if rgba_input is not None else torch.ones(
                image.shape[:3], device=image.device, dtype=image.dtype
            )
        )
        image = _drop_alpha(image)
        transparent_input = (
            rgba_input if rgba_input is not None
            else torch.cat([image, alpha.unsqueeze(-1)], dim=-1)
        )
        n = image.shape[0]
        try:
            parsed = json.loads(sequence_json) if sequence_json else []
            if not isinstance(parsed, list):
                raise ValueError("sequence_json must be a JSON array")
            seq = [int(i) for i in parsed]
        except (TypeError, ValueError, json.JSONDecodeError):
            seq = []

        if not seq or any(i < 0 or i >= n for i in seq):
            seq = list(range(n))  # untouched / invalid -> pass every frame through once, in order

        idx_tensor = torch.tensor(seq, dtype=torch.long, device=image.device)
        out = image[idx_tensor]
        transparent_out = transparent_input[idx_tensor]
        transparency_mask = 1.0 - transparent_out[..., 3]
        info = f"{n} input frame(s) -> {out.shape[0]} output frame(s)"
        if source_fps and source_fps > 0:
            info += f" (~{out.shape[0] / source_fps:.2f}s @ {source_fps:g}fps)"

        if _save_preview_images is None:
            return (out, info, transparent_out, transparency_mask)

        input_refs = _save_preview_images(image, "VPS_retime_in", max_frames=max_preview_frames)
        # Thumbnails can be capped by max_preview_frames, but the true
        # frame count is sent separately so the widget can still track
        # and edit every frame (via the curve/timeline) even without a
        # thumbnail for it — see frame_retimer.js for how this is used.
        return {
            "ui": {"vps_retimer_frames": input_refs, "vps_total_frames": [n], "vps_source_fps": [source_fps]},
            "result": (out, info, transparent_out, transparency_mask),
        }


NODE_CLASS_MAPPINGS = {"VideoPixelSnapperFrameRetimer": VideoPixelSnapperFrameRetimer}
NODE_DISPLAY_NAME_MAPPINGS = {"VideoPixelSnapperFrameRetimer": "Frame Retimer (Video Pixel Snapper)"}
