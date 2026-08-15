"""
Frame Retimer — edit the duration (repeat count) of each frame in an
IMAGE batch: drop unwanted frames, hold others longer, or drive the
whole thing from a keyframed duration curve.

Generic — works on any image batch, not specific to pixel art — but
ships in this pack since that's where it was asked for.

The actual editing happens in the DOM widget (web/frame_retimer.js): it
loads thumbnails of the input batch and lets you build the frame-by-
frame duration list client-side (frame strip with per-frame duration,
or a keyframed curve applied across the whole sequence), with an
instant preview of the resulting order/count and a play/pause scrub —
all without re-running the graph. The `durations_json` widget is what
the JS writes to; re-running the node is only needed to get the real
retimed IMAGE tensor for downstream nodes.
"""
import json
import torch

try:
    from .video_pixel_snapper import _save_preview_images
except Exception:
    _save_preview_images = None


class VideoPixelSnapperFrameRetimer:
    CATEGORY = "image/transform"
    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("image", "info")
    FUNCTION = "run"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "durations_json": ("STRING", {
                    "default": "[]", "multiline": True,
                    "tooltip": "Written by the widget below — not meant to be typed by hand. JSON array, "
                                "one non-negative integer per input frame: 0 drops that frame, N repeats/holds "
                                "it N times in the output. Left as [] (or the wrong length for the current "
                                "input), every frame passes through once unchanged."
                }),
                "max_preview_frames": ("INT", {"default": 100, "min": 1, "max": 500,
                                                "tooltip": "How many input-frame thumbnails the widget loads "
                                                            "for editing. Each is written as a PNG file, so very "
                                                            "high values on a long batch cost real disk/time."}),
            },
        }

    def run(self, image, durations_json, max_preview_frames):
        n = image.shape[0]
        try:
            durations = json.loads(durations_json) if durations_json else []
            durations = [max(0, int(d)) for d in durations]
        except Exception:
            durations = []

        if len(durations) != n:
            durations = [1] * n  # untouched / mismatched -> pass every frame through once

        indices = []
        for i, d in enumerate(durations):
            indices.extend([i] * d)
        if not indices:
            indices = list(range(n))  # never return an empty batch

        idx_tensor = torch.tensor(indices, dtype=torch.long, device=image.device)
        out = image[idx_tensor]
        info = f"{n} input frame(s) -> {out.shape[0]} output frame(s)"

        if _save_preview_images is None:
            return (out, info)

        input_refs = _save_preview_images(image, "VPS_retime_in", max_frames=max_preview_frames)
        return {"ui": {"vps_retimer_frames": input_refs}, "result": (out, info)}


NODE_CLASS_MAPPINGS = {"VideoPixelSnapperFrameRetimer": VideoPixelSnapperFrameRetimer}
NODE_DISPLAY_NAME_MAPPINGS = {"VideoPixelSnapperFrameRetimer": "Frame Retimer (Video Pixel Snapper)"}
