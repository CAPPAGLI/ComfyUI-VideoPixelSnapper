"""Row-major RGBA sprite-sheet assembly without resampling or blending."""

import math

import torch


class VideoPixelSnapperSpriteSheet:
    CATEGORY = "Video Pixel Snapper"
    RETURN_TYPES = ("IMAGE", "STRING", "INT", "INT", "INT", "INT")
    RETURN_NAMES = (
        "sprite_sheet", "info", "frame_width", "frame_height", "columns", "rows"
    )
    FUNCTION = "run"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE", {
                    "tooltip": "RGB or RGBA frame batch. Connect Frame Retimer's "
                               "transparent_image to preserve hard alpha."
                }),
                "columns": ("INT", {
                    "default": 8, "min": 1, "max": 256,
                    "tooltip": "Frames per row. Placement is row-major."
                }),
                "padding": ("INT", {
                    "default": 0, "min": 0, "max": 64,
                    "tooltip": "Transparent pixels between frame cells. Frames are never resized."
                }),
            },
        }

    def run(self, image, columns, padding):
        if image.ndim != 4 or image.shape[-1] not in (3, 4):
            raise ValueError(
                "Sprite Sheet expects an IMAGE batch shaped (frames,H,W,3|4); "
                f"got {tuple(image.shape)}"
            )
        frame_count, frame_h, frame_w, channels = image.shape
        if frame_count < 1:
            raise ValueError("Sprite Sheet requires at least one frame")
        columns = max(1, min(int(columns), int(frame_count)))
        rows = int(math.ceil(frame_count / columns))
        padding = max(0, int(padding))
        sheet_h = rows * frame_h + max(0, rows - 1) * padding
        sheet_w = columns * frame_w + max(0, columns - 1) * padding
        sheet = torch.zeros(
            (1, sheet_h, sheet_w, channels),
            device=image.device,
            dtype=image.dtype,
        )
        for index in range(frame_count):
            row, column = divmod(index, columns)
            y = row * (frame_h + padding)
            x = column * (frame_w + padding)
            sheet[0, y:y + frame_h, x:x + frame_w] = image[index]

        alpha_note = "RGBA hard alpha preserved" if channels == 4 else "RGB (opaque export)"
        info = (
            f"frames={frame_count} layout={columns}x{rows} "
            f"frame={frame_w}x{frame_h} sheet={sheet_w}x{sheet_h} "
            f"padding={padding}; {alpha_note}"
        )
        return (sheet, info, frame_w, frame_h, columns, rows)


NODE_CLASS_MAPPINGS = {
    "VideoPixelSnapperSpriteSheet": VideoPixelSnapperSpriteSheet,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "VideoPixelSnapperSpriteSheet": "Sprite Sheet (Video Pixel Snapper)",
}
