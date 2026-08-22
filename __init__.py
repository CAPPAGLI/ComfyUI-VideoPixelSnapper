from .video_pixel_snapper import (
    NODE_CLASS_MAPPINGS as _VPS_CLASSES,
    NODE_DISPLAY_NAME_MAPPINGS as _VPS_NAMES,
)
from .frame_retimer import (
    NODE_CLASS_MAPPINGS as _RETIMER_CLASSES,
    NODE_DISPLAY_NAME_MAPPINGS as _RETIMER_NAMES,
)
from .temporal_denoise import (
    NODE_CLASS_MAPPINGS as _DENOISE_CLASSES,
    NODE_DISPLAY_NAME_MAPPINGS as _DENOISE_NAMES,
)
from .sprite_sheet import (
    NODE_CLASS_MAPPINGS as _SHEET_CLASSES,
    NODE_DISPLAY_NAME_MAPPINGS as _SHEET_NAMES,
)

NODE_CLASS_MAPPINGS = {
    **_VPS_CLASSES, **_RETIMER_CLASSES, **_DENOISE_CLASSES, **_SHEET_CLASSES
}
NODE_DISPLAY_NAME_MAPPINGS = {
    **_VPS_NAMES, **_RETIMER_NAMES, **_DENOISE_NAMES, **_SHEET_NAMES
}

# Serves everything under ./web/ (video_pixel_snapper.js, frame_retimer.js)
# as a ComfyUI frontend extension. Standard, long-standing convention for
# custom-node JS assets — this part is low-risk even if a widget itself
# doesn't render correctly on your frontend version.
WEB_DIRECTORY = "./web"

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
