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
from .palette_analyzer import (
    NODE_CLASS_MAPPINGS as _ANALYZER_CLASSES,
    NODE_DISPLAY_NAME_MAPPINGS as _ANALYZER_NAMES,
)
from .selective_outline import (
    NODE_CLASS_MAPPINGS as _SELOUT_CLASSES,
    NODE_DISPLAY_NAME_MAPPINGS as _SELOUT_NAMES,
)
from .rgba_loader import (
    NODE_CLASS_MAPPINGS as _RGBA_CLASSES,
    NODE_DISPLAY_NAME_MAPPINGS as _RGBA_NAMES,
)

NODE_CLASS_MAPPINGS = {
    **_VPS_CLASSES, **_RETIMER_CLASSES, **_DENOISE_CLASSES,
    **_SHEET_CLASSES, **_ANALYZER_CLASSES, **_SELOUT_CLASSES,
    **_RGBA_CLASSES,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    **_VPS_NAMES, **_RETIMER_NAMES, **_DENOISE_NAMES,
    **_SHEET_NAMES, **_ANALYZER_NAMES, **_SELOUT_NAMES,
    **_RGBA_NAMES,
}

# Serves everything under ./web/ (video_pixel_snapper.js, frame_retimer.js)
# as a ComfyUI frontend extension. Standard, long-standing convention for
# custom-node JS assets — this part is low-risk even if a widget itself
# doesn't render correctly on your frontend version.
WEB_DIRECTORY = "./web"
__version__ = "2.6.2"

print(
    f"[VideoPixelSnapper] v{__version__} loaded: "
    f"{len(NODE_CLASS_MAPPINGS)} nodes "
    f"(RGBA loader={'yes' if 'VideoPixelSnapperLoadRGBA' in NODE_CLASS_MAPPINGS else 'no'})"
)

__all__ = [
    "NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS",
    "WEB_DIRECTORY", "__version__",
]
