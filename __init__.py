from .video_pixel_snapper import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

# Serves everything under ./web/ (video_pixel_snapper.js) as a ComfyUI
# frontend extension. Standard, long-standing convention for custom-node
# JS assets — this part is low-risk even if the widget itself doesn't
# render correctly on your frontend version.
WEB_DIRECTORY = "./web"

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
