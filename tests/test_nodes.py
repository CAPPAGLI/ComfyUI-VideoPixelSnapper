import base64
import io
import json
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import torch
from PIL import Image

from video_pixel_snapper import (
    VideoPixelSnapper,
    VideoPixelSnapperEditor,
    _estimate_phase,
    _prepare_mask,
    _preview_fingerprint,
    cell_stats,
    despeckle_indices,
    estimate_phase_for_frame,
    nearest_palette_index,
    process_batch,
)
from frame_retimer import VideoPixelSnapperFrameRetimer
from sprite_sheet import VideoPixelSnapperSpriteSheet
from palette_analyzer import VideoPixelSnapperPaletteCoverage
from rgba_loader import _list_input_images, _load_rgba_path
from selective_outline import (
    VideoPixelSnapperSelectiveOutline,
    apply_selective_outline,
)
from temporal_denoise import (
    VideoPixelSnapperTemporalCleanup,
    VideoPixelSnapperTemporalCleanupAdvanced,
    _align_guide_to_snapper,
    _raft_bidirectional_flow,
)


class FrontendSyntaxTests(unittest.TestCase):
    @unittest.skipUnless(shutil.which("node"), "Node.js is unavailable")
    def test_live_editor_is_parsed_as_an_es_module(self):
        # `node --check file.js` can silently parse under CommonJS/package
        # defaults and missed a real trailing-brace corruption in v2.5.0.
        # A .mjs suffix forces the same ES-module grammar browsers use.
        source = Path(__file__).resolve().parents[1] / "web" / "video_pixel_snapper.js"
        with tempfile.NamedTemporaryFile(suffix=".mjs", delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(source.read_bytes())
        try:
            completed = subprocess.run(
                [shutil.which("node"), "--check", str(temporary)],
                capture_output=True, text=True,
            )
        finally:
            temporary.unlink(missing_ok=True)
        self.assertEqual(completed.returncode, 0, completed.stderr)

    @unittest.skipUnless(shutil.which("node"), "Node.js is unavailable")
    def test_live_median_ignores_hidden_rgb_under_zero_alpha(self):
        source_path = Path(__file__).resolve().parents[1] / "web" / "video_pixel_snapper.js"
        source = source_path.read_text(encoding="utf-8")
        source = source.replace(
            'import { app } from "../../scripts/app.js";',
            'const app = globalThis.__testApp;',
        ).replace(
            'import { api } from "../../scripts/api.js";',
            'const api = globalThis.__testApi;',
        )
        source += "\nglobalThis.__blockMedianReduce = blockMedianReduce;\n"
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            module = folder / "editor.mjs"
            runner = folder / "runner.mjs"
            module.write_text(source, encoding="utf-8")
            runner.write_text(f'''\nclass ImageData {{\n  constructor(w, h) {{ this.width=w; this.height=h; this.data=new Uint8ClampedArray(w*h*4); }}\n}}\nglobalThis.ImageData=ImageData;\nglobalThis.__testApp={{registerExtension(){{}}}};\nglobalThis.__testApi={{}};\nawait import({json.dumps(module.as_uri())});\nconst bytes=new Uint8ClampedArray([255,0,255,0, 200,10,20,255]);\nconst canvas={{width:2,height:1,getContext(){{return {{getImageData(){{return {{data:bytes}};}}}};}}}};\nconst out=globalThis.__blockMedianReduce(canvas,1,1,null);\nif (out.data[0]!==200 || out.data[1]!==10 || out.data[2]!==20 || out.data[3]!==255) {{\n  throw new Error('hidden transparent RGB contaminated median: '+Array.from(out.data));\n}}\n''', encoding="utf-8")
            completed = subprocess.run(
                [shutil.which("node"), str(runner)], capture_output=True, text=True
            )
        self.assertEqual(completed.returncode, 0, completed.stderr)


class RGBALoaderTests(unittest.TestCase):
    def test_input_listing_is_recursive_and_does_not_require_get_input_files(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "nested").mkdir()
            (root / "A.PNG").write_bytes(b"")
            (root / "nested" / "b.webp").write_bytes(b"")
            (root / "nested" / "ignore.txt").write_text("x")
            listed = _list_input_images(root)
        self.assertEqual(listed, ["A.PNG", "nested/b.webp"])

    def test_loader_retains_alpha_and_sanitizes_only_fully_hidden_rgb(self):
        pixels = torch.tensor([
            [[255, 0, 255, 0], [255, 255, 255, 0]],
            [[200, 20, 10, 128], [10, 40, 90, 255]],
        ], dtype=torch.uint8).numpy()
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as handle:
            path = Path(handle.name)
        try:
            Image.fromarray(pixels, mode="RGBA").save(path)
            rgb, rgba, foreground, transparency, info = _load_rgba_path(
                path, sanitize_hidden_rgb=True
            )
        finally:
            path.unlink(missing_ok=True)
        self.assertTrue(torch.equal(rgba[0, 0, 0, :3], torch.zeros(3)))
        self.assertTrue(torch.equal(rgba[0, 0, 1, :3], torch.zeros(3)))
        self.assertTrue(torch.allclose(
            rgba[0, 1, 0], torch.tensor([200, 20, 10, 128]) / 255.0
        ))
        self.assertTrue(torch.equal(rgb, rgba[..., :3]))
        self.assertTrue(torch.equal(foreground, rgba[..., 3]))
        self.assertTrue(torch.equal(transparency, 1.0 - foreground))
        self.assertIn("hidden_alpha0=2", info)
        self.assertIn("partial_alpha=1", info)

    def test_loader_can_preserve_hidden_rgb_when_requested(self):
        pixels = torch.tensor([[[253, 0, 251, 0]]], dtype=torch.uint8).numpy()
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as handle:
            path = Path(handle.name)
        try:
            Image.fromarray(pixels, mode="RGBA").save(path)
            _rgb, rgba, _fg, _tr, info = _load_rgba_path(
                path, sanitize_hidden_rgb=False
            )
        finally:
            path.unlink(missing_ok=True)
        self.assertTrue(torch.allclose(
            rgba[0, 0, 0, :3], torch.tensor([253, 0, 251]) / 255.0
        ))
        self.assertIn("hidden_rgb=preserved", info)


class GridPhaseTests(unittest.TestCase):
    def test_edge_index_is_converted_to_cell_start(self):
        block = 8
        for phase in (0, 1, 3, 7):
            signal = torch.zeros(block * 7 - 1)
            # A cell beginning at `phase` has its preceding boundary at
            # phase-1 (mod block). edge_signal[i] is between i and i+1.
            boundary_mod = (phase - 1) % block
            signal[boundary_mod::block] = 1.0
            self.assertEqual(_estimate_phase(signal, block), phase)

    def test_block_one_saliency_is_finite(self):
        image = torch.rand(8, 9, 3)
        colors, saliency = cell_stats(image, block=1)
        self.assertEqual(colors.shape, (8, 9, 3))
        self.assertTrue(torch.isfinite(saliency).all())
        self.assertTrue(torch.equal(saliency, torch.zeros_like(saliency)))

    def test_phase_on_a_real_checker_grid(self):
        block, phase_x, phase_y = 6, 3, 2
        gh, gw = 7, 9
        yy, xx = torch.meshgrid(torch.arange(gh), torch.arange(gw), indexing="ij")
        cells = ((xx + yy) % 2).float().unsqueeze(-1).expand(-1, -1, 3)
        core = cells.repeat_interleave(block, 0).repeat_interleave(block, 1)
        image = torch.zeros(phase_y + gh * block, phase_x + gw * block, 3)
        image[phase_y:, phase_x:] = core
        self.assertEqual(estimate_phase_for_frame(image, block), (phase_x, phase_y))


class CoreNodeTests(unittest.TestCase):
    def test_oklab_preserves_orange_hue_where_rgb_prefers_salmon(self):
        sample = torch.tensor([[228, 149, 75]], dtype=torch.float32) / 255.0
        palette = torch.tensor([
            [239, 125, 87],   # salmon: nearest in raw RGB
            [234, 138, 46],   # orange: perceptually nearest
            [228, 166, 114],
        ], dtype=torch.float32) / 255.0
        rgb_index = nearest_palette_index(
            sample, palette, color_distance="rgb_legacy"
        )
        oklab_index = nearest_palette_index(
            sample, palette, color_distance="oklab"
        )
        self.assertEqual(int(rgb_index[0]), 0)
        self.assertEqual(int(oklab_index[0]), 1)

        image = sample.view(1, 1, 1, 3).expand(2, 2, 2, 3).clone()
        custom = palette.view(1, 1, 3, 3)
        common = dict(
            pixel_size=1.0, grid_detection_mode="first_frame",
            cell_method="majority", k_colors=3, accent_slots=0,
            sample_frames=1, dither="none", despeckle=False,
            output_scale_mode="manual", output_scale=1, seed=1,
            custom_palette=custom,
        )
        rgb_out = VideoPixelSnapper().run(
            image, color_distance="rgb_legacy", **common
        )[0]
        oklab_out = VideoPixelSnapper().run(
            image, color_distance="oklab", **common
        )[0]
        self.assertTrue(torch.allclose(rgb_out[0, 0, 0], palette[0]))
        self.assertTrue(torch.allclose(oklab_out[0, 0, 0], palette[1]))

    def test_rgba_input_and_custom_palette(self):
        image = torch.rand(3, 24, 32, 4)
        image[..., 3] = 1.0
        palette = torch.tensor(
            [[[[0.0, 0.0, 0.0, 1.0], [1.0, 1.0, 1.0, 1.0], [1.0, 0.0, 0.0, 1.0]]]]
        )
        out, preview, info, transparent, transparency = VideoPixelSnapper().run(
            image,
            pixel_size=4.0,
            grid_detection_mode="average_across_frames",
            cell_method="majority",
            k_colors=8,
            accent_slots=1,
            sample_frames=3,
            dither="none",
            despeckle=False,
            output_scale_mode="manual",
            output_scale=1,
            seed=42,
            custom_palette=palette,
        )
        self.assertEqual(out.shape[-1], 3)
        self.assertEqual(preview.shape[-1], 3)
        preview_centers = preview[0, 16, 16::32]
        expected_order = palette[0, 0, :, :3]
        self.assertTrue(torch.equal(preview_centers, expected_order))
        self.assertEqual(transparent.shape[-1], 4)
        self.assertEqual(float(transparency.sum()), 0.0)
        self.assertTrue(torch.equal(transparent[..., :3], out))
        self.assertIn("palette=custom (3 colors)", info)
        self.assertIn("color_distance=oklab", info)
        self.assertIn("source=32x24", info)

    def test_live_editor_exports_optional_postprocess_mask_for_widget(self):
        raw = torch.rand(1, 8, 10, 3)
        snapped = torch.rand(1, 4, 5, 4)
        snapped[..., 3] = 1.0
        palette = torch.rand(1, 1, 4, 3)
        postprocess = torch.zeros(1, 4, 5)
        postprocess[:, 0, 1] = 0.49
        postprocess[:, 1, 2] = 0.51
        calls = []

        def fake_save(batch, prefix, max_frames):
            calls.append((prefix, batch.clone(), max_frames))
            return [{"filename": prefix + ".png", "subfolder": "", "type": "temp"}]

        with patch("video_pixel_snapper._save_preview_images", side_effect=fake_save):
            result = VideoPixelSnapperEditor().run(
                raw, snapped, palette, 1,
                info=("grid=2px phase=(0,0) cells=5x4 palette=custom "
                      "scale=1x -> 5x4 color_distance=oklab"),
                postprocess_mask=postprocess,
            )
        self.assertEqual(
            result["ui"]["vps_postprocess_active"], [True]
        )
        self.assertEqual(
            result["ui"]["vps_postprocess_masks"][0]["filename"],
            "VPS_postprocess.png",
        )
        saved_mask = next(batch for prefix, batch, _ in calls if prefix == "VPS_postprocess")
        self.assertEqual(saved_mask.shape, (1, 4, 5, 3))
        self.assertEqual(float(saved_mask[0, 0, 1, 0]), 0.0)
        self.assertEqual(float(saved_mask[0, 1, 2, 0]), 1.0)
        self.assertTrue(torch.equal(result["result"][1], snapped))

    def test_live_editor_preview_preserves_snapped_alpha_and_original_mask(self):
        raw = torch.rand(1, 4, 5, 3)
        original_transparency = torch.zeros(1, 4, 5)
        original_transparency[:, 0, :] = 1.0
        snapped = torch.rand(1, 4, 5, 4)
        snapped[..., 3] = 1.0
        snapped[:, :, 0, 3] = 0.0
        palette = torch.rand(1, 1, 3, 3)
        saved = {}

        def fake_save(batch, prefix, max_frames):
            saved[prefix] = batch.clone()
            return [{"filename": prefix + ".png", "subfolder": "", "type": "temp"}]

        with patch("video_pixel_snapper._save_preview_images", side_effect=fake_save):
            result = VideoPixelSnapperEditor().run(
                raw, snapped, palette, 1,
                original_transparency_mask=original_transparency,
            )
        self.assertEqual(saved["VPS_raw"].shape[-1], 4)
        self.assertTrue(torch.equal(
            saved["VPS_raw"][..., 3], 1.0 - original_transparency
        ))
        self.assertEqual(saved["VPS_frame"].shape[-1], 4)
        self.assertTrue(torch.equal(saved["VPS_frame"], snapped))
        self.assertTrue(torch.equal(result["result"][1], snapped))
        self.assertTrue(torch.equal(
            result["result"][2], 1.0 - snapped[..., 3]
        ))

    def test_live_editor_rgb_preview_uses_background_metadata_for_alpha(self):
        blue = torch.tensor([0.0, 0.0, 1.0])
        red = torch.tensor([1.0, 0.0, 0.0])
        raw = torch.rand(1, 4, 5, 3)
        snapped = blue.view(1, 1, 1, 3).expand(1, 4, 5, 3).clone()
        snapped[:, 1:3, 1:4] = red
        palette = torch.stack([blue, red]).view(1, 1, 2, 3)
        saved = {}

        def fake_save(batch, prefix, max_frames):
            saved[prefix] = batch.clone()
            return [{"filename": prefix + ".png", "subfolder": "", "type": "temp"}]

        with patch("video_pixel_snapper._save_preview_images", side_effect=fake_save):
            result = VideoPixelSnapperEditor().run(
                raw, snapped, palette, 1,
                info=("grid=1px phase=(0,0) cells=5x4 palette=custom "
                      "scale=1x -> 5x4 mask=on bg=#0000ff "
                      "background_mode=solid"),
            )
        preview = saved["VPS_frame"]
        self.assertEqual(preview.shape[-1], 4)
        self.assertEqual(float(preview[0, 0, 0, 3]), 0.0)
        self.assertEqual(float(preview[0, 1, 1, 3]), 1.0)
        self.assertTrue(torch.equal(preview, result["result"][1]))

    def test_live_editor_committed_png_materializes_exact_output(self):
        raw = torch.rand(1, 4, 5, 3)
        snapped = torch.zeros(1, 4, 5, 4)
        snapped[..., 3] = 1.0
        palette = torch.rand(1, 1, 3, 3)
        committed_u8 = torch.zeros(4, 5, 4, dtype=torch.uint8)
        committed_u8[..., 0] = 211
        committed_u8[..., 1] = 97
        committed_u8[..., 2] = 43
        committed_u8[..., 3] = 255
        committed_u8[0, :, :3] = 0
        committed_u8[0, :, 3] = 0
        buffer = io.BytesIO()
        Image.fromarray(committed_u8.numpy(), mode="RGBA").save(buffer, format="PNG")
        payload = json.dumps({
            "version": 1,
            "batch": 1,
            "frame": 0,
            "width": 5,
            "height": 4,
            "raw_hash": _preview_fingerprint(raw),
            "png": "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii"),
        })
        result = VideoPixelSnapperEditor().run(
            raw, snapped, palette, 1, committed_live_png=payload
        )
        expected = committed_u8.float().unsqueeze(0) / 255.0
        self.assertTrue(torch.equal(result["result"][0], expected[..., :3]))
        self.assertTrue(torch.equal(result["result"][1], expected))
        self.assertTrue(torch.equal(result["result"][2], 1.0 - expected[..., 3]))
        self.assertIn("exact committed browser Live PNG", result["result"][3])
        self.assertEqual(result["ui"]["vps_commit_active"], [True])

    def test_live_editor_accepts_browser_fingerprint_roundtrip_mismatch(self):
        raw = torch.rand(1, 2, 3, 3)
        snapped = torch.rand(1, 2, 3, 3)
        palette = torch.rand(1, 1, 2, 3)
        rgba = torch.zeros(2, 3, 4, dtype=torch.uint8)
        rgba[..., 0] = 77
        rgba[..., 3] = 255
        buffer = io.BytesIO()
        Image.fromarray(rgba.numpy(), mode="RGBA").save(buffer, format="PNG")
        payload = json.dumps({
            "version": 1, "batch": 1, "frame": 0,
            "width": 3, "height": 2, "raw_width": 3, "raw_height": 2,
            "raw_hash": "deadbeef",
            "png": "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii"),
        })
        result = VideoPixelSnapperEditor().run(
            raw, snapped, palette, 1, committed_live_png=payload
        )
        expected = rgba.float().unsqueeze(0) / 255.0
        self.assertTrue(torch.equal(result["result"][1], expected))
        self.assertIn("mismatch_accepted", result["result"][3])
        self.assertEqual(result["ui"]["vps_commit_active"], [True])

    def test_live_editor_bad_commit_fails_soft_to_snapped(self):
        raw = torch.rand(1, 2, 3, 3)
        snapped = torch.rand(1, 2, 3, 3)
        palette = torch.rand(1, 1, 2, 3)
        result = VideoPixelSnapperEditor().run(
            raw, snapped, palette, 1, committed_live_png="not-json"
        )
        self.assertTrue(torch.equal(result["result"][0], snapped))
        self.assertTrue(torch.equal(result["result"][1][..., :3], snapped))
        self.assertIn("commit rejected, pass-through Snapped", result["result"][3])
        self.assertEqual(result["ui"]["vps_commit_active"], [False])

    def test_editor_emits_rich_grid_metadata(self):
        raw = torch.rand(2, 32, 40, 4)
        snapped = torch.rand(2, 15, 20, 3)
        palette = torch.rand(1, 32, 96, 3)
        info = ("grid=4px phase=(1,2) cells=8x6 palette=auto masked "
                "(3 colors, 1 accent, 1 bg) color_distance=oklab "
                "scale=3x -> 24x18 mask=on bg=#0011aa "
                "mask_threshold=0.5 cell_threshold=0.25")
        result = VideoPixelSnapperEditor().run(raw, snapped, palette, 2, info)
        self.assertEqual(result["ui"]["vps_grid"], [4, 1, 2, 8, 6, 3])
        self.assertEqual(result["ui"]["vps_background"], ["#0011aa"])
        self.assertEqual(result["ui"]["vps_color_distance"], ["oklab"])
        self.assertEqual(result["result"][0].shape[-1], 3)
        self.assertEqual(result["result"][1].shape[-1], 4)
        self.assertEqual(result["result"][2].shape, snapped.shape[:3])

    def test_mask_broadcast_resize_and_invert(self):
        mask = torch.tensor([[0.0, 1.0], [1.0, 0.0]])
        prepared = _prepare_mask(mask, batch=3, height=4, width=4, device="cpu")
        self.assertEqual(prepared.shape, (3, 4, 4))
        inverted = _prepare_mask(mask, batch=1, height=2, width=2, device="cpu", invert=True)
        self.assertTrue(torch.equal(inverted[0], 1.0 - mask))

    def test_masked_palette_has_one_background_color(self):
        blue = torch.tensor([0.0, 0.0, 1.0])
        red = torch.tensor([1.0, 0.0, 0.0])
        green = torch.tensor([0.0, 1.0, 0.0])
        image = blue.view(1, 1, 1, 3).expand(3, 16, 16, 3).clone()
        mask = torch.zeros(3, 16, 16)
        mask[:, 4:12, 4:12] = 1.0

        # Several compositing/fringe shades outside the confident mask must
        # not become palette entries.
        image[:, 2:4, :] = torch.tensor([0.10, 0.0, 0.90])
        image[:, 12:14, :] = torch.tensor([0.20, 0.0, 0.80])
        for y in range(4, 12, 2):
            for x in range(4, 12, 2):
                image[:, y:y + 2, x:x + 2] = red if ((x + y) // 2) % 2 else green

        background_image = blue.view(1, 1, 1, 3).expand(1, 16, 16, 3).clone()
        out, preview, info, transparent, transparency = VideoPixelSnapper().run(
            image,
            pixel_size=2.0,
            grid_detection_mode="average_across_frames",
            cell_method="majority",
            k_colors=8,
            accent_slots=1,
            sample_frames=3,
            dither="none",
            despeckle=False,
            output_scale_mode="manual",
            output_scale=1,
            seed=42,
            mask_threshold=0.5,
            mask_cell_threshold=0.25,
            invert_mask=False,
            foreground_mask=mask,
            background_image=background_image,
        )
        swatches = preview[0, 16, 16::32]
        self.assertEqual(int(torch.isclose(swatches, blue).all(dim=1).sum()), 1)
        # The purple/blue compositing fringe must not survive as several
        # separate background-like palette entries.
        non_background = swatches[~torch.isclose(swatches, blue).all(dim=1)]
        self.assertTrue((non_background[:, 2] < 0.1).all())
        self.assertTrue(torch.equal(out[:, 0, 0], blue.expand(3, 3)))
        self.assertEqual(transparent.shape[-1], 4)
        self.assertTrue((transparent[:, 0, 0, 3] == 0).all())
        self.assertTrue((transparent[:, 3, 3, 3] == 1).all())
        self.assertTrue((transparency[:, 0, 0] == 1).all())
        self.assertTrue((transparency[:, 3, 3] == 0).all())
        self.assertIn("1 bg", info)
        self.assertIn("mask=on", info)

    def test_embedded_rgba_alpha_drives_transparent_output_without_mask(self):
        blue = torch.tensor([0.0, 0.0, 1.0])
        red = torch.tensor([1.0, 0.0, 0.0])
        rgba = torch.zeros(1, 12, 12, 4)
        rgba[..., :3] = blue
        rgba[:, 3:9, 3:9, :3] = red
        rgba[:, 3:9, 3:9, 3] = 1.0
        custom = torch.stack([red, torch.tensor([0.2, 0.0, 0.0])]).view(1, 1, 2, 3)
        out, _preview, info, transparent, transparency = VideoPixelSnapper().run(
            rgba, pixel_size=3.0,
            grid_detection_mode="average_across_frames",
            cell_method="majority", k_colors=4, accent_slots=0,
            sample_frames=1, dither="none", despeckle=False,
            output_scale_mode="manual", output_scale=1, seed=1,
            mask_threshold=0.5, mask_cell_threshold=0.25,
            invert_mask=False, background_mode="transparent",
            custom_palette=custom,
        )
        self.assertEqual(float(transparent[0, 0, 0, 3]), 0.0)
        self.assertEqual(float(transparent[0, 1, 1, 3]), 1.0)
        self.assertEqual(float(transparency[0, 0, 0]), 1.0)
        self.assertFalse(torch.equal(out[0, 0, 0], red))
        self.assertIn("mask_source=embedded_alpha", info)
        self.assertIn("background_mode=transparent", info)

    def test_transparent_mode_uses_unique_hidden_key_and_hard_alpha(self):
        blue = torch.tensor([0.0, 0.0, 1.0])
        red = torch.tensor([1.0, 0.0, 0.0])
        image = blue.view(1, 1, 1, 3).expand(2, 12, 12, 3).clone()
        image[:, 3:9, 3:9] = red
        mask = torch.zeros(2, 12, 12)
        mask[:, 3:9, 3:9] = 1.0
        custom = red.view(1, 1, 1, 3)
        out, _preview, info, transparent, transparency = VideoPixelSnapper().run(
            image, pixel_size=3.0, grid_detection_mode="average_across_frames",
            cell_method="majority", k_colors=4, accent_slots=0,
            sample_frames=2, dither="none", despeckle=False,
            output_scale_mode="manual", output_scale=1, seed=1,
            mask_threshold=0.5, mask_cell_threshold=0.25,
            invert_mask=False, background_mode="transparent",
            custom_palette=custom, foreground_mask=mask,
        )
        self.assertIn("background_mode=transparent", info)
        self.assertFalse(torch.equal(out[0, 0, 0], red))
        self.assertEqual(float(transparent[0, 0, 0, 3]), 0.0)
        self.assertEqual(float(transparent[0, 2, 2, 3]), 1.0)
        self.assertEqual(float(transparency[0, 0, 0]), 1.0)
        self.assertEqual(float(transparency[0, 2, 2]), 0.0)

    def test_transparent_mode_forces_scale_one_and_shares_rgba_storage(self):
        image = torch.rand(2, 12, 12, 3)
        mask = torch.ones(2, 12, 12)
        out, _preview, info, transparent, _transparency = VideoPixelSnapper().run(
            image, pixel_size=3.0,
            grid_detection_mode="average_across_frames",
            cell_method="majority", k_colors=4, accent_slots=0,
            sample_frames=2, dither="none", despeckle=False,
            output_scale_mode="manual", output_scale=6, seed=1,
            background_mode="transparent", foreground_mask=mask,
        )
        match = re.search(r"cells=(\d+)x(\d+)", info)
        self.assertIsNotNone(match)
        cells_w, cells_h = map(int, match.groups())
        self.assertEqual(out.shape, (2, cells_h, cells_w, 3))
        self.assertEqual(transparent.shape, (2, cells_h, cells_w, 4))
        self.assertEqual(
            out.untyped_storage().data_ptr(),
            transparent.untyped_storage().data_ptr(),
        )
        requested = f"{cells_w * 6}x{cells_h * 6}"
        self.assertIn(
            f"transparent_scale=forced_1x(requested={requested})", info
        )

    def test_transparent_mode_requires_mask(self):
        image = torch.rand(2, 8, 8, 3)
        with self.assertRaisesRegex(ValueError, "requires foreground_mask"):
            VideoPixelSnapper().run(
                image, pixel_size=2.0,
                grid_detection_mode="average_across_frames",
                cell_method="majority", k_colors=4, accent_slots=0,
                sample_frames=2, dither="none", despeckle=False,
                output_scale_mode="manual", output_scale=1, seed=1,
                background_mode="transparent",
            )

    def test_masked_majority_ignores_background_pixels_inside_foreground_cell(self):
        red = torch.tensor([1.0, 0.0, 0.0])
        blue = torch.tensor([0.0, 0.0, 1.0])
        image = blue.view(1, 1, 1, 3).expand(1, 2, 2, 3).clone()
        image[0, 0, 0] = red
        mask = torch.zeros(1, 2, 2)
        mask[0, 0, 0] = 1.0
        palette = torch.stack([red, blue])
        out = process_batch(
            image, 2, 0, 0, palette, "majority", "none", False, 1, 1,
            foreground_mask=mask, mask_threshold=0.5,
            mask_cell_threshold=0.2, background_index=1,
        )
        self.assertTrue(torch.equal(out[0, 0, 0], red))

    def test_despeckle_does_not_use_arbitrary_four_way_tie(self):
        # Center differs from all neighbors, but all four neighbors also
        # differ from one another. It must remain unchanged.
        grid = torch.tensor([[[0, 1, 0], [2, 9, 3], [0, 4, 0]]])
        out = despeckle_indices(grid, k_colors=10)
        self.assertEqual(int(out[0, 1, 1]), 9)

    def test_despeckle_uses_neighbor_consensus(self):
        grid = torch.tensor([[[0, 2, 0], [2, 9, 2], [0, 3, 0]]])
        out = despeckle_indices(grid, k_colors=10)
        self.assertEqual(int(out[0, 1, 1]), 2)

    def test_automatic_frame_chunking_matches_one_frame_calls(self):
        torch.manual_seed(7)
        images = torch.rand(17, 12, 14, 3)  # 17 forces the internal max-16 split
        palette = torch.tensor(
            [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [1.0, 0.0, 0.0], [0.0, 0.5, 1.0]]
        )
        batched = process_batch(
            images, 2, 0, 0, palette, "majority", "bayer2", True, 6, 7
        )
        separate = torch.cat([
            process_batch(frame[None], 2, 0, 0, palette, "majority", "bayer2", True, 6, 7)
            for frame in images
        ])
        self.assertTrue(torch.equal(batched, separate))


class TemporalCleanupTests(unittest.TestCase):
    def test_raft_directions_are_sequential_and_flows_shrink_per_chunk(self):
        calls = []

        class FakeRaft:
            def __call__(self, a, b, num_flow_updates):
                calls.append((a.shape[0], a.shape[-2:]))
                flow = torch.ones(
                    (a.shape[0], 2, a.shape[-2], a.shape[-1]),
                    device=a.device, dtype=a.dtype,
                )
                return [flow]

        guide = torch.rand(5, 32, 48, 3)
        fake_loader = (FakeRaft(), lambda a, b: (a, b))
        with patch("temporal_denoise._load_raft_small", return_value=fake_loader):
            forward, backward, f_cost, b_cost = _raft_bidirectional_flow(
                guide, 4, 6, flow_scale=1.0, updates=3, batch_size=2,
                keep_model_loaded=True, compute_device="cpu",
            )

        # Two chunks, each direction executed separately. The old path made
        # two calls with an effective batch of four and retained 128px flows.
        self.assertEqual(calls, [(2, (128, 128))] * 4)
        self.assertEqual(forward.shape, (4, 2, 4, 6))
        self.assertEqual(backward.shape, (4, 2, 4, 6))
        self.assertTrue(torch.allclose(forward[:, 0], torch.full((4, 4, 6), 6 / 128)))
        self.assertTrue(torch.allclose(forward[:, 1], torch.full((4, 4, 6), 4 / 128)))
        self.assertIsNone(f_cost)
        self.assertIsNone(b_cost)

    def test_guide_crop_scales_phase_from_core_source_resolution(self):
        guide = torch.rand(2, 24, 40, 3)  # 2x a 20x12 core source
        info = (
            "grid=3px phase=(2,1) cells=5x3 source=20x12 "
            "palette=auto (8 colors, 1 accent) scale=1x -> 5x3"
        )
        cropped, note = _align_guide_to_snapper(guide, info, 3, 5)
        # Core crop is x=2..17, y=1..10; at 2x guide resolution:
        # x=4..34 (30 px), y=2..20 (18 px).
        self.assertEqual(cropped.shape, (2, 18, 30, 3))
        self.assertIn("grid_crop=(4,2)-(34,20)", note)
        self.assertIn("scale1", note)

    @staticmethod
    def run_block(frames, guide=None, window=5, agreement=0.6,
                  data_tolerance=0.05, max_color_distance=1.732,
                  search_radius=2, patch_radius=1, snapper_info="",
                  silhouette=True, silhouette_agreement=0.75,
                  silhouette_min_support=3, edge_colors=True,
                  edge_radius=2, edge_agreement=0.6,
                  edge_min_support=3, edge_max_color_distance=0.8,
                  edge_medoid=True, edge_cluster_radius=0.35,
                  feature_edges=False, feature_radius=1,
                  feature_contrast_threshold=0.08,
                  feature_hysteresis=False, feature_hold_frames=2,
                  feature_hold_radius=0.45):
        if guide is None:
            guide = frames
        return VideoPixelSnapperTemporalCleanupAdvanced().run(
            frames, guide, "integer_block_matching", "cpu", window, agreement,
            data_tolerance, max_color_distance,
            silhouette, silhouette_agreement, silhouette_min_support,
            edge_colors, edge_radius, edge_agreement,
            edge_min_support, edge_max_color_distance,
            edge_medoid, edge_cluster_radius,
            0.20, 1.25, 0.90,
            search_radius, patch_radius, 0.15,
            0.5, 8, 4, False,
            feature_edges, feature_radius, feature_contrast_threshold,
            feature_hysteresis, feature_hold_frames, feature_hold_radius,
            snapper_info,
        )

    def test_isolated_static_flicker_is_corrected(self):
        frames = torch.zeros(5, 3, 3, 3)
        frames[2, 1, 1] = torch.tensor([1.0, 0.0, 0.0])
        guide = torch.zeros_like(frames)
        (
            out, confidence, info, preview, changed,
            silhouette_changed, edge_changed, feature_changed, hysteresis_actions,
            transparent, transparency,
        ) = self.run_block(
            frames, guide, search_radius=0, patch_radius=0
        )
        self.assertTrue(torch.equal(out, torch.zeros_like(out)))
        self.assertGreater(float(confidence[2, 1, 1]), 0.0)
        self.assertEqual(preview.shape, frames.shape)
        self.assertEqual(changed.shape, frames.shape[:3])
        self.assertEqual(silhouette_changed.shape, frames.shape[:3])
        self.assertEqual(feature_changed.shape, frames.shape[:3])
        self.assertEqual(hysteresis_actions.shape, frames.shape[:3])
        self.assertEqual(transparent.shape[-1], 4)
        self.assertEqual(float(transparency.sum()), 0.0)
        self.assertEqual(float(changed[2, 1, 1]), 1.0)
        self.assertIn("backend=integer_block_matching", info)
        self.assertIn("device=cpu", info)

    def test_cleanup_preserves_input_hard_alpha_even_when_rgb_key_collides(self):
        rgba = torch.zeros(3, 5, 5, 4)
        rgba[:, 1:4, 1:4, 3] = 1.0
        result = self.run_block(
            rgba, rgba.clone(), search_radius=0, patch_radius=0,
            snapper_info="mask=on bg=#000000",
        )
        self.assertEqual(result[0].shape[-1], 3)
        self.assertTrue(torch.equal(result[9][..., 3], rgba[..., 3]))
        self.assertTrue(torch.equal(result[10], 1.0 - rgba[..., 3]))
        self.assertIn("alpha=preserved+topology", result[2])

    def test_one_frame_silhouette_tooth_is_removed_without_color_gate(self):
        black = torch.tensor([0.0, 0.0, 0.0])
        green = torch.tensor([0.0, 1.0, 0.0])
        frames = black.view(1, 1, 1, 3).expand(5, 6, 7, 3).clone()
        frames[:, 2:4, 2:5] = green
        frames[2, 1, 3] = green  # transient one-cell spine tooth
        guide = frames.clone()   # photometrically disagrees at that tooth
        result = self.run_block(
            frames, guide, search_radius=0, patch_radius=0,
            snapper_info="mask=on bg=#000000",
            max_color_distance=0.10,  # ordinary color gate cannot cross fg/bg
        )
        out, _confidence, info, _preview, changed, silhouette, _edge, _feature, _hold, rgba, transparency = result
        self.assertTrue(torch.equal(out[2, 1, 3], black))
        self.assertEqual(float(rgba[2, 1, 3, 3]), 0.0)
        self.assertEqual(float(transparency[2, 1, 3]), 1.0)
        self.assertEqual(float(rgba[2, 2, 3, 3]), 1.0)
        self.assertEqual(float(silhouette[2, 1, 3]), 1.0)
        self.assertEqual(float(changed[2, 1, 3]), 1.0)
        self.assertIn("silhouette=on", info)
        self.assertIn("silhouette_replaced=1", info)
        # Persistent body cells remain untouched.
        self.assertTrue(torch.equal(out[:, 2:4, 2:5], frames[:, 2:4, 2:5]))

    def test_edge_palette_flicker_uses_geometry_confidence(self):
        black = torch.tensor([0.0, 0.0, 0.0])
        green = torch.tensor([0.0, 1.0, 0.0])
        pink = torch.tensor([1.0, 0.0, 1.0])
        frames = black.view(1, 1, 1, 3).expand(5, 7, 8, 3).clone()
        frames[:, 2:5, 2:6] = green
        frames[2, 2, 3] = pink  # internal color flicker near silhouette
        guide = frames.clone()   # strict photometric confidence rejects it
        result = self.run_block(
            frames, guide, search_radius=0, patch_radius=0,
            snapper_info="mask=on bg=#000000",
            max_color_distance=0.10,
            edge_colors=True, edge_radius=2, edge_agreement=0.6,
            edge_min_support=3, edge_max_color_distance=1.75,
        )
        out, _confidence, info, _preview, changed, _silhouette, edge, _feature, _hold, _rgba, _transparency = result
        self.assertTrue(torch.equal(out[2, 2, 3], green))
        self.assertEqual(float(edge[2, 2, 3]), 1.0)
        self.assertEqual(float(changed[2, 2, 3]), 1.0)
        self.assertIn("edge_colors=on", info)
        self.assertIn("edge_replaced=1", info)

    def test_edge_medoid_handles_all_unique_palette_shades(self):
        black = torch.tensor([0.0, 0.0, 0.0])
        base = torch.tensor([0.5, 0.5, 0.5])
        shades = [0.20, 0.30, 0.60, 0.40, 0.50]
        frames = black.view(1, 1, 1, 3).expand(5, 7, 8, 3).clone()
        frames[:, 2:5, 2:6] = base
        for t, shade in enumerate(shades):
            frames[t, 2, 3] = shade
        guide = frames.clone()
        result = self.run_block(
            frames, guide, search_radius=0, patch_radius=0,
            snapper_info="mask=on bg=#000000",
            edge_colors=True, edge_radius=2, edge_agreement=0.9,
            edge_min_support=3, edge_max_color_distance=1.0,
            edge_medoid=True, edge_cluster_radius=0.5,
        )
        out, _confidence, _info, _preview, _changed, _silhouette, edge, _feature, _hold, _rgba, _transparency = result
        expected = torch.tensor([0.4, 0.4, 0.4])
        self.assertTrue(torch.allclose(out[2, 2, 3], expected))
        self.assertEqual(float(edge[2, 2, 3]), 1.0)

    def test_internal_palette_boundary_uses_observed_medoid(self):
        black = torch.tensor([0.0, 0.0, 0.0])
        green = torch.tensor([0.05, 0.65, 0.20])
        beige = torch.tensor([0.80, 0.65, 0.35])
        shades = [
            torch.tensor([0.05, 0.45, 0.15]),
            torch.tensor([0.05, 0.55, 0.17]),
            torch.tensor([0.05, 0.85, 0.24]),
            torch.tensor([0.05, 0.65, 0.20]),
            torch.tensor([0.05, 0.75, 0.22]),
        ]
        frames = black.view(1, 1, 1, 3).expand(5, 13, 15, 3).clone()
        frames[:, 1:12, 1:14] = green
        frames[:, 2:11, 7:14] = beige
        for t, shade in enumerate(shades):
            frames[t, 6, 6] = shade  # all-unique ripple at green/beige border
        result = self.run_block(
            frames, frames.clone(), search_radius=0, patch_radius=0,
            snapper_info="mask=on bg=#000000",
            max_color_distance=0.10,
            edge_colors=True, edge_radius=2, edge_agreement=0.9,
            edge_min_support=3, edge_max_color_distance=1.0,
            edge_medoid=True, edge_cluster_radius=0.5,
            feature_edges=True, feature_radius=1,
            feature_contrast_threshold=0.08,
        )
        out, _confidence, info, _preview, changed, _silhouette, edge, feature, _hold, _rgba, _transparency = result
        # The temporal medoid is copied from frame 3; no RGB mean is created.
        self.assertTrue(torch.equal(out[2, 6, 6], shades[3]))
        self.assertEqual(float(changed[2, 6, 6]), 1.0)
        self.assertEqual(float(feature[2, 6, 6]), 1.0)
        self.assertEqual(float(edge[2, 6, 6]), 0.0)
        self.assertIn("features=on", info)
        self.assertIn("feature_replaced=", info)

    def test_feature_hysteresis_reduces_medoid_chatter_and_expires(self):
        black = torch.tensor([0.0, 0.0, 0.0])
        green = torch.tensor([0.05, 0.65, 0.20])
        beige = torch.tensor([0.80, 0.65, 0.35])
        values = [0.35, 0.55, 0.80, 0.45, 0.70, 0.40, 0.75, 0.50, 0.65]
        frames = black.view(1, 1, 1, 3).expand(9, 13, 15, 3).clone()
        frames[:, 1:12, 1:14] = green
        frames[:, 2:11, 7:14] = beige
        for t, value in enumerate(values):
            frames[t, 6, 6] = torch.tensor([0.05, value, 0.18])
        kwargs = dict(
            search_radius=0, patch_radius=0,
            snapper_info="mask=on bg=#000000",
            max_color_distance=0.10,
            edge_colors=True, edge_radius=2, edge_agreement=0.9,
            edge_min_support=2, edge_max_color_distance=1.5,
            edge_medoid=True, edge_cluster_radius=0.8,
            feature_edges=True, feature_radius=2,
            feature_contrast_threshold=0.05,
        )
        base = self.run_block(frames, frames.clone(), **kwargs)
        held = self.run_block(
            frames, frames.clone(), feature_hysteresis=True,
            feature_hold_frames=3, feature_hold_radius=0.8, **kwargs,
        )
        base_keys = (base[0][:, 6, 6] * 255).round().to(torch.int64)
        held_keys = (held[0][:, 6, 6] * 255).round().to(torch.int64)
        base_transitions = (base_keys[1:] != base_keys[:-1]).any(dim=-1).sum()
        held_transitions = (held_keys[1:] != held_keys[:-1]).any(dim=-1).sum()
        self.assertEqual(int(base_transitions), 5)
        self.assertEqual(int(held_transitions), 1)
        self.assertEqual(int(held[8][:, 6, 6].sum()), 4)
        # The three-frame cap eventually releases the old label rather than
        # freezing the detail forever.
        self.assertFalse(torch.equal(held[0][6, 6, 6], held[0][7, 6, 6]))
        input_colors = {tuple(v.tolist()) for v in frames.reshape(-1, 3)}
        self.assertTrue(all(
            tuple(v.tolist()) in input_colors for v in held[0].reshape(-1, 3)
        ))
        self.assertIn("hysteresis=on(max=3)", held[2])
        self.assertIn("feature_held=4", held[2])

    def test_propagated_feature_region_restores_fully_missing_dot(self):
        black = torch.tensor([0.0, 0.0, 0.0])
        green = torch.tensor([0.05, 0.65, 0.20])
        dot = torch.tensor([0.15, 0.10, 0.05])
        frames = black.view(1, 1, 1, 3).expand(5, 13, 15, 3).clone()
        frames[:, 1:12, 1:14] = green
        frames[:, 6, 7] = dot
        frames[2, 6, 7] = green  # no current-frame boundary remains here
        kwargs = dict(
            search_radius=0, patch_radius=0,
            snapper_info="mask=on bg=#000000",
            data_tolerance=0.01, max_color_distance=1.0,
            edge_colors=True, edge_radius=2,
            edge_min_support=2, edge_max_color_distance=1.0,
            edge_medoid=True, edge_cluster_radius=0.8,
            feature_edges=True, feature_radius=0,
            feature_contrast_threshold=0.05,
        )
        base = self.run_block(frames, frames.clone(), **kwargs)
        held = self.run_block(
            frames, frames.clone(), feature_hysteresis=True,
            feature_hold_frames=3, feature_hold_radius=0.8, **kwargs,
        )
        self.assertTrue(torch.equal(base[0][2, 6, 7], green))
        self.assertTrue(torch.equal(held[0][2, 6, 7], dot))
        self.assertEqual(float(held[8][2, 6, 7]), 1.0)
        self.assertEqual(float(held[7][2, 6, 7]), 1.0)

    def test_feature_hysteresis_does_not_cross_scene_cut(self):
        black = torch.tensor([0.0, 0.0, 0.0])
        green = torch.tensor([0.05, 0.65, 0.20])
        beige = torch.tensor([0.80, 0.65, 0.35])
        values = [0.35, 0.55, 0.80, 0.45, 0.70, 0.40, 0.75, 0.50, 0.65]
        frames = black.view(1, 1, 1, 3).expand(9, 13, 15, 3).clone()
        frames[:, 1:12, 1:14] = green
        frames[:, 2:11, 7:14] = beige
        for t, value in enumerate(values):
            frames[t, 6, 6] = torch.tensor([0.05, value, 0.18])
        guide = torch.zeros_like(frames)
        guide[4:] = 1.0  # cut between frames 3 and 4
        held = self.run_block(
            frames, guide, search_radius=0, patch_radius=0,
            snapper_info="mask=on bg=#000000",
            max_color_distance=0.10,
            edge_colors=True, edge_radius=2, edge_agreement=0.9,
            edge_min_support=2, edge_max_color_distance=1.5,
            edge_medoid=True, edge_cluster_radius=0.8,
            feature_edges=True, feature_radius=2,
            feature_contrast_threshold=0.05,
            feature_hysteresis=True, feature_hold_frames=3,
            feature_hold_radius=0.8,
        )
        self.assertEqual(float(held[8][4].sum()), 0.0)
        self.assertIn("scene_cuts=1", held[2])

    def test_feature_pass_requires_locked_background_metadata(self):
        green = torch.tensor([0.05, 0.65, 0.20])
        pink = torch.tensor([1.0, 0.0, 1.0])
        frames = green.view(1, 1, 1, 3).expand(5, 7, 7, 3).clone()
        frames[2, 3, 3] = pink
        result = self.run_block(
            frames, frames.clone(), search_radius=0, patch_radius=0,
            snapper_info="", max_color_distance=0.10,
            edge_colors=False, edge_max_color_distance=1.75,
            feature_edges=True,
        )
        out, _confidence, info, _preview, _changed, _silhouette, _edge, feature, _hold, _rgba, _transparency = result
        self.assertTrue(torch.equal(out[2, 3, 3], pink))
        self.assertEqual(float(feature.sum()), 0.0)
        self.assertIn("features=off(no background metadata)", info)

    def test_thin_internal_line_far_from_background_is_restored(self):
        black = torch.tensor([0.0, 0.0, 0.0])
        green = torch.tensor([0.05, 0.65, 0.20])
        line = torch.tensor([0.15, 0.10, 0.05])
        frames = black.view(1, 1, 1, 3).expand(5, 13, 15, 3).clone()
        frames[:, 1:12, 1:14] = green
        frames[:, 3:10, 7] = line
        frames[2, 6, 7] = green  # one missing cell in a thin muscle/fold line
        result = self.run_block(
            frames, frames.clone(), search_radius=0, patch_radius=0,
            snapper_info="mask=on bg=#000000",
            max_color_distance=0.10,
            edge_colors=True, edge_radius=2,
            edge_min_support=3, edge_max_color_distance=1.0,
            feature_edges=True, feature_radius=1,
            feature_contrast_threshold=0.08,
        )
        out, _confidence, _info, _preview, changed, _silhouette, edge, feature, _hold, _rgba, _transparency = result
        self.assertTrue(torch.equal(out[2, 6, 7], line))
        self.assertEqual(float(changed[2, 6, 7]), 1.0)
        self.assertEqual(float(feature[2, 6, 7]), 1.0)
        self.assertEqual(float(edge[2, 6, 7]), 0.0)

    def test_preset_frontend_runs_with_five_visible_controls(self):
        frames = torch.zeros(3, 3, 3, 3)
        result = VideoPixelSnapperTemporalCleanup().run(
            frames, frames.clone(), "strong", "fast",
            "integer_block_matching", "cpu", 2,
            "mask=on bg=#000000",
        )
        self.assertEqual(len(result), 11)
        self.assertIn("preset=strong", result[2])
        self.assertIn("device=cpu", result[2])

    def test_detail_lock_preset_enables_internal_feature_pass(self):
        frames = torch.zeros(3, 5, 5, 3)
        result = VideoPixelSnapperTemporalCleanup().run(
            frames, frames.clone(), "detail_lock", "fast",
            "integer_block_matching", "cpu", 2,
            "mask=on bg=#000000",
        )
        self.assertIn("preset=detail_lock", result[2])
        self.assertIn("features=on", result[2])
        self.assertIn("hysteresis=off(user)", result[2])

    def test_stability_lock_preset_enables_bounded_hysteresis(self):
        frames = torch.zeros(3, 5, 5, 3)
        result = VideoPixelSnapperTemporalCleanup().run(
            frames, frames.clone(), "stability_lock", "fast",
            "integer_block_matching", "cpu", 2,
            "mask=on bg=#000000",
        )
        self.assertEqual(len(result), 11)
        self.assertIn("preset=stability_lock", result[2])
        self.assertIn("features=on", result[2])
        self.assertIn("hysteresis=on(max=3)", result[2])

    def test_maximum_lock_uses_wider_longer_stability_settings(self):
        frames = torch.zeros(3, 5, 5, 3)
        result = VideoPixelSnapperTemporalCleanup().run(
            frames, frames.clone(), "maximum_lock", "fast",
            "integer_block_matching", "cpu", 2,
            "mask=on bg=#000000",
        )
        self.assertEqual(len(result), 11)
        self.assertIn("preset=maximum_lock", result[2])
        self.assertIn("window=9", result[2])
        self.assertIn("hysteresis=on(max=6)", result[2])

    def test_tied_motion_aligned_vote_keeps_current(self):
        a = torch.tensor([0.0, 0.0, 0.0])
        b = torch.tensor([1.0, 1.0, 1.0])
        c = torch.tensor([1.0, 0.0, 0.0])
        frames = torch.stack([a, b, c, b, a]).view(5, 1, 1, 3)
        guide = c.view(1, 1, 1, 3).expand_as(frames).clone()
        out, _confidence, _info, *_ = self.run_block(
            frames, guide, agreement=0.34,
            data_tolerance=1.0, search_radius=0, patch_radius=0,
        )
        self.assertTrue(torch.equal(out[2], frames[2]))

    def test_moving_sprite_flicker_is_fixed_without_trail(self):
        n, h, w = 5, 8, 12
        black = torch.tensor([0.0, 0.0, 0.0])
        white = torch.tensor([1.0, 1.0, 1.0])
        gray = torch.tensor([0.65, 0.65, 0.65])
        guide = black.view(1, 1, 1, 3).expand(n, h, w, 3).clone()
        frames = guide.clone()
        for t in range(n):
            x = 2 + t
            guide[t, 3:5, x:x + 2] = white
            frames[t, 3:5, x:x + 2] = white
        # One-frame wrong palette label on a material point moving right.
        frames[2, 3, 4] = gray
        out, _confidence, _info, *_ = self.run_block(
            frames, guide, search_radius=2, patch_radius=1,
            data_tolerance=0.03, max_color_distance=0.8,
        )
        self.assertTrue(torch.equal(out[2, 3, 4], white))
        # Every frame must retain exactly its current moving silhouette: no
        # stale white trail may remain behind it.
        self.assertTrue(torch.equal(out == white, guide == white))

    def test_output_colors_always_come_from_snapped_input(self):
        torch.manual_seed(4)
        choices = torch.tensor([
            [0.1, 0.2, 0.3], [0.8, 0.2, 0.1], [0.2, 0.7, 0.4]
        ])
        idx = torch.randint(0, len(choices), (7, 3, 4))
        frames = choices[idx]
        out, _confidence, _info, *_ = self.run_block(
            frames, frames.clone(), search_radius=0, patch_radius=0
        )
        input_colors = {tuple(v.tolist()) for v in frames.reshape(-1, 3)}
        self.assertTrue(all(tuple(v.tolist()) in input_colors for v in out.reshape(-1, 3)))

    def test_untouched_float_palette_values_are_bit_exact(self):
        color = torch.tensor([0.21403, 0.21017, 0.07011])
        frames = color.view(1, 1, 1, 3).expand(5, 2, 3, 3).clone()
        out, _confidence, _info, *_ = self.run_block(
            frames, frames.clone(), search_radius=0, patch_radius=0
        )
        self.assertTrue(torch.equal(out, frames))


class PaletteCoverageTests(unittest.TestCase):
    @staticmethod
    def palette_image(hexes):
        colors = torch.tensor([
            [int(value[i:i + 2], 16) for i in (1, 3, 5)]
            for value in hexes
        ], dtype=torch.float32) / 255.0
        return colors.view(1, 1, len(colors), 3)

    def test_reports_missing_lavender_without_learning_background(self):
        lavender = torch.tensor([208, 160, 220], dtype=torch.float32) / 255.0
        orange = torch.tensor([234, 138, 46], dtype=torch.float32) / 255.0
        image = orange.view(1, 1, 1, 3).expand(2, 4, 4, 3).clone()
        image[:, 1:3, 1:3] = lavender
        mask = torch.zeros(2, 4, 4)
        mask[:, 1:3, 1:3] = 1.0
        master_hex = ["#000000", "#ABB7CD", "#5B3A8C", "#FFFFFF"]
        result = VideoPixelSnapperPaletteCoverage().run(
            image, self.palette_image(master_hex),
            subpalette_size=2, suggestion_count=4, sample_frames=2,
            missing_threshold=0.06, duplicate_threshold=0.03,
            heatmap_limit=0.12, foreground_mask=mask,
            snapper_info=(
                "grid=1px phase=(0,0) cells=4x4 source=4x4 "
                "mask=on bg=#FF00FF mask_threshold=0.5 cell_threshold=0.25"
            ),
        )
        heatmap, subpalette, missing, info = result
        self.assertEqual(heatmap.shape, (2, 4, 4, 3))
        self.assertTrue(torch.equal(heatmap[:, 0, 0], torch.zeros(2, 3)))
        missing_colors = {
            tuple(value.tolist()) for value in missing[:, 16, 16::32].reshape(-1, 3)
        }
        self.assertIn(tuple(lavender.tolist()), missing_colors)
        self.assertNotIn(tuple(orange.tolist()), missing_colors)
        master_colors = {
            tuple(value.tolist())
            for value in self.palette_image(master_hex).reshape(-1, 3)
        }
        self.assertTrue(all(
            tuple(value.tolist()) in master_colors
            for value in subpalette[:, 16, 16::32].reshape(-1, 3)
        ))
        self.assertIn("#D0A0DC", info)
        self.assertIn("nearest #ABB7CD", info)
        self.assertIn("exact_grid", info)
        gap_share = float(re.search(r"gap_cells@[^=]+=([0-9.]+)%", info).group(1))
        suggestion_shares = [
            float(value) for value in re.findall(r"\(([0-9.]+)% cells;", info)
        ]
        self.assertLessEqual(sum(suggestion_shares), gap_share + 0.02)

    def test_embedded_alpha_excludes_hidden_transparent_rgb(self):
        red = torch.tensor([1.0, 0.0, 0.0])
        magenta = torch.tensor([1.0, 0.0, 1.0])
        rgba = torch.zeros(1, 4, 4, 4)
        rgba[..., :3] = magenta  # Photoshop-style hidden RGB under alpha=0
        rgba[:, 1:3, 1:3, :3] = red
        rgba[:, 1:3, 1:3, 3] = 1.0
        master = self.palette_image(["#FF0000", "#000000"])
        heat, _sub, missing, info = VideoPixelSnapperPaletteCoverage().run(
            rgba, master, subpalette_size=2, suggestion_count=4,
            sample_frames=1, missing_threshold=0.06,
            duplicate_threshold=0.02, heatmap_limit=0.12,
            analysis_pixel_size=1.0,
        )
        self.assertEqual(float(heat[0, 0, 0].sum()), 0.0)
        self.assertIn("gap_cells@0.060=0.00%", info)
        self.assertIn("suggestions=none", info)
        self.assertEqual(float(missing.sum()), 0.0)

    def test_standalone_manual_grid_does_not_require_snapper_node(self):
        yy, xx = torch.meshgrid(torch.arange(3), torch.arange(3), indexing="ij")
        cells = ((xx + yy) % 2).float().unsqueeze(-1).expand(-1, -1, 3)
        image = cells.repeat_interleave(2, 0).repeat_interleave(2, 1).unsqueeze(0)
        master = self.palette_image(["#000000", "#FFFFFF"])
        heat, _sub, _missing, info = VideoPixelSnapperPaletteCoverage().run(
            image, master, subpalette_size=2, suggestion_count=2,
            sample_frames=1, missing_threshold=0.06,
            duplicate_threshold=0.02, heatmap_limit=0.12,
            analysis_pixel_size=2.0,
        )
        self.assertEqual(heat.shape, (1, 3, 3, 3))
        self.assertIn("grid_source=standalone_manual", info)
        self.assertIn("exact_grid block=2 phase=(0,0) cells=3x3", info)
        self.assertIn("gap_cells@0.060=0.00%", info)

    def test_near_black_resize_residue_is_not_a_missing_color(self):
        image = torch.tensor([15, 2, 6], dtype=torch.float32).view(
            1, 1, 1, 3
        ) / 255.0
        master = self.palette_image(["#000000", "#181425", "#FFFFFF"])
        heat, _sub, _missing, info = VideoPixelSnapperPaletteCoverage().run(
            image, master, subpalette_size=2, suggestion_count=8,
            sample_frames=1, missing_threshold=0.06,
            duplicate_threshold=0.02, heatmap_limit=0.12,
            snapper_info="grid=1px phase=(0,0) cells=1x1 source=1x1",
        )
        self.assertEqual(float(heat.sum()), 1.0)  # zero error renders blue
        self.assertIn("gap_cells@0.060=0.00%", info)
        self.assertIn("suggestions=none", info)

    def test_lists_near_duplicate_master_slots(self):
        image = torch.tensor([90, 105, 136], dtype=torch.float32).view(
            1, 1, 1, 3
        ) / 255.0
        master = self.palette_image([
            "#5A6988", "#566C86", "#181425", "#FFFFFF"
        ])
        _heat, _sub, _missing, info = VideoPixelSnapperPaletteCoverage().run(
            image, master, subpalette_size=2, suggestion_count=0,
            sample_frames=1, missing_threshold=0.06,
            duplicate_threshold=0.02, heatmap_limit=0.12,
            snapper_info="grid=1px phase=(0,0) cells=1x1 source=1x1",
        )
        self.assertIn("#5A6988/#566C86", info)
        self.assertIn("suggestions=none", info)


class SelectiveOutlineTests(unittest.TestCase):
    @staticmethod
    def palette():
        return torch.tensor([[[
            [0.0, 0.0, 0.0],
            [0.12, 0.02, 0.03],
            [0.28, 0.05, 0.03],
            [0.48, 0.10, 0.05],
            [0.75, 0.20, 0.10],
            [1.00, 0.50, 0.30],
        ]]], dtype=torch.float32)

    @staticmethod
    def rgba_sprite(internal_line=False):
        image = torch.zeros(1, 9, 9, 4)
        image[0, 1:8, 1:8, 3] = 1.0
        image[0, 2:7, 2:7, :3] = torch.tensor([0.75, 0.20, 0.10])
        if internal_line:
            image[0, 3:6, 4, :3] = 0.0
        return image

    def test_outer_selout_uses_palette_only_and_preserves_rgba(self):
        image = self.rgba_sprite(internal_line=True)
        rgb, rgba, changed, info = VideoPixelSnapperSelectiveOutline().run(
            image, self.palette(), "outer_only", "manual", "top_left",
            "balanced", 0, "white_is_foreground",
        )
        self.assertEqual(rgb.shape, (1, 9, 9, 3))
        self.assertTrue(torch.equal(rgba[..., 3], image[..., 3]))
        self.assertEqual(int(changed.sum()), 24)
        # The requested outer-only pass must not recolor the internal line.
        self.assertTrue(torch.equal(rgb[0, 4, 4], torch.zeros(3)))
        palette_colors = {
            tuple(value.tolist()) for value in self.palette().reshape(-1, 3)
        }
        self.assertTrue(all(
            tuple(value.tolist()) in palette_colors
            for value in rgb[changed.bool()].reshape(-1, 3)
        ))
        # Manual top-left lighting creates a lighter outline there than on the
        # bottom/right shadow side, while remaining in the palette.
        self.assertGreater(float(rgb[0, 1, 4].sum()), float(rgb[0, 7, 4].sum()))
        self.assertIn("scope=outer_only", info)
        self.assertIn("replacement_colors=palette_only", info)

    def test_internal_toggle_recolors_thin_black_line(self):
        image = self.rgba_sprite(internal_line=True)
        outer = apply_selective_outline(
            image, self.palette(), "outer_only", "manual", "top_left",
            "balanced", 0,
        )
        all_lines = apply_selective_outline(
            image, self.palette(), "outer_and_internal", "manual", "top_left",
            "balanced", 0,
        )
        self.assertTrue(torch.equal(outer[0][0, 4, 4], torch.zeros(3)))
        self.assertFalse(torch.equal(all_lines[0][0, 4, 4], torch.zeros(3)))
        self.assertGreater(int(all_lines[2].sum()), int(outer[2].sum()))

    def test_auto_light_detects_bright_top_and_shades_bottom_darker(self):
        image = self.rgba_sprite()
        image[0, 2:4, 2:7, :3] = torch.tensor([1.0, 0.50, 0.30])
        rgb, _rgba, _changed, info = apply_selective_outline(
            image, self.palette(), "outer_only", "auto", "bottom_right",
            "balanced", 0,
        )
        self.assertIn("lighting=auto[top", info)
        self.assertGreater(float(rgb[0, 1, 4].sum()), float(rgb[0, 7, 4].sum()))

    def test_comfy_transparency_mask_convention_is_supported(self):
        image = self.rgba_sprite()[..., :3]
        transparency = torch.ones(1, 9, 9)
        transparency[:, 1:8, 1:8] = 0.0
        _rgb, rgba, changed, info = apply_selective_outline(
            image, self.palette(), "outer_only", "manual", "top_left",
            "subtle", 0, transparency, "white_is_transparent",
        )
        self.assertTrue(torch.equal(rgba[..., 3], 1.0 - transparency))
        self.assertEqual(int(changed.sum()), 24)
        self.assertIn("foreground=mask(transparency)", info)
        self.assertIn("alpha=hard_from_mask", info)

    def test_soft_foreground_mask_exports_hard_alpha(self):
        image = self.rgba_sprite()[..., :3]
        foreground = torch.zeros(1, 9, 9)
        foreground[:, 1:8, 1:8] = 0.75
        foreground[:, 0, :] = 0.2
        _rgb, rgba, _changed, info = apply_selective_outline(
            image, self.palette(), "outer_only", "manual", "top_left",
            "balanced", 0, foreground, "white_is_foreground",
        )
        self.assertTrue(torch.equal(
            torch.unique(rgba[..., 3]), torch.tensor([0.0, 1.0])
        ))
        self.assertEqual(float(rgba[0, 1, 1, 3]), 1.0)
        self.assertEqual(float(rgba[0, 0, 0, 3]), 0.0)
        self.assertIn("alpha=hard_from_mask", info)

    def test_plain_rgb_flat_background_fallback_keeps_background_opaque(self):
        image = torch.ones(1, 9, 9, 3)
        image[:, 1:8, 1:8] = 0.0
        image[:, 2:7, 2:7] = torch.tensor([0.75, 0.20, 0.10])
        rgb, rgba, changed, info = apply_selective_outline(
            image, self.palette(), "outer_only", "manual", "top_left",
            "balanced", 0,
        )
        self.assertEqual(int(changed.sum()), 24)
        self.assertTrue(torch.equal(rgb[:, 0, 0], torch.ones(1, 3)))
        self.assertEqual(float(rgba[..., 3].min()), 1.0)
        self.assertIn("foreground=auto_flat_border", info)

    def test_scaled_input_matches_native_then_nearest_result(self):
        native = self.rgba_sprite(internal_line=True)
        native_result = apply_selective_outline(
            native, self.palette(), "outer_and_internal", "manual", "top_left",
            "balanced", 0,
        )
        scale = 3
        enlarged = native.repeat_interleave(scale, 1).repeat_interleave(scale, 2)
        scaled_result = apply_selective_outline(
            enlarged, self.palette(), "outer_and_internal", "manual", "top_left",
            "balanced", 0, input_scale=scale,
        )
        for actual, expected in zip(scaled_result[:3], native_result[:3]):
            expected = expected.repeat_interleave(scale, 1).repeat_interleave(scale, 2)
            self.assertTrue(torch.equal(actual, expected))
        self.assertIn("input_scale=3x", scaled_result[3])
        self.assertIn("scale_processing=logical_then_nearest", scaled_result[3])

    def test_scaled_rgb_and_transparency_mask_match_native(self):
        rgba = self.rgba_sprite()
        native_rgb = rgba[..., :3]
        native_transparency = 1.0 - rgba[..., 3]
        native_result = apply_selective_outline(
            native_rgb, self.palette(), "outer_only", "manual", "top_left",
            "balanced", 0, native_transparency, "white_is_transparent",
        )
        scale = 3
        enlarged_rgb = native_rgb.repeat_interleave(scale, 1).repeat_interleave(scale, 2)
        enlarged_mask = native_transparency.repeat_interleave(scale, 1).repeat_interleave(scale, 2)
        scaled_result = apply_selective_outline(
            enlarged_rgb, self.palette(), "outer_only", "manual", "top_left",
            "balanced", 0, enlarged_mask, "white_is_transparent",
            input_scale=scale,
        )
        for actual, expected in zip(scaled_result[:3], native_result[:3]):
            expected = expected.repeat_interleave(scale, 1).repeat_interleave(scale, 2)
            self.assertTrue(torch.equal(actual, expected))

    def test_node_reads_scaled_grid_from_snapper_info(self):
        native = self.rgba_sprite()
        scale = 4
        enlarged = native.repeat_interleave(scale, 1).repeat_interleave(scale, 2)
        result = VideoPixelSnapperSelectiveOutline().run(
            enlarged, self.palette(), "outer_only", "manual", "top_left",
            "balanced", 0, "white_is_foreground",
            snapper_info=(
                "grid=5px phase=(0,1) cells=9x9 source=45x45 "
                "palette=custom (6 colors) color_distance=oklab "
                "scale=4x -> 36x36"
            ),
            input_pixel_scale=0,
        )
        self.assertEqual(result[0].shape, enlarged[..., :3].shape)
        self.assertIn("input_scale=4x", result[3])
        self.assertIn("scale_source=snapper_info", result[3])
        # Every logical output pixel must remain an exact 4x4 block.
        reduced = result[0][:, scale // 2::scale, scale // 2::scale]
        reconstructed = reduced.repeat_interleave(scale, 1).repeat_interleave(scale, 2)
        self.assertTrue(torch.equal(result[0], reconstructed))

    def test_wrong_declared_scale_rejects_non_nearest_input(self):
        image = self.rgba_sprite().repeat_interleave(3, 1).repeat_interleave(3, 2)
        image[0, 4, 4, 0] = 0.5  # break one pixel inside a nominal 3x3 block
        with self.assertRaisesRegex(ValueError, "does not match an exact nearest-neighbor"):
            apply_selective_outline(
                image, self.palette(), "outer_only", "manual", "top_left",
                "balanced", 0, input_scale=3,
            )

    def test_dark_nonblack_pixels_above_threshold_are_not_outline(self):
        image = self.rgba_sprite()
        dark = torch.tensor([24, 20, 37], dtype=torch.float32) / 255.0
        image[0, 1, 4, :3] = dark
        rgb, _rgba, changed, _info = apply_selective_outline(
            image, self.palette(), "outer_only", "manual", "top_left",
            "balanced", 12,
        )
        self.assertTrue(torch.equal(rgb[0, 1, 4], dark))
        self.assertEqual(float(changed[0, 1, 4]), 0.0)


class TransparentPipelineTests(unittest.TestCase):
    def test_hard_alpha_survives_cleanup_editor_retimer_and_sheet(self):
        blue = torch.tensor([0.0, 0.0, 1.0])
        red = torch.tensor([1.0, 0.0, 0.0])
        raw = blue.view(1, 1, 1, 3).expand(3, 8, 8, 3).clone()
        raw[:, 2:6, 2:6] = red
        mask = torch.zeros(3, 8, 8)
        mask[:, 2:6, 2:6] = 1.0
        rgb, palette, info, transparent, _mask = VideoPixelSnapper().run(
            raw, pixel_size=2.0,
            grid_detection_mode="average_across_frames",
            cell_method="majority", k_colors=4, accent_slots=0,
            sample_frames=3, dither="none", despeckle=False,
            output_scale_mode="manual", output_scale=1, seed=2,
            mask_threshold=0.5, mask_cell_threshold=0.25,
            invert_mask=False, background_mode="transparent",
            foreground_mask=mask,
        )
        cleanup = VideoPixelSnapperTemporalCleanup().run(
            transparent, raw, "maximum_lock", "fast",
            "integer_block_matching", "cpu", 2, info,
        )
        self.assertTrue(torch.equal(cleanup[9][..., 3], transparent[..., 3]))
        editor = VideoPixelSnapperEditor().run(
            raw, cleanup[9], palette, 3, info
        )["result"]
        self.assertTrue(torch.equal(editor[1][..., 3], transparent[..., 3]))
        self.assertEqual(
            editor[1].untyped_storage().data_ptr(),
            cleanup[9].untyped_storage().data_ptr(),
        )
        _rgb_retimed, _retime_info, rgba_retimed, _retime_mask = (
            VideoPixelSnapperFrameRetimer().run(
                editor[1], json.dumps([2, 0, 2]), 0.0, 3
            )
        )
        sheet = VideoPixelSnapperSpriteSheet().run(
            rgba_retimed, columns=2, padding=0
        )[0]
        self.assertEqual(sheet.shape[-1], 4)
        self.assertTrue(torch.equal(
            sheet[0, 0:4, 0:4, 3], transparent[2, ..., 3]
        ))
        self.assertEqual(float(sheet[0, 4:8, 4:8, 3].sum()), 0.0)


class SpriteSheetTests(unittest.TestCase):
    def test_rgba_sheet_is_row_major_exact_and_empty_slots_are_transparent(self):
        frames = torch.zeros(5, 2, 3, 4)
        for i in range(5):
            frames[i, ..., 0] = i / 4.0
            frames[i, ..., 3] = 1.0
        sheet, info, frame_w, frame_h, columns, rows = (
            VideoPixelSnapperSpriteSheet().run(frames, columns=3, padding=1)
        )
        self.assertEqual(sheet.shape, (1, 5, 11, 4))
        self.assertEqual((frame_w, frame_h, columns, rows), (3, 2, 3, 2))
        self.assertTrue(torch.equal(sheet[0, 0:2, 0:3], frames[0]))
        self.assertTrue(torch.equal(sheet[0, 0:2, 4:7], frames[1]))
        self.assertTrue(torch.equal(sheet[0, 3:5, 4:7], frames[4]))
        self.assertEqual(float(sheet[0, 3:5, 8:11, 3].sum()), 0.0)
        self.assertEqual(float(sheet[0, :, 3, 3].sum()), 0.0)  # padding column
        self.assertIn("layout=3x2", info)
        self.assertIn("RGBA hard alpha preserved", info)


class FrameRetimerTests(unittest.TestCase):
    def test_sequence_reorders_and_repeats_without_blending(self):
        image = torch.arange(4.0).view(4, 1, 1, 1).expand(-1, 1, 1, 3)
        out, info, transparent, transparency = VideoPixelSnapperFrameRetimer().run(
            image, json.dumps([2, 0, 2, 3]), 10.0, 4
        )
        self.assertEqual(out[:, 0, 0, 0].tolist(), [2.0, 0.0, 2.0, 3.0])
        self.assertEqual(transparent.shape[-1], 4)
        self.assertEqual(float(transparency.sum()), 0.0)
        self.assertIn("4 input frame(s) -> 4 output frame(s)", info)

    def test_rgba_retiming_preserves_alpha_without_blending(self):
        rgba = torch.zeros(3, 2, 2, 4)
        rgba[0, ..., :3] = 0.2
        rgba[1, ..., :3] = 0.5
        rgba[2, ..., :3] = 0.8
        rgba[0, 0, 0, 3] = 1.0
        rgba[1, 0, 1, 3] = 1.0
        rgba[2, 1, 1, 3] = 1.0
        rgb, _info, transparent, mask = VideoPixelSnapperFrameRetimer().run(
            rgba, json.dumps([2, 0, 2]), 0.0, 3
        )
        self.assertTrue(torch.equal(rgb, rgba[[2, 0, 2], ..., :3]))
        self.assertTrue(torch.equal(transparent, rgba[[2, 0, 2]]))
        self.assertTrue(torch.equal(mask, 1.0 - rgba[[2, 0, 2], ..., 3]))

    def test_non_array_json_falls_back_to_identity(self):
        image = torch.rand(3, 2, 2, 3)
        out, _, _transparent, _mask = VideoPixelSnapperFrameRetimer().run(
            image, '{"0": 2}', 0.0, 3
        )
        self.assertTrue(torch.equal(out, image))


if __name__ == "__main__":
    unittest.main()
