import json
import unittest
from unittest.mock import patch

import torch

from video_pixel_snapper import (
    VideoPixelSnapper,
    VideoPixelSnapperEditor,
    _estimate_phase,
    _prepare_mask,
    cell_stats,
    despeckle_indices,
    estimate_phase_for_frame,
    process_batch,
)
from frame_retimer import VideoPixelSnapperFrameRetimer
from temporal_denoise import (
    VideoPixelSnapperTemporalCleanup,
    VideoPixelSnapperTemporalCleanupAdvanced,
    _align_guide_to_snapper,
    _raft_bidirectional_flow,
)


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
    def test_rgba_input_and_custom_palette(self):
        image = torch.rand(3, 24, 32, 4)
        palette = torch.tensor(
            [[[[0.0, 0.0, 0.0, 1.0], [1.0, 1.0, 1.0, 1.0], [1.0, 0.0, 0.0, 1.0]]]]
        )
        out, preview, info = VideoPixelSnapper().run(
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
        self.assertIn("palette=custom (3 colors)", info)
        self.assertIn("source=32x24", info)

    def test_editor_emits_rich_grid_metadata(self):
        raw = torch.rand(2, 32, 40, 4)
        snapped = torch.rand(2, 15, 20, 3)
        palette = torch.rand(1, 32, 96, 3)
        info = ("grid=4px phase=(1,2) cells=8x6 palette=auto masked "
                "(3 colors, 1 accent, 1 bg) scale=3x -> 24x18 "
                "mask=on bg=#0011aa mask_threshold=0.5 cell_threshold=0.25")
        result = VideoPixelSnapperEditor().run(raw, snapped, palette, 2, info)
        self.assertEqual(result["ui"]["vps_grid"], [4, 1, 2, 8, 6, 3])
        self.assertEqual(result["ui"]["vps_background"], ["#0011aa"])
        self.assertEqual(result["result"][0].shape[-1], 3)

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
        out, preview, info = VideoPixelSnapper().run(
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
        self.assertIn("1 bg", info)
        self.assertIn("mask=on", info)

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
                  feature_contrast_threshold=0.08):
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
            snapper_info,
        )

    def test_isolated_static_flicker_is_corrected(self):
        frames = torch.zeros(5, 3, 3, 3)
        frames[2, 1, 1] = torch.tensor([1.0, 0.0, 0.0])
        guide = torch.zeros_like(frames)
        (
            out, confidence, info, preview, changed,
            silhouette_changed, edge_changed, feature_changed,
        ) = self.run_block(
            frames, guide, search_radius=0, patch_radius=0
        )
        self.assertTrue(torch.equal(out, torch.zeros_like(out)))
        self.assertGreater(float(confidence[2, 1, 1]), 0.0)
        self.assertEqual(preview.shape, frames.shape)
        self.assertEqual(changed.shape, frames.shape[:3])
        self.assertEqual(silhouette_changed.shape, frames.shape[:3])
        self.assertEqual(feature_changed.shape, frames.shape[:3])
        self.assertEqual(float(changed[2, 1, 1]), 1.0)
        self.assertIn("backend=integer_block_matching", info)
        self.assertIn("device=cpu", info)

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
        out, _confidence, info, _preview, changed, silhouette, _edge, _feature = result
        self.assertTrue(torch.equal(out[2, 1, 3], black))
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
        out, _confidence, info, _preview, changed, _silhouette, edge, _feature = result
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
        out, _confidence, _info, _preview, _changed, _silhouette, edge, _feature = result
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
        out, _confidence, info, _preview, changed, _silhouette, edge, feature = result
        # The temporal medoid is copied from frame 3; no RGB mean is created.
        self.assertTrue(torch.equal(out[2, 6, 6], shades[3]))
        self.assertEqual(float(changed[2, 6, 6]), 1.0)
        self.assertEqual(float(feature[2, 6, 6]), 1.0)
        self.assertEqual(float(edge[2, 6, 6]), 0.0)
        self.assertIn("features=on", info)
        self.assertIn("feature_replaced=", info)

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
        out, _confidence, info, _preview, _changed, _silhouette, _edge, feature = result
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
        out, _confidence, _info, _preview, changed, _silhouette, edge, feature = result
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
        self.assertEqual(len(result), 8)
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


class FrameRetimerTests(unittest.TestCase):
    def test_sequence_reorders_and_repeats_without_blending(self):
        image = torch.arange(4.0).view(4, 1, 1, 1).expand(-1, 1, 1, 3)
        out, info = VideoPixelSnapperFrameRetimer().run(
            image, json.dumps([2, 0, 2, 3]), 10.0, 4
        )
        self.assertEqual(out[:, 0, 0, 0].tolist(), [2.0, 0.0, 2.0, 3.0])
        self.assertIn("4 input frame(s) -> 4 output frame(s)", info)

    def test_non_array_json_falls_back_to_identity(self):
        image = torch.rand(3, 2, 2, 3)
        out, _ = VideoPixelSnapperFrameRetimer().run(image, '{"0": 2}', 0.0, 3)
        self.assertTrue(torch.equal(out, image))


if __name__ == "__main__":
    unittest.main()
