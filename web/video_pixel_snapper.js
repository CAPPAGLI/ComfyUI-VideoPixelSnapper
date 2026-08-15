/**
 * Video Pixel Snapper — live palette editor widget.
 *
 * Adds a DOM widget under the node showing the raw input frame, the
 * node's own output, and a live re-color preview that updates as you
 * edit the palette client-side — no need to re-run the graph to see
 * palette changes.
 *
 * Diagnostics go to the browser console only (search for
 * "[VideoPixelSnapper]"); nothing is drawn on top of the page.
 */
import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

function log(line) { console.log("[VideoPixelSnapper] " + line); }

function refUrl(ref) {
  const params = new URLSearchParams({ filename: ref.filename, subfolder: ref.subfolder || "", type: ref.type || "temp" });
  return "/view?" + params.toString();
}
function hexToRgb(hex) {
  const v = parseInt(hex.replace("#", ""), 16);
  return [(v >> 16) & 255, (v >> 8) & 255, v & 255];
}
function rgbToHex(r, g, b) {
  return "#" + [r, g, b].map((x) => x.toString(16).padStart(2, "0")).join("");
}
function normalizeHex(value) {
  const hex = String(value || "").trim().toLowerCase();
  return /^#[0-9a-f]{6}$/.test(hex) ? hex : null;
}
function palettesHaveSameColors(a, b) {
  if (a.length !== b.length) return false;
  const aa = a.map((hex) => hex.toLowerCase()).sort();
  const bb = b.map((hex) => hex.toLowerCase()).sort();
  return aa.every((hex, i) => hex === bb[i]);
}
function loadImage(url) {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.crossOrigin = "anonymous";
    img.onload = () => resolve(img);
    img.onerror = reject;
    img.src = url;
  });
}
function fitRect(srcW, srcH, boxW, boxH) {
  const scale = Math.min(boxW / srcW, boxH / srcH);
  return { scale, w: srcW * scale, h: srcH * scale };
}

// Exact nearest-color search is O(pixels * colors). Fine for typical
// snapped-resolution images, but "large video + many colors" (reported
// in practice) can push that into hundreds of ms. Above a size
// threshold, fall back to a quantized lookup table: bin each channel to
// 5 bits (32 levels), precompute the nearest palette index per bin once
// (O(bins * colors), a one-time cost per palette change), then every
// pixel is an O(1) table lookup. Benchmarked: ~7x faster at 800x1000px /
// 96 colors, at the cost of ~10-15% of pixels picking an adjacent
// (still close, per direct measurement — worst case ~20/441 units off)
// color instead of the exact nearest one. Below the threshold this
// isn't used at all, so typical-size images stay pixel-exact.
const LUT_BITS = 5;
const LUT_SHIFT = 8 - LUT_BITS;
const LUT_SIZE = 1 << LUT_BITS;
const LUT_THRESHOLD = 15_000_000; // pixels * colors

function buildPaletteLUT(paletteRGB) {
  const lut = new Int32Array(LUT_SIZE * LUT_SIZE * LUT_SIZE);
  for (let ri = 0; ri < LUT_SIZE; ri++) {
    const r = ri << LUT_SHIFT;
    for (let gi = 0; gi < LUT_SIZE; gi++) {
      const g = gi << LUT_SHIFT;
      for (let bi = 0; bi < LUT_SIZE; bi++) {
        const b = bi << LUT_SHIFT;
        let best = 0, bestD = Infinity;
        for (let p = 0; p < paletteRGB.length; p++) {
          const [pr, pg, pb] = paletteRGB[p];
          const d = (r - pr) ** 2 + (g - pg) ** 2 + (b - pb) ** 2;
          if (d < bestD) { bestD = d; best = p; }
        }
        lut[(ri << (2 * LUT_BITS)) | (gi << LUT_BITS) | bi] = best;
      }
    }
  }
  return lut;
}

// Reduces a full-resolution RAW canvas down to (targetW x targetH) — one
// color per output cell, taken as the per-channel MEDIAN of the raw
// pixels that fall in that cell. Palette-independent (only depends on
// the raw image + grid), so it's cached once per frame and reused
// across any number of palette edits — see computeLiveCanvas for why
// this replaced re-classifying the already-quantized snapped cache.
//
// `grid`, if given (from Video Pixel Snapper's `info` output, wired
// into this node's optional `info` input), is {block, phaseX, phaseY}
// — the REAL detected grid, so cell boundaries line up exactly with
// what the core node actually used. Without it, falls back to a naive
// centered block mapping, which can be a cell or two off from the true
// alignment. Either way this is still a MEDIAN reduction, not the core
// node's actual majority-vote-with-confidence-margin algorithm, so an
// exact pixel-for-pixel match with "Snapped" isn't guaranteed even
// with the right grid — the real, authoritative result always comes
// from re-running the graph. This is a fast, close approximation for
// live editing feedback, not a reimplementation of the whole pipeline.
function blockMedianReduce(srcCanvas, targetW, targetH, grid) {
  const srcW = srcCanvas.width, srcH = srcCanvas.height;
  const sctx = srcCanvas.getContext("2d", { willReadFrequently: true });
  const src = sctx.getImageData(0, 0, srcW, srcH).data;
  const out = new ImageData(targetW, targetH);

  function medianCell(x0, y0, x1, y1, oi) {
    const rs = [], gs = [], bs = [];
    for (let y = y0; y < y1; y++) {
      for (let x = x0; x < x1; x++) {
        const i = (y * srcW + x) * 4;
        rs.push(src[i]); gs.push(src[i + 1]); bs.push(src[i + 2]);
      }
    }
    if (!rs.length) return;
    rs.sort((a, b) => a - b); gs.sort((a, b) => a - b); bs.sort((a, b) => a - b);
    // torch.median uses the lower middle element for an even sample count.
    const mid = (rs.length - 1) >> 1;
    out.data[oi] = rs[mid]; out.data[oi + 1] = gs[mid]; out.data[oi + 2] = bs[mid]; out.data[oi + 3] = 255;
  }

  if (grid && grid.block > 0) {
    const block = grid.block;
    const px = ((grid.phaseX % block) + block) % block;
    const py = ((grid.phaseY % block) + block) % block;
    for (let cy = 0; cy < targetH; cy++) {
      const y0 = py + cy * block, y1 = Math.min(srcH, y0 + block);
      if (y0 >= srcH) continue;
      for (let cx = 0; cx < targetW; cx++) {
        const x0 = px + cx * block, x1 = Math.min(srcW, x0 + block);
        if (x0 >= srcW) continue;
        medianCell(x0, y0, x1, y1, (cy * targetW + cx) * 4);
      }
    }
    return out;
  }

  // fallback: naive centered division (used when no `info` is connected)
  const blockW = srcW / targetW, blockH = srcH / targetH;
  for (let cy = 0; cy < targetH; cy++) {
    const y0 = Math.floor(cy * blockH);
    const y1 = Math.max(y0 + 1, Math.floor((cy + 1) * blockH));
    for (let cx = 0; cx < targetW; cx++) {
      const x0 = Math.floor(cx * blockW);
      const x1 = Math.max(x0 + 1, Math.floor((cx + 1) * blockW));
      medianCell(x0, y0, x1, y1, (cy * targetW + cx) * 4);
    }
  }
  return out;
}

app.registerExtension({
  name: "VideoPixelSnapper.LiveEditor",

  async beforeRegisterNodeDef(nodeType, nodeData, appRef) {
    if (nodeType.comfyClass !== "VideoPixelSnapperEditor" && nodeData?.name !== "VideoPixelSnapperEditor") return;
    log("node def matched — extension is active");

    const origOnNodeCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      origOnNodeCreated?.apply(this, arguments);
      const node = this;
      log(`node instance created (id ${node.id})`);

      const uid = node.id;
      const wrap = document.createElement("div");
      wrap.style.cssText =
        "display:flex;flex-direction:column;gap:6px;padding:8px;background:#1b1e24;" +
        "border-radius:6px;width:100%;min-width:280px;box-sizing:border-box;font-family:monospace;";
      wrap.innerHTML = `
        <div style="display:flex;justify-content:space-between;align-items:center;flex-shrink:0;">
          <div style="font-size:10px;color:#8b909c;text-transform:uppercase;letter-spacing:.05em;">
            Live palette editor
          </div>
          <div style="display:flex;gap:2px;">
            <button class="vps-h-minus" title="Shrink preview" style="font-size:10px;padding:2px 6px;">−</button>
            <button class="vps-h-plus" title="Grow preview" style="font-size:10px;padding:2px 6px;">+</button>
          </div>
        </div>

        <div style="display:flex;align-items:center;justify-content:center;gap:8px;flex-shrink:0;">
          <button class="vps-prev" title="Previous frame" style="font-size:11px;padding:3px 8px;">&larr;</button>
          <span class="vps-framecount" style="font-size:10px;color:#8b909c;min-width:56px;text-align:center;">—</span>
          <button class="vps-next" title="Next frame" style="font-size:11px;padding:3px 8px;">&rarr;</button>
        </div>

        <div class="vps-previews" style="display:flex;gap:4px;flex-shrink:0;"
             title="Scroll to zoom, middle-click drag to pan, double-click to reset view">
          <div style="flex:1;min-width:0;">
            <div style="font-size:8px;color:#8b909c;text-align:center;">Original</div>
            <canvas data-src="raw" style="width:100%;height:84px;display:block;background:#111;border-radius:3px;
                           image-rendering:pixelated;touch-action:none;"></canvas>
          </div>
          <div style="flex:1;min-width:0;">
            <div style="font-size:8px;color:#8b909c;text-align:center;">Snapped (auto palette)</div>
            <canvas data-src="snapped" style="width:100%;height:84px;display:block;background:#111;border-radius:3px;
                           image-rendering:pixelated;touch-action:none;"></canvas>
          </div>
          <div style="flex:1;min-width:0;">
            <div class="vps-live-label" style="font-size:8px;color:#8b909c;text-align:center;">Live (your edit)</div>
            <canvas data-src="live" style="width:100%;height:84px;display:block;background:#111;border-radius:3px;
                           image-rendering:pixelated;touch-action:none;"></canvas>
          </div>
        </div>

        <div style="display:flex;gap:4px;flex-shrink:0;">
          <button class="vps-tool" data-tool="pick" title="Click a preview to add the color under the cursor"
                  style="flex:1;font-size:10px;padding:4px;">Pick</button>
          <button class="vps-tool" data-tool="delete" title="Click a preview to remove the closest palette color"
                  style="flex:1;font-size:10px;padding:4px;">Delete</button>
          <button class="vps-tool" data-tool="replace" title="Click a color to select it, then click another (or use manual entry) to replace it"
                  style="flex:1;font-size:10px;padding:4px;">Replace</button>
          <button class="vps-tool" data-tool="measure" title="Click two points on Original to measure a distance and derive pixel_size"
                  style="flex:1;font-size:10px;padding:4px;">Measure</button>
          <div class="vps-pick-swatch" style="width:20px;height:20px;border-radius:3px;border:1px solid #3a3f4a;display:none;flex-shrink:0;"></div>
        </div>

        <div class="vps-measure-panel" style="display:none;flex-direction:column;gap:4px;flex-shrink:0;
                    background:#14161a;border-radius:4px;padding:6px;">
          <div class="vps-measure-info" style="font-size:9px;color:#8b909c;line-height:1.4;">
            Click two points on "Original" to measure a distance (e.g. top and bottom of a character).
          </div>
          <div style="display:flex;gap:4px;align-items:center;">
            <span style="font-size:9px;color:#8b909c;">Target size (cells):</span>
            <input type="number" class="vps-measure-target" value="192" min="1" step="1"
                   style="width:60px;font-size:10px;padding:2px;background:#111;color:#eee;border:1px solid #333;border-radius:3px;">
          </div>
          <div style="display:flex;gap:4px;">
            <button class="vps-measure-apply" disabled style="flex:1;font-size:10px;padding:4px;">Apply to pixel_size</button>
            <button class="vps-measure-clear" style="flex:1;font-size:10px;padding:4px;">Clear points</button>
          </div>
        </div>

        <div style="display:flex;gap:4px;align-items:center;flex-shrink:0;">
          <input type="color" class="vps-colorpick" value="#ffffff" style="width:26px;height:22px;padding:0;border:none;background:none;cursor:pointer;flex-shrink:0;">
          <input type="text" class="vps-hexinput" value="#ffffff" style="width:64px;font-size:10px;padding:3px;background:#111;color:#eee;border:1px solid #333;border-radius:3px;flex-shrink:0;">
          <button class="vps-addcolor" title="Adds the color as a new palette entry" style="flex:1;font-size:10px;padding:4px;">+ Add</button>
        </div>
        <div style="display:flex;gap:4px;flex-shrink:0;">
          <button class="vps-replace-apply" disabled title="Replaces the color selected with the Replace tool"
                  style="flex:1;font-size:10px;padding:4px;">Replace selected</button>
        </div>
        <div style="display:flex;gap:4px;flex-shrink:0;">
          <button class="vps-apply-saved-subst" title="Re-applies remembered from-color -> to-color substitutions to whichever palette entries are currently closest to each remembered 'from' — safer than reusing a whole saved palette verbatim on a different image/scene"
                  style="flex:2;font-size:10px;padding:4px;">Apply saved substitutions</button>
          <button class="vps-clear-saved-subst" title="Forgets all remembered substitutions (stored in this browser only)"
                  style="flex:1;font-size:10px;padding:4px;">Forget</button>
        </div>

        <div class="vps-palette" style="display:grid;grid-template-columns:repeat(auto-fill,20px);grid-auto-rows:20px;
                    gap:3px;max-height:100px;overflow-y:auto;padding:2px;background:#14161a;border-radius:4px;flex-shrink:0;"></div>
        <div class="vps-palette-empty" style="font-size:9px;color:#8b909c;">Palette is empty.</div>

        <div style="display:flex;gap:4px;flex-shrink:0;">
          <button class="vps-reset" style="flex:1;font-size:10px;padding:4px;">Reset</button>
        </div>
        <div style="display:flex;gap:4px;flex-shrink:0;">
          <input type="text" class="vps-subfolder" placeholder="subfolder" value="vps_palettes"
                 style="flex:1;min-width:0;font-size:10px;padding:3px;background:#111;color:#eee;border:1px solid #333;border-radius:3px;">
          <input type="text" class="vps-filename" placeholder="filename" value="palette_${uid}"
                 style="flex:1;min-width:0;font-size:10px;padding:3px;background:#111;color:#eee;border:1px solid #333;border-radius:3px;">
        </div>
        <div style="display:flex;gap:4px;flex-shrink:0;">
          <button class="vps-save" title="Uploads to ComfyUI's input/ folder via /upload/image"
                  style="flex:1;font-size:10px;padding:4px;">Save as custom_palette</button>
        </div>
        <div class="vps-status" style="font-size:9px;color:#8b909c;line-height:1.4;flex-shrink:0;">
          Run the node at least once to get a preview.
        </div>
      `;
      node.addDOMWidget(`vps_live_editor_${uid}`, "vps_live_editor", wrap, {});

      node._vps = {
        rawRefs: [], frameRefs: [],
        frameIdx: 0,
        rawCache: {}, snappedCache: {},   // frameIdx -> {canvas,w,h}, loaded lazily
        rawReducedCache: {},              // frameIdx -> ImageData: raw pixels block-reduced to cell
                                           // resolution, palette-INDEPENDENT (see computeLiveCanvas)
        grid: null,                       // {block,phaseX,phaseY,cellsW,cellsH,scale}
                                           // from the optional core `info` connection
        backgroundHex: null,              // locked mask background, when reported by core
        originalPalette: [],
        livePalette: [],
        paletteDirty: false,              // preserve edits on re-run, but refresh untouched palettes
        liveCache: { key: null, data: null },  // memoized live-recolor result
        tool: null,                       // 'pick' | 'delete' | 'replace' | 'measure' | null
        replaceSource: null,              // index into livePalette pending replacement
        measurePoints: [],                // up to 2 {sx, sy} points in RAW source-pixel space
        view: { zoom: 1, panX: 0, panY: 0 },
      };
      const st = node._vps;

      const boxHeight = { px: 84 };
      const canvases = Array.from(wrap.querySelectorAll(".vps-previews canvas"));
      const boxOf = (key) => wrap.querySelector(`canvas[data-src="${key}"]`);
      const paletteRow = wrap.querySelector(".vps-palette");
      const paletteEmpty = wrap.querySelector(".vps-palette-empty");
      const statusEl = wrap.querySelector(".vps-status");
      const pickSwatch = wrap.querySelector(".vps-pick-swatch");
      const colorPick = wrap.querySelector(".vps-colorpick");
      const hexInput = wrap.querySelector(".vps-hexinput");
      const frameCounter = wrap.querySelector(".vps-framecount");
      const liveLabel = wrap.querySelector(".vps-live-label");
      const replaceApplyBtn = wrap.querySelector(".vps-replace-apply");
      const measurePanel = wrap.querySelector(".vps-measure-panel");
      const measureInfo = wrap.querySelector(".vps-measure-info");
      const measureTargetInput = wrap.querySelector(".vps-measure-target");
      const measureApplyBtn = wrap.querySelector(".vps-measure-apply");

      // Recomputing a full nearest-color remap on every mousemove (while
      // hovering the live preview with a tool active) was the cause of
      // both the lag and the "adding doesn't seem to update" reports —
      // it wasn't actually stuck, just re-doing an O(pixels*colors) pass
      // far more often than needed. Memoize on (frame, palette contents)
      // so it only recomputes when something that actually changes the
      // result changes.
      function getRawReduced() {
        const cached = st.rawReducedCache[st.frameIdx];
        if (cached) return cached;
        const raw = st.rawCache[st.frameIdx];
        const snapped = st.snappedCache[st.frameIdx];
        if (!raw || !snapped) return null;

        // The snapped preview can be an integer nearest-neighbor upscale
        // (for example 117x81 for a 13x9 cell grid). Reducing RAW directly
        // to snapped.w/snapped.h would then create fake sub-cells and make
        // the Live panel disagree with the Python node. Rich grid metadata
        // supplies the real cell dimensions; old workflows without the
        // optional `info` wire retain the previous best-effort fallback.
        const targetW = st.grid?.cellsW || snapped.w;
        const targetH = st.grid?.cellsH || snapped.h;
        const reduced = blockMedianReduce(raw.canvas, targetW, targetH, st.grid);
        st.rawReducedCache[st.frameIdx] = reduced;
        return reduced;
      }

      // With no palette edits, Live must be an exact visual control: use
      // the authoritative Python-generated Snapped frame itself. Merely
      // sharing a palette was not enough before, because the browser used
      // a median cell representative while Python may have used majority,
      // center weighting, dithering, and/or despeckle.
      //
      // Once the palette differs, classify the block-reduced RAW image.
      // That is intentionally retained instead of recoloring the old
      // snapped cache: a newly added color could never win against pixels
      // that had already been quantized under the old palette.
      function computeLiveCanvas() {
        if (!st.livePalette.length) return null;
        const cacheKey = st.frameIdx + "::" + st.livePalette.join(",");
        if (st.liveCache.key === cacheKey) return st.liveCache.data;

        const snapped = st.snappedCache[st.frameIdx];
        const exactBaseline = !st.paletteDirty ||
          palettesHaveSameColors(st.livePalette, st.originalPalette);
        if (snapped && exactBaseline) {
          liveLabel.textContent = "Live (exactly matches Snapped)";
          liveLabel.style.color = "#5fb3a3";
          st.liveCache = { key: cacheKey, data: snapped };
          return snapped;
        }

        liveLabel.textContent = "Live (edited preview)";
        liveLabel.style.color = "#e0a458";
        const reduced = getRawReduced();
        if (!reduced) return null;
        const { width: w, height: h, data: id } = reduced;
        const out = new ImageData(w, h);
        const paletteRGB = st.livePalette.map(hexToRgb);
        const workload = (w * h) * paletteRGB.length;

        if (workload > LUT_THRESHOLD) {
          const lut = buildPaletteLUT(paletteRGB);
          for (let i = 0; i < id.length; i += 4) {
            const idx = ((id[i] >> LUT_SHIFT) << (2 * LUT_BITS)) |
                        ((id[i + 1] >> LUT_SHIFT) << LUT_BITS) |
                        (id[i + 2] >> LUT_SHIFT);
            const best = lut[idx];
            out.data[i] = paletteRGB[best][0]; out.data[i + 1] = paletteRGB[best][1];
            out.data[i + 2] = paletteRGB[best][2]; out.data[i + 3] = 255;
          }
        } else {
          for (let i = 0; i < id.length; i += 4) {
            let best = 0, bestD = Infinity;
            const r = id[i], g = id[i + 1], b = id[i + 2];
            for (let p = 0; p < paletteRGB.length; p++) {
              const [pr, pg, pb] = paletteRGB[p];
              const d = (r - pr) ** 2 + (g - pg) ** 2 + (b - pb) ** 2;
              if (d < bestD) { bestD = d; best = p; }
            }
            out.data[i] = paletteRGB[best][0]; out.data[i + 1] = paletteRGB[best][1];
            out.data[i + 2] = paletteRGB[best][2]; out.data[i + 3] = 255;
          }
        }

        // Mask-aware core processing locks background cells to one exact
        // color. Preserve those same cells in the edited approximation by
        // sampling the authoritative Snapped frame at each cell center.
        if (st.backgroundHex && snapped) {
          const [br, bg, bb] = hexToRgb(st.backgroundHex);
          const snappedData = snapped.canvas.getContext("2d", { willReadFrequently: true })
            .getImageData(0, 0, snapped.w, snapped.h).data;
          for (let cy = 0; cy < h; cy++) {
            const sy = Math.min(snapped.h - 1, Math.floor((cy + 0.5) * snapped.h / h));
            for (let cx = 0; cx < w; cx++) {
              const sx = Math.min(snapped.w - 1, Math.floor((cx + 0.5) * snapped.w / w));
              const si = (sy * snapped.w + sx) * 4;
              if (Math.abs(snappedData[si] - br) <= 1 &&
                  Math.abs(snappedData[si + 1] - bg) <= 1 &&
                  Math.abs(snappedData[si + 2] - bb) <= 1) {
                const oi = (cy * w + cx) * 4;
                out.data[oi] = br; out.data[oi + 1] = bg;
                out.data[oi + 2] = bb; out.data[oi + 3] = 255;
              }
            }
          }
        }

        const c = document.createElement("canvas");
        c.width = w; c.height = h;
        c.getContext("2d").putImageData(out, 0, 0);
        const result = { canvas: c, w, h };
        st.liveCache = { key: cacheKey, data: result };
        return result;
      }

      function currentSource(key) {
        if (key === "live") return computeLiveCanvas();
        return key === "raw" ? st.rawCache[st.frameIdx] : st.snappedCache[st.frameIdx];
      }

      // screen-space geometry for a box, given the shared pan/zoom view
      function geom(source, boxW, boxH) {
        const base = fitRect(source.w, source.h, boxW, boxH);
        const scale = base.scale * st.view.zoom;
        const x = boxW / 2 - (source.w / 2) * scale + st.view.panX;
        const y = boxH / 2 - (source.h / 2) * scale + st.view.panY;
        return { scale, x, y };
      }
      function screenToSourcePixel(key, offsetX, offsetY) {
        const source = currentSource(key);
        if (!source) return null;
        const canvas = boxOf(key);
        const g = geom(source, canvas.width, canvas.height);
        const sx = Math.floor((offsetX - g.x) / g.scale);
        const sy = Math.floor((offsetY - g.y) / g.scale);
        if (sx < 0 || sy < 0 || sx >= source.w || sy >= source.h) return null;
        return { source, sx, sy };
      }

      function redrawBox(key) {
        const canvas = boxOf(key);
        const boxW = canvas.clientWidth || 90, boxH = canvas.clientHeight || boxHeight.px;
        if (canvas.width !== boxW) canvas.width = boxW;
        if (canvas.height !== boxH) canvas.height = boxH;
        const ctx = canvas.getContext("2d");
        ctx.imageSmoothingEnabled = false;
        ctx.clearRect(0, 0, boxW, boxH);
        const source = currentSource(key);
        if (!source) return;
        const g = geom(source, boxW, boxH);
        ctx.drawImage(source.canvas, 0, 0, source.w, source.h, g.x, g.y, source.w * g.scale, source.h * g.scale);

        if (key === "raw" && st.measurePoints.length) {
          const toScreen = (p) => ({ x: g.x + p.sx * g.scale, y: g.y + p.sy * g.scale });
          const pts = st.measurePoints.map(toScreen);
          ctx.strokeStyle = "#e0a458"; ctx.fillStyle = "#e0a458"; ctx.lineWidth = 1.5;
          if (pts.length === 2) {
            ctx.beginPath(); ctx.moveTo(pts[0].x, pts[0].y); ctx.lineTo(pts[1].x, pts[1].y); ctx.stroke();
          }
          pts.forEach((p) => { ctx.beginPath(); ctx.arc(p.x, p.y, 3, 0, Math.PI * 2); ctx.fill(); });
        }
      }
      function redrawAll() { redrawBox("raw"); redrawBox("snapped"); redrawBox("live"); }

      // --- frame paging ---
      async function ensureFrameLoaded(i) {
        if (!st.snappedCache[i] && st.frameRefs[i]) {
          const img = await loadImage(refUrl(st.frameRefs[i]));
          const c = document.createElement("canvas");
          c.width = img.naturalWidth; c.height = img.naturalHeight;
          c.getContext("2d").drawImage(img, 0, 0);
          st.snappedCache[i] = { canvas: c, w: c.width, h: c.height };
        }
        if (!st.rawCache[i] && st.rawRefs[i]) {
          const img = await loadImage(refUrl(st.rawRefs[i]));
          const c = document.createElement("canvas");
          c.width = img.naturalWidth; c.height = img.naturalHeight;
          c.getContext("2d").drawImage(img, 0, 0);
          st.rawCache[i] = { canvas: c, w: c.width, h: c.height };
        }
      }
      async function goToFrame(i) {
        const n = st.frameRefs.length;
        if (n === 0) return;
        st.frameIdx = ((i % n) + n) % n;
        frameCounter.textContent = `${st.frameIdx + 1} / ${n}`;
        await ensureFrameLoaded(st.frameIdx);
        redrawAll();
      }
      wrap.querySelector(".vps-prev").addEventListener("click", () => goToFrame(st.frameIdx - 1));
      wrap.querySelector(".vps-next").addEventListener("click", () => goToFrame(st.frameIdx + 1));

      // --- preview height control (the node grows to fit, it's a normal DOM widget) ---
      function applyHeight() {
        canvases.forEach((c) => { c.style.height = boxHeight.px + "px"; });
        redrawAll();
      }
      wrap.querySelector(".vps-h-plus").addEventListener("click", () => {
        boxHeight.px = Math.min(400, boxHeight.px + 40);
        applyHeight();
      });
      wrap.querySelector(".vps-h-minus").addEventListener("click", () => {
        boxHeight.px = Math.max(60, boxHeight.px - 40);
        applyHeight();
      });

      // --- palette ---
      function renderPaletteRow() {
        paletteRow.innerHTML = "";
        paletteEmpty.style.display = st.livePalette.length ? "none" : "block";
        st.livePalette.forEach((hex, i) => {
          const sw = document.createElement("div");
          const selected = i === st.replaceSource;
          sw.style.cssText = `background:${hex};border-radius:2px;cursor:pointer;` +
            (selected
              ? `box-shadow:0 0 0 2px #0a0b0d, 0 0 0 4px #5fb3a3;`
              : `box-shadow:inset 0 0 0 1px rgba(255,255,255,0.15);`);
          sw.title = st.tool === "replace" ? `${hex} — click to select for replacement` : `${hex} — click to remove`;
          sw.addEventListener("click", () => {
            if (st.tool === "replace") {
              st.replaceSource = i;
              renderPaletteRow();
              updateReplaceApplyState();
              statusEl.textContent = `Replacing ${st.livePalette[i]} — click another color anywhere, or use manual entry + "Replace selected".`;
              return;
            }
            st.livePalette.splice(i, 1);
            st.paletteDirty = true;
            renderPaletteRow();
            redrawBox("live");
          });
          paletteRow.appendChild(sw);
        });
      }
      function addColorToPalette(value) {
        const hex = normalizeHex(value);
        if (!hex) {
          statusEl.textContent = `Invalid color "${value}" — use a six-digit value such as #ff8800.`;
          return;
        }
        if (!st.livePalette.some((h) => h.toLowerCase() === hex)) {
          st.livePalette.push(hex);
          st.paletteDirty = true;
          renderPaletteRow();
          redrawBox("live");
          // Best-effort hardening: if something in the host page's own
          // render cycle (e.g. ComfyUI's newer Vue-based "Nodes 2.0"
          // rendering, still beta) defers/batches DOM updates in a way
          // that leaves an imperative canvas draw visually stale until
          // the next reactive tick, forcing one more redraw on the next
          // animation frame should catch it. Harmless no-op otherwise.
          requestAnimationFrame(() => redrawBox("live"));
          statusEl.textContent = `Added ${hex} to the palette (${st.livePalette.length} colors).`;
        } else {
          statusEl.textContent = `${hex} is already in the palette.`;
        }
      }
      function removeNearestFromPalette(hex) {
        if (!st.livePalette.length) return;
        const [r, g, b] = hexToRgb(hex);
        let best = 0, bestD = Infinity;
        st.livePalette.forEach((h, i) => {
          const [pr, pg, pb] = hexToRgb(h);
          const d = (r - pr) ** 2 + (g - pg) ** 2 + (b - pb) ** 2;
          if (d < bestD) { bestD = d; best = i; }
        });
        const removed = st.livePalette[best];
        st.livePalette.splice(best, 1);
        st.paletteDirty = true;
        renderPaletteRow();
        redrawBox("live");
        statusEl.textContent = `Removed ${removed} from the palette (${st.livePalette.length} colors).`;
      }
      function nearestPaletteIndex(hex) {
        const [r, g, b] = hexToRgb(hex);
        let best = 0, bestD = Infinity;
        st.livePalette.forEach((h, i) => {
          const [pr, pg, pb] = hexToRgb(h);
          const d = (r - pr) ** 2 + (g - pg) ** 2 + (b - pb) ** 2;
          if (d < bestD) { bestD = d; best = i; }
        });
        return best;
      }
      function updateReplaceApplyState() { replaceApplyBtn.disabled = st.replaceSource == null; }

      // --- persistent "this color was intentionally replaced" memory ---
      // Deliberately NOT encoded into the exported palette PNG/metadata:
      // that would mean patching raw PNG chunk structure (real risk of a
      // subtle encoding bug) and, more importantly, the core node's
      // custom_palette loader just reads unique pixel colors from
      // whatever image it's given — any extra encoded pixels would leak
      // into the actual palette used for processing unless the Python
      // side were also taught to ignore them. localStorage keeps this
      // entirely on the editing side, with zero risk to the real
      // pipeline, at the cost of not traveling with the file itself.
      const SUBST_KEY = "vps_color_substitutions_v1";
      function loadSubstitutions() {
        try { return JSON.parse(localStorage.getItem(SUBST_KEY) || "[]"); }
        catch (err) { return []; }
      }
      function saveSubstitutions(list) {
        try { localStorage.setItem(SUBST_KEY, JSON.stringify(list.slice(-200))); }
        catch (err) { log("failed to save substitution memory: " + (err?.message || err)); }
      }
      function recordSubstitution(fromHex, toHex) {
        const list = loadSubstitutions();
        const i = list.findIndex((r) => r.from === fromHex);
        const entry = { from: fromHex, to: toHex, ts: Date.now() };
        if (i >= 0) list[i] = entry; else list.push(entry);
        saveSubstitutions(list);
      }
      function applySavedSubstitutions() {
        const rules = loadSubstitutions();
        if (!rules.length) { statusEl.textContent = "No saved substitutions yet — use Replace at least once first."; return; }
        const THRESHOLD = 40; // RGB-space distance; a rough "close enough to be the same intended color" cutoff
        let applied = 0;
        rules.forEach((rule) => {
          const [fr, fg, fb] = hexToRgb(rule.from);
          let best = -1, bestD = Infinity;
          st.livePalette.forEach((h, i) => {
            const [pr, pg, pb] = hexToRgb(h);
            const d = Math.hypot(fr - pr, fg - pg, fb - pb);
            if (d < bestD) { bestD = d; best = i; }
          });
          if (best >= 0 && bestD <= THRESHOLD) { st.livePalette[best] = rule.to; applied++; }
        });
        if (applied) st.paletteDirty = true;
        renderPaletteRow();
        redrawBox("live");
        statusEl.textContent = `Applied ${applied} of ${rules.length} saved substitution(s) ` +
          `(others had no close-enough match in the current palette — that's expected on a very different scene).`;
      }
      wrap.querySelector(".vps-apply-saved-subst").addEventListener("click", applySavedSubstitutions);
      wrap.querySelector(".vps-clear-saved-subst").addEventListener("click", () => {
        saveSubstitutions([]);
        statusEl.textContent = "Forgot all saved color substitutions.";
      });

      function applyReplace(value) {
        if (st.replaceSource == null || st.replaceSource >= st.livePalette.length) return;
        const targetHex = normalizeHex(value);
        if (!targetHex) {
          statusEl.textContent = `Invalid replacement color "${value}" — use #rrggbb.`;
          return;
        }
        const idx = st.replaceSource;
        const oldHex = st.livePalette[idx];
        st.livePalette[idx] = targetHex;
        st.paletteDirty = true;
        st.replaceSource = null;
        recordSubstitution(oldHex, targetHex);
        renderPaletteRow();
        updateReplaceApplyState();
        redrawBox("live");
        statusEl.textContent = `Replaced ${oldHex} → ${targetHex}. Remembered — "Apply saved substitutions" will try to reapply this on future palettes.`;
      }
      function handleReplaceClick(hex) {
        if (!st.livePalette.length) return;
        if (st.replaceSource == null) {
          st.replaceSource = nearestPaletteIndex(hex);
          renderPaletteRow();
          updateReplaceApplyState();
          statusEl.textContent = `Replacing ${st.livePalette[st.replaceSource]} — click another color anywhere, or use manual entry + "Replace selected".`;
        } else {
          applyReplace(hex);
        }
      }
      replaceApplyBtn.addEventListener("click", () => applyReplace(hexInput.value));

      async function loadPaletteFromRef(ref) {
        const img = await loadImage(refUrl(ref));
        const c = document.createElement("canvas");
        c.width = img.naturalWidth; c.height = img.naturalHeight;
        const ctx = c.getContext("2d", { willReadFrequently: true });
        ctx.drawImage(img, 0, 0);
        const y = Math.floor(c.height / 2);
        const row = ctx.getImageData(0, y, c.width, 1).data;

        // palette_preview is a horizontal strip of exact flat swatches.
        // Scan the full row and keep first-seen order. The old every-Nth-
        // pixel sampler could skip an entire 32 px swatch once the strip
        // was wide, and its hard 64-color cap contradicted the core node's
        // documented 256-color limit.
        const colors = [];
        const seen = new Set();
        for (let x = 0; x < c.width; x++) {
          const i = x * 4;
          if (row[i + 3] < 8) continue;
          const hex = rgbToHex(row[i], row[i + 1], row[i + 2]);
          if (!seen.has(hex)) {
            seen.add(hex);
            colors.push(hex);
            if (colors.length >= 256) break;
          }
        }
        return colors;
      }

      // --- measure: two clicks on "Original" -> pixel_size = distance / target cells ---
      function computeMeasurement() {
        if (st.measurePoints.length < 2) return null;
        const [a, b] = st.measurePoints;
        const dist = Math.hypot(b.sx - a.sx, b.sy - a.sy);
        const target = Math.max(1, parseFloat(measureTargetInput.value) || 1);
        const pixelSize = dist / target;
        const rounded = Math.max(1, Math.round(pixelSize));
        const actualCells = dist / rounded;
        return { dist, target, pixelSize, rounded, actualCells };
      }
      function updateMeasureInfo() {
        const m = computeMeasurement();
        if (!m) {
          measureInfo.textContent = st.measurePoints.length === 1
            ? "Click a second point on Original."
            : "Click two points on Original to measure a distance (e.g. top and bottom of a character).";
          measureApplyBtn.disabled = true;
          return;
        }
        measureInfo.textContent =
          `Distance: ${m.dist.toFixed(1)}px, target ${m.target} cells → pixel_size ≈ ${m.pixelSize.toFixed(2)}. ` +
          `The node rounds this to a whole pixel on run (→ ${m.rounded}), giving ≈${m.actualCells.toFixed(1)} cells ` +
          `instead of exactly ${m.target} — pixel-art grids can't use fractional block sizes, so this is as close as it gets.`;
        measureApplyBtn.disabled = false;
      }
      measureTargetInput.addEventListener("input", updateMeasureInfo);
      // The pixel_size widget lives on the CORE VideoPixelSnapper node,
      // not on this editor node — walk this node's inputs to find it.
      // node.getInputNode is a standard LiteGraph graph-traversal API;
      // wrapped defensively since it's not something testable from here.
      function findCoreNode() {
        try {
          if (!node.inputs) return null;
          for (let i = 0; i < node.inputs.length; i++) {
            const upstream = node.getInputNode ? node.getInputNode(i) : null;
            if (upstream && (upstream.comfyClass === "VideoPixelSnapper" ||
                              upstream.type === "VideoPixelSnapper")) {
              return upstream;
            }
          }
        } catch (err) {
          log(`findCoreNode failed: ${err?.message || err}`);
        }
        return null;
      }

      wrap.querySelector(".vps-measure-clear").addEventListener("click", () => {
        st.measurePoints = [];
        updateMeasureInfo();
        redrawBox("raw");
      });
      measureApplyBtn.addEventListener("click", () => {
        const m = computeMeasurement();
        if (!m) return;
        const core = findCoreNode();
        const w = core?.widgets?.find((x) => x.name === "pixel_size");
        if (w) {
          w.value = m.rounded;
          w.callback?.(w.value, appRef.canvas, core, undefined, undefined);
          core.setDirtyCanvas?.(true, true);
          statusEl.textContent = `pixel_size on "${core.title || "Video Pixel Snapper"}" set to ${m.rounded}. Re-run to apply.`;
        } else {
          statusEl.textContent = `Couldn't find a connected Video Pixel Snapper node to set pixel_size on ` +
            `automatically (this only works if this editor's inputs trace back to it directly) — set it to ` +
            `${m.rounded} manually instead.`;
        }
      });

      // --- tools: Pick / Delete / Replace / Measure ---
      function setTool(tool) {
        st.tool = st.tool === tool ? null : tool;
        if (st.tool !== "replace" && st.replaceSource != null) {
          st.replaceSource = null;
          updateReplaceApplyState();
        }
        measurePanel.style.display = st.tool === "measure" ? "flex" : "none";
        if (st.tool === "measure") updateMeasureInfo();
        wrap.querySelectorAll(".vps-tool").forEach((b) => {
          b.style.background = b.dataset.tool === st.tool ? "#5fb3a3" : "";
          b.style.color = b.dataset.tool === st.tool ? "#0d1a17" : "";
        });
        const active = !!st.tool;
        const cursorFor = { delete: "not-allowed", replace: "crosshair", measure: "crosshair" };
        canvases.forEach((c) => { c.style.cursor = active ? (cursorFor[st.tool] || "copy") : "grab"; });
        pickSwatch.style.display = (active && st.tool !== "measure") ? "block" : "none";
        renderPaletteRow();
      }
      wrap.querySelectorAll(".vps-tool").forEach((b) => b.addEventListener("click", () => setTool(b.dataset.tool)));

      // --- per-canvas interaction: hover preview, click (tool action),
      //     wheel zoom, middle-mouse pan, double-click reset ---
      canvases.forEach((canvas) => {
        const key = canvas.dataset.src;

        canvas.addEventListener("mousemove", (e) => {
          if (!st.tool || st.tool === "measure") return;
          const hit = screenToSourcePixel(key, e.offsetX, e.offsetY);
          if (!hit) return;
          const d = hit.source.canvas.getContext("2d").getImageData(hit.sx, hit.sy, 1, 1).data;
          pickSwatch.style.background = rgbToHex(d[0], d[1], d[2]);
        });

        canvas.addEventListener("click", (e) => {
          if (!st.tool) return;
          const hit = screenToSourcePixel(key, e.offsetX, e.offsetY);
          if (!hit) return;

          if (st.tool === "measure") {
            if (key !== "raw") {
              statusEl.textContent = "Measure only works on the Original preview (it needs true source-pixel coordinates).";
              return;
            }
            st.measurePoints.push({ sx: hit.sx, sy: hit.sy });
            if (st.measurePoints.length > 2) st.measurePoints.shift();
            updateMeasureInfo();
            redrawBox("raw");
            return;
          }

          const d = hit.source.canvas.getContext("2d").getImageData(hit.sx, hit.sy, 1, 1).data;
          const hex = rgbToHex(d[0], d[1], d[2]);
          if (st.tool === "pick") addColorToPalette(hex);
          else if (st.tool === "delete") removeNearestFromPalette(hex);
          else if (st.tool === "replace") handleReplaceClick(hex);
        });

        canvas.addEventListener("wheel", (e) => {
          e.preventDefault(); e.stopPropagation();
          const source = currentSource(key);
          if (!source) return;
          const g0 = geom(source, canvas.width, canvas.height);
          const srcX = (e.offsetX - g0.x) / g0.scale, srcY = (e.offsetY - g0.y) / g0.scale;
          const dir = e.deltaY < 0 ? 1.15 : 1 / 1.15;
          st.view.zoom = Math.min(20, Math.max(1, st.view.zoom * dir));
          const base = fitRect(source.w, source.h, canvas.width, canvas.height);
          const newScale = base.scale * st.view.zoom;
          st.view.panX = e.offsetX - canvas.width / 2 + (source.w / 2) * newScale - srcX * newScale;
          st.view.panY = e.offsetY - canvas.height / 2 + (source.h / 2) * newScale - srcY * newScale;
          redrawAll();
        }, { passive: false });

        canvas.addEventListener("dblclick", () => {
          st.view.zoom = 1; st.view.panX = 0; st.view.panY = 0;
          redrawAll();
        });
      });

      // middle-mouse pan, shared across all 3 boxes
      let panning = false, panStart = null;
      const previewsRow = wrap.querySelector(".vps-previews");
      previewsRow.addEventListener("pointerdown", (e) => {
        if (e.button !== 1) return;
        e.preventDefault();
        panning = true;
        panStart = { x: e.clientX, y: e.clientY, panX: st.view.panX, panY: st.view.panY };
      });
      window.addEventListener("pointermove", (e) => {
        if (!panning) return;
        st.view.panX = panStart.panX + (e.clientX - panStart.x);
        st.view.panY = panStart.panY + (e.clientY - panStart.y);
        redrawAll();
      });
      window.addEventListener("pointerup", () => { panning = false; });

      // --- manual color entry ---
      colorPick.addEventListener("input", () => { hexInput.value = colorPick.value; });
      hexInput.addEventListener("change", () => {
        if (/^#[0-9a-fA-F]{6}$/.test(hexInput.value)) colorPick.value = hexInput.value;
      });
      wrap.querySelector(".vps-addcolor").addEventListener("click", () => addColorToPalette(hexInput.value));

      wrap.querySelector(".vps-reset").addEventListener("click", () => {
        st.livePalette = st.originalPalette.slice();
        st.paletteDirty = false;
        st.replaceSource = null;
        updateReplaceApplyState();
        renderPaletteRow();
        redrawBox("live");
        statusEl.textContent = "Palette reset to the auto-detected one.";
      });

      wrap.querySelector(".vps-save").addEventListener("click", async () => {
        if (!st.livePalette.length) return;
        const swatch = 32;
        const c = document.createElement("canvas");
        c.width = swatch * st.livePalette.length; c.height = swatch;
        const ctx = c.getContext("2d");
        st.livePalette.forEach((hex, i) => { ctx.fillStyle = hex; ctx.fillRect(i * swatch, 0, swatch, swatch); });
        const blob = await new Promise((res) => c.toBlob(res, "image/png"));
        const filename = (wrap.querySelector(".vps-filename").value || `palette_${uid}`).replace(/[^\w\-]/g, "_") + ".png";
        const subfolder = (wrap.querySelector(".vps-subfolder").value || "").replace(/[^\w\-\/]/g, "_");
        const form = new FormData();
        form.append("image", blob, filename);
        form.append("type", "input");
        if (subfolder) form.append("subfolder", subfolder);
        form.append("overwrite", "true");
        try {
          const resp = await api.fetchApi("/upload/image", { method: "POST", body: form });
          if (resp.ok) {
            const data = await resp.json();
            const shown = data.subfolder ? `${data.subfolder}/${data.name}` : data.name;
            statusEl.textContent = `Saved as "${shown}" in input/ — pick it in LoadImage → custom_palette and re-run the graph.`;
          } else {
            statusEl.textContent = `Upload failed (HTTP ${resp.status}).`;
          }
        } catch (err) {
          statusEl.textContent = "Upload failed: " + (err?.message || err);
        }
      });

      new ResizeObserver(() => redrawAll()).observe(wrap);

      node._vpsInternal = { goToFrame, loadPaletteFromRef, redrawAll, renderPaletteRow, statusEl, frameCounter };
    };

    const origOnExecuted = nodeType.prototype.onExecuted;
    nodeType.prototype.onExecuted = function (message) {
      origOnExecuted?.apply(this, arguments);
      const node = this;
      const internal = node._vpsInternal;
      if (!internal) { log(`WARNING: onExecuted before widget existed for node ${node.id}`); return; }
      (async () => {
        try {
          const raw = message?.vps_raw || [];
          const frames = message?.vps_frames || [];
          const palette = message?.vps_palette;
          // [block, phaseX, phaseY, cellsW?, cellsH?, scale?], only
          // present when the editor's optional `info` input is connected.
          const gridMsg = message?.vps_grid;
          const backgroundMsg = message?.vps_background;
          if (!frames.length || !palette?.length) {
            internal.statusEl.textContent =
              "No preview in the node's output — open ComfyUI's Logs panel (Ctrl+`) and look for " +
              "'[VideoPixelSnapper] preview save failed'.";
            log(`node ${node.id}: no vps_frames/vps_palette in message`);
            return;
          }
          node._vps.rawRefs = raw;
          node._vps.frameRefs = frames;
          node._vps.frameIdx = 0;
          node._vps.rawCache = {};
          node._vps.snappedCache = {};
          node._vps.rawReducedCache = {};
          node._vps.grid = Array.isArray(gridMsg) ? {
            block: gridMsg[0], phaseX: gridMsg[1], phaseY: gridMsg[2],
            cellsW: gridMsg[3] || null, cellsH: gridMsg[4] || null,
            scale: gridMsg[5] || 1,
          } : null;
          node._vps.backgroundHex = Array.isArray(backgroundMsg)
            ? (backgroundMsg[0] || null)
            : (backgroundMsg || null);
          node._vps.liveCache = { key: null, data: null };
          node._vps.measurePoints = [];
          node._vps.view = { zoom: 1, panX: 0, panY: 0 };
          node._vps.originalPalette = await internal.loadPaletteFromRef(palette[0]);
          // Refresh an untouched working palette when k_colors, seed, the
          // source clip, or custom_palette changes. Once the user edits it,
          // preserve those edits across incidental ComfyUI re-executions;
          // Reset explicitly accepts the newest auto palette again.
          if (!node._vps.paletteDirty || !node._vps.livePalette.length) {
            node._vps.livePalette = node._vps.originalPalette.slice();
            node._vps.paletteDirty = false;
          }
          internal.renderPaletteRow();
          await internal.goToFrame(0);
          internal.statusEl.textContent = `${node._vps.livePalette.length} colors, ${frames.length} frame(s). ` +
            `Live exactly matches Snapped until the palette colors change. ` +
            `Pick/Delete/Replace/Measure — click any preview.`;
          log(`node ${node.id}: preview loaded OK (${node._vps.livePalette.length} colors, ${frames.length} frames)`);
        } catch (err) {
          internal.statusEl.textContent = "Preview load failed: " + (err?.message || err);
          log(`node ${node.id}: ERROR — ${err?.message || err}`);
        }
      })();
    };
  },
});
