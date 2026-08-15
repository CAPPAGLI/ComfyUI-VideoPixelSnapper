/**
 * Frame Retimer widget.
 *
 * Everything here operates on the already-loaded input-frame thumbnails
 * client-side (reorder/repeat/drop), so the frame strip, output count,
 * and play/pause preview all update instantly without re-running the
 * graph. The `durations_json` STRING widget is what gets written for
 * the actual node execution to read.
 *
 * The duration array always covers every real input frame (from
 * `vps_total_frames`), not just however many thumbnails were loaded
 * (`max_preview_frames` can cap that for performance) — frames beyond
 * the thumbnail cap have no strip cell but are still editable via the
 * curve and still get written correctly into durations_json.
 */
import { app } from "../../scripts/app.js";

function log(line) { console.log("[FrameRetimer] " + line); }

function refUrl(ref) {
  const params = new URLSearchParams({ filename: ref.filename, subfolder: ref.subfolder || "", type: ref.type || "temp" });
  return "/view?" + params.toString();
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

app.registerExtension({
  name: "VideoPixelSnapper.FrameRetimer",

  async beforeRegisterNodeDef(nodeType, nodeData, appRef) {
    if (nodeType.comfyClass !== "VideoPixelSnapperFrameRetimer" && nodeData?.name !== "VideoPixelSnapperFrameRetimer") return;
    log("node def matched — extension is active");

    const origOnNodeCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      origOnNodeCreated?.apply(this, arguments);
      const node = this;
      const uid = node.id;

      // Hide the raw JSON widget from the node body — it's internal
      // state the JS writes to, not something meant to be read/typed.
      // Best-effort: if your ComfyUI frontend doesn't respect
      // computeSize overrides this way, worst case it just shows as a
      // normal (small, single-line — no more giant textarea) field.
      const hideRawWidget = () => {
        const w = node.widgets?.find((x) => x.name === "durations_json");
        if (w) {
          w.computeSize = () => [0, -4];
          w.draw = () => {};
        }
      };
      hideRawWidget();

      const wrap = document.createElement("div");
      wrap.style.cssText =
        "display:flex;flex-direction:column;gap:6px;padding:8px;background:#1b1e24;" +
        "border-radius:6px;min-width:320px;font-family:monospace;";
      wrap.innerHTML = `
        <div style="font-size:10px;color:#8b909c;text-transform:uppercase;letter-spacing:.05em;">
          Frame Retimer
        </div>

        <div style="font-size:9px;color:#8b909c;text-transform:uppercase;letter-spacing:.05em;">
          Frames &amp; duration curve — drag points, click empty space to add a key, double-click a point to remove it
        </div>
        <div class="fr-strip-aligned" style="position:relative;height:28px;flex-shrink:0;"></div>
        <canvas class="fr-curve" style="width:100%;height:90px;display:block;background:#111;border-radius:4px 4px 0 0;touch-action:none;cursor:crosshair;"></canvas>
        <div style="display:flex;gap:4px;flex-shrink:0;margin-top:-4px;">
          <button class="fr-curve-reset" style="flex:1;font-size:10px;padding:4px;">Reset curve</button>
        </div>

        <div style="display:flex;gap:6px;align-items:center;flex-shrink:0;border-top:1px solid #2c313a;padding-top:6px;margin-top:2px;">
          <div class="fr-selected-thumb" style="width:32px;height:32px;background:#111;border-radius:3px;
               background-size:cover;background-position:center;flex-shrink:0;"></div>
          <span class="fr-selected-label" style="font-size:10px;color:#8b909c;flex:1;">No frame selected</span>
          <span style="font-size:10px;color:#8b909c;">Duration:</span>
          <input type="number" class="fr-duration-input" min="0" step="1" value="1"
                 style="width:48px;font-size:10px;padding:3px;background:#111;color:#eee;border:1px solid #333;border-radius:3px;">
        </div>
        <div style="display:flex;gap:4px;flex-shrink:0;">
          <button class="fr-dup" title="+1 (hold one tick longer)" style="flex:1;font-size:10px;padding:4px;">Duplicate</button>
          <button class="fr-del" title="Set duration to 0 (drop this frame)" style="flex:1;font-size:10px;padding:4px;">Delete</button>
          <button class="fr-restore" title="Reset to 1" style="flex:1;font-size:10px;padding:4px;">Restore</button>
        </div>

        <div style="display:flex;justify-content:space-between;align-items:center;flex-shrink:0;margin-top:4px;">
          <span class="fr-output-count" style="font-size:10px;color:#8b909c;">—</span>
          <div style="display:flex;gap:4px;align-items:center;">
            <span style="font-size:9px;color:#8b909c;">fps</span>
            <input type="number" class="fr-fps" min="1" max="60" value="12" style="width:40px;font-size:10px;padding:2px;background:#111;color:#eee;border:1px solid #333;border-radius:3px;">
            <button class="fr-play" style="font-size:10px;padding:4px 10px;">▶ Preview</button>
          </div>
        </div>
        <canvas class="fr-preview" style="width:100%;height:90px;display:block;background:#111;border-radius:4px;image-rendering:pixelated;"></canvas>

        <div class="fr-status" style="font-size:9px;color:#8b909c;line-height:1.4;flex-shrink:0;">
          Run the node once to load frame thumbnails.
        </div>
      `;
      node.addDOMWidget(`fr_editor_${uid}`, "fr_editor", wrap, {});

      const st = {
        frameRefs: [],          // thumbnail refs (may be capped by max_preview_frames)
        thumbs: [],             // Image objects, index-aligned with frameRefs
        totalFrames: 0,         // TRUE total input frame count (from vps_total_frames)
        durations: [],          // length == totalFrames
        selected: -1,
        curveKeys: [{ x: 0, y: 1 }, { x: 1, y: 1 }],
        curveMaxY: 4,
        draggingKey: null,
        playing: false,
        playTimer: null,
      };

      const stripAligned = wrap.querySelector(".fr-strip-aligned");
      const selThumb = wrap.querySelector(".fr-selected-thumb");
      const selLabel = wrap.querySelector(".fr-selected-label");
      const durationInput = wrap.querySelector(".fr-duration-input");
      const statusEl = wrap.querySelector(".fr-status");
      const outputCountEl = wrap.querySelector(".fr-output-count");
      const curveCanvas = wrap.querySelector(".fr-curve");
      const previewCanvas = wrap.querySelector(".fr-preview");
      const fpsInput = wrap.querySelector(".fr-fps");
      const playBtn = wrap.querySelector(".fr-play");

      function durationsWidget() { return node.widgets?.find((w) => w.name === "durations_json"); }
      function syncWidget() {
        const w = durationsWidget();
        if (w) {
          w.value = JSON.stringify(st.durations);
          w.callback?.(w.value, appRef.canvas, node, undefined, undefined);
        }
        node.setDirtyCanvas?.(true, true);
      }
      function updateOutputCount() {
        const total = st.durations.reduce((a, b) => a + b, 0);
        outputCountEl.textContent = `${st.totalFrames} in -> ${total} out`;
      }

      // --- geometry shared between the curve canvas and the aligned strip ---
      function curveGeom() {
        const w = curveCanvas.width, h = curveCanvas.height;
        const pad = 10;
        return {
          w, h, pad,
          toScreen: (x, y) => ({ sx: pad + x * (w - 2 * pad), sy: h - pad - (y / st.curveMaxY) * (h - 2 * pad) }),
          fromScreen: (sx, sy) => ({
            x: Math.min(1, Math.max(0, (sx - pad) / (w - 2 * pad))),
            y: Math.min(st.curveMaxY, Math.max(0, ((h - pad - sy) / (h - 2 * pad)) * st.curveMaxY)),
          }),
        };
      }
      // x-position (CSS px, matching the curve canvas's displayed width)
      // for frame index i out of totalFrames — used to align the strip.
      function frameX(i) {
        const boxW = curveCanvas.clientWidth || 300;
        const pad = 10;
        const x = st.totalFrames <= 1 ? 0 : i / (st.totalFrames - 1);
        return pad + x * (boxW - 2 * pad);
      }

      function renderStrip() {
        stripAligned.innerHTML = "";
        const boxW = curveCanvas.clientWidth || 300;
        stripAligned.style.width = boxW + "px";
        st.frameRefs.forEach((ref, i) => {
          const cell = document.createElement("div");
          const sx = frameX(i);
          cell.style.cssText = `position:absolute;left:${sx}px;top:0;transform:translateX(-50%);` +
            "width:22px;height:22px;cursor:pointer;border-radius:3px;" +
            `box-shadow:${i === st.selected ? "0 0 0 2px #0a0b0d, 0 0 0 3px #5fb3a3" : "inset 0 0 0 1px rgba(255,255,255,0.15)"};`;
          const img = st.thumbs[i];
          if (img) {
            cell.style.backgroundImage = `url(${img.src})`;
            cell.style.backgroundSize = "cover";
            cell.style.backgroundPosition = "center";
          } else {
            cell.style.background = "#222";
          }
          const d = st.durations[i] ?? 1;
          if (d === 0) cell.style.opacity = "0.35";
          cell.title = `Frame ${i + 1}: ×${d}`;
          cell.addEventListener("click", () => selectFrame(i));
          stripAligned.appendChild(cell);
        });
        updateOutputCount();
      }

      function selectFrame(i) {
        st.selected = i;
        const img = st.thumbs[i];
        selThumb.style.backgroundImage = img ? `url(${img.src})` : "";
        selLabel.textContent = img ? `Frame ${i + 1} / ${st.totalFrames}` : `Frame ${i + 1} / ${st.totalFrames} (no thumbnail — beyond max_preview_frames)`;
        durationInput.value = st.durations[i] ?? 1;
        renderStrip();
      }

      function setSelectedDuration(v) {
        if (st.selected < 0) return;
        st.durations[st.selected] = Math.max(0, Math.round(v));
        durationInput.value = st.durations[st.selected];
        renderStrip();
        syncWidget();
      }
      durationInput.addEventListener("change", () => setSelectedDuration(parseInt(durationInput.value, 10) || 0));
      wrap.querySelector(".fr-dup").addEventListener("click", () => setSelectedDuration((st.durations[st.selected] ?? 1) + 1));
      wrap.querySelector(".fr-del").addEventListener("click", () => setSelectedDuration(0));
      wrap.querySelector(".fr-restore").addEventListener("click", () => setSelectedDuration(1));

      // --- duration curve ---
      function curveValueAt(x) {
        const keys = st.curveKeys.slice().sort((a, b) => a.x - b.x);
        if (x <= keys[0].x) return keys[0].y;
        if (x >= keys[keys.length - 1].x) return keys[keys.length - 1].y;
        for (let i = 0; i < keys.length - 1; i++) {
          const a = keys[i], b = keys[i + 1];
          if (x >= a.x && x <= b.x) {
            const t = b.x === a.x ? 0 : (x - a.x) / (b.x - a.x);
            return a.y + (b.y - a.y) * t;
          }
        }
        return 1;
      }

      function applyCurveToAllFrames() {
        const n = st.totalFrames;
        if (!n) return;
        for (let i = 0; i < n; i++) {
          const x = n === 1 ? 0 : i / (n - 1);
          st.durations[i] = Math.max(0, Math.round(curveValueAt(x)));
        }
        renderStrip();
        if (st.selected >= 0) durationInput.value = st.durations[st.selected];
        syncWidget();
      }

      function renderCurve() {
        const boxW = curveCanvas.clientWidth || 300, boxH = curveCanvas.clientHeight || 90;
        curveCanvas.width = boxW; curveCanvas.height = boxH;
        const ctx = curveCanvas.getContext("2d");
        ctx.clearRect(0, 0, boxW, boxH);
        const g = curveGeom();

        const ref1 = g.toScreen(0, 1);
        ctx.strokeStyle = "#2c313a"; ctx.lineWidth = 1;
        ctx.beginPath(); ctx.moveTo(g.pad, ref1.sy); ctx.lineTo(boxW - g.pad, ref1.sy); ctx.stroke();

        ctx.strokeStyle = "#5fb3a3"; ctx.lineWidth = 1.5;
        ctx.beginPath();
        for (let i = 0; i <= 100; i++) {
          const x = i / 100;
          const p = g.toScreen(x, curveValueAt(x));
          if (i === 0) ctx.moveTo(p.sx, p.sy); else ctx.lineTo(p.sx, p.sy);
        }
        ctx.stroke();

        st.curveKeys.forEach((k) => {
          const p = g.toScreen(k.x, k.y);
          ctx.fillStyle = "#e0a458";
          ctx.beginPath(); ctx.arc(p.sx, p.sy, 4, 0, Math.PI * 2); ctx.fill();
        });
        renderStrip(); // keep strip x-alignment in sync with the curve's current width
      }

      function keyAtScreen(sx, sy) {
        const g = curveGeom();
        for (const k of st.curveKeys) {
          const p = g.toScreen(k.x, k.y);
          if (Math.hypot(p.sx - sx, p.sy - sy) < 8) return k;
        }
        return null;
      }
      function eventToCanvasXY(e) {
        const rect = curveCanvas.getBoundingClientRect();
        return {
          sx: (e.clientX - rect.left) * (curveCanvas.width / rect.width),
          sy: (e.clientY - rect.top) * (curveCanvas.height / rect.height),
        };
      }

      curveCanvas.addEventListener("pointerdown", (e) => {
        const { sx, sy } = eventToCanvasXY(e);
        const hit = keyAtScreen(sx, sy);
        if (hit) { st.draggingKey = hit; return; }
        const g = curveGeom();
        const { x, y } = g.fromScreen(sx, sy);
        const nk = { x, y };
        st.curveKeys.push(nk);
        st.draggingKey = nk;
        renderCurve();
        applyCurveToAllFrames(); // live-apply: adding a key already changes the shape
      });
      curveCanvas.addEventListener("dblclick", (e) => {
        const { sx, sy } = eventToCanvasXY(e);
        const hit = keyAtScreen(sx, sy);
        if (hit && st.curveKeys.length > 2) {
          st.curveKeys = st.curveKeys.filter((k) => k !== hit);
          renderCurve();
          applyCurveToAllFrames();
        }
      });
      window.addEventListener("pointermove", (e) => {
        if (!st.draggingKey) return;
        const { sx, sy } = eventToCanvasXY(e);
        const g = curveGeom();
        const { x, y } = g.fromScreen(sx, sy);
        st.draggingKey.x = x; st.draggingKey.y = y;
        renderCurve();
        applyCurveToAllFrames(); // live-apply while dragging, per request — no separate "Apply" click needed
      });
      window.addEventListener("pointerup", () => { st.draggingKey = null; });
      new ResizeObserver(() => renderCurve()).observe(curveCanvas);

      wrap.querySelector(".fr-curve-reset").addEventListener("click", () => {
        st.curveKeys = [{ x: 0, y: 1 }, { x: 1, y: 1 }];
        renderCurve();
        applyCurveToAllFrames();
      });

      // --- output preview ---
      function outputSequence() {
        const seq = [];
        st.durations.forEach((d, i) => { for (let r = 0; r < d; r++) seq.push(i); });
        return seq;
      }
      let previewIdx = 0;
      function drawPreviewFrame() {
        const seq = outputSequence();
        const ctx = previewCanvas.getContext("2d");
        const boxW = previewCanvas.clientWidth || 200, boxH = previewCanvas.clientHeight || 90;
        previewCanvas.width = boxW; previewCanvas.height = boxH;
        ctx.imageSmoothingEnabled = false;
        ctx.clearRect(0, 0, boxW, boxH);
        if (!seq.length) return;
        previewIdx = previewIdx % seq.length;
        const img = st.thumbs[seq[previewIdx]];
        if (!img) return; // frame beyond thumbnail cap -- nothing to draw, harmless
        const scale = Math.min(boxW / img.naturalWidth, boxH / img.naturalHeight);
        const w = img.naturalWidth * scale, h = img.naturalHeight * scale;
        ctx.drawImage(img, (boxW - w) / 2, (boxH - h) / 2, w, h);
      }
      function stopPlayback() {
        st.playing = false; playBtn.textContent = "▶ Preview";
        if (st.playTimer) clearInterval(st.playTimer);
        st.playTimer = null;
      }
      function startPlayback() {
        const seq = outputSequence();
        if (!seq.length) return;
        st.playing = true; playBtn.textContent = "⏸ Stop";
        const fps = Math.max(1, Math.min(60, parseInt(fpsInput.value, 10) || 12));
        st.playTimer = setInterval(() => { previewIdx++; drawPreviewFrame(); }, 1000 / fps);
      }
      playBtn.addEventListener("click", () => { st.playing ? stopPlayback() : startPlayback(); });
      fpsInput.addEventListener("change", () => { if (st.playing) { stopPlayback(); startPlayback(); } });

      node._frInternal = { st, renderStrip, renderCurve, selectFrame, drawPreviewFrame, statusEl, hideRawWidget };
    };

    const origOnExecuted = nodeType.prototype.onExecuted;
    nodeType.prototype.onExecuted = function (message) {
      origOnExecuted?.apply(this, arguments);
      const node = this;
      const internal = node._frInternal;
      if (!internal) { log(`WARNING: onExecuted before widget existed for node ${node.id}`); return; }
      (async () => {
        try {
          internal.hideRawWidget();
          const refs = message?.vps_retimer_frames || [];
          const total = message?.vps_total_frames?.[0] ?? refs.length;
          if (!refs.length) {
            internal.statusEl.textContent =
              "No thumbnails in the node's output — check ComfyUI's Logs panel (Ctrl+`) for a " +
              "'preview save failed' line.";
            return;
          }
          const st = internal.st;
          st.frameRefs = refs;
          st.totalFrames = total;
          st.thumbs = await Promise.all(refs.map((r) => loadImage(refUrl(r))));
          // keep existing edits if the TRUE total still matches (re-run after an edit);
          // otherwise reset to identity (1 each)
          if (st.durations.length !== total) {
            st.durations = new Array(total).fill(1);
          }
          internal.renderCurve();  // also renders the aligned strip
          internal.drawPreviewFrame();
          internal.selectFrame(0);
          const capNote = refs.length < total ? ` (only the first ${refs.length} have thumbnails — raise max_preview_frames to see more, or use the curve, which covers all ${total})` : "";
          internal.statusEl.textContent = `${total} frame(s) total${capNote}. Click a thumbnail to edit its duration, or use the curve above — it applies live as you drag.`;
          log(`node ${node.id}: loaded ${refs.length}/${total} frame thumbnails`);
        } catch (err) {
          internal.statusEl.textContent = "Thumbnail load failed: " + (err?.message || err);
          log(`node ${node.id}: ERROR — ${err?.message || err}`);
        }
      })();
    };
  },
});
