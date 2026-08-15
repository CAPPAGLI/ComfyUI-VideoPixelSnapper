/**
 * Frame Retimer widget.
 *
 * Two layers, composed:
 *  1. Pacing (duration curve, or a single frame's duration) — regenerates
 *     the whole output sequence from scratch: each source frame repeated
 *     N times, in original order. Good for broad timing/speed-ramping.
 *  2. Timeline — further edits whatever sequence pacing produced: drag
 *     to reorder entries, select a range + Duplicate to create a loop,
 *     Delete to cut. Touching pacing again regenerates the sequence and
 *     discards timeline edits — intentional, not a bug.
 *
 * Everything here operates on already-loaded input-frame thumbnails
 * client-side, so the whole preview updates instantly without
 * re-running the graph. `sequence_json` is what gets written for the
 * actual node execution to read.
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

      // Hide the raw JSON widget from the node body — internal state
      // the JS writes to. Best-effort (see video_pixel_snapper.js for
      // the same trick); worst case it just shows as a small field.
      const hideRawWidget = () => {
        const w = node.widgets?.find((x) => x.name === "sequence_json");
        if (w) { w.computeSize = () => [0, -4]; w.draw = () => {}; }
      };
      hideRawWidget();

      const wrap = document.createElement("div");
      wrap.style.cssText =
        "display:flex;flex-direction:column;gap:6px;padding:8px;background:#1b1e24;" +
        "border-radius:6px;min-width:320px;font-family:monospace;";
      wrap.innerHTML = `
        <div style="display:flex;justify-content:space-between;align-items:center;flex-shrink:0;">
          <div style="font-size:10px;color:#8b909c;text-transform:uppercase;letter-spacing:.05em;">
            Frame Retimer — Pacing
          </div>
          <div style="display:flex;gap:6px;align-items:center;">
            <span style="font-size:9px;color:#8b909c;">smooth</span>
            <input type="checkbox" class="fr-smooth-toggle">
            <span style="font-size:9px;color:#8b909c;">max</span>
            <input type="number" class="fr-maxy-input" min="1" step="1" value="4"
                   style="width:36px;font-size:10px;padding:2px;background:#111;color:#eee;border:1px solid #333;border-radius:3px;">
          </div>
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

        <div style="font-size:9px;color:#8b909c;text-transform:uppercase;letter-spacing:.05em;margin-top:4px;border-top:1px solid #2c313a;padding-top:6px;">
          Timeline — drag to reorder, click (shift-click for a range), Duplicate to loop, Delete to cut
        </div>
        <div class="fr-timeline" style="display:flex;gap:2px;overflow-x:auto;padding:4px;background:#111;border-radius:4px;min-height:32px;"></div>
        <div style="display:flex;gap:4px;flex-shrink:0;">
          <button class="fr-tl-duplicate" style="flex:1;font-size:10px;padding:4px;">Duplicate selection</button>
          <button class="fr-tl-delete" style="flex:1;font-size:10px;padding:4px;">Delete selection</button>
        </div>

        <div style="display:flex;justify-content:space-between;align-items:center;flex-shrink:0;margin-top:4px;">
          <span class="fr-output-count" style="font-size:10px;color:#8b909c;">—</span>
          <div style="display:flex;gap:4px;align-items:center;">
            <button class="fr-h-minus" title="Shrink preview" style="font-size:10px;padding:2px 6px;">−</button>
            <button class="fr-h-plus" title="Grow preview" style="font-size:10px;padding:2px 6px;">+</button>
            <span style="font-size:9px;color:#8b909c;">fps</span>
            <input type="number" class="fr-fps" min="1" max="60" value="12" style="width:40px;font-size:10px;padding:2px;background:#111;color:#eee;border:1px solid #333;border-radius:3px;">
            <button class="fr-play" style="font-size:10px;padding:4px 10px;">▶ Preview</button>
          </div>
        </div>
        <canvas class="fr-preview" style="width:100%;height:120px;display:block;background:#111;border-radius:4px;image-rendering:pixelated;"></canvas>

        <div class="fr-status" style="font-size:9px;color:#8b909c;line-height:1.4;flex-shrink:0;">
          Run the node once to load frame thumbnails.
        </div>
      `;
      node.addDOMWidget(`fr_editor_${uid}`, "fr_editor", wrap, {});

      const st = {
        frameRefs: [], thumbs: [], totalFrames: 0, sourceFps: 0,
        sequence: [],                 // canonical: array of source-frame indices, output order
        selected: -1,                 // selected SOURCE frame (pacing panel)
        curveKeys: [{ x: 0, y: 1 }, { x: 1, y: 1 }],
        curveMaxY: 4,
        smooth: false,
        draggingKey: null,
        tlSelection: null,            // {start,end} positions within st.sequence
        dragSrcPos: null,
        previewHeight: 120,
        playing: false, playTimer: null,
      };

      // Restore the sequence saved in the workflow. onNodeCreated can run
      // before ComfyUI applies widgets_values, so this is called both here
      // and again from onExecuted when state is still empty.
      function restoreSequenceFromWidget() {
        const widget = node.widgets?.find((x) => x.name === "sequence_json");
        if (!widget) return false;
        try {
          const parsed = Array.isArray(widget.value)
            ? widget.value
            : JSON.parse(widget.value || "[]");
          if (!Array.isArray(parsed) || !parsed.length) return false;
          const seq = parsed.map((v) => Number(v));
          if (!seq.every((v) => Number.isInteger(v))) return false;
          st.sequence = seq;
          return true;
        } catch (err) {
          log(`node ${node.id}: couldn't restore sequence_json — ${err?.message || err}`);
          return false;
        }
      }
      restoreSequenceFromWidget();

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
      const timelineEl = wrap.querySelector(".fr-timeline");
      const smoothToggle = wrap.querySelector(".fr-smooth-toggle");
      const maxYInput = wrap.querySelector(".fr-maxy-input");

      function fmtTime(frames) {
        if (!st.sourceFps || st.sourceFps <= 0) return "";
        return ` (~${(frames / st.sourceFps).toFixed(2)}s)`;
      }

      function syncSequence() {
        const w = node.widgets?.find((x) => x.name === "sequence_json");
        if (w) {
          w.value = JSON.stringify(st.sequence);
          w.callback?.(w.value, appRef.canvas, node, undefined, undefined);
        }
        node.setDirtyCanvas?.(true, true);
        renderTimeline();
        updateOutputCount();
      }
      function updateOutputCount() {
        outputCountEl.textContent = `${st.totalFrames} in -> ${st.sequence.length} out${fmtTime(st.sequence.length)}`;
      }

      // ---------------- geometry shared by curve + aligned strip ----------------
      function curveGeom() {
        const w = curveCanvas.width, h = curveCanvas.height, pad = 10;
        return {
          w, h, pad,
          toScreen: (x, y) => ({ sx: pad + x * (w - 2 * pad), sy: h - pad - (y / st.curveMaxY) * (h - 2 * pad) }),
          fromScreen: (sx, sy) => ({
            x: Math.min(1, Math.max(0, (sx - pad) / (w - 2 * pad))),
            y: Math.min(st.curveMaxY, Math.max(0, ((h - pad - sy) / (h - 2 * pad)) * st.curveMaxY)),
          }),
        };
      }
      function frameX(i) {
        const boxW = curveCanvas.clientWidth || 300, pad = 10;
        const x = st.totalFrames <= 1 ? 0 : i / (st.totalFrames - 1);
        return pad + x * (boxW - 2 * pad);
      }

      // count how many times source frame i currently appears in the sequence
      function occurrences(i) { return st.sequence.reduce((a, s) => a + (s === i ? 1 : 0), 0); }

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
          if (img) { cell.style.backgroundImage = `url(${img.src})`; cell.style.backgroundSize = "cover"; cell.style.backgroundPosition = "center"; }
          else cell.style.background = "#222";
          const d = occurrences(i);
          const badge = document.createElement("div");
          badge.style.cssText = "position:absolute;bottom:0;right:0;background:rgba(10,11,13,0.85);" +
            "color:#e8e6df;font-size:7px;padding:0 2px;border-radius:2px 0 0 0;line-height:1.3;";
          badge.textContent = "×" + d;
          if (d === 0) { badge.style.color = "#d9695f"; cell.style.opacity = "0.4"; }
          cell.appendChild(badge);
          cell.title = `Frame ${i + 1}: ${d} occurrence(s) in output`;
          cell.addEventListener("click", () => selectFrame(i));
          stripAligned.appendChild(cell);
        });
      }

      function selectFrame(i) {
        st.selected = i;
        const img = st.thumbs[i];
        selThumb.style.backgroundImage = img ? `url(${img.src})` : "";
        selLabel.textContent = `Frame ${i + 1} / ${st.totalFrames} — ${occurrences(i)}x in output`;
        durationInput.value = occurrences(i);
        renderStrip();
      }

      // Editing pacing (duration curve or a single frame's duration)
      // REGENERATES the whole sequence: each source frame repeated its
      // duration count, in original order. This intentionally discards
      // any prior timeline reordering/looping — see module docstring.
      function regenerateFromDurations(durations) {
        const seq = [];
        durations.forEach((d, i) => { for (let r = 0; r < d; r++) seq.push(i); });
        st.sequence = seq.length ? seq : Array.from({ length: st.totalFrames }, (_, i) => i);
        st.tlSelection = null;
        syncSequence();
      }
      function setSelectedDuration(v) {
        if (st.selected < 0) return;
        const durations = Array.from({ length: st.totalFrames }, (_, i) => occurrences(i));
        durations[st.selected] = Math.max(0, Math.round(v));
        durationInput.value = durations[st.selected];
        regenerateFromDurations(durations);
        renderStrip();
      }
      durationInput.addEventListener("change", () => setSelectedDuration(parseInt(durationInput.value, 10) || 0));
      wrap.querySelector(".fr-dup").addEventListener("click", () => setSelectedDuration(occurrences(st.selected) + 1));
      wrap.querySelector(".fr-del").addEventListener("click", () => setSelectedDuration(0));
      wrap.querySelector(".fr-restore").addEventListener("click", () => setSelectedDuration(1));

      // ---------------- duration curve ----------------
      function easeT(t) { return st.smooth ? t * t * (3 - 2 * t) : t; } // smoothstep vs linear
      function curveValueAt(x) {
        const keys = st.curveKeys.slice().sort((a, b) => a.x - b.x);
        if (x <= keys[0].x) return keys[0].y;
        if (x >= keys[keys.length - 1].x) return keys[keys.length - 1].y;
        for (let i = 0; i < keys.length - 1; i++) {
          const a = keys[i], b = keys[i + 1];
          if (x >= a.x && x <= b.x) {
            const t = b.x === a.x ? 0 : easeT((x - a.x) / (b.x - a.x));
            return a.y + (b.y - a.y) * t;
          }
        }
        return 1;
      }
      function applyCurveLive() {
        const n = st.totalFrames;
        if (!n) return;
        const durations = [];
        for (let i = 0; i < n; i++) {
          const x = n === 1 ? 0 : i / (n - 1);
          durations.push(Math.max(0, Math.round(curveValueAt(x))));
        }
        regenerateFromDurations(durations);
        if (st.selected >= 0) { durationInput.value = occurrences(st.selected); }
        renderStrip();
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
        renderStrip();
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
        return { sx: (e.clientX - rect.left) * (curveCanvas.width / rect.width), sy: (e.clientY - rect.top) * (curveCanvas.height / rect.height) };
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
        applyCurveLive();
      });
      curveCanvas.addEventListener("dblclick", (e) => {
        const { sx, sy } = eventToCanvasXY(e);
        const hit = keyAtScreen(sx, sy);
        if (hit && st.curveKeys.length > 2) {
          st.curveKeys = st.curveKeys.filter((k) => k !== hit);
          renderCurve(); applyCurveLive();
        }
      });
      window.addEventListener("pointermove", (e) => {
        if (!st.draggingKey) return;
        const { sx, sy } = eventToCanvasXY(e);
        const g = curveGeom();
        const { x, y } = g.fromScreen(sx, sy);
        st.draggingKey.x = x; st.draggingKey.y = y;
        renderCurve();
        applyCurveLive();
      });
      window.addEventListener("pointerup", () => { st.draggingKey = null; });
      new ResizeObserver(() => renderCurve()).observe(curveCanvas);

      wrap.querySelector(".fr-curve-reset").addEventListener("click", () => {
        st.curveKeys = [{ x: 0, y: 1 }, { x: 1, y: 1 }];
        renderCurve(); applyCurveLive();
      });
      smoothToggle.addEventListener("change", () => { st.smooth = smoothToggle.checked; renderCurve(); applyCurveLive(); });
      maxYInput.addEventListener("change", () => {
        const v = Math.max(1, parseFloat(maxYInput.value) || 4);
        st.curveMaxY = v;
        st.curveKeys.forEach((k) => { k.y = Math.min(k.y, v); }); // keep existing keys on-chart
        renderCurve(); applyCurveLive();
      });

      // ---------------- timeline: reorder + loop (range duplicate) + cut ----------------
      function renderTimeline() {
        timelineEl.innerHTML = "";
        st.sequence.forEach((srcIdx, pos) => {
          const block = document.createElement("div");
          block.draggable = true;
          const inSel = st.tlSelection && pos >= st.tlSelection.start && pos <= st.tlSelection.end;
          block.style.cssText = "width:22px;height:22px;flex:0 0 auto;border-radius:2px;cursor:grab;" +
            `box-shadow:${inSel ? "0 0 0 2px #0a0b0d, 0 0 0 3px #e0a458" : "inset 0 0 0 1px rgba(255,255,255,.12)"};`;
          const img = st.thumbs[srcIdx];
          if (img) { block.style.backgroundImage = `url(${img.src})`; block.style.backgroundSize = "cover"; block.style.backgroundPosition = "center"; }
          else block.style.background = "#222";
          block.title = `output pos ${pos + 1}: source frame ${srcIdx + 1}`;

          block.addEventListener("dragstart", (e) => { st.dragSrcPos = pos; e.dataTransfer.effectAllowed = "move"; });
          block.addEventListener("dragend", () => { st.dragSrcPos = null; });
          block.addEventListener("dragover", (e) => e.preventDefault());
          block.addEventListener("drop", (e) => {
            e.preventDefault();
            if (st.dragSrcPos === null) return;
            const srcPos = st.dragSrcPos;

            // Insert before/after the target according to which half was
            // dropped on. The previous pos-1 formula made dragging a block
            // onto its immediate right-hand neighbor a silent no-op.
            const rect = block.getBoundingClientRect();
            const afterTarget = e.clientX >= rect.left + rect.width / 2;
            let insertAt = pos + (afterTarget ? 1 : 0);
            const [moved] = st.sequence.splice(srcPos, 1);
            if (srcPos < insertAt) insertAt -= 1;
            insertAt = Math.max(0, Math.min(st.sequence.length, insertAt));
            st.sequence.splice(insertAt, 0, moved);
            st.dragSrcPos = null;
            st.tlSelection = null;
            syncSequence();
            statusEl.textContent = `Reordered — moved output position ${srcPos + 1} to ${insertAt + 1}.`;
          });
          block.addEventListener("click", (e) => {
            if (e.shiftKey && st.tlSelection) {
              st.tlSelection = { start: Math.min(st.tlSelection.start, pos), end: Math.max(st.tlSelection.end, pos) };
            } else {
              st.tlSelection = { start: pos, end: pos };
            }
            renderTimeline();
          });
          timelineEl.appendChild(block);
        });
      }
      wrap.querySelector(".fr-tl-duplicate").addEventListener("click", () => {
        if (!st.tlSelection) { statusEl.textContent = "Select a block (or shift-click for a range) first."; return; }
        const { start, end } = st.tlSelection;
        const chunk = st.sequence.slice(start, end + 1);
        st.sequence.splice(end + 1, 0, ...chunk);
        st.tlSelection = { start: end + 1, end: end + chunk.length };
        syncSequence();
        statusEl.textContent = `Looped ${chunk.length} frame(s) — click Duplicate again to repeat further.`;
      });
      wrap.querySelector(".fr-tl-delete").addEventListener("click", () => {
        if (!st.tlSelection) { statusEl.textContent = "Select a block (or shift-click for a range) first."; return; }
        const { start, end } = st.tlSelection;
        st.sequence.splice(start, end - start + 1);
        st.tlSelection = null;
        if (!st.sequence.length) st.sequence = Array.from({ length: st.totalFrames }, (_, i) => i);
        syncSequence();
        renderStrip();
        statusEl.textContent = "Removed selected range from the timeline.";
      });

      // ---------------- preview height ----------------
      function applyPreviewHeight() {
        previewCanvas.style.height = st.previewHeight + "px";
        drawPreviewFrame();
      }
      wrap.querySelector(".fr-h-plus").addEventListener("click", () => { st.previewHeight = Math.min(400, st.previewHeight + 40); applyPreviewHeight(); });
      wrap.querySelector(".fr-h-minus").addEventListener("click", () => { st.previewHeight = Math.max(60, st.previewHeight - 40); applyPreviewHeight(); });

      // ---------------- output preview (play/pause) ----------------
      let previewIdx = 0;
      function drawPreviewFrame() {
        const seq = st.sequence;
        const ctx = previewCanvas.getContext("2d");
        const boxW = previewCanvas.clientWidth || 200, boxH = previewCanvas.clientHeight || st.previewHeight;
        previewCanvas.width = boxW; previewCanvas.height = boxH;
        ctx.imageSmoothingEnabled = false;
        ctx.clearRect(0, 0, boxW, boxH);
        if (!seq.length) return;
        previewIdx = previewIdx % seq.length;
        const img = st.thumbs[seq[previewIdx]];
        if (!img) return;
        const scale = Math.min(boxW / img.naturalWidth, boxH / img.naturalHeight);
        const w = img.naturalWidth * scale, h = img.naturalHeight * scale;
        ctx.drawImage(img, (boxW - w) / 2, (boxH - h) / 2, w, h);
      }
      function stopPlayback() { st.playing = false; playBtn.textContent = "▶ Preview"; if (st.playTimer) clearInterval(st.playTimer); st.playTimer = null; }
      function startPlayback() {
        if (!st.sequence.length) return;
        st.playing = true; playBtn.textContent = "⏸ Stop";
        const fps = Math.max(1, Math.min(60, parseInt(fpsInput.value, 10) || 12));
        st.playTimer = setInterval(() => { previewIdx++; drawPreviewFrame(); }, 1000 / fps);
      }
      playBtn.addEventListener("click", () => { st.playing ? stopPlayback() : startPlayback(); });
      fpsInput.addEventListener("change", () => { if (st.playing) { stopPlayback(); startPlayback(); } });
      new ResizeObserver(() => drawPreviewFrame()).observe(previewCanvas);

      node._frInternal = {
        st, renderStrip, renderCurve, renderTimeline, selectFrame,
        drawPreviewFrame, applyPreviewHeight, updateOutputCount,
        restoreSequenceFromWidget, statusEl, hideRawWidget,
      };
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
          const fps = message?.vps_source_fps?.[0] ?? 0;
          if (!refs.length) {
            internal.statusEl.textContent =
              "No thumbnails in the node's output — check ComfyUI's Logs panel (Ctrl+`) for a " +
              "'preview save failed' line.";
            return;
          }
          const st = internal.st;
          st.frameRefs = refs;
          st.totalFrames = total;
          st.sourceFps = fps;
          st.thumbs = await Promise.all(refs.map((r) => loadImage(refUrl(r))));

          // On a freshly loaded workflow, hydrate client-side state from
          // sequence_json before validating it. Without this, the backend
          // applied the saved retime once, but the widget displayed an
          // identity timeline and could overwrite the saved sequence.
          if (!st.sequence.length) internal.restoreSequenceFromWidget();

          // Keep an existing/saved sequence if it references valid source
          // frames for this batch; otherwise reset safely to identity.
          const valid = st.sequence.length && st.sequence.every(
            (i) => Number.isInteger(i) && i >= 0 && i < total
          );
          if (!valid) st.sequence = Array.from({ length: total }, (_, i) => i);
          internal.renderCurve();  // also renders the aligned strip
          internal.renderTimeline();
          internal.updateOutputCount();
          internal.applyPreviewHeight();
          internal.selectFrame(0);
          const capNote = refs.length < total ? ` (only the first ${refs.length} have thumbnails — raise max_preview_frames to see more; pacing/timeline still cover all ${total})` : "";
          internal.statusEl.textContent = `${total} frame(s) total${capNote}. Curve/duration = pacing (regenerates the sequence); timeline below = reorder & loop on top of that.`;
          log(`node ${node.id}: loaded ${refs.length}/${total} frame thumbnails, fps=${fps}`);
        } catch (err) {
          internal.statusEl.textContent = "Thumbnail load failed: " + (err?.message || err);
          log(`node ${node.id}: ERROR — ${err?.message || err}`);
        }
      })();
    };
  },
});
