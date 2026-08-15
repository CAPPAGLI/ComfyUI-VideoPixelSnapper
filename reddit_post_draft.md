# Where to post
r/comfyui is the right fit. r/StableDiffusion works too if you want more
reach, but tends to be less about tools/workflows and more about outputs —
optional cross-post later if the first one lands well, not required.

# Suggested flair
"Workflow Included" or "Custom Nodes" (whichever your version of the sub uses).

---

# Title (pick one)

Option A (leads with the problem — usually does better on r/comfyui):
> AI video models have no idea what a pixel grid is — made a ComfyUI node to fix the flickering/jittery mess when you try to turn video into actual pixel art

Option B (shorter, more neutral):
> Video Pixel Snapper — a ComfyUI node pack for turning AI-generated video into stable, temporally-consistent pixel art

---

# Body

If you've tried getting an AI video model to output "pixel art" and then
run it through a normal pixel-art snapper frame by frame, you've probably
seen this: the grid jumps around between frames, the palette flickers,
and outlines turn into speckled noise instead of staying crisp. Makes
sense — the model has no concept of a pixel grid, and single-image
pixel-art tools weren't built with that in mind either, since they
re-detect everything independently on every frame.

I made a node pack that estimates the grid and palette **once** from the
whole clip instead of per frame, then applies that fixed result across
every frame. A few other things came out of actually using it:

- Grid size *and* phase (alignment, not just cell size) detected from a
  sample of frames, not guessed per-frame
- A `majority`/`center_weighted` cell-reduction mode with a
  confidence-margin fallback, so ambiguous (anti-aliased) edge cells
  blend instead of snapping to a random nearby color — this is most of
  what kills the "speckled outline" look
- Reserved palette slots for rare-but-important colors (eyes, thin
  highlights) that plain frequency-based quantization tends to drop
- An in-graph live palette editor: see the raw frame, the auto-processed
  result, and a live re-colored preview side by side, pick/delete/replace
  colors by clicking the image directly, save the edited palette back in
  as a `custom_palette` input
- A separate frame retimer node — hold frames longer, drop them, reorder,
  loop a section, with a keyframed curve for pacing (not AI
  interpolation — it only ever duplicates/reorders real frames, so it
  won't undo the pixel-art result)

Repo: [LINK] — MIT licensed, feedback/issues welcome. This started as a
personal itch-scratch project so there's still a "not built yet" list in
the README (per-frame transforms, a shake preset, a couple other things) —
happy to hear what people would actually use before I guess at priorities
myself.

---

# Notes for you (not part of the post)
- Replace [LINK] with your actual GitHub URL once it's pushed.
- If you want a screenshot/gif, Reddit posts with visible before/after
  images get way more traction than text-only — even a single side-by-side
  frame (jittery single-image-snapper output vs. this) would make the
  problem instantly obvious without reading a word.
- You don't need to defend or oversell anything in comments — if someone
  points out a rough edge, "yeah, that's on the list, PRs welcome" is a
  completely normal response on that sub.
