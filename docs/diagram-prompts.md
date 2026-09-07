# Image-generation prompts

Historical Claude-specific prompts for the README header banner and the "how it works" explainer diagram.
They are design assets, not a current architecture specification. Codex uses the
bounded supervisor described in `skills/autoresearch/references/codex.md`. Future
diagrams should distinguish instructed scorer/noise rules from runtime checks.
Paste into an image model (GPT-Image / Imagen / etc.). Suggested output paths:
`imgs/autoresearch-banner.png` (banner) and `imgs/how-it-works.png` (diagram).

---

## 1. Header banner (3:1 wide)

> Minimal wide banner (3:1), dark charcoal background. Center: the text
> "autoresearch" in a bold geometric lowercase sans. To the left of the wordmark,
> a circular loop-arrow motif built from a ring of small dots — most dots a warm
> goldenrod, a few faded gray (kept vs. discarded experiments) — with a thin
> rising sparkline threading through the ring and exiting to the right as one
> clean upward line, symbolizing a metric improving over many iterations. A small
> padlock icon sits on the ring (the locked eval harness). Subtitle text below in
> a lighter weight: "Autonomous experiment loop for Claude Code — try, measure,
> keep, repeat." Flat, high-contrast, no photorealism, no robots, no 3D render.

---

## 2. How it works (isometric explainer)

> Isometric technical illustration, hand-drawn vector style: bold black outlines,
> flat color fills with subtle cel-shading, on a light warm-gray background. Clean
> sans-serif labels.
>
> COMPOSITION — a CIRCULAR loop (a flywheel/cycle, NOT a straight pipeline), with
> four numbered stations placed clockwise around a central ring, connected by
> thick numbered ribbon-arrows that flow ① → ② → ③ → ④ and then a bold RETURN
> arrow ④ → ① that closes the loop. The closed ring is the whole point — this is a
> machine that keeps turning. Exact clockwise placement:
> - TOP station = ① HYPOTHESIS
> - RIGHT station = ② RUN (locked harness)
> - BOTTOM station = ③ JUDGE (keep / discard)
> - LEFT station = ④ LOOP ENFORCER (Stop hook)
> At the center of the ring, draw a stack of documents labeled "state" — a
> "autoresearch.jsonl" ledger, a "worklog.md" notebook, and a small dashboard
> table — the shared memory every station reads and writes.
>
> ① HYPOTHESIS (top): a person in an orange sweater at a desk pinning ONE sticky
> note to a board covered in crossed-out past ideas; a caption ribbon reads "one
> atomic change". A small "ideas backlog" tray sits on the desk.
>
> ② RUN — LOCKED HARNESS (right): a sturdy machine/black-box labeled
> "autoresearch.sh" with a bold amber PADLOCK on its front panel and a red ribbon
> banner reading "OFF LIMITS". A strip of paper feeds out of its output slot
> printed with "METRIC r2=…" lines. A small stopwatch icon and a "SEED" dial on
> the side (reproducibility + per-experiment time cap).
>
> ③ JUDGE — KEEP / DISCARD (bottom): a balance scale weighing a metric value
> against a dashed horizontal line labeled "noise floor". Two chutes lead away: a
> GREEN chute stamped "keep → git commit" and a RED chute stamped "discard → git
> revert". A caption reads "only past the noise floor". One small experiment token
> is being re-run three times (a "×3 seeds" tag) before the scale tips —
> confirming a borderline win.
>
> ④ LOOP ENFORCER — STOP HOOK (left): a big ratchet/flywheel gear with a pawl that
> visibly PREVENTS it from stopping, labeled "Stop hook". Beside it a fuel gauge
> labeled "budget: maxRuns / maxSeconds / target" — when the needle hits empty, a
> small "/autoresearch off" switch flips and the loop parks. A little side-panel
> shows two guardian icons labeled "PreCompact" and "SessionStart" catching a
> falling "context" cloud (surviving compaction).
>
> The ④ → ① RETURN arrow is the thickest, clearly closing the circle, with a small
> looping "∞ within budget" tag. Ribbon-arrow colors: goldenrod for the main flow,
> green and red splitting at the JUDGE station. Along the very bottom, below the
> ring, a thin measuring line with the caption "loops until the budget or the
> target is hit — not before". Overall palette: goldenrod #FFA300, keep-green,
> discard-red, amber lock, slate labels, on light warm-gray. No photorealism, no
> heavy 3D render — keep the flat illustrated cel-shaded look.
