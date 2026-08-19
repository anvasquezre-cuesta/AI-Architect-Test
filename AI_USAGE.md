# AI Assistance Disclosure

In the spirit of transparency I expect of an AI/ML engineering role, this document
states how I used AI tooling while completing this assessment. Nothing in the
assignment prohibits or restricts the use of AI tools (I checked the full brief), so
I provide this disclosure voluntarily.

## Summary

I used an AI coding assistant (Claude Code) as a development accelerator. The
engineering decisions, the architecture, and the final judgment are mine. The AI did
not originate the technical direction: I made, challenged, and approved every library
choice, design tradeoff, and structural decision, and I recorded each one (together
with the reasoning behind it) in [`DECISION_LOG.md`](DECISION_LOG.md). I wrote that log
before the corresponding code, so the git history itself shows the reasoning coming
first and the implementation second.

## What I owned (ideas, decisions, architecture)

- Framing and strategy: I read this as a debug-and-complete exercise (not a greenfield
  build), and I noticed that roughly 60% of the grade sits in bug-fixing, architecture,
  and engineering practice.
- Technology decisions and their rationale: keeping the pinned stack; edit-distance
  fuzzy matching over embeddings (with a Jaro-Winkler corroboration metric); deterministic
  spaCy NER at the extraction step for injection safety; Qdrant for the service and
  scaling story; MiniLM embeddings; and a local open model (Qwen via Ollama) instead of a
  hosted provider.
- Architecture: the hybrid Protocol plus dependency-injection pattern that satisfies both
  the unit tests and the architecture tests at the same time.
- Review and acceptance: I rejected and re-drove work that was not good enough (for
  instance, I sent the component diagram back several times until it read the way I
  wanted), and I decided what ships.

## What the AI accelerated (execution under my direction)

- Turning my decisions into code, and refactoring the services into the class plus
  Protocol plus dependency-injection structure I chose.
- Running the test suite, capturing the initial failing baseline, and iterating each
  layer to green.
- Mechanical scaffolding: the Dockerfile, the docker-compose file, the config and
  environment plumbing, and drafting documentation prose from my decisions.
- Running the application end to end and reporting results, which is how I found and
  fixed different runtime-only defects (the retrieval threshold and the empty-key handling).

## Verification

I verified everything myself: the full test suite passes (61 of 61), and I exercised
every API endpoint against the running container, including real RAG answer generation
with a local open model. I reviewed, corrected, and accepted the AI's output rather than
taking it on trust.
