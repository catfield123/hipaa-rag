# Current Chunking Strategy (HIPAA RAG)

This document explains how chunking currently works in this project and shows concrete examples based on the implementation in:

- `backend/app/services/pdf_parser.py`
- `backend/app/services/chunking.py`

## 1) End-to-end flow

1. PDF text is parsed into a tree of legal nodes (`part`, `subpart`, `section`, `paragraph`, `subparagraph`, `text`).
2. Each `section` is chunked independently.
3. Leaf descendants of that section become "atoms".
4. Atoms are greedily grouped into retrieval chunks by token thresholds.
5. Each chunk gets:
   - plain content (`content`)
   - contextualized content (`content_with_context`)
   - source/page span
   - metadata (included node ids/labels, dominant quote node, etc.)

## 2) Parsing and structure extraction

The parser (`HIPAAPdfParser`) first builds a legal hierarchy:

- `PART <num>` lines become `part` nodes.
- `SUBPART <letter>` lines become `subpart` nodes.
- `§ <num>` lines become `section` nodes.
- Marker lines like `(a)`, `(1)`, `(i)` become paragraph/subparagraph nodes.
- Non-marker text is stored as `text` nodes under the section.

The section body is split into logical blocks in `_build_logical_blocks()`:

- A new block starts when a marker like `(a)` appears.
- Lines without marker continue the current block.
- Blocks are normalized and assigned page ranges.

## 3) What is an atom?

In `RetrievalChunker._build_atoms()`, an atom is a normalized leaf node text with metadata:

- node key and parent key
- source label (for example `45 CFR § 160.203(a)(1)(ii)`)
- token count
- page start/end
- node type and marker

Only leaf descendants are used, so chunking works with the finest available legal text units.

## 4) Chunk grouping rules

Configured defaults:

- `min_tokens = 50`
- `target_tokens = 140`
- `hard_cap_tokens = 220`

Greedy algorithm behavior:

1. Start a chunk with the first atom.
2. Keep appending atoms while close to `target_tokens`.
3. If current chunk is still below `min_tokens`, prefer appending (unless cap would break).
4. If adding the next atom exceeds `target_tokens`:
   - still append if total stays under `hard_cap_tokens` and atom is small (`atom < min_tokens`)
   - otherwise flush current chunk and start a new one.
5. If an atom alone is larger than `target_tokens`, split it with `_split_large_text()` (paragraph/sentence windows).
6. At section end, if remainder is tiny (`< min_tokens`), attempt merge into previous chunk in same section (if within cap).

## 5) Splitting oversized atoms

If a single atom is too large for the target:

1. Try paragraph-like splitting by blank lines (`\n\n`) and pack windows near `target_tokens`.
2. If that fails, split by sentence boundaries (`(?<=[.!?;])\s+`) into windows near `target_tokens`.
3. Each split window becomes an individual chunk with char offsets.

This keeps retrieval granularity reasonable even for long legal blocks.

## 6) Context construction (`content_with_context`)

Each chunk stores both:

- `content`: pure chunk text
- `content_with_context`: breadcrumb-like legal context + content

Context is built from ancestors and section labels, joined by `" > "`.
Example shape:

`45 CFR Part 160 GENERAL > 45 CFR § 160.203 General rule and exceptions. > 45 CFR § 160.203(a)`

This gives dense embeddings more legal positioning signals than raw text alone.

## 7) Metadata included per chunk

Important metadata fields:

- `section_source_label`
- `node_type` (or `mixed`)
- `marker` (if single-marker chunk)
- `included_node_keys`
- `included_source_labels`
- `quote_node_key` (largest atom in chunk)
- plus DB-level ids (`start_node_id`, `end_node_id`, `quote_node_id`) added during ingestion

This metadata is later used for quote extraction and debug traces.

## 8) Worked examples

## Example A: Regular section with multiple short blocks

Assume atoms in order (token counts):

- Atom 1: 90
- Atom 2: 70
- Atom 3: 180
- Atom 4: 140
- Atom 5: 60

With `min=120`, `target=450`, `cap=850`:

- Start with Atom 1 (90)
- Add Atom 2 -> 160 (good)
- Add Atom 3 -> 340 (good)
- Add Atom 4 -> 480 (above target, but Atom 4 is not "small"; flush before it)
- Chunk #1 = Atoms 1-3 (340)
- Start Chunk #2 with Atom 4 (140)
- Add Atom 5 -> 200
- Chunk #2 = Atoms 4-5 (200)

## Example B: Very small trailing remainder

Suppose final atoms create:

- Chunk #N = 420 tokens
- trailing remainder = 50 tokens

Remainder is `< min_tokens`, so chunker tries merging:

- if same section and `420 + 50 <= 850`, merge into Chunk #N
- otherwise keep as its own tiny chunk

## Example C: Single huge atom (1100 tokens)

Atom exceeds `hard_cap_tokens=850`, so it is split:

- paragraph windows or sentence windows near 450 tokens each
- maybe resulting in chunks around the target window (for example ~110, ~130, ~90 tokens)

Each split chunk keeps proper source/page data and contextual header.

## 9) Strengths of current strategy

- Respects legal hierarchy (section-scoped chunking).
- Prevents tiny under-informative chunks (`min_tokens`).
- Prevents huge unwieldy chunks (`hard_cap_tokens`).
- Preserves context explicitly for embedding quality.
- Tracks rich metadata for audits/debugging/quotes.

## 10) Known limitations

- Greedy grouping is local; it does not optimize global semantic coherence.
- Marker nesting depth is heuristic-based (`marker_depth`), not legal-grammar perfect.
- `quote_node_key` picks largest atom, which may not always be the best citation span.
- Token estimate is whitespace-based (`len(text.split())`), not model-token exact.

## 11) Tuning knobs

If retrieval is too noisy or too coarse, start with:

- Lower `target_tokens` (e.g. 300-380) for finer granularity.
- Lower `hard_cap_tokens` (e.g. 650-750) to reduce broad mixed chunks.
- Keep `min_tokens` moderate (80-140) to avoid over-fragmentation.

For this corpus size, these knobs usually have larger impact than adding complex chunking logic.
