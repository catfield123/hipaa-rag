"""CFR display labels (must stay aligned with ``app.ingest.chunking`` regexes)."""

from __future__ import annotations

PART_LINE = "PART {part_no} {part_title}"
SUBPART_LINE = "Subpart {letter} - {title}"
SECTION_LINE = "§ {sec_no} {title}"
MARKER_PARENS = "({token})"

STRUCTURAL_SECTION_BODY = "{section_header}\n\n{body}"
