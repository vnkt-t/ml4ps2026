"""Compare every cited BibTeX record with separately checked primary metadata.

The frozen reference snapshot records primary URLs and version/year choices.
This is an offline identity/metadata check, not a claim that every URL was
retrieved again when this program ran or that citation relevance is automated.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
import unicodedata

REPO = Path(__file__).resolve().parents[1]


def normalized(value):
    # Strip BibTeX grouping/accents while retaining letters, digits and order.
    value = value.replace("ı", "i")
    value = unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]", "", value.lower())


def author_names(value):
    names = []
    for author in value.split(" and "):
        parts = author.split(",")
        author = " ".join(parts[1:] + parts[:1]) if len(parts) > 1 else author
        names.append(normalized(author))
    return names


def fields(body):
    result = {}
    position = 0
    while True:
        match = re.search(r"(\w+)\s*=\s*\{", body[position:])
        if match is None:
            return result
        start = position + match.end()
        end, depth = start, 1
        while end < len(body) and depth:
            depth += (body[end] == "{") - (body[end] == "}")
            end += 1
        if depth:
            raise ValueError("unbalanced BibTeX field")
        result[match[1].lower()] = body[start:end-1]
        position = end


def run():
    tex_path = REPO/"paper/draft_ml4ps.tex"
    bib_path = REPO/"paper/references.bib"
    snapshot_path = REPO/"paper/citations_primary_snapshot.json"
    keys = {key.strip() for block in re.findall(r"\\cite\w*\{([^}]+)\}", tex_path.read_text())
            for key in block.split(",")}
    records = {}
    for match in re.finditer(r"(?ms)^@\w+\{([^,]+),(.*?)^\}", bib_path.read_text()):
        if match[1] in records:
            raise ValueError("duplicate BibTeX key: "+match[1])
        records[match[1]] = fields(match[2])
    reference = json.loads(snapshot_path.read_text())["records"]
    failures, checked = [], {}
    if keys != set(reference):
        failures.append("cited-key set differs from primary metadata snapshot")
    for key in sorted(keys):
        if key not in records or key not in reference:
            failures.append("missing record: "+key)
            continue
        expected, actual = reference[key], records[key]
        outcomes = {}
        for field, value in expected["fields"].items():
            if field == "author":
                outcomes[field] = author_names(actual.get(field, "")) == author_names(value)
            elif field in {"url", "doi", "eprint", "year", "volume"}:
                outcomes[field] = actual.get(field, "").strip() == value.strip()
            elif field == "pages":
                outcomes[field] = actual.get(field, "").replace("--", "-") == value.replace("--", "-")
            else:
                outcomes[field] = normalized(actual.get(field, "")) == normalized(value)
            if not outcomes[field]:
                failures.append(key+"/"+field)
        checked[key] = {"fields": outcomes, "primary_sources": expected["primary_sources"],
                        "version_note": expected.get("version_note", "")}
    report = {"status": "PASS" if not failures else "FAIL", "citations": len(keys),
              "metadata_fields_checked": sum(len(row["fields"]) for row in checked.values()),
              "failures": failures, "records": checked,
              "scope": "All cited titles, ordered full author lists, chosen years, URLs and listed identifiers/publication details; typography normalized",
              "input_sha256": {str(p.relative_to(REPO)): hashlib.sha256(p.read_bytes()).hexdigest()
                               for p in [tex_path, bib_path, snapshot_path, Path(__file__)]}}
    (REPO/"paper/citation_check.json").write_text(json.dumps(report, indent=2)+"\n")
    print(f"{report['status']}: {len(keys)} citations, {report['metadata_fields_checked']} metadata fields")
    if failures:
        raise SystemExit("Citation mismatches: "+", ".join(failures))


if __name__ == "__main__":
    run()
