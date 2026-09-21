"""Apply existing document/citation checks to the alternate without editing the primary."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from analysis import audit_submission_artifacts as audit_module


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run():
    (REPO/"tmp").mkdir(exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix="alternate_document_audit_", dir=REPO/"tmp"))
    (stage/"paper").mkdir()
    (stage/"analysis").mkdir()
    for ext in ("tex", "pdf", "log"):
        shutil.copy2(REPO/f"paper/draft_ml4ps_alternate.{ext}", stage/f"paper/draft_ml4ps.{ext}")
    for name in ("generated", "figures", "references.bib", "citations_primary_snapshot.json",
                 "neurips_2026.sty", "neurips_2026.official.sty"):
        (stage/"paper"/name).symlink_to(REPO/"paper"/name)
    (stage/"analysis/build_submission_assets.py").symlink_to(REPO/"analysis/build_submission_assets.py")
    shutil.copy2(REPO/"analysis/audit_submission_citations.py", stage/"analysis/audit_submission_citations.py")
    audit_module.REPO = stage
    audit = audit_module.Audit()
    try:
        audit_module.audit_submission_files(audit)
    finally:
        audit_module.REPO = REPO
    for script in ("build_alternate_figure.py", "build_robustness_assets.py"):
        process = subprocess.run([sys.executable, str(REPO/"analysis"/script), "--check"],
                                 cwd=REPO, capture_output=True, text=True)
        audit.check(process.returncode == 0, script+" current assets", output=process.stdout+process.stderr)
    process = subprocess.run([sys.executable, "analysis/audit_submission_citations.py"], cwd=stage,
                             capture_output=True, text=True)
    audit.check(process.returncode == 0, "alternate citation metadata check", output=process.stdout+process.stderr)
    citation = json.loads((stage/"paper/citation_check.json").read_text())
    citation["input_sha256"] = {name.replace("paper/draft_ml4ps.tex", "paper/draft_ml4ps_alternate.tex"): value
                                for name, value in citation["input_sha256"].items()}
    (REPO/"paper/alternate_citation_check.json").write_text(json.dumps(citation, indent=2)+"\n")
    tex = (REPO/"paper/draft_ml4ps_alternate.tex").read_text()
    # Submission prose and generated tables must retain the template's font and spacing.
    format_sources = [REPO/"paper/draft_ml4ps_alternate.tex"] + sorted((REPO/"paper/generated").glob("*.tex"))
    forbidden = re.compile(r"\\(?:tiny|scriptsize|footnotesize|small|large|Large|LARGE|huge|Huge|fontsize|linespread|geometry|setstretch|resizebox|scalebox)\b|\\(?:setlength|addtolength)\b")
    overrides = {str(path.relative_to(REPO)): forbidden.findall(re.sub(r"(?m)%.*$", "", path.read_text()))
                 for path in format_sources}
    audit.check(not any(overrides.values()), "no manuscript or table font/spacing overrides", matches=overrides)
    pdf_text = subprocess.check_output(["pdftotext", "-layout", str(REPO/"paper/draft_ml4ps_alternate.pdf"), "-"], text=True)
    macro_text = (REPO/"paper/generated/robustness_numbers.tex").read_text()
    macros = dict(re.findall(r"\\newcommand\{\\([A-Za-z]+)\}\{([^{}]+)\}", macro_text))
    used = set(re.findall(r"\\([A-Za-z]+)", tex))
    for name in sorted(used & macros.keys()):
        audit.check(bool(re.search(r"(?<![\d.])"+re.escape(macros[name])+r"(?!\d|\.\d)", pdf_text)), "smooth macro printed: "+name)
    table = (REPO/"paper/generated/robustness_table.tex").read_text()
    for value in set(re.findall(r"\d+\.\d+", table)):
        audit.check(bool(re.search(r"(?<![\d.])"+re.escape(value)+r"(?!\d|\.\d)", pdf_text)), "robustness table value printed: "+value)
    # Strip review line numbers before checking text that wraps across lines.
    prose = " ".join(re.sub(r"(?m)^\s*\d+\s+", "", pdf_text).split())
    for sentence in [
        "Codex-assisted code was subject to these checks; scientific design, analysis, and interpretation were author-directed.",
        "We certify a cell as belonging to the largest-error set when its lower bound exceeds enough of the other upper bounds.",
    ]:
        audit.check(sentence in prose, "requested sentence preserved: "+sentence)
    inputs = {name.replace("paper/draft_ml4ps.", "paper/draft_ml4ps_alternate."): value for name, value in audit.inputs.items()}
    for name in ["analysis/audit_alternate_manuscript.py", "analysis/build_robustness_assets.py",
                 "paper/generated/robustness_numbers.tex", "paper/generated/robustness_table.tex",
                 "paper/generated/robustness_manifest.json", "results/smooth_robustness/verification.json"]:
        inputs[name] = sha(REPO/name)
    report = {"status": "FAIL" if audit.failures else "PASS", "created_utc": datetime.now(timezone.utc).isoformat(),
              "scope": "Mechanical format, anonymous metadata, generated assets/numbers and citation checks for the alternate manuscript. This does not substitute for visual inspection or experimental verification.",
              "document_checks": sum(audit.counts.values()), "failures": audit.failures,
              "pdf": audit.details.get("PDF"), "input_sha256": inputs,
              "citations": citation["citations"], "citation_metadata_fields": citation["metadata_fields_checked"]}
    (REPO/"paper/alternate_prose_review.json").write_text(json.dumps(report, indent=2, sort_keys=True)+"\n")
    print(f"{report['status']}: {report['document_checks']} document checks, {report['citations']} citations")
    if audit.failures:
        raise SystemExit(json.dumps(audit.failures, indent=2))


if __name__ == "__main__":
    run()
