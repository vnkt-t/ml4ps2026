PYTHON ?= .venv/bin/python

.PHONY: test assets assets-check paper audit verify

test:
	$(PYTHON) -m pytest -q -W error::RuntimeWarning

# Builds in a new directory; does not overwrite frozen assets or manuscript.
assets:
	$(PYTHON) analysis/reproduce_paper_assets.py

assets-check:
	$(PYTHON) analysis/build_submission_assets.py --check
	$(PYTHON) analysis/build_alternate_figure.py --check
	$(PYTHON) analysis/build_robustness_assets.py --check

# Explicit opt-in source compilation; the submission PDF remains frozen.
paper: assets-check
	latexmk -cd -pdf -interaction=nonstopmode -halt-on-error paper/draft_ml4ps.tex

audit: assets-check
	$(PYTHON) analysis/audit_submission_artifacts.py
	$(PYTHON) analysis/audit_alternate_manuscript.py

verify: test audit
