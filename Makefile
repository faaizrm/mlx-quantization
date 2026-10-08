PYTHON ?= .venv/bin/python
UV ?= uv

.PHONY: report check build benchmark
report:
	$(PYTHON) scripts/report_results.py

check:
	$(PYTHON) -m ruff check .
	$(PYTHON) -m ruff format --check .
	$(PYTHON) -m pytest -q

build:
	$(UV) build --wheel --out-dir dist

benchmark:
	$(PYTHON) scripts/prepare_data.py
	$(PYTHON) scripts/smoke_generate.py
	$(PYTHON) scripts/run_study.py --quick
	$(PYTHON) scripts/run_study.py
	$(PYTHON) scripts/run_specdec.py --quick
	$(PYTHON) scripts/run_specdec.py
	$(PYTHON) scripts/run_kv_cache.py --quick
	$(PYTHON) scripts/run_kv_cache.py
	$(PYTHON) scripts/diagnose_kv_cache.py
	$(PYTHON) scripts/run_fine_grained.py --quick
	$(PYTHON) scripts/run_fine_grained.py
	$(MAKE) report
