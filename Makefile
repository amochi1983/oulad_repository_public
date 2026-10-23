PY ?= python
.PHONY: reproduce verify test
reproduce:
	$(PY) reproduce.py --rebuild-features --refit-association --retrain-core
verify:
	$(PY) reproduce.py
test:
	$(PY) -m pytest tests -q -p no:cacheprovider
