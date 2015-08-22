VENV_DIR=venv

setup: requirements.txt
	virtualenv $(VENV_DIR)
	$(VENV_DIR)/bin/pip install -r requirements.txt

teardown:
	rm -rf $(VENV_DIR)

dist:
	python setup.py sdist

clean:
	rm -rf build dist *.egg-info

PHONY: setup clean
