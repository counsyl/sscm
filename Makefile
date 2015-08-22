setup: requirements.txt
	virtualenv env
	env/bin/pip install -r requirements.txt

dist:
	python setup.py sdist

clean:
	rm -rf build dist *.egg-info

PHONY: setup clean
