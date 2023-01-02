.PHONY: init clean lint test docs dist install release

init:  ## initialize virtual environment
	pip --disable-pip-version-check --quiet install pipenv
	pipenv lock
	pipenv sync --bare --dev

clean: ## remove all non source artifacts
	rm -fr build/
	rm -fr dist/
	rm -fr .eggs/
	find . -name '*.egg-info' -exec rm -fr {} +
	find . -name '*.egg' -exec rm -f {} +
	find . -name '*.pyc' -exec rm -f {} +
	find . -name '*.pyo' -exec rm -f {} +
	find . -name '__pycache__' -exec rm -fr {} +
	rm -fr .pytest_cache
	rm -fr .mypy_cache
	rm -fr docs/_build/*

style: 
	pipenv run isort setup.py quaternions tests
	pipenv run black --line-length 79 setup.py quaternions tests

lint: 
	pipenv run flake8 setup.py quaternions tests
	pipenv run mypy quaternions

test: ## run the full suite of unit tests
	pipenv run pytest tests

docs: ## generate Sphinx HTML documentation, including API docs
	pipenv run $(MAKE) -C docs clean
	pipenv run $(MAKE) -C docs html

dist: clean ## build distribution packages
	pipenv run python setup.py sdist bdist_wheel

release: dist ## publish to PyPi
	twine upload dist/*
