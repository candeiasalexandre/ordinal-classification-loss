.PHONY: clean
clean: clean-env clean-build


.PHONY: clean-env
clean-env: clean-build
	-rm -rf .venv poetry.lock


.PHONY: clean-build
clean-build:
	@rm -fr pip-wheel-metadata
	@rm -fr build/
	@rm -fr dist/
	@rm -fr .eggs/
	@find . -name '*.egg-info' -exec rm -fr {} +
	@find . -name '*.egg' -exec rm -f {} +

.PHONY: setup-dev
setup-dev: clean
	poetry config virtualenvs.in-project true 
	poetry install

.PHONY: tests
tests:
	poetry run pytest -s tests