
.PHONY: *

# Admin
dev:
	pip install --progress-bar off torch gym pycodestyle >> log_install.txt
	python setup.py develop

lint:
	pycodestyle --max-line-length=160 --ignore=W605 cherry/

lint-examples:
	pycodestyle examples/ --max-line-length=80

lint-tests:
	pycodestyle tests/ --max-line-length=180

tests:
	OMP_NUM_THREADS=1 \
	MKL_NUM_THREADS=1 \
	python -W ignore -m unittest discover -s 'tests/' -p '*_tests.py' -v
	make lint

predocs:
	cp ./README.md docs/index.md
	cp ./CHANGELOG.md docs/changelog.md

docs: predocs
	mkdocs serve

docs-deploy: predocs
	mkdocs gh-deploy

# https://dev.to/neshaz/a-tutorial-for-tagging-releases-in-git-147e
release:
	echo 'Do not forget to bump the CHANGELOG.md'
	echo 'Tagging v'$(shell python -c 'print(open("cherry/_version.py").read()[15:-2])')
	sleep 3
	git tag -a v$(shell python -c 'print(open("cherry/_version.py").read()[15:-2])')
	git push origin --tags

publish:
	python setup.py sdist  # Create package
	twine upload --repository-url https://upload.pypi.org/legacy/ dist/*  # Push to PyPI
