#makefile

install:
	pip install --upgrade pip &&\
	pip install -r requirements/test_requirements.txt

format:
	black *.py

lint:
	pylint --disable=R,C *.py
	pylint --disable=R,C iris_model/*.py
	pylint --disable=R,C iris_model/processing/*.py
	pylint --disable=R,C iris_model/config/*.py

test:
	python -m pytest tests/test_*.py

all: install lint test format