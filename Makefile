install:
	python -m pip install --upgrade pip
	pip install -r requirements.txt

build:
	python model_training.py

run:
	streamlit run webapp.py