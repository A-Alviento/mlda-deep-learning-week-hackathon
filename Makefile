install:
	python -m pip install --upgrade pip
	pip install -r requirements.txt
	python -m spacy download en_core_web_sm

build:
	python model_training.py

run:
	streamlit run webapp.py