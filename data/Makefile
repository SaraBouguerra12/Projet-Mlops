```makefile
.PHONY: setup train register serve api drift test pipeline ui

setup:
	pip install -r requirements.txt

ui:
	mlflow ui --host 0.0.0.0 --port 5000

train:
	python src/train.py

register:
	python src/register_best_model.py

serve:
	mlflow models serve -m models:/vehicule_prediction_model/Production --port 1234 --no-conda

api:
	uvicorn src.serve_api:app --host 0.0.0.0 --port 8000 --reload

drift:
	python src/simulate_drift.py

test:
	python src/test_api.py

pipeline:
	make train
	make register
	make drift
	make test
```
