# MLOps-Activity7
.\venv\Scripts\activate
docker-compose build
docker-compose up -d

cd airflow\dags
do it from docker desktop ->   python weather_pipeline.py



mlflow server --host 127.0.0.1 --port 8080

then run python mlflow.py


dvc repro
(to run dvc ymal)


To run fronend nad backend

cd backend
cd app
python main.py

cd frontend
npx http-server .