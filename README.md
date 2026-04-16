# ML Sentiment API

## Setup

```bash
git clone <repo>
cd ml-sentiment-api
pip install -r requirements.txt
python src/train.py
uvicorn api.app:app --reload
```

## Docker

```bash
docker build -t sentiment-api .
docker run -p 8000:8000 sentiment-api
```

## API test

```bash
curl -X POST http://localhost:8000/predict \
-H "Content-Type: application/json" \
-d '{"text": "This is great"}'
```
