# Image Caption Generator

Full-stack AI app: upload an image, get an auto-generated caption.

**Stack:** React (frontend) + FastAPI (backend) + TensorFlow/Keras VGG16+LSTM (model)

---

## Quick Start

### 1. Train / Setup Backend

```bash
cd backend
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Auto-download dataset & train (needs Kaggle API key)
python train.py

# OR just download dataset
python download_dataset.py

# Start API server (needs working/best_model.h5 + tokenizer.pkl)
python main.py
```

### 2. Start Frontend

```bash
cd frontend
npm install
npm start
```

Open http://localhost:3000

---

## Project Structure

```
backend/
  main.py              - FastAPI inference server
  train.py             - Full training pipeline (auto-downloads dataset)
  download_dataset.py  - Standalone dataset downloader
  requirements.txt     - Python deps
  .env.example         - Env vars template

frontend/
  package.json         - React deps
  public/index.html    - HTML entry
  src/
    App.js             - Main React component
    App.css            - Styles
    index.js           - Entry point
```

## Env Vars

| Var | Default | Purpose |
|-----|---------|---------|
| WORKING_DIR | ./working | Model + tokenizer path |
| BASE_DIR | ./flickr8k | Dataset path |
| REACT_APP_API_URL | http://localhost:8000 | Backend URL |
