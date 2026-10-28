# Telco Customer Churn — End-to-End AIOps Case Study

Binary classification case study (will the customer churn?) built as a full
AIOps pipeline: train → save model → serve via FastAPI → containerize →
CI pipeline that proves the prediction endpoint actually works.

## Dataset

**Kaggle: [Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)**
(search "Telco Customer Churn" by `blastchar` if the link goes stale).

1. Download `WA_Fn-UseC_-Telco-Customer-Churn.csv` from Kaggle (manually, or
   via the Kaggle API — see below).
2. Place it at `data/WA_Fn-UseC_-Telco-Customer-Churn.csv`.

Kaggle API alternative (needs `~/.kaggle/kaggle.json` token):
```bash
pip install kaggle
kaggle datasets download -d blastchar/telco-customer-churn -p data --unzip
```

## Pipeline order

1. **Train** — open `notebooks/train_churn_model.ipynb` in Jupyter or
   upload to Colab. Run all cells. It writes:
   - `models/model.joblib` — the full sklearn Pipeline (preprocessing + classifier)
   - `models/model_metadata.json` — chosen hyperparameters, feature list, test metrics
2. **Commit the model artifacts** (`models/model.joblib`,
   `models/model_metadata.json`) — the API and CI both need them, and CI has
   no Kaggle access, so the trained model must live in the repo.
3. **Serve locally** (VSCode / terminal):
   ```bash
   python -m venv .venv && source .venv/bin/activate
   pip install -r requirements.txt
   uvicorn app.main:app --reload
   ```
   Open http://127.0.0.1:8000/docs and try `/predict`.
4. **Docker**:
   ```bash
   docker build -t churn-api .
   docker run -p 8000:8000 churn-api
   curl -X POST http://127.0.0.1:8000/predict -H "Content-Type: application/json" -d @sample_request.json
   ```
5. **GitHub**: `git init`, commit, `gh repo create` (see below), push.
6. **CI** (`.github/workflows/ci.yml`) runs on every push: installs deps,
   runs `pytest` against the FastAPI app, builds the Docker image, starts
   the container, and curls `/predict` to prove the end-to-end prediction
   path works — not just unit tests.

## Creating the GitHub repo (when you're ready)

```bash
git init
git add .
git commit -m "Initial churn prediction AIOps pipeline"
gh repo create telco-churn-aiops --public --source=. --remote=origin --push
```

## Project layout

```
data/            raw CSV (gitignored except README)
notebooks/       training notebook (run in Jupyter/Colab)
models/          trained model.joblib + metadata.json (commit after training)
app/             FastAPI service
tests/           pytest tests for the API
Dockerfile       container for the API
.github/workflows/ci.yml   CI: test -> build -> run -> verify prediction
```
