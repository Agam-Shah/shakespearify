# Shakespearify ✨
Modern English to Shakespearean English Translator using BART Transformer and Custom Tokenizer.

---

## 📌 Project Overview
Shakespearify is an end-to-end machine learning project designed to translate modern English into Shakespearean English.  
It includes model training (NLP), evaluation (ROUGE, BLEU scores), and a complete production-ready deployment with API, frontend, CI/CD, Docker, and MLOps pipelines.

---

## 🚀 Features
- Custom-trained **RobertaTokenizerFast**.
- Fine-tuned **BART model** on Shakespearean dataset.
- Evaluation using **ROUGE** and **BLEU** metrics.
- Full **REST API** with **FastAPI**.
- **Frontend app** using **React.js** (or simple HTML if you prefer).
- **Dockerized** backend + frontend.
- **GitHub Actions** for automatic CI/CD workflows.
- **MLOps** best practices: logging, versioning, reproducibility.

---

## 🛠️ Tech Stack
| Area              | Tech Used                          |
|-------------------|------------------------------------|
| Machine Learning  | PyTorch, HuggingFace Transformers |
| Web Backend       | FastAPI                            |
| Frontend          | React.js / HTML                    |
| Deployment        | Docker, Docker-Compose             |
| MLOps             | GitHub Actions, Monitoring (future)|
| Data              | Pandas, HuggingFace Datasets       |

---

## 📂 Project Structure
```bash
shakespearify/
│
├── app/                   # API & frontend
│   ├── api/                # FastAPI backend
│   └── frontend/           # Frontend code
│
├── config/                 # Configuration files (yaml)
│
├── data/
│   ├── raw/                # Raw input data
│   ├── processed/          # Cleaned, tokenized data
│   └── external/           # External datasets or resources
│
├── model/
│   ├── final_model/        # Trained model files
│   └── tokenizer/          # Custom tokenizer
│
├── notebooks/              # EDA, Training, Evaluation notebooks
│
├── scripts/                # Python scripts (training, evaluation, inference)
│
├── tests/                  # Unit tests
│
├── .github/workflows/      # CI/CD pipelines (GitHub Actions)
│
├── Dockerfile              # Dockerfile for API
├── docker-compose.yml      # Compose for full system
├── README.md               # Project overview
├── requirements.txt        # Python dependencies
├── setup.py                # Package setup
└── .gitignore              # Files/folders to ignore

🔥 Quickstart
Clone the repo
git clone https://github.com/Agam-Shah/shakespearify.git
cd shakespearify

Install dependencies
pip install -r requirements.txt

Train model
python scripts/model_training.py

Run API
uvicorn app.api.app:app --reload
Access frontend (Instructions will be added if React or HTML frontend setup is ready)

📈 Future Enhancements
Model optimization for latency.
Continuous training pipelines.
Frontend UI improvements.
Model monitoring in production.
Kubernetes deployment for scale.

🤝 Contributing
Pull requests are welcome.
For major changes, please open an issue first to discuss what you would like to change.

📄 License
MIT License

📬 Contact
Agam Shah - LinkedIn - GitHub