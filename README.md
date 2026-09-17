# AI StudyMate 🎓
### Personalized Generative AI Learning Assistant

**Domain:** Education and E-Learning

AI StudyMate is a Python/Streamlit prototype based on the proposed project. It combines:
- document upload
- lightweight RAG retrieval
- Generative AI answers through Hugging Face
- AI quiz generation
- personalized study-plan generation
- quiz-performance tracking and weak-topic detection
- Hugging Face + Kaggle educational datasets

## 1. Project structure

```text
AI_StudyMate/
├── app.py
├── requirements.txt
├── .env.example
├── README.md
├── data/
│   ├── sciq.csv                 # created by download script
│   └── kaggle_ai2/              # downloaded Kaggle files
├── scripts/
│   └── download_datasets.py
└── src/
    ├── analytics.py
    ├── datasets.py
    ├── document_utils.py
    ├── llm.py
    └── rag.py
```

## 2. Install

Create a virtual environment:

```bash
python -m venv .venv
```

Windows:

```bash
.venv\Scripts\activate
```

macOS/Linux:

```bash
source .venv/bin/activate
```

Install packages:

```bash
pip install -r requirements.txt
```

## 3. Download datasets

Run:

```bash
python scripts/download_datasets.py
```

This attempts to download:

- Hugging Face `allenai/sciq`
- Kaggle `allenai/ai2-science-questions`

Kaggle requires your Kaggle account authentication/API credentials.

## 4. Enable Generative AI

Create a Hugging Face access token and set it as an environment variable.

Windows PowerShell:

```powershell
$env:HF_TOKEN="YOUR_TOKEN"
```

macOS/Linux:

```bash
export HF_TOKEN="YOUR_TOKEN"
```

Optional model:

```bash
export HF_MODEL="Qwen/Qwen2.5-7B-Instruct"
```

Without `HF_TOKEN`, the application still runs in **Demo Mode** and performs local retrieval plus rule-based planning/analytics.

## 5. Run

```bash
streamlit run app.py
```

Then open the local Streamlit address shown in the terminal.

## 6. How the project maps to the proposal

| Proposal feature | Implementation |
|---|---|
| Upload study material | PDF/TXT/DOCX uploader |
| Ask Your Notes | RAG retrieval + LLM |
| RAG | TF-IDF based local retriever |
| Generative AI | Hugging Face Inference API |
| Quiz generation | LLM + SciQ/Kaggle sources |
| Study plan | LLM-generated personalized schedule |
| Weak-topic detection | Local performance analytics |
| Educational datasets | Hugging Face SciQ + Kaggle AI2 Science Questions |
| Bilingual extension | Prompt layer can be extended for Urdu/Roman Urdu |

## 7. Important academic note

The public datasets are mainly used for **evaluation, demonstration, and quiz/question resources**. The student's own uploaded course material is the primary knowledge source for the RAG tutor.

For a final-year or production version, replace the lightweight TF-IDF retriever with embeddings + FAISS/Chroma, add authentication/database persistence, and add proper quiz-answer storage and evaluation.

## 8. Dataset sources

Hugging Face:
https://huggingface.co/datasets/allenai/sciq

Kaggle:
https://www.kaggle.com/datasets/allenai/ai2-science-questions
