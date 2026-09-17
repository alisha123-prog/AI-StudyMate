import os
from datetime import date
from pathlib import Path

import pandas as pd
import streamlit as st

from src.document_utils import extract_text_from_upload, chunk_text
from src.rag import SimpleRAG
from src.llm import generate_with_hf, demo_answer, generate_quiz_with_hf, generate_study_plan_with_hf
from src.datasets import load_sciq, load_kaggle_ai2
from src.analytics import performance_summary, weak_topics

st.set_page_config(page_title="AI StudyMate", page_icon="🎓", layout="wide")

BASE = Path(__file__).parent
DATA_DIR = BASE / "data"

if "rag" not in st.session_state:
    st.session_state.rag = SimpleRAG()
if "quiz_history" not in st.session_state:
    st.session_state.quiz_history = []
if "messages" not in st.session_state:
    st.session_state.messages = []

st.title("🎓 AI StudyMate")
st.caption("Personalized Generative AI Learning Assistant | Education & E-Learning")

with st.sidebar:
    st.header("⚙️ Settings")
    hf_model = st.text_input(
        "Hugging Face model",
        value=os.getenv("HF_MODEL", "Qwen/Qwen2.5-7B-Instruct"),
        help="Requires HF_TOKEN for live Generative AI. Without a token, the app runs in Demo Mode."
    )
    st.session_state.hf_model = hf_model
    st.divider()
    st.info(
        "Upload your own course material for RAG-based answers. "
        "Public datasets are used for educational QA/quiz evaluation, not as a replacement for your course material."
    )

tabs = st.tabs([
    "📚 My Materials", "💬 Ask Your Notes", "📝 Quiz Generator",
    "📅 Study Plan", "📊 Progress", "🗃️ Datasets"
])

with tabs[0]:
    st.subheader("Upload course material")
    uploads = st.file_uploader(
        "Upload PDF, TXT, or DOCX files",
        type=["pdf", "txt", "docx"],
        accept_multiple_files=True
    )
    if st.button("Index Materials", type="primary"):
        if not uploads:
            st.warning("Please upload at least one file.")
        else:
            added = 0
            for up in uploads:
                try:
                    text = extract_text_from_upload(up)
                    chunks = chunk_text(text)
                    st.session_state.rag.add_documents(chunks, source=up.name)
                    added += len(chunks)
                except Exception as e:
                    st.error(f"{up.name}: {e}")
            st.success(f"Indexed {added} text chunks.")
    if st.session_state.rag.documents:
        st.write(f"**Indexed chunks:** {len(st.session_state.rag.documents)}")
        sources = sorted(set(d["source"] for d in st.session_state.rag.documents))
        st.write("**Sources:** " + ", ".join(sources))

with tabs[1]:
    st.subheader("💬 Ask Your Notes")
    for m in st.session_state.messages:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])

    question = st.chat_input("Ask something about your uploaded material...")
    if question:
        st.session_state.messages.append({"role": "user", "content": question})
        context_items = st.session_state.rag.retrieve(question, k=4)
        context = "\n\n".join(
            f"[Source: {x['source']}]\n{x['text']}" for x in context_items
        )
        if context:
            answer = generate_with_hf(
                question, context, st.session_state.hf_model
            ) if os.getenv("HF_TOKEN") else demo_answer(question, context)
        else:
            answer = "Please upload and index your study material first."
        st.session_state.messages.append({"role": "assistant", "content": answer})
        st.rerun()

with tabs[2]:
    st.subheader("📝 AI Quiz Generator")
    source = st.radio("Quiz source", ["My uploaded notes", "Hugging Face SciQ dataset", "Kaggle AI2 Science Questions"])
    difficulty = st.select_slider("Difficulty", options=["Easy", "Medium", "Hard"], value="Medium")
    num_q = st.slider("Number of questions", 3, 10, 5)

    if st.button("Generate Quiz", type="primary"):
        context = ""
        if source == "My uploaded notes":
            if not st.session_state.rag.documents:
                st.warning("Upload and index notes first.")
            else:
                context = "\n\n".join(
                    x["text"] for x in st.session_state.rag.retrieve("important concepts exam questions", k=6)
                )
        elif source == "Hugging Face SciQ dataset":
            df = load_sciq()
            if df.empty:
                st.error("SciQ is not downloaded. Run: python scripts/download_datasets.py")
            else:
                sample = df.sample(min(num_q, len(df)), random_state=42)
                context = sample.to_csv(index=False)
        else:
            df = load_kaggle_ai2()
            if df.empty:
                st.error("Kaggle AI2 dataset is not downloaded. Configure Kaggle authentication and run the dataset script.")
            else:
                sample = df.sample(min(num_q, len(df)), random_state=42)
                context = sample.to_csv(index=False)

        if context:
            if os.getenv("HF_TOKEN"):
                quiz = generate_quiz_with_hf(context, difficulty, num_q, st.session_state.hf_model)
            else:
                quiz = demo_answer(
                    "Create a short quiz from the following educational material. "
                    "If it is already a multiple-choice dataset, reuse the questions and options.",
                    context
                )
            st.markdown(quiz)
            st.session_state.last_quiz = quiz
            st.session_state.last_quiz_source = source

    st.caption("Tip: In a production version, store each selected answer and score to build the adaptive learning profile.")

with tabs[3]:
    st.subheader("📅 Personalized Study Plan")
    col1, col2 = st.columns(2)
    with col1:
        exam_date = st.date_input("Exam date", value=date.today())
        hours = st.number_input("Available study hours/day", 1.0, 12.0, 2.0, 0.5)
    with col2:
        subjects = st.text_area("Subjects/topics", "Python, Database Systems, Cybersecurity")
        level = st.selectbox("Current level", ["Beginner", "Intermediate", "Advanced"])

    if st.button("Create Study Plan", type="primary"):
        if os.getenv("HF_TOKEN"):
            plan = generate_study_plan_with_hf(
                subjects, str(exam_date), hours, level, st.session_state.hf_model
            )
        else:
            plan = (
                f"### Study Plan\n\n"
                f"**Exam:** {exam_date}  \n**Daily time:** {hours} hours  \n**Level:** {level}\n\n"
                f"1. Divide the listed subjects into equal topic blocks.\n"
                f"2. Spend 60% of each session learning/revising and 40% practicing.\n"
                f"3. End each day with 10–15 minutes of active recall.\n"
                f"4. Every third day, take a short quiz and revisit weak topics.\n"
                f"5. In the final 2 days, focus on revision and practice rather than new topics.\n"
            )
        st.markdown(plan)

with tabs[4]:
    st.subheader("📊 Learning Progress & Weak Topics")
    st.write("Add quiz results manually to demonstrate the adaptive-learning component.")
    topic = st.text_input("Topic", "Cryptography")
    score = st.number_input("Score (%)", 0, 100, 70)
    if st.button("Save Result"):
        st.session_state.quiz_history.append({"topic": topic, "score": score})
        st.success("Result saved.")

    if st.session_state.quiz_history:
        df = pd.DataFrame(st.session_state.quiz_history)
        st.dataframe(df, use_container_width=True)
        summary = performance_summary(df)
        st.metric("Average Score", f"{summary:.1f}%")
        weak = weak_topics(df)
        if weak:
            st.warning("Weak topics: " + ", ".join(weak))
        else:
            st.success("No weak topic detected yet.")

with tabs[5]:
    st.subheader("🗃️ Project Data Sources")
    st.markdown(
        "- **Hugging Face:** `allenai/sciq` — science questions with supporting passages and answers.\n"
        "- **Kaggle:** `allenai/ai2-science-questions` — multiple-choice science assessment questions."
    )
    st.write("Use the download script to fetch both datasets into the local `data/` folder.")
    if st.button("Load local dataset statistics"):
        sciq = load_sciq()
        ai2 = load_kaggle_ai2()
        st.write(f"SciQ rows: **{len(sciq):,}**")
        st.write(f"Kaggle AI2 rows: **{len(ai2):,}**")
