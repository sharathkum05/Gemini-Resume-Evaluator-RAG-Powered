import streamlit as st
import os
import PyPDF2 as pdf
import google.generativeai as genai
import faiss
from dotenv import load_dotenv
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import json

# --- Load API key from .env ---
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# --- Prompts ---
input_prompt = """
Hey Act Like a skilled or very experienced ATS (Application Tracking System)
with a deep understanding of tech field, software engineering, data science, data analyst,
and big data engineer. Your task is to evaluate the resume based on the given job description.
You must consider the job market is very competitive and you should provide
best assistance for improving the resumes. Assign the percentage Matching based 
on JD and the missing keywords with high accuracy.

resume: {text}

description: {jd}

I want the response in one single string having the structure:
{{"JD Match":"%","MissingKeywords":[],"Profile Summary":""}}
"""

# --- Utilities ---
def extract_text_from_pdf(uploaded_file):
    reader = pdf.PdfReader(uploaded_file)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text

def chunk_text(text, chunk_size=300):
    words = text.split()
    return [" ".join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]

def create_embeddings(chunks):
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform(chunks).toarray()
    return vectors, vectorizer

def build_faiss_index(vectors):
    dim = vectors.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(np.array(vectors).astype('float32'))
    return index

def retrieve_relevant_chunks(query, vectorizer, chunks, index, k=3):
    query_vec = vectorizer.transform([query]).toarray().astype('float32')
    _, I = index.search(query_vec, k)
    return [chunks[i] for i in I[0]]

def get_gemini_response(prompt):
    model = genai.GenerativeModel(model_name="models/gemini-2.0-flash")
    response = model.generate_content(prompt)
    return response.text

# --- Streamlit UI ---
st.set_page_config(page_title="Gemini Resume Evaluator", layout="centered")
st.title("📄 Gemini Resume Evaluator (RAG-Powered)")
st.markdown("Paste a Job Description and upload your Resume. Gemini will evaluate your resume using ATS logic.")

# Inputs
job_description = st.text_area("🔍 Paste the Job Description")
uploaded_file = st.file_uploader("📎 Upload your Resume (PDF)", type="pdf")

if st.button("📊 Evaluate Resume"):
    if uploaded_file and job_description:
        with st.spinner("Processing Resume with RAG..."):
            resume_text = extract_text_from_pdf(uploaded_file)
            chunks = chunk_text(resume_text)
            vectors, vectorizer = create_embeddings(chunks)
            index = build_faiss_index(np.array(vectors))

            # Retrieve top resume chunks based on job description
            top_chunks = retrieve_relevant_chunks(job_description, vectorizer, chunks, index, k=3)
            context = "\n".join(top_chunks)

            # Inject into prompt
            final_prompt = input_prompt.format(text=context, jd=job_description)

            # Gemini response
            raw_response = get_gemini_response(final_prompt)

        st.subheader("✅ Gemini ATS Evaluation")

        # Try parsing JSON from Gemini
        try:
            clean_text = raw_response.strip().strip("`json").strip("```")
            parsed_json = json.loads(clean_text)

            # Display nicely
            st.markdown(f"### 🎯 JD Match: `{parsed_json.get('JD Match', 'N/A')}`")
            
            missing_keywords = parsed_json.get("MissingKeywords", [])
            if missing_keywords:
                st.markdown("### ❌ Missing Keywords:")
                st.write(", ".join(missing_keywords))
            else:
                st.markdown("✅ No missing keywords found!")

            summary = parsed_json.get("Profile Summary", "No summary provided.")
            st.markdown("### 🧾 Profile Summary:")
            st.write(summary)

        except Exception as e:
            st.error("❌ Couldn't parse Gemini's response. Here's the raw output:")
            st.code(raw_response)

    else:
        st.warning("Please paste a job description and upload your resume.")