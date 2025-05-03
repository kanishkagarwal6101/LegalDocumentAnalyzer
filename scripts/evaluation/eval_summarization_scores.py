# scripts/summarization/eval_summaries.py

import pandas as pd
from transformers import pipeline
import evaluate
import nltk
import re
from PyPDF2 import PdfReader
import torch

# Setup
nltk.download("punkt")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
summarizer = pipeline("summarization", model="facebook/bart-large-cnn", device=0 if torch.cuda.is_available() else -1)
rouge = evaluate.load("rouge")

def chunk_contract(pdf_path, max_chunks=None):
    reader = PdfReader(pdf_path)
    text = "\n".join([page.extract_text() for page in reader.pages if page.extract_text()])
    clause_chunks = re.split(r"\n?\s*\d+\.\s+", text)
    clause_chunks = [chunk.strip() for chunk in clause_chunks if chunk.strip()]
    if max_chunks:
        clause_chunks = clause_chunks[:max_chunks]
    return clause_chunks

def summarize_clause(clause):
    try:
        input_length = len(clause.split())
        if input_length < 15:
            return clause.strip()
        summary = summarizer(
            clause,
            max_length=min(50, int(0.7 * input_length)),
            min_length=min(20, max(5, int(0.4 * input_length))),
            do_sample=False
        )[0]["summary_text"]
        return summary.strip()
    except Exception as e:
        print(f"⚠️ Summarization failed: {str(e)}. Using raw clause.")
        return clause.strip()

def evaluate_summaries(clauses, contract_name):
    generated_summaries = []
    for clause in clauses:
        generated_summaries.append(summarize_clause(clause))

    # Compute ROUGE
    rouge_scores = rouge.compute(predictions=generated_summaries, references=clauses, use_stemmer=True)
    print(f"\n📈 Summarization ROUGE scores for {contract_name}:")
    print(f"ROUGE-1: {rouge_scores['rouge1']*100:.2f}%")
    print(f"ROUGE-2: {rouge_scores['rouge2']*100:.2f}%")
    print(f"ROUGE-L: {rouge_scores['rougeL']*100:.2f}%")

    # Save summaries
    df = pd.DataFrame({
        "Original Clause": clauses,
        "Generated Summary": generated_summaries
    })
    csv_name = f"{contract_name.lower()}_summaries.csv"
    df.to_csv(csv_name, index=False)
    print(f"📁 Saved summarized output to {csv_name}")

# ✅ Independent Contractor Evaluation
clauses_independent = chunk_contract("/content/independent_contractor_agreement.pdf")
evaluate_summaries(clauses_independent, contract_name="Independent_Contract")

# ✅ Shuttle Contract Evaluation
clauses_shuttle = chunk_contract("/content/SampleContract-Shuttle.pdf", max_chunks=24)
evaluate_summaries(clauses_shuttle, contract_name="Shuttle_Contract")