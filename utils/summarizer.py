# scripts/summarization/summarize_contracts.py

from transformers import pipeline
from PyPDF2 import PdfReader
import pandas as pd
import re
import nltk
import torch

nltk.download("punkt")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#  Summarizer model
summarizer = pipeline("summarization", model="facebook/bart-large-cnn", device=0 if torch.cuda.is_available() else -1)

def summarize_contract(pdf_path, output_csv, max_clauses=None):
    reader = PdfReader(pdf_path)
    text = "\n".join([page.extract_text() for page in reader.pages if page.extract_text()])

    # Smart clause splitting
    clause_chunks = re.split(r"\n?\s*\d+\.\s+", text)
    clause_chunks = [chunk.strip() for chunk in clause_chunks if chunk.strip()]
    if max_clauses:
        clause_chunks = clause_chunks[:max_clauses]

    print(f"✅ Total clauses detected: {len(clause_chunks)}")

    summaries = []

    for clause in clause_chunks:
        try:
            input_length = len(clause.split())
            if input_length < 15:
                summaries.append(clause.strip())
            else:
                summary = summarizer(
                    clause,
                    max_length=min(50, int(0.7 * input_length)),
                    min_length=min(20, max(5, int(0.4 * input_length))),
                    do_sample=False
                )[0]["summary_text"]
                summaries.append(summary.strip())
        except Exception as e:
            print(f"⚠️ Summarization failed for one clause: {str(e)}")
            summaries.append(clause.strip())

    df = pd.DataFrame({
        "Clause": clause_chunks,
        "Summary": summaries
    })
    df.to_csv(output_csv, index=False)
    print(f"📁 Saved summarized output to {output_csv}")

# Summarize Independent Contractor Agreement
summarize_contract(
    pdf_path="/content/independent_contractor_agreement.pdf",
    output_csv="independent_summaries.csv"
)

# s Summarize Shuttle Contract
summarize_contract(
    pdf_path="/content/SampleContract-Shuttle.pdf",
    output_csv="shuttle_summaries.csv",
    max_clauses=24  # Limit to first 24
)