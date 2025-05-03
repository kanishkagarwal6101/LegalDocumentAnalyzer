# scripts/evaluation/evaluate_independent_contract.py

from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from PyPDF2 import PdfReader
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import pandas as pd
import torch
import re
import nltk

nltk.download("punkt")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model paths
model_paths = {
    "LegalBERT": "Kanishkagarwal6101/LegalBERT_LegalAnalyzer",
    "BERT": "Kanishkagarwal6101/BERT_LegalAnalyzer",
    "Longformer": "Kanishkagarwal6101/Longformer_LegalAnalyzer"
}

# Label mappings
label2id = {
    "Business": 0, "Confidentiality": 1, "Consumers": 2, "Declarations": 3, "Economy": 4,
    "Education": 5, "Employment": 6, "Environment": 7, "External Relations": 8, "Fairness": 9,
    "Health": 10, "IP & Rights": 11, "Indemnification": 12, "Legal Governance": 13,
    "Miscellaneous": 14, "Payment": 15, "Social": 16, "Termination": 17
}
id2simple = {
    "Business": "Business Matters",
    "Confidentiality": "Confidentiality Terms",
    "Consumers": "Consumer Protection",
    "Declarations": "Legal Declarations",
    "Economy": "Economic Issues",
    "Education": "Education Topics",
    "Employment": "Employment Terms",
    "Environment": "Environmental Concerns",
    "External Relations": "International Matters",
    "Fairness": "Fairness Obligations",
    "Health": "Health Provisions",
    "IP & Rights": "Intellectual Property",
    "Indemnification": "Liability Protection",
    "Legal Governance": "Legal Rules",
    "Miscellaneous": "Other Provisions",
    "Payment": "Payment Terms",
    "Social": "Social Responsibilities",
    "Termination": "Ending Agreements"
}

# Summarizer
summarizer = pipeline("summarization", model="facebook/bart-large-cnn", device=0 if torch.cuda.is_available() else -1)

# Load Independent Contractor Agreement
reader = PdfReader("/content/independent_contractor_agreement.pdf")
text = "\n".join([page.extract_text() for page in reader.pages if page.extract_text()])

# Correct chunking
clause_chunks = re.split(r"\n?\s*\d+\.\s+", text)
clause_chunks = [chunk.strip() for chunk in clause_chunks if chunk.strip()]
print(f" Total clauses found: {len(clause_chunks)}")

# Expected labels for major clauses (optional for scoring)
expected_labels = [
    "Business", "Employment", "Termination", "Payment", "Confidentiality",
    "IP & Rights", "Indemnification", "Legal Governance", "Miscellaneous"
]
expected_ids = [label2id[label] for label in expected_labels]

# Evaluation for each model
for model_name, model_path in model_paths.items():
    print(f"\n🔵 Evaluating {model_name}")

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path).to(device)

    predicted_ids = []
    summaries = []

    for clause in clause_chunks:
        max_len = 4096 if "longformer" in model_name.lower() else 512

        inputs = tokenizer(clause, return_tensors="pt", truncation=True, padding=True, max_length=max_len).to(device)
        with torch.no_grad():
            logits = model(**inputs).logits
            pred_id = torch.argmax(logits, dim=1).item()

        predicted_ids.append(pred_id)

        # Summarization
        input_length = len(clause.split())
        if input_length < 15:
            summaries.append(clause.strip())
        else:
            try:
                summary = summarizer(
                    clause,
                    max_length=min(50, int(0.7 * input_length)),
                    min_length=min(20, max(5, int(0.4 * input_length))),
                    do_sample=False
                )[0]["summary_text"]
                summaries.append(summary.strip())
            except Exception:
                summaries.append(clause.strip())

    predicted_labels = [id2simple[label] for label in [k for k, v in label2id.items() if v in predicted_ids]]

    # Accuracy scores (only if expected available)
    acc, prec, rec, f1 = "-", "-", "-", "-"
    if len(expected_ids) == len(predicted_ids):
        acc = accuracy_score(expected_ids, predicted_ids)
        prec, rec, f1, _ = precision_recall_fscore_support(expected_ids, predicted_ids, average="weighted", zero_division=0)

    print(f"✅ Accuracy: {acc if isinstance(acc, str) else acc*100:.2f}%")
    print(f"✅ Precision: {prec if isinstance(prec, str) else prec*100:.2f}%")
    print(f"✅ Recall: {rec if isinstance(rec, str) else rec*100:.2f}%")
    print(f"✅ F1 Score: {f1 if isinstance(f1, str) else f1*100:.2f}%")

    df = pd.DataFrame({
        "Clause": clause_chunks[:len(predicted_labels)],
        "Predicted Class Description": predicted_labels,
        "Summary": summaries
    })
    df.to_csv(f"{model_name}_independent_contract_eval.csv", index=False)
    print(f"📁 Saved {model_name}_independent_contract_eval.csv")