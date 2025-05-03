

import torch
import numpy as np
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datasets import load_dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from tqdm import tqdm

#  Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load dataset
dataset = load_dataset("Kanishkagarwal6101/Legal_Analyzer_Final")
df = pd.DataFrame(dataset["train"])

#  Split into train/test
train_df, test_df = train_test_split(df, test_size=0.1, random_state=42, stratify=df["label"])
print(f"✅ Train size: {len(train_df)}, Test size: {len(test_df)}")

# Model paths
model_paths = {
    "LegalBERT": "Kanishkagarwal6101/LegalBERT_LegalAnalyzer",
    "BERT": "Kanishkagarwal6101/BERT_LegalAnalyzer",
    "Longformer": "Kanishkagarwal6101/Longformer_LegalAnalyzer"
}

#  18-class map
id2label = {
    0: "Business", 1: "Confidentiality", 2: "Consumers", 3: "Declarations", 4: "Economy",
    5: "Education", 6: "Employment", 7: "Environment", 8: "External Relations", 9: "Fairness",
    10: "Health", 11: "IP & Rights", 12: "Indemnification", 13: "Legal Governance",
    14: "Miscellaneous", 15: "Payment", 16: "Social", 17: "Termination"
}

#  Evaluate
results = []

for model_name, model_path in model_paths.items():
    print(f"\n🔵 Evaluating model: {model_name}")

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path).to(device)
    model.eval()

    # Set batch size depending on model
    if "longformer" in model_name.lower():
        batch_size = 2
    else:
        batch_size = 8

    all_preds = []

    for i in tqdm(range(0, len(test_df), batch_size)):
        batch_texts = list(test_df["text"][i:i+batch_size])
        inputs = tokenizer(
            batch_texts,
            truncation=True,
            padding="max_length",
            max_length=4096 if "longformer" in model_name.lower() else 512,
            return_tensors="pt"
        ).to(device)

        with torch.no_grad():
            logits = model(**inputs).logits
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            all_preds.extend(preds)

    # Ground truth
    labels = test_df["label"].values

    # Metrics
    acc = accuracy_score(labels, all_preds)
    prec, rec, f1, _ = precision_recall_fscore_support(labels, all_preds, average="weighted", zero_division=0)

    results.append({
        "Model": model_name,
        "Accuracy": round(acc * 100, 2),
        "Precision": round(prec * 100, 2),
        "Recall": round(rec * 100, 2),
        "F1": round(f1 * 100, 2)
    })

#  Create results table
df_results = pd.DataFrame(results)
print("\n📊 Final Results Table:\n")
print(df_results)

# Save
df_results.to_csv("test_set_evaluation_results.csv", index=False)
print("\n📁 Saved evaluation to test_set_evaluation_results.csv")