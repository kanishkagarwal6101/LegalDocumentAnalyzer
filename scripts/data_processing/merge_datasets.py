# scripts/data_processing/merge_datasets.py

# Merge CUAD, EURLEX, UNFAIR ToS datasets
# Downsample large classes, fix labels, augment real clauses
# Prepare final Legal_Analyzer_Final dataset

import pandas as pd
import random
import nltk
from datasets import load_dataset, Dataset
from sklearn.utils import resample

nltk.download('punkt_tab')

#  Load datasets
ledgar = load_dataset("lex_glue", "ledgar")["train"]
eurlex = load_dataset("lex_glue", "eurlex")["train"]
unfair_tos = load_dataset("lex_glue", "unfair_tos")["train"]

#  Mapping definitions
ledgar_label_map = {
    "Confidentiality": "Confidentiality", "Non-Disparagement": "Confidentiality",
    "Termination": "Termination", "Severability": "Termination", "Death": "Termination",
    "Payments": "Payment", "Fees": "Payment", "Base Salary": "Payment",
    "Indemnifications": "Indemnification", "Indemnity": "Indemnification",
    "Jurisdictions": "Legal Governance", "Governing Laws": "Legal Governance",
    "Submission To Jurisdiction": "Legal Governance", "Consent To Jurisdiction": "Legal Governance",
    "Warranties": "Declarations", "Representations": "Declarations",
    "Intellectual Property": "IP & Rights", "Licenses": "IP & Rights",
    "Conflicts": "Miscellaneous", "Assignments": "Miscellaneous",
    "Modifications": "Miscellaneous", "Miscellaneous": "Miscellaneous",
    "Entire Agreements": "Miscellaneous"
}

eurlex_label_map = {
    1: "Legal Governance", 2: "Legal Governance", 3: "Business", 4: "Consumers",
    5: "Economy", 6: "Education", 7: "Employment", 8: "Environment",
    9: "External Relations", 10: "Social", 11: "Health"
}

unfair_label = "Fairness"

label2id = {
    "Business": 0, "Confidentiality": 1, "Consumers": 2, "Declarations": 3, "Economy": 4,
    "Education": 5, "Employment": 6, "Environment": 7, "External Relations": 8, "Fairness": 9,
    "Health": 10, "IP & Rights": 11, "Indemnification": 12, "Legal Governance": 13,
    "Miscellaneous": 14, "Payment": 15, "Social": 16, "Termination": 17
}

# Merge clauses
merged_data = []

# LEDGAR
for ex in ledgar:
    label = ledgar.features["label"].int2str(ex["label"])
    mapped = ledgar_label_map.get(label, None)
    if mapped:
        merged_data.append({"text": ex["text"], "label": mapped})

# EURLEX (sentences)
from nltk.tokenize import sent_tokenize
eurlex_subset = eurlex.select(range(5000))
for ex in eurlex_subset:
    sentences = sent_tokenize(ex["text"])
    mapped_labels = [eurlex_label_map.get(l) for l in ex["labels"] if l in eurlex_label_map]
    if mapped_labels:
        for sent in sentences:
            merged_data.append({"text": sent, "label": mapped_labels[0]})

# UNFAIR TOS
for ex in unfair_tos:
    merged_data.append({"text": ex["text"], "label": unfair_label})

random.shuffle(merged_data)
df = pd.DataFrame(merged_data)

# Downsample large classes (max 3000)
df_balanced = df.groupby("label").apply(lambda x: resample(x, n_samples=min(len(x), 3000), random_state=42)).reset_index(drop=True)

# Label integer mapping
df_balanced["label"] = df_balanced["label"].map(label2id)

#  Add augmented real contract clauses
df_real = pd.read_csv("/content/smart_ensemble_employment_eval.csv")[["Clause", "Your Label"]]
df_real.columns = ["text", "label"]

# Create augmentations (synonyms, compression, noise)
synonyms = {
    "Employee": "Staff Member", "Employer": "Company", "Contract": "Agreement",
    "shall": "will", "may": "might", "position": "role", "duties": "responsibilities",
    "terminate": "end", "confidential": "private", "liable": "responsible"
}

def apply_synonyms(text):
    for word, sub in synonyms.items():
        text = text.replace(word, sub)
    return text

def compress_text(text):
    words = text.split()
    if len(words) > 20:
        return " ".join(words[:10]) + " ... " + " ".join(words[-10:])
    return text

def add_suffix(text):
    suffixes = [
        " This clause is governed by HR policy.",
        " Refer to company manual for exceptions.",
        " Subject to change based on business needs.",
        " Applicable within employment jurisdiction.",
        " As outlined in the role handbook."
    ]
    return text + random.choice(suffixes)

augmented_rows = []
for _ in range(9):
    for _, row in df_real.iterrows():
        method = random.choice([apply_synonyms, compress_text, add_suffix])
        augmented_rows.append({
            "text": method(row["text"]).strip(),
            "label": row["label"]
        })
df_augmented = pd.DataFrame(augmented_rows)

#  Merge all
df_real["label"] = df_real["label"].map(label2id)
df_augmented["label"] = df_augmented["label"].map(label2id)

df_final = pd.concat([df_balanced, df_real, df_augmented], ignore_index=True)
df_final = df_final.sample(frac=1, random_state=42).reset_index(drop=True)

print("Final merged dataset:", df_final.shape)

#  Save and upload
final_ds = Dataset.from_pandas(df_final, preserve_index=False)
final_ds.push_to_hub("Kanishkagarwal6101/Legal_Analyzer_Final", private=False)