# 📝 Legal Document Analyzer  

This repository contains code for the **Legal Document Analyzer**, an NLP-based tool designed to extract, summarize, and classify clauses from legal documents. The project compares the performance of transformer-based models on key legal datasets to automate the process of understanding complex legal texts.  

## 🚀 Overview  
Legal documents are often lengthy, complex, and filled with technical jargon, making them difficult to understand for both legal professionals and the general public. The Legal Document Analyzer automates the process of:  

- **Extracting** key clauses.  
- **Summarizing** long legal clauses.  
- **Classifying** sections into predefined categories:  

## 🧠 Models  
We fine-tune and compare the following models:  

- **BERT** – General-purpose model for contextual understanding.  
- **Legal-BERT** – Pre-trained on legal texts for better domain-specific performance.  
- **Longformer** – Optimized for long document processing.  

## 📚 Datasets  
- **Unfair_TOS** – Sentence-level labels focused
solely on unfairness detection.  
- **LexGLUE - LEDGAR** – Legal text classification dataset.  
- **EUR-Lex** – European Union legal documents annotated with categories.  

## 🛠️ Requirements  
-transformers
-datasets
-evaluate
-scikit-learn
-PyPDF2
-nltk
-torch
-rouge_score
