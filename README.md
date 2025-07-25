# 🧠 BERT for Question Classification

[![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/YOUR_USERNAME/question_classification_bert/blob/main/DL_BERT_for_question_classification_task.ipynb)

This notebook explores fine-tuning and evaluating BERT-based models for a **question classification** task in natural language. The project includes multiple experiments using different pre-trained encoders and training configurations.

---

## 📌 Project Goals

- Classify English questions into 6 coarse-grained semantic categories (e.g., PERSON, LOCATION, etc.)
- Fine-tune models like **BERT-base**, **DistilBERT**, and **RoBERTa** using PyTorch and HuggingFace Transformers
- Compare single-model training vs multi-model training with hyperparameter tuning
- Evaluate performance with metrics such as **accuracy**, **precision**, **recall**, and **F1-score**
- Visualize training curves and confusion matrices

---

## 🧪 Models Used

- `bert-base-cased` (trained for 3 and 5 epochs)
- `distilbert-base-cased` (with hyperparameter tuning)
- `roberta-large`

All models are integrated via `transformers` and trained with a custom PyTorch pipeline.

---

## 📊 Dataset

A dataset of English natural language questions, annotated with 6 semantic categories. Examples:

- `"Who is Galileo?"` → `PERSON`  
- `"Where is Rome?"` → `LOCATION`

The dataset is loaded via Google Drive using `pandas`.

---

## 🔧 Notebook Highlights

- Tokenization and preprocessing with HuggingFace `AutoTokenizer`
- Definition of custom PyTorch classifier
- Training and evaluation loops with metric tracking
- Plots for training loss and accuracy
- Model comparison and discussion

