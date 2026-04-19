# Urdu Neural NLP Pipeline — Assignment 2

This repository contains a complete neural NLP pipeline for Urdu, implemented as part of CS-4063 (Natural Language Processing).

## Project Structure

```text
i23-2545-NLP-Assignment2/
├── i23-2545_Assignment2_DS-A.ipynb    # Main notebook with all chunks
├── data/                               # Raw and processed datasets
│   ├── raw.txt
│   ├── cleaned.txt
│   ├── Metadata.json
│   ├── article_topics.json
│   ├── pos_train.conll
│   └── ner_train.conll
├── embeddings/                         # Generated embedding matrices
│   ├── word2idx.json
│   ├── embeddings_w2v.npy
│   └── ppmi_matrix.npy
├── models/                             # Saved model weights
│   ├── bilstm_pos.pt
│   ├── bilstm_ner.pt
│   └── transformer_cls.pt
└── README.md                           # This file
```

## Features

1.  **Word Embeddings**:
    *   TF-IDF Term-Document Matrix.
    *   PPMI Co-occurrence Matrix with t-SNE visualization.
    *   Skip-gram Word2Vec (C3 condition) with analogy and MRR evaluation.
2.  **Sequence Labeling (BiLSTM-CRF)**:
    *   Custom rule-based POS and NER (BIO) annotation.
    *   2-layer BiLSTM model with a from-scratch Linear-Chain CRF.
    *   Early stopping and ablation studies (frozen vs. fine-tuned).
3.  **Topic Classification (Transformer)**:
    *   From-scratch Transformer Encoder (4 layers, 4 heads).
    *   Sinusoidal positional encoding and Pre-LN blocks.
    *   Attention heatmap visualization and comparative analysis.

## Reproduction Instructions

### 1. Environment Setup
Create a virtual environment and install dependencies:
```bash
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows
pip install torch numpy matplotlib seaborn scikit-learn tqdm
```

### 2. Run the Notebook
Open `i23-2545_Assignment2_DS-A.ipynb` in Jupyter or VS Code and run all cells sequentially.

### 3. Verification
*   Embedding nearest neighbours and analogies will be displayed in Part 1.
*   POS/NER performance metrics (P/R/F1) and ablation tables are in Part 2.
*   Transformer training plots, confusion matrices, and attention heatmaps are in Part 3.

## Author
*   **Name**: Munazat (i23-2545)
*   **Assignment**: NLP Assignment 2