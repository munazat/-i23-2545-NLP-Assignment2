# Urdu Neural NLP Pipeline — Assignment 2
**Student:** Munazat (i23-2545)
**Course:** CS-4063 (Natural Language Processing)

This repository contains a complete neural NLP pipeline for the Urdu language, featuring word embedding generation, sequence labeling (POS & NER), and topic classification using a Transformer model.

## Project Structure

```text
i23-2545-NLP-Assignment2/
├── i23-2545_Assignment2_DS-A.ipynb    # Complete implementation (Part 1, 2, & 3)
├── data/                               # Dataset files
│   ├── raw.txt                         # Raw BBC Urdu corpus
│   ├── cleaned.txt                     # Pre-processed corpus
│   ├── Metadata.json                   # Article metadata
│   ├── article_topics.json             # Topic labels
│   ├── pos_train.conll                 # POS training data
│   └── ner_train.conll                 # NER training data
├── embeddings/                         # Models and Visualizations
│   ├── word2idx.json                   # Vocabulary mapping
│   ├── attention_heatmaps.png          # Transformer attention plots
│   ├── bilstm_training.png             # BiLSTM loss/accuracy curves
│   ├── transformer_training.png        # Transformer loss/accuracy curves
│   └── tsne_ppmi.png                   # Embedding visualization
├── models/                             # Saved model weights
│   ├── bilstm_pos.pt                   # POS model
│   ├── bilstm_ner.pt                   # NER model
│   └── transformer_cls.pt              # Topic classifier
└── README.md                           # This file
```

## Key Components

### 1. Word Embeddings (Part 1)
- Implementation of **TF-IDF**, **PPMI**, and **Skip-gram (Word2Vec)**.
- Nearest neighbor queries and analogy tests (semantic/syntactic).
- Dimensionality analysis (C1-C4 conditions).

### 2. Sequence Labeling (Part 2)
- **BiLSTM-CRF** architecture for Part-of-Speech (POS) tagging and Named Entity Recognition (NER).
- Custom Linear-Chain CRF layer with Viterbi decoding.
- Entity-level evaluation (conlleval style) for NER entities (PER, LOC, ORG, MISC).
- Ablation study comparing frozen vs. fine-tuned embeddings and architectural variants.

### 3. Topic Classification (Part 3)
- From-scratch **Transformer Encoder** implementation (4 layers, 4 heads).
- Sinusoidal positional encodings and Multi-head Self-Attention.
- Visualization of attention heatmaps demonstrating model focus on salience.

## Reproduction Instructions

1. **Environment**:
   ```bash
   pip install torch numpy matplotlib seaborn scikit-learn tqdm
   ```
2. **Execution**:
   Open the `i23-2545_Assignment2_DS-A.ipynb` notebook and run all cells. The models will load pre-trained embeddings and execute the full evaluation suite.
3. **Outputs**:
   Metrics and visualizations will be generated in the `embeddings/` folder and displayed inline.

---
*Verified for final submission.*