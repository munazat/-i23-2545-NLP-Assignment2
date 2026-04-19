# CS-4063: Natural Language Processing
## Assignment 2 — Neural NLP Pipeline
### A Continuation of the BBC Urdu NLP Pipeline

**Institution:** FAST National University of Computer & Emerging Sciences  
**Total Marks:** 75 | **Framework:** PyTorch (from scratch)  
**Due Date:** 8 April 2026, 11:59 PM | **Prerequisite:** Assignment 1 complete

---

## ⚠️ Restrictions — Read Before You Start

- No pretrained models, no Gensim, no HuggingFace
- Everything must be implemented **from scratch in PyTorch**
- The following built-ins are **not allowed:**
  - `nn.Transformer`
  - `nn.MultiheadAttention`
  - `nn.TransformerEncoder`

---

## Required Input Files

| File | Used In | Purpose |
|------|---------|---------|
| `cleaned.txt` | All parts | Primary training corpus |
| `raw.txt` | Parts 1 & 2 | Ablation baseline |
| `Metadata.json` | Part 3 | Topic labels for classification |

---

## Part 1 — Word Embeddings (25 Marks)

### 1.1 TF-IDF Weighting (4 Marks)

- Build a term–document matrix from `cleaned.txt`
- Restrict vocabulary to the **10,000 most frequent tokens**; all others → `<UNK>`
- Compute TF-IDF weights:

```
TF-IDF(w, d) = TF(w, d) × log( N / (1 + df(w)) )
```

- Save the resulting matrix as `tfidf_matrix.npy`
- Report the **top-10 most discriminative words** per topic category

---

### 1.2 Pointwise Mutual Information — PMI (5 Marks)

- Build a word–word co-occurrence matrix with symmetric context window **k = 5**
- Apply **Positive PMI (PPMI)**:

```
PPMI(w1, w2) = max( 0, log2( P(w1,w2) / (P(w1) × P(w2)) ) )
```

- Save as `ppmi_matrix.npy`
- Produce a **2-D t-SNE visualization** of the 200 most frequent tokens, colour-coded by semantic category (politics, sports, geography), with a legend
- Report the **top-5 nearest neighbours** by cosine similarity for at least 10 query words

---

### 2.1 Skip-gram Word2Vec Implementation (9 Marks)

Train a Skip-gram Word2Vec model on `cleaned.txt`. Requirements:

- Separate **centre (V)** and **context (U)** embedding matrices, both of dimension `|V| × d`
- Noise distribution: `Pn(w) ∝ f(w)^(3/4)` with **K = 10** noise samples per positive pair
- Binary cross-entropy loss over context window **k = 5**:

```
L = -log σ(u_o^T v_c) - Σ log σ(-u_wk^T v_c)
```

- Train for **at least 5 epochs**, batch size **≥ 512**
- Plot training loss curve
- Save averaged final embeddings `½(V + U)` as `embeddings_w2v.npy`

**Required Hyperparameters:**

| Parameter | Value |
|-----------|-------|
| d (embedding dim) | 100 |
| k (window size) | 5 |
| K (noise samples) | 10 |
| η (learning rate) | 0.001 (Adam) |

---

### 2.2 Evaluation (7 Marks)

#### Nearest Neighbours & Analogy (4 Marks)
- Report **top-10 nearest neighbours** for each query word:
  `Pakistan, Hukumat, Adalat, Maeeshat, Fauj, Sehat, Taleem, Aabadi`
- Construct **10 analogy tests** using: `v(b) - v(a) + v(c)`; report top-3 candidates per test
- At least **5 analogies must be correct**
- Write 2–3 sentences assessing the quality of semantic relationships

#### Four-Condition Comparison (3 Marks)

| ID | Condition | Description |
|----|-----------|-------------|
| C1 | PPMI baseline | PPMI-weighted co-occurrence vectors |
| C2 | Skip-gram on raw.txt | Word2Vec trained on unprocessed corpus |
| C3 | Skip-gram on cleaned.txt | Word2Vec trained on cleaned corpus |
| C4 | Skip-gram, d = 200 | Condition C3 with doubled embedding dimension |

- For each condition: report **top-5 neighbours** for 5 query words and **MRR** on 20 manually labelled word pairs
- Discuss which condition yields the best embeddings and whether increasing d helps

---

## Part 2 — Sequence Labeling: POS Tagging & NER (25 Marks)

### 3. Dataset Preparation (5 Marks)

1. Randomly select **500 sentences** from `cleaned.txt` (at least 100 from each of 3 topic categories)

2. **POS Annotation** — assign one of the 12 tags below to every token:

   `NOUN` `VERB` `ADJ` `ADV` `PRON` `DET` `CONJ` `POST` `NUM` `PUNC` `UNK`

   - Use a rule-based tagger built on the stemmer/lemmatizer from Assignment 1
   - Include a hand-crafted lexicon of **at least 200 entries per major category**

3. **NER Annotation** — annotate using the BIO scheme:

   `B-PER` `I-PER` `B-LOC` `I-LOC` `B-ORG` `I-ORG` `B-MISC` `I-MISC` `O`

   - Seed gazetteer must cover: **50 Pakistani persons**, **50 locations**, **30 organisations**

4. Split **70 / 15 / 15** (train / val / test), stratified by topic. Report class-label distribution for both tasks.

---

### 4. BiLSTM Sequence Labeler (10 Marks)

Build a **2-layer bidirectional LSTM** sequence labeler:

- Initialize with Word2Vec embeddings from Part 1 (condition C3)
- Evaluate in both **frozen** and **fine-tuned** embedding modes; report validation F1 for each
- Concatenate forward and backward hidden states: `h_t = [→h_t ∥ ←h_t]`
- Apply **dropout p = 0.5** between LSTM layers
- **NER:** CRF output layer with learnable tag-transition matrix; Viterbi decoding
- **POS:** Linear classifier with cross-entropy loss
- Handle variable-length sequences — padding must not contribute to loss
- Optimizer: **Adam** (η = 1e-3, weight decay = 1e-4)
- **Early stopping** on validation F1 with patience of 5 epochs
- Plot training and validation loss per epoch

---

### 5. Evaluation (10 Marks)

#### 5.1 POS Tagging (4 Marks)
- Token-level **accuracy** and **macro-F1** on test set
- **Confusion matrix** over all 12 tags
- Identify the **3 most confused tag pairs** with at least 2 example sentences each
- Compare **frozen vs. fine-tuned** embedding modes in a summary table

#### 5.2 NER (4 Marks)
- Entity-level **precision, recall, and F1** per type (PER, LOC, ORG, MISC) and overall using `conlleval`
- Compare results **with and without the CRF** output layer
- Error analysis: **5 false positives** and **5 false negatives** with explanations

#### 5.3 Ablation Study (2 Marks)

| ID | Change | What Is Tested |
|----|--------|----------------|
| A1 | Unidirectional LSTM only | Value of backward context |
| A2 | No dropout | Effect of dropout regularization |
| A3 | Random embedding initialization | Contribution of pre-trained embeddings |
| A4 | Softmax instead of CRF (NER) | Whether structured decoding improves NER |

Run each ablation independently on the same data split. Report numeric results and discuss each finding.

---

## Part 3 — Transformer Encoder for Topic Classification (20 Marks)

### 6. Dataset Preparation (2 Marks)

Assign each article from `Metadata.json` to one of 5 categories:

| # | Category | Indicative Keywords |
|---|----------|---------------------|
| 1 | Politics | election, government, minister, parliament |
| 2 | Sports | cricket, match, team, player, score |
| 3 | Economy | inflation, trade, bank, GDP, budget |
| 4 | International | UN, treaty, foreign, bilateral, conflict |
| 5 | Health & Society | hospital, disease, vaccine, flood, education |

- Represent each article as a token-ID sequence from `cleaned.txt`, padded/truncated to **256 tokens**
- Split **70 / 15 / 15**, stratified by category. Report class distribution.

---

### 7. Transformer Encoder (10 Marks)

Implement a Transformer encoder for **5-class topic classification**. Each component must be a separate, self-contained module. **No PyTorch built-in Transformer classes.**

| Component | Specification |
|-----------|--------------|
| Scaled dot-product attention | Accepts optional padding mask; returns output + attention weights |
| Multi-head self-attention | h = 4 heads, d_model = 128, d_k = d_v = 32 per head |
| Position-wise FFN | Two linear layers, ReLU, inner dim d_ff = 512 |
| Sinusoidal positional encoding | Fixed (non-learned) buffer, added to input embeddings |
| Stacked encoder blocks | 4 blocks with Pre-Layer Normalization |
| Classification head | Learned [CLS] token → MLP (128 → 64 → 5) |

**Sinusoidal PE formula:**
```
PE(pos, 2i)   = sin( pos / 10000^(2i/d) )
PE(pos, 2i+1) = cos( pos / 10000^(2i/d) )
```

**Pre-LN encoder block:**
```
x ← x + Dropout( MultiHead( LN(x) ) )
x ← x + Dropout( FFN( LN(x) ) )
```

**Training Requirements:**
- Optimizer: **AdamW** (η = 5e-4, weight decay = 0.01)
- **Cosine LR schedule** with 50 warmup steps
- Train for **20 epochs**
- Plot training and validation loss and accuracy per epoch

---

### 8. Evaluation (8 Marks)

#### 8.1 Results (4 Marks)
- Test **accuracy** and **macro-F1**
- 5×5 **confusion matrix**
- For 3 correctly classified articles, plot **attention weight heatmaps** from at least 2 heads of the final encoder layer

#### 8.2 BiLSTM vs. Transformer Comparison (4 Marks)

Write 10–15 sentences addressing all five questions:

1. Which model achieves higher accuracy, and by how much?
2. Which model converged in fewer epochs?
3. Which model was faster to train per epoch, and why?
4. What do the attention heatmaps reveal about Transformer focus?
5. Given a dataset of only 200–300 articles, which architecture is more appropriate and why?

---

## GitHub Submission (5 Marks)

All code must be in a **public GitHub repository** before the deadline.

**Repository name:** `i23-XXXX-NLP-Assignment2`

Include the GitHub URL in both your report and your notebook.

| Criteria | Marks |
|----------|-------|
| Public repository with correct name and folder structure | 1 |
| All code committed (notebook + scripts) | 2 |
| Meaningful commit history (≥ 5 commits — no single bulk commit) | 1 |
| README.md with reproduction instructions | 1 |

---

## Submission

### File Naming
```
i23-XXXX_Assignment2_DS-X.zip
```

### Zip Contents
```
i23-XXXX_Assignment2_DS-X/
├── i23-XXXX_Assignment2_DS-X.ipynb    # All cells executed
├── report.pdf                          # 2–3 pages, Times New Roman 12pt
├── embeddings/
│   ├── tfidf_matrix.npy
│   ├── ppmi_matrix.npy
│   ├── embeddings_w2v.npy
│   └── word2idx.json
├── models/
│   ├── bilstm_pos.pt
│   ├── bilstm_ner.pt
│   └── transformer_cls.pt
└── data/
    ├── pos_train.conll
    ├── pos_test.conll
    ├── ner_train.conll
    └── ner_test.conll
```

### Report Format
- **Format:** PDF only (no .md or .docx accepted)
- **Font:** Times New Roman, 12pt
- **Line Spacing:** 1.5
- **Length:** 2–3 pages
- **Sections:** Overview, Part 1 Results, Part 2 Results, Part 3 Results, Conclusion

---

## Grading Summary

| Part | Marks |
|------|-------|
| Part 1 — Word Embeddings | 25 |
| Part 2 — BiLSTM Sequence Labeling | 25 |
| Part 3 — Transformer Encoder | 20 |
| GitHub Submission | 5 |
| **Total** | **75** |

---

> ⚠️ Any notebook cell with **no output receives zero marks**.  
> ⚠️ All figures must include **axis labels and a title**.

**Good Luck!**
