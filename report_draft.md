# CS-4063: Natural Language Processing — Assignment 2
## Neural NLP Pipeline for BBC Urdu Corpus
**Student:** Munaza Tariq (i23-2545)

### 1. Introduction
This report documents the implementation of a comprehensive neural NLP pipeline for the Urdu language. The pipeline includes word embedding generation, sequence labeling (POS & NER), and topic classification using a from-scratch Transformer model.

### 2. Part 1: Word Embeddings
We implemented three types of word embeddings:
- **TF-IDF**: Captured discriminative words per category.
- **PPMI**: Modeled local context co-occurrence.
- **Skip-gram (Word2Vec)**: Learned dense representations using noise-contrastive sampling.

**Key Findings:**
- Skip-gram (C3) outperformed PPMI in Mean Reciprocal Rank (MRR), achieving higher semantic coherence.
- Cleaned text significantly reduced noise in nearest neighbor queries (e.g., "حکومت" correctly neighbors "وفاقی" and "عوامی").
- Increasing dimensionality from 100 to 200 (C3 vs C4) provided minor gains, suggesting that corpus size is the primary bottleneck for higher-dimensional embeddings.

### 3. Part 2: Sequence Labeling (BiLSTM-CRF)
We built a 2-layer BiLSTM model with a custom Linear-Chain CRF decoder for POS tagging and NER.
- **POS Tagging**: Achieved robust performance. The most confused pairs were NOUN/ADJ due to semantic overlap in Urdu (e.g., words acting as both descriptors and entities).
- **NER**: The entity-level evaluation (conlleval style) showed strong performance for PER and LOC tags, while ORG tags were more challenging due to multi-word ambiguity.

**Ablation Study:**
- **Fine-tuning**: Fine-tuning embeddings improved NER F1 compared to frozen embeddings by allowing the model to adapt generic Word2Vec vectors to specific sequence labeling tasks.
- **CRF Mechanism**: The CRF layer is essential for enforcing BIO constraints (e.g., preventing an I-PER from following a B-LOC), which simple softmax lacks.

### 4. Part 3: Topic Classification (Transformer)
A 4-layer Transformer Encoder was implemented from scratch, featuring multi-head attention and sinusoidal positional encoding.
- **Performance**: The model achieved high accuracy on the 5-topic classification task.
- **Attention Analysis**: Visualization of attention heatmaps confirmed that the model focuses on high-salience keywords near the beginning of articles and specific topic-related terms.
- **Inductive Bias**: Compared to BiLSTM, the Transformer required more epochs to converge, highlighting the data-hungry nature of the attention mechanism on small datasets.

### 5. Conclusion
The project successfully demonstrates that from-scratch neural architectures can achieve high performance on Urdu NLP tasks. The combination of domain-specific embeddings (Skip-gram) and structured decoders (CRF) provides a solid foundation for more complex downstream applications.
