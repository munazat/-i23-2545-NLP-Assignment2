# CS-4063: Natural Language Processing
## Assignment 1 — Language Modeling for Urdu News Articles

**Due Date:** 15 February 2026, 11:59 PM

---

## Overview

An end-to-end NLP pipeline for the Urdu language using real-world BBC Urdu news data. The assignment covers data scraping, preprocessing, custom linguistic tools, and statistical language model-based text generation.

---

## Learning Outcomes

- Scrape structured real-world Urdu datasets
- Design data pipelines for low-resource languages
- Handle Unicode normalization challenges
- Implement custom linguistic preprocessing tools
- Train statistical language models
- Generate coherent Urdu articles
- Evaluate NLP systems

---

## Part 1 — BBC Urdu Dataset Collection & Preprocessing

### 1. Dataset Collection

- **Source:** [https://www.bbc.com/urdu](https://www.bbc.com/urdu)
- **Volume:** 200–300 complete BBC Urdu news articles

### 2. Storage Format

#### `Metadata.json`
- Unique article number per entry
- No article body included
- Numbering must match `raw.txt` and `cleaned.txt`

#### `raw.txt`
- All scraped article bodies in one file
- No cleaning or normalization applied
- One article per block, each starting with its article number

#### `cleaned.txt`
- Fully processed and cleaned data
- Normalized, sentence-segmented, lemmatized, stemmed tokens
- Numbers replaced with `<NUM>`
- Article numbering must match `raw.txt` and `Metadata.json`

---

### 3. Data Cleaning & Normalization

#### 3.1 Diacritics Removal
Remove diacritics to ensure uniform word representation and reduce data sparsity.
```
عِلَم  →  علم
```

#### 3.2 Noise Removal
Remove URLs, emojis, navigation text, English sentences, and Roman Urdu using regex-based rules.
```
Before: بریکنگ نیوز Visit www.bbc.com 😊
After:  بریکنگ نیوز
```

#### 3.3 Removal of Non-Urdu Text
Filter out English phrases, Roman Urdu, and metadata text using script-based filtering.
```
Before: یہ رپورٹ breaking news کے بارے میں ہے
After:  یہ رپورٹ کے بارے میں ہے
```

#### 3.4 Sentence Segmentation
Split long paragraphs using Urdu punctuation marks (full stop, question mark, exclamation mark).
```
Before: پاکستان میں بارش ہوئی لوگ متاثر ہوئے
After:  پاکستان میں بارش ہوئی۔
        لوگ متاثر ہوئے۔
```

#### 3.5 Whitespace & Formatting Normalization
Remove extra spaces, line breaks, and inconsistent formatting to ensure correct tokenization.

#### 3.6 Custom Linguistic Processing

| Tool | Description |
|------|-------------|
| **Custom Urdu Tokenizer** | Handles word boundaries, punctuation, postpositions, and replaces numbers with `<NUM>` |
| **Custom Urdu Stemmer** | Removes common suffixes to obtain root forms |
| **Custom Urdu Lemmatizer** | Handles plural and gender normalization |

> All components must be implemented from scratch — no pretrained NLP libraries.

---

### 4. Part 1 Output Files

| File | Description |
|------|-------------|
| `Metadata.json` | Metadata for 200–300 articles, no article body |
| `raw.txt` | All scraped article bodies, unprocessed, numbered |
| `cleaned.txt` | Fully preprocessed, tokenized, stemmed, lemmatized articles |

---

## Part 2 — BBC-Style Urdu News Article Generation

### 1. Language Models to Implement

| Model | Purpose |
|-------|---------|
| Unigram | Backoff support and perplexity evaluation only |
| Bigram | Article generation |
| Trigram (with backoff) | Article generation: Trigram → Bigram → Unigram |

### 2. Smoothing Techniques

Choose one of:
- **Laplace (Add-One) Smoothing**
- **Add-k Smoothing** (k is a small positive constant)

### 3. Article Generation System

- Language model selection (Bigram or Trigram)
- Seed prompt input (5–8 Urdu words — single-word prompts not allowed)
- Automatic article generation with proper RTL display

**Valid seed prompt example:** `پاکستان میں مہنگائی کی شرح میں`  
**Invalid seed prompt example:** `پاکستان`

### 4. Generation Constraints

| Constraint | Value |
|------------|-------|
| Minimum length | 200 words |
| Target length | 250 words |
| Maximum length | 300 words |
| Minimum sentences | 5 |
| Forced termination | Applied at 300 words if no EOS token |

> Generated text must be original — no copied or memorized sentences.

### 5. Required Outputs

- **3** complete Urdu news articles using the **Bigram** model
- **3** complete Urdu news articles using the **Trigram** model with backoff
- **5** Urdu news headlines overall

### 6. Evaluation & Analysis

**Per-article evaluation:**
- Fluency
- Grammatical correctness
- Coherence and readability
- Perplexity score

**Comparative analysis across:**
- Raw text pipeline vs. Cleaned text pipeline
- Bigram model vs. Trigram model with backoff

---

## Bonus (Optional)

Implement a GUI-based interface using Tkinter, Streamlit, or any other framework for additional credit (+5 marks).

---

## Submission

### File Naming
```
i23-XXXX_Assignment1_DS-X.zip
```

### Zip Contents
```
i23-XXXX_Assignment1_DS-X/
├── i23-XXXX_Assignment1_DS-X.ipynb   # All outputs must be visible
├── Metadata.json
├── raw.txt
├── cleaned.txt
└── report.pdf
```

### Report Format
- **Font:** Times New Roman, 12pt
- **Line Spacing:** 1.5
- **Length:** 2–3 pages including plots
- **Sections:** Overview, Data Collection & Preprocessing, Language Model Training, Article Generation, Evaluation & Comparison, Conclusion

---

## Grading Rubric (70 Marks Total)

### Part 1 — Dataset Collection & Preprocessing (30 Marks)

| Component | Marks |
|-----------|-------|
| Scraping 200–300 BBC Urdu articles | 2 |
| Correct JSON metadata structure | 2 |
| Consistency across JSON, raw.txt, cleaned.txt | 2 |
| raw.txt compliance (no preprocessing, correct format) | 4 |
| Diacritics removal | 2 |
| Noise removal (URLs, emojis, web artifacts) | 2 |
| Non-Urdu / Roman Urdu / English removal | 2 |
| Sentence segmentation | 2 |
| Whitespace and formatting normalization | 2 |
| Custom Urdu tokenizer (word boundaries, `<NUM>`) | 4 |
| Custom Urdu stemmer | 3 |
| Custom Urdu lemmatizer | 3 |

### Part 2 — Article Generation (30 Marks)

| Component | Marks |
|-----------|-------|
| Unigram model (backoff + evaluation) | 2 |
| Bigram model | 4 |
| Trigram model with backoff | 4 |
| Smoothing (Laplace or Add-k) | 2 |
| Unseen n-gram handling (no zero probabilities) | 2 |
| Model selection + valid seed prompt handling | 2 |
| Length constraints + EOS handling | 2 |
| Correct article outputs (3 articles per model) | 2 |
| Fluency and grammatical correctness | 2 |
| Coherence, readability, BBC-style tone | 2 |
| Originality (no memorized sentences) | 1 |
| Perplexity calculation | 1 |
| Raw vs. cleaned pipeline comparison | 1 |
| Bigram vs. Trigram comparison | 1 |

### Bonus — UI (5 Marks)

| Component | Marks |
|-----------|-------|
| Functional frontend interface | 2 |
| Model selection + seed prompt input via UI | 2 |
| Proper RTL display | 1 |

### Report & Documentation (10 Marks)

| Component | Marks |
|-----------|-------|
| Complete, well-structured, properly formatted report | 2 |
| Dataset collection and preprocessing explanation | 2 |
| Language model implementation and smoothing description | 2 |
| Article generation process and constraints | 2 |
| Evaluation discussion and comparison | 2 |

---

> ⚠️ **Plagiarism** of any kind results in zero marks. All code and report content must be your own work.  
> ⚠️ All notebook cells **must show outputs**. Missing outputs result in mark deductions.

**Good Luck!**
