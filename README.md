# 💬 Natural Language Processing 2 — AIML Module Project

> An end-to-end NLP deep learning project across two domains: **IMDB Sentiment Analysis** using an Embedding + LSTM model, and **News Headline Sarcasm Detection** using a Bidirectional LSTM with GloVe pre-trained word embeddings and Word2Vec.

---

## 📁 Project Structure

```
sentiment-sarcasm-detection-lstm-glove/
│
├── NLP2_Project.ipynb                  # Main Jupyter Notebook
├── NLP2_Project.html                   # HTML export of the notebook
│
├── Sarcasm_Headlines_Dataset.json      # Sarcasm detection dataset (Part B)
├── glove.6B.zip                        # GloVe pre-trained embeddings (Part B)
│   └── glove.6B.200d.txt               # 200-dimensional GloVe vectors
│
└── README.md
```

> **Note:** The IMDB dataset (Part A) is loaded directly via `keras.datasets.imdb` — no separate file needed.

---

## 🗂️ Project Overview

| Part | Domain | Dataset | Technique | Marks |
|------|--------|---------|-----------|-------|
| Part A | Digital Content & Entertainment | IMDB Reviews (Keras built-in) | Embedding + LSTM | — |
| Part B | Social Media Analytics | Sarcasm Headlines JSON | Bidirectional LSTM + GloVe | — |

---

## 🎬 Part A — IMDB Sentiment Analysis

**Domain:** Digital Content and Entertainment  
**Context:** Build a text classification model that analyses customer sentiments based on IMDB movie reviews. The model uses a deep learning Embedding layer followed by an LSTM classifier to predict whether a review is **positive** or **negative**.

---

### Dataset: IMDB (via `keras.datasets.imdb`)

- **Total reviews:** 50,000 (25,000 train + 25,000 test)
- **Labels:** Binary — `0` = Negative, `1` = Positive
- **Vocabulary size:** Top 10,000 most frequent words (`num_words=10,000`)
- **Encoding:** Each review is a sequence of integer word indices
- **Padding:** Sequences padded/truncated to `maxlen=300` (post-padding)
- **Average review length:** Computed from raw data; standard deviation also reported
- **Reviews below length 301:** Majority of reviews fall within this threshold

---

### Steps & Tasks

**1. Import & Analyse the Dataset**
- Load IMDB via `keras.datasets.imdb` with `num_words=10,000`
- Concatenate train and test sets for EDA

**2. Sequence Padding**
- Calculate average and standard deviation of review lengths
- Pad all sequences to `maxlen=300` using `pad_sequences` (post-padding)

**3. Data Analysis**
- Print shape of features and labels: `(50000, 300)`
- Print a sample feature value and its corresponding label

**4. Decode a Review**
- Retrieve IMDB word index dictionary
- Build reverse index (`index → word`)
- Decode a padded integer sequence back to readable English text

**5. Model Design, Training & Tuning**

```
Architecture (Embedding + LSTM):
  Input (integer sequences, maxlen=300)
  → Embedding(input_dim=10000, output_dim=100, input_length=300)
  → LSTM(units=200, dropout=0.2, recurrent_dropout=0.2, return_sequences=True)
  → [Additional layers as tuned]
  → Dense(1, activation='sigmoid')
```

- Train/test split: **80:20** (`train_test_split`)
- Training: `batch_size=300`, `epochs=4`, `validation_split=0.1`
- Plot: Training Loss vs. Validation Loss over epochs
- Plot: Training Accuracy vs. Validation Accuracy over epochs

**6. Prediction on Sample**
- Decode a test review back to text
- Pass through trained model and print `positive` / `negative` sentiment prediction

---

## 📰 Part B — News Headline Sarcasm Detection

**Domain:** Social Media Analytics  
**Context:** Past sarcasm detection research used Twitter datasets which are noisy in labels and language. This project uses a cleaner, higher-quality dataset collected from two contrasting news sources — a satirical site and a factual news site — to build a **Bidirectional LSTM** classifier that detects whether a headline is sarcastic or not.

---

### Dataset: `Sarcasm_Headlines_Dataset.json`

- **Format:** JSON Lines (one record per line) — read with `pd.read_json(..., lines=True)`
- **Sources:** `theonion.com` (sarcastic) and `huffingtonpost.com` (not sarcastic)

| Column | Description |
|--------|-------------|
| `headline` | News headline text |
| `is_sarcastic` | Target — `1` = Sarcastic, `0` = Not Sarcastic |
| `article_link` | URL of article → **dropped** (non-predictive) |

#### Target Distribution

| Label | Meaning |
|-------|---------|
| `0` | Not sarcastic (HuffPost) |
| `1` | Sarcastic (The Onion) |

> ⚠️ **Duplicates:** Duplicate headlines detected and removed before modelling.

---

### Steps & Tasks

**1. Read & Explore the Data**
- Read JSON Lines file → DataFrame
- Check `.info()`, null values, and class distribution bar chart

**2. Data Preprocessing**
- Drop `article_link` column
- Remove duplicate headlines
- Custom text cleaning pipeline:
  - `split_into_words()` — tokenise by whitespace
  - `lowercase()` — normalise to lowercase
  - `remove_special_characters()` — strip punctuation and noise
  - NLTK stopword removal
- Store cleaned text in `cleaned_headline` column

**3. Sequence Length Analysis**
- Compute length of each cleaned headline
- Stats: max length = **66 words**, mean and std reported

**4. Parameter Definition**
```python
MAX_FEATURES  = 14505   # vocabulary size (excluding single-occurrence words)
MAX_LEN       = 66      # max sequence length
EMBEDDING_SIZE = 200    # Word2Vec / GloVe vector dimension
```

**5. Tokenisation & Word Index**
- Fit `Tokenizer` on cleaned headlines
- Analyse word frequency distribution — identify single-occurrence words
- Initial vocabulary: full unique word count
- Reduced vocabulary: `14,505` (excluding hapax legomena)

**6. Feature & Label Creation**
- `texts_to_sequences` → integer sequences
- `pad_sequences` to `maxlen=66` (post-padding)
- Shape: `(N_samples, 66)`

**7. Vocabulary Analysis (Word2Vec)**
- Train `gensim.Word2Vec` on headline tokens (`window=5`, `min_count=1`, `EMBEDDING_DIM=200`)

**8. GloVe Embedding Matrix**
- Load `glove.6B.200d.txt` (200-dimensional pre-trained vectors)
- Build embedding matrix of shape `(vocab_size, 200)` mapping each word index to its GloVe vector

**9. Bidirectional LSTM Model**

```
Architecture:
  Input (padded sequences, maxlen=66)
  → Embedding(vocab_size, 200, weights=[embedding_matrix])
  → Bidirectional(LSTM(512, return_sequences=True))
  → Dense(256, activation='relu')
  → Dropout(0.5)
  → Dense(1, activation='sigmoid')
```

- Compiled with `binary_crossentropy` loss (binary classification)
- OOV tokens handled with `<OOV>` token in tokenizer

**10. Model Training & Evaluation**
- Train/test split: **85:15** (`test_size=0.15`, `random_state=42`)
- Training: `batch_size=128`, `epochs=5`, `validation_data=(X_test_padded, y_test)`
- Plot: Training Loss vs. Validation Loss over epochs
- Plot: Training Accuracy vs. Validation Accuracy over epochs
- Share insights on model convergence and performance

---

## 🛠️ Tech Stack

- **Language:** Python 3.x
- **Notebook:** Jupyter Notebook (Google Colab)
- **Libraries:**

```
pandas             # Data handling
numpy              # Array operations
matplotlib         # Loss/accuracy plots
nltk               # Stopword removal, text preprocessing
scikit-learn       # train_test_split
tensorflow / keras # Sequential model, Embedding, LSTM, Bidirectional, Dense
gensim             # Word2Vec embeddings (Part B)
re                 # Regex-based text cleaning
logging            # Modular preprocessing functions
```

---

## ⚙️ Setup & Usage

```bash
# Clone the repository
git clone https://github.com/<your-username>/nlp2-aiml-project.git
cd nlp2-aiml-project

# Install dependencies
pip install pandas numpy matplotlib nltk scikit-learn tensorflow gensim jupyter

# Download NLTK stopwords
python -c "import nltk; nltk.download('stopwords')"

# Launch the notebook
jupyter notebook NLP2_Project.ipynb
```

> **Google Colab users:** The notebook uses `google.colab.drive.mount()` — update the base path (`project_path`) to match your Drive folder structure or local directory.

---

## 📈 Key Findings (Summary)

**Part A — IMDB Sentiment:**
- 50,000 reviews padded to length 300; most reviews fall under 301 words
- Embedding (output_dim=100) + LSTM (units=200, dropout=0.2) architecture trained on 80% of data
- Bidirectional information flow is not required here as sentence-level sentiment can be inferred left-to-right

**Part B — Sarcasm Detection:**
- Sarcasm detection from clean, well-labelled news headline data — significantly less noisy than Twitter-based datasets
- Vocabulary pruned from full unique-word count to **14,505** by removing hapax legomena (words appearing only once)
- GloVe `200d` pre-trained embeddings initialise the embedding layer — leveraging large-scale external semantic knowledge
- Bidirectional LSTM (512 units) captures both forward and backward context in headlines, which is important for sarcasm where tone often depends on the full phrase
- Train/test split: 85:15 with padded sequences of `maxlen=66`

---

## 📋 Submission Checklist

- [x] `.ipynb` notebook with all code, outputs, and markdown explanations
- [x] `.html` export of the notebook
- [x] All code cells have visible outputs
- [x] Insights documented after every analysis step
- [x] No plagiarism

---

## 📄 License

This project was completed as part of the **Great Learning AIML Programme** coursework.  
Educational use only.

---

*Made with 💬 and Python*
