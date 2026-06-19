# Abstractive Text Summarisation using LSTM Seq2Seq (Encoder-Decoder)

Implementation of a sequence-to-sequence LSTM architecture for abstractive text summarisation on the BBC News dataset. The model learns to generate short, human-readable summaries from long news articles using an encoder-decoder design with teacher forcing.

[![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)](https://tensorflow.org)
[![LinkedIn](https://img.shields.io/badge/linkedin-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/harishdeepak/)

---

## Problem

Abstractive summarisation — unlike extractive methods — requires generating new sentences that capture the meaning of a document, not just copying existing phrases. This project implements a classic neural sequence-to-sequence architecture that reads a full news article and generates a short summary word-by-word.

---

## Dataset

**BBC News Articles and Summaries**

| Split | Samples |
|---|---|
| Total | 4,450 article–summary pairs |
| Train | 3,115 (70%) |
| Validation | 1,335 (30%) |

- Articles: full BBC news body text
- Summaries: short headline-style abstractive summaries
- Maximum article length: **500 tokens** (longer articles truncated/padded)
- Maximum summary length: **100 tokens**

---

## Preprocessing Pipeline

Two parallel preprocessing streams — one for articles, one for summaries:

```
Raw Article Text                        Raw Summary Text
    ↓                                       ↓
Contraction expansion                   Contraction expansion
(90+ mappings: "can't"→"cannot")        (same mapping dict)
    ↓                                       ↓
Regex cleaning                          Regex cleaning
(remove special chars, digits,          (remove special chars, digits)
 extra whitespace)
    ↓                                       ↓
NLTK stopword removal                   ← NOT applied to summaries
(preserve article content words)           (summaries need connective words)
    ↓                                       ↓
Keras Tokeniser                         Keras Tokeniser
(article vocab: ~24,225 unique tokens)  (summary vocab: ~15,775 unique tokens)
    ↓                                       ↓
Zero-padding to 500 tokens              <START>/<END> token injection
                                        + zero-padding to 100 tokens
```

**Key preprocessing decision:** Stopword removal is applied only to articles (reduces vocabulary noise) but NOT to summaries (summaries need conjunctions and prepositions to be grammatical).

---

## Model Architecture

### Training Model

```
Encoder:
  Input: (batch, 500)  ← article token sequence
  Embedding: (batch, 500, 100)  ← 100-dim word vectors, vocab=24,225
  LSTM(300 units, return_sequences=True, return_state=True)
    → output: (batch, 500, 300)
    → state_h: (batch, 300)   ← hidden state (context vector)
    → state_c: (batch, 300)   ← cell state

Decoder:
  Input: (batch, 100)  ← summary token sequence (teacher forcing: shifted by 1)
  Embedding: (batch, 100, 100)  ← 100-dim word vectors, vocab=15,775
  LSTM(300 units, initial_state=[state_h, state_c])
    → output: (batch, 100, 300)
  TimeDistributed(Dense(15,775, activation='softmax'))
    → output: (batch, 100, 15,775)  ← per-timestep word probability distribution
```

**Total parameters: ~9.7M**

| Component | Parameters |
|---|---|
| Encoder Embedding | 24,225 × 100 = 2.42M |
| Encoder LSTM | ~481k |
| Decoder Embedding | 15,775 × 100 = 1.58M |
| Decoder LSTM | ~481k |
| TimeDistributed Dense | 300 × 15,775 = 4.73M |
| **Total** | **~9.7M** |

### Training Configuration

| Parameter | Value |
|---|---|
| Loss | Sparse Categorical Cross-Entropy |
| Optimizer | RMSprop |
| Batch size | 128 |
| Epochs trained | 1 |
| Train loss (epoch 1) | 8.44 |
| Val loss (epoch 1) | 6.61 |

### Inference Model

Two separate Keras models are built for autoregressive (one-token-at-a-time) decoding:

```
Encoder inference model:
  Input: article (batch, 500) → outputs (state_h, state_c)

Decoder inference model:
  Inputs: one token + (state_h, state_c)
  Outputs: next token probabilities + updated (state_h, state_c)

Inference loop:
  1. Encode article → context vector (h, c)
  2. Feed <START> token + context to decoder
  3. Sample highest-probability token at each step
  4. Feed predicted token back as next input
  5. Stop at <END> token or max_len=100
```

---

## Teacher Forcing

During training, the decoder receives the **ground-truth** summary token at each step (shifted by one position) rather than its own previous prediction. This stabilises training by avoiding error accumulation early in learning but means the model must learn autoregressive inference separately (hence the split inference models).

---

## Why Only 1 Epoch?

The notebook trained for 1 epoch due to compute constraints. At epoch 1:
- Train loss: 8.44 → indicates the model is beginning to learn token patterns
- Val loss: 6.61 → val loss is lower than train loss (expected early in training with teacher forcing)

More epochs with a learning rate schedule would be required to produce coherent summaries. The project demonstrates architecture implementation and preprocessing design rather than fully converged summarisation quality.

---

## File Structure

| File | Contents |
|---|---|
| `SummarizingtextusingNLPalgorithms.ipynb` | Full pipeline: preprocessing, model definition, training, inference model construction |

---

## Tech Stack

- **Framework**: TensorFlow / Keras
- **NLP preprocessing**: NLTK (stopwords), regex, custom contraction dictionary
- **Data**: pandas (CSV loading), NumPy
- **Tokenisation**: `keras.preprocessing.text.Tokenizer`
- **Padding**: `keras.preprocessing.sequence.pad_sequences`

---

## References

- Bahdanau et al. (2015), *Neural Machine Translation by Jointly Learning to Align and Translate* — foundational Seq2Seq attention paper
- Sutskever et al. (2014), *Sequence to Sequence Learning with Neural Networks* — original encoder-decoder architecture
- BBC News dataset (sourced from Kaggle)
