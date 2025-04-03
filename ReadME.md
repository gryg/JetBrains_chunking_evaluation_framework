# Text Chunking and Retrieval System Evaluation

This repository contains the evaluation of various text chunking strategies for retrieval-based systems. The goal was to understand how different chunking parameters affect retrieval quality using standard metrics like precision, recall, and more advanced MTEB-style metrics.

## Project Overview

In retrieval-based systems, breaking documents into smaller chunks is essential for effective retrieval. This project explores how different chunking strategies and parameters affect retrieval quality. Three chunking strategies were evaluated:

1. **FixedTokenChunker**: Splits text into fixed-size token chunks
2. **RecursiveCharacterTextSplitter**: Recursively splits text on a list of separators
3. **SentenceChunker**: Splits text by sentences and then groups into chunks

Each chunker was evaluated with different parameters:
- Chunk sizes: 200, 400, and 600 tokens
- Chunk overlap: 0, 100, and 200 tokens
- Number of retrieved chunks: 5

Two embedding models were used:
- **MiniLM**: Sentence-transformers all-MiniLM-L6-v2 model
- **E5-small**: Microsoft's E5-small-v2 model

## Methodology

The evaluation pipeline includes:
1. Loading a corpus and questions with golden (reference) excerpts
2. Chunking the corpus using different strategies and parameters
3. Embedding chunks using the selected model
4. Retrieving the most relevant chunks for each question
5. Evaluating retrieval quality using precision, recall, IoU, F1, nDCG@5, and MAP metrics

## Results Analysis

### Key Findings

#### 1. Effect of Embedding Model

The E5-small model consistently outperformed MiniLM across almost all configurations:

```
                    MiniLM          E5-small
Metric          Average Score    Average Score    Improvement
--------        -------------    -------------    -----------
Precision           0.0654           0.0724          +10.7%
Recall              0.9521           0.9699          +1.9%
nDCG@5              0.8369           0.8652          +3.4%
MAP                 0.8022           0.8273          +3.1%
```

**Finding:** E5-small demonstrates better performance for retrieval tasks, with more significant improvements in precision and ranking metrics (nDCG@5, MAP) than in recall.

#### 2. Effect of Chunk Size

For both embedding models, as chunk size increases:
- Precision decreases
- Recall increases or remains high
- Ranking metrics (nDCG@5, MAP) generally increase

**Finding:** Larger chunk sizes reduce precision but improve recall and ranking quality. This is because larger chunks capture more context and are likely to contain the relevant information, albeit with more irrelevant content mixed in.

#### 3. Effect of Chunk Overlap

Increasing overlap generally improves all metrics:
- Most evident in the 600 token chunk size
- With MiniLM, 400 tokens with 200 overlap had the best F1 score (0.1413)
- With E5-small, 200 tokens with overlap showed the best precision

**Finding:** Adding overlap helps maintain continuity between chunks and reduces the chance of splitting relevant information across chunk boundaries.

#### 4. Best Chunking Strategy by Metric

**For Precision (when exact matching is important):**
- Best: RecursiveCharacterTextSplitter with 200 token size (0.1197 with E5-small)
- The recursive approach potentially creates more semantically coherent chunks

**For Recall (when finding all relevant information is critical):**
- Best: SentenceChunker with 600 token size and 200 overlap (0.9913 with E5-small)
- Sentence-based chunking with overlap ensures comprehensive coverage

**For Ranking (nDCG@5 and MAP):**
- Best: SentenceChunker with 600 token size and 200 overlap (0.9260/0.9149 with E5-small)
- This configuration provides the most effective ranking of relevant chunks

## Detailed Comparison Tables

### MiniLM Model Results

| Chunker | Size | Overlap | Retrieved | Precision | Recall | IoU | F1 | nDCG@5 | MAP |
|---------|------|---------|-----------|-----------|--------|-----|-----|--------|-----|
| FixedTokenChunker | 200 | 0 | 5 | 0.0918 | 0.9078 | 0.0909 | 0.1637 | 0.7887 | 0.7542 |
| FixedTokenChunker | 200 | 100 | 5 | 0.1111 | 0.8985 | 0.1099 | 0.1934 | 0.7865 | 0.7516 |
| FixedTokenChunker | 400 | 0 | 5 | 0.0551 | 0.9327 | 0.0549 | 0.1027 | 0.8159 | 0.7811 |
| FixedTokenChunker | 400 | 100 | 5 | 0.0618 | 0.9535 | 0.0616 | 0.1145 | 0.8367 | 0.8073 |
| FixedTokenChunker | 400 | 200 | 5 | 0.0696 | 0.9663 | 0.0694 | 0.1280 | 0.8587 | 0.8287 |
| FixedTokenChunker | 600 | 0 | 5 | 0.0419 | 0.9542 | 0.0418 | 0.0796 | 0.8313 | 0.8049 |
| FixedTokenChunker | 600 | 100 | 5 | 0.0459 | 0.9700 | 0.0459 | 0.0868 | 0.8745 | 0.8480 |
| FixedTokenChunker | 600 | 200 | 5 | 0.0477 | 0.9475 | 0.0475 | 0.0898 | 0.8514 | 0.8191 |
| RecursiveCharacterTextSplitter | 200 | 0 | 5 | 0.1030 | 0.9338 | 0.1022 | 0.1816 | 0.7869 | 0.7405 |
| RecursiveCharacterTextSplitter | 200 | 100 | 5 | 0.1019 | 0.9349 | 0.1013 | 0.1800 | 0.7848 | 0.7386 |
| RecursiveCharacterTextSplitter | 400 | 0 | 5 | 0.0742 | 0.9412 | 0.0739 | 0.1353 | 0.8177 | 0.7803 |
| RecursiveCharacterTextSplitter | 400 | 100 | 5 | 0.0752 | 0.9433 | 0.0750 | 0.1372 | 0.8125 | 0.7730 |
| RecursiveCharacterTextSplitter | 400 | 200 | 5 | 0.0777 | 0.9478 | 0.0775 | 0.1413 | 0.8152 | 0.7815 |
| RecursiveCharacterTextSplitter | 600 | 0 | 5 | 0.0559 | 0.9553 | 0.0558 | 0.1043 | 0.8527 | 0.8159 |
| RecursiveCharacterTextSplitter | 600 | 100 | 5 | 0.0560 | 0.9632 | 0.0559 | 0.1045 | 0.8386 | 0.7996 |
| RecursiveCharacterTextSplitter | 600 | 200 | 5 | 0.0589 | 0.9562 | 0.0587 | 0.1095 | 0.8599 | 0.8246 |
| SentenceChunker | 200 | 0 | 5 | 0.0845 | 0.8955 | 0.0836 | 0.1518 | 0.7355 | 0.6862 |
| SentenceChunker | 200 | 100 | 5 | 0.1032 | 0.9197 | 0.1024 | 0.1818 | 0.7897 | 0.7420 |
| SentenceChunker | 400 | 0 | 5 | 0.0537 | 0.9622 | 0.0536 | 0.1004 | 0.8559 | 0.8227 |
| SentenceChunker | 400 | 100 | 5 | 0.0575 | 0.9631 | 0.0574 | 0.1071 | 0.8484 | 0.8242 |
| SentenceChunker | 400 | 200 | 5 | 0.0638 | 0.9729 | 0.0637 | 0.1180 | 0.8633 | 0.8275 |
| SentenceChunker | 600 | 0 | 5 | 0.0392 | 0.9627 | 0.0391 | 0.0745 | 0.8482 | 0.8186 |
| SentenceChunker | 600 | 100 | 5 | 0.0409 | 0.9765 | 0.0409 | 0.0778 | 0.8796 | 0.8535 |
| SentenceChunker | 600 | 200 | 5 | 0.0430 | 0.9583 | 0.0429 | 0.0814 | 0.8962 | 0.8739 |

### E5-small Model Results

| Chunker | Size | Overlap | Retrieved | Precision | Recall | IoU | F1 | nDCG@5 | MAP |
|---------|------|---------|-----------|-----------|--------|-----|-----|--------|-----|
| FixedTokenChunker | 200 | 0 | 5 | 0.0956 | 0.9475 | 0.0950 | 0.1705 | 0.8162 | 0.7723 |
| FixedTokenChunker | 200 | 100 | 5 | 0.1140 | 0.9450 | 0.1131 | 0.1994 | 0.8225 | 0.7836 |
| FixedTokenChunker | 400 | 0 | 5 | 0.0576 | 0.9819 | 0.0575 | 0.1075 | 0.8340 | 0.7882 |
| FixedTokenChunker | 400 | 100 | 5 | 0.0628 | 0.9653 | 0.0626 | 0.1163 | 0.8492 | 0.8124 |
| FixedTokenChunker | 400 | 200 | 5 | 0.0711 | 0.9612 | 0.0709 | 0.1304 | 0.8555 | 0.8209 |
| FixedTokenChunker | 600 | 0 | 5 | 0.0434 | 0.9780 | 0.0434 | 0.0822 | 0.8831 | 0.8557 |
| FixedTokenChunker | 600 | 100 | 5 | 0.0459 | 0.9895 | 0.0459 | 0.0869 | 0.9053 | 0.8840 |
| FixedTokenChunker | 600 | 200 | 5 | 0.0492 | 0.9800 | 0.0491 | 0.0927 | 0.8985 | 0.8742 |
| RecursiveCharacterTextSplitter | 200 | 0 | 5 | 0.1197 | 0.9427 | 0.1189 | 0.2071 | 0.8285 | 0.7820 |
| RecursiveCharacterTextSplitter | 200 | 100 | 5 | 0.1192 | 0.9540 | 0.1186 | 0.2069 | 0.8287 | 0.7741 |
| RecursiveCharacterTextSplitter | 400 | 0 | 5 | 0.0808 | 0.9695 | 0.0807 | 0.1468 | 0.8240 | 0.7693 |
| RecursiveCharacterTextSplitter | 400 | 100 | 5 | 0.0821 | 0.9770 | 0.0820 | 0.1490 | 0.8316 | 0.7790 |
| RecursiveCharacterTextSplitter | 400 | 200 | 5 | 0.0819 | 0.9685 | 0.0817 | 0.1487 | 0.8590 | 0.8172 |
| RecursiveCharacterTextSplitter | 600 | 0 | 5 | 0.0589 | 0.9855 | 0.0589 | 0.1097 | 0.8638 | 0.8260 |
| RecursiveCharacterTextSplitter | 600 | 100 | 5 | 0.0586 | 0.9882 | 0.0586 | 0.1092 | 0.8578 | 0.8168 |
| RecursiveCharacterTextSplitter | 600 | 200 | 5 | 0.0655 | 0.9750 | 0.0654 | 0.1211 | 0.8526 | 0.8025 |
| SentenceChunker | 200 | 0 | 5 | 0.0903 | 0.9489 | 0.0898 | 0.1619 | 0.8081 | 0.7580 |
| SentenceChunker | 200 | 100 | 5 | 0.1059 | 0.9497 | 0.1051 | 0.1867 | 0.8459 | 0.8026 |
| SentenceChunker | 400 | 0 | 5 | 0.0531 | 0.9767 | 0.0530 | 0.0995 | 0.8761 | 0.8309 |
| SentenceChunker | 400 | 100 | 5 | 0.0587 | 0.9777 | 0.0587 | 0.1094 | 0.8952 | 0.8670 |
| SentenceChunker | 400 | 200 | 5 | 0.0658 | 0.9784 | 0.0658 | 0.1217 | 0.8944 | 0.8657 |
| SentenceChunker | 600 | 0 | 5 | 0.0391 | 0.9804 | 0.0391 | 0.0745 | 0.9065 | 0.8830 |
| SentenceChunker | 600 | 100 | 5 | 0.0415 | 0.9855 | 0.0415 | 0.0788 | 0.9046 | 0.8919 |
| SentenceChunker | 600 | 200 | 5 | 0.0453 | 0.9913 | 0.0453 | 0.0857 | 0.9260 | 0.9149 |

## Trade-off Analysis: Precision vs. Recall

There is a clear trade-off between precision and recall across configurations:

- **High Precision Configurations** (better when the exact matching of information is important):
  - RecursiveCharacterTextSplitter with 200 tokens
  - FixedTokenChunker with 200 tokens and 100 overlap
  - Generally, smaller chunks with some overlap

- **High Recall Configurations** (better when finding all relevant information is critical):
  - SentenceChunker with 600 tokens and 200 overlap
  - FixedTokenChunker with 600 tokens and 100 overlap
  - Generally, larger chunks with significant overlap

- **Balanced Performance** (good F1 score):
  - RecursiveCharacterTextSplitter with 200 tokens using E5-small (F1: 0.2071)
  - FixedTokenChunker with 200 tokens and 100 overlap using E5-small (F1: 0.1994)

## Ranking Quality Analysis

The nDCG@5 and MAP metrics provide insight into the ranking quality of the retrieved chunks:

- **Best Overall Ranking**: SentenceChunker with 600 tokens and 200 overlap using E5-small (nDCG@5: 0.9260, MAP: 0.9149)
- **Worst Ranking**: SentenceChunker with 200 tokens and no overlap using MiniLM (nDCG@5: 0.7355, MAP: 0.6862)

Ranking quality is particularly important when the system will only present a few top results to the user.

## Recommendations

Based on the results analysis, the following recommendations can be made:

1. **For General Purpose Retrieval**:
   - Use E5-small embeddings when possible
   - SentenceChunker with 400 tokens and 200 overlap provides a good balance of metrics
   - This configuration delivers good ranking quality while maintaining reasonable precision

2. **For Question Answering Systems**:
   - Use SentenceChunker with 600 tokens and 200 overlap with E5-small
   - This maximizes recall and ranking quality, ensuring relevant information is retrieved and ranked highly

3. **For Search Applications with Limited Display Space**:
   - Use RecursiveCharacterTextSplitter with 200 tokens and E5-small
   - This prioritizes precision while still maintaining good ranking

4. **For Computational Efficiency**:
   - FixedTokenChunker with 400 tokens and 100 overlap using MiniLM
   - This provides a good balance of performance and computational cost

## Future Work

Further improvements could be explored by:

1. Testing with a wider range of embedding models (e.g., OpenAI embeddings, BGE, E5-large)
2. Exploring hybrid chunking approaches that combine sentence awareness with fixed token limits
3. Implementing semantic chunking based on topic shifts rather than fixed size
4. Testing the effect of different retrievers beyond simple cosine similarity (e.g., hybrid dense-sparse retrieval)
5. Evaluating performance on specific domains (e.g., medical, legal, technical documentation)

## Implementation Details

The evaluation pipeline is implemented using:
- Python with NumPy and scikit-learn for core functionality
- HuggingFace Transformers and SentenceTransformers for embeddings
- Custom metrics implementation for MTEB-style evaluation

## Installation and Setup

### Prerequisites
- Python 3.8 or higher
- pip (Python package installer)

### Dependencies Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/chunking-eval.git
cd chunking-eval
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows, use: venv\Scripts\activate
```

3. Install the package:

You can install the package with different feature sets depending on your needs:

**Basic Installation (core functionality only):**
```bash
pip install -e .
```

**Full Installation (all features):**
```bash
pip install -e ".[all]"
```

**Installation with Specific Features:**
```bash
# For sentence-transformers support:
pip install -e ".[sentence-transformers]"

# For HuggingFace transformers support:
pip install -e ".[huggingface]"

# For tokenizers support:
pip install -e ".[tokenizers]"

# For sentence chunking support:
pip install -e ".[sentence-chunker]"
```

4. If using the SentenceChunker, download the spaCy model:
```bash
python -m spacy download en_core_web_sm
```

### Dependencies Details

The package requires the following core dependencies:
- numpy (>=1.20.0)
- pandas (>=1.3.0)
- scikit-learn (>=1.0.0)

Optional dependencies include:
- sentence-transformers (>=2.2.0) - For SentenceTransformer embedding
- transformers (>=4.20.0) and torch (>=1.10.0) - For HuggingFace and E5 embeddings
- tiktoken (>=0.3.0) - For token counting in FixedTokenChunker
- spacy (>=3.4.0) - For sentence splitting in SentenceChunker

### Project Structure
```
chunking-eval/
├── chunkers/
│   ├── __init__.py
│   ├── base_chunker.py            # Base class for all chunking strategies
│   ├── fixed_token_chunker.py     # Fixed-size token chunking implementation
│   ├── recursive_chunker.py       # Recursive text splitting implementation
│   └── sentence_chunker.py        # Sentence-based chunking implementation
├── embeddings/
│   ├── __init__.py
│   ├── base_provider.py           # Base class for embedding providers
│   ├── e5_provider.py             # Microsoft E5 embedding provider
│   ├── huggingface.py             # HuggingFace transformer embeddings
│   └── sentence_transformers.py   # SentenceTransformers embedding provider
├── evaluation/
│   ├── __init__.py
│   ├── data_loader.py             # Utilities for loading data
│   ├── enhanced_metrics.py        # MTEB-style evaluation metrics
│   ├── enhanced_pipeline.py       # Extended evaluation pipeline
│   ├── helpers.py                 # Helper functions for analysis
│   ├── metrics.py                 # Basic evaluation metrics
│   ├── pipeline.py                # Core evaluation pipeline
│   └── retrieval.py               # Retrieval system implementations
├── data/
│   ├── wikitexts.md               # Text corpora for evaluation
│   ├── pubmed.md                  # Text corpora for evaluation
│   ├── finance.md                 # Text corpora for evaluation
│   ├── chatlogs.md                # Text corpora for evaluation
│   └── questions_df.csv           # Questions with reference excerpts
├── run_evaluation.py              # Main script for running evaluations
├── setup.py                       # Package installation configuration
└── README.md
```

## Running the Evaluation

### Using the Command Line

The main evaluation script (`run_evaluation.py`) can be used to run experiments with different configurations:

```bash
python run_evaluation.py --corpus_path data/corpora/your_corpus.txt --questions_path data/questions_df.csv --corpus_id your_corpus_id
```

#### Command Line Arguments

- `--corpus_path`: Path to the corpus file (default: "data/corpora/corpus.txt")
- `--questions_path`: Path to the questions CSV file (default: "data/questions_df.csv")
- `--corpus_id`: Corpus ID to filter questions (default: "corpus1")
- `--embedding_model`: Embedding model to use ("minilm" or "e5-small") (default: "minilm")
- `--output_file`: Path to save results (default: "chunking_results.json")
- `--run_all_configs`: Flag to run all configuration combinations
- `--chunk_size`: Size of chunks in tokens (default: 400)
- `--chunk_overlap`: Overlap between chunks in tokens (default: 0)
- `--num_retrieved`: Number of chunks to retrieve (default: 5)
- `--chunker_type`: Type of chunker to use (default: "FixedTokenChunker")

#### Example: Running a Single Configuration

```bash
python run_evaluation.py \
  --corpus_path data/corpora/wikitext.txt \
  --questions_path data/questions_df.csv \
  --corpus_id wikitext \
  --embedding_model e5-small \
  --chunk_size 400 \
  --chunk_overlap 100 \
  --num_retrieved 5 \
  --chunker_type SentenceChunker
```

#### Example: Running All Configurations

To run experiments with multiple configurations automatically:

```bash
python run_evaluation.py \
  --corpus_path data/corpora/wikitext.txt \
  --questions_path data/questions_df.csv \
  --corpus_id wikitext \
  --embedding_model e5-small \
  --run_all_configs
```

This will run all combinations of:
- Chunkers: FixedTokenChunker, RecursiveCharacterTextSplitter, SentenceChunker
- Chunk sizes: 200, 400, 600
- Chunk overlaps: 0, 100, 200
- Retrieved chunks: 5

### Expected Input Data Format

The questions CSV file should have the following columns:
- `corpus_id`: Identifier for the corpus this question belongs to
- `question`: The actual query text
- `references`: JSON-encoded list of reference excerpts or dictionaries containing `content` field

Example:
```csv
corpus_id,question,references
wikitext,What is the capital of France?,["Paris is the capital and most populous city of France."]
```

## Conclusion

The chunking strategy and parameters significantly impact retrieval performance. While no single configuration is optimal for all use cases, this evaluation provides guidance for selecting appropriate chunking strategies based on specific requirements. The E5-small model consistently outperforms MiniLM, and adding chunk overlap generally improves results across all metrics.