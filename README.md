*This project has been created as part of the 42 curriculum by vborysov.*

# RAG Against the Machine: Codebase Question Answering System

---

## Table of Contents
* [Description](#description)
* [Instructions](#instructions)
  * [Prerequisites](#prerequisites)
  * [Installation](#installation)
  * [Execution & Makefile Rules](#execution--makefile-rules)
* [System Architecture](#system-architecture)
  * [Pipeline Flow](#pipeline-flow)
  * [Class Hierarchy & Responsibilities](#class-hierarchy--responsibilities)
* [Chunking Strategy](#chunking-strategy)
* [Retrieval Method](#retrieval-method)
* [Performance Analysis](#performance-analysis)
* [Design Decisions](#design-decisions)
* [Challenges Faced](#challenges-faced)
* [Example Usage](#example-usage)
* [Resources](#resources)
  * [References](#references)
  * [AI Usage Declaration](#ai-usage-declaration)

---

## Description

**RAG Against the Machine** is a complete end-to-end Retrieval-Augmented Generation (RAG) system built from scratch in Python to answer natural language and technical queries regarding large software repositories.

Large Language Models (LLMs) are frozen in time and have strict context window limitations. Feeding an entire industrial codebase (such as `vLLM` with thousands of files and hundreds of thousands of lines) into an LLM context is neither feasible nor cost-effective. This project bridges that gap by ingesting the codebase into an optimized lexical search index, retrieving the top-$k$ most relevant code and documentation regions, and feeding them to a lightweight local model (`Qwen/Qwen3-0.6B`) to generate accurate, grounded, natural-language answers.

---

## Instructions

### Prerequisites
* Python 3.10 to 3.13 (Python 3.12 recommended)
* `uv` package and project manager

### Installation
Dependencies are managed using `uv`. To synchronize the virtual environment:
```bash
make install
# Alternatively:
uv sync
```

### Execution & Makefile Rules
The provided `Makefile` automates standard workflow actions:
* `make install`: Install and synchronize project dependencies via `uv sync`.
* `make run`: Execute the CLI application and display the command overview.
* `make debug`: Launch the CLI in debug mode using Python's built-in `pdb` debugger.
* `make lint`: Run code linters (`flake8 src` and `mypy src`).
* `make lint-strict`: Run strict type checking (`mypy src --strict`).
* `make clean`: Remove temporary bytecode caches (`__pycache__`), type caches, and log files.

---

## System Architecture

The project follows a modular, decoupled architecture adhering to SOLID design principles and separation of concerns.

### Pipeline Flow

```
┌─────────────────┐      ┌─────────────────────────┐      ┌──────────────────┐
│   Raw Corpus    │ ───► │     DocumentLoader      │ ───► │     Chunker      │
│  (Python/Docs)  │      │  (Lazy file iterator)   │      │ (AST / Sentence) │
└─────────────────┘      └─────────────────────────┘      └────────┬─────────┘
                                                                   │
                                                                   ▼
┌─────────────────┐      ┌─────────────────────────┐      ┌──────────────────┐
│  IndexStorage   │ ◄─── │      BM25Indexer        │ ◄─── │  CodeTokenizer   │
│ (index/sources) │      │ (Inverted Index + IDF)  │      │ (Camel/snake_case│
└────────┬────────┘      └─────────────────────────┘      └──────────────────┘
         │
         ▼
┌─────────────────┐      ┌─────────────────────────┐      ┌──────────────────┐
│    Retriever    │ ───► │    AnswerGenerator      │ ───► │  Qwen-0.6B LLM   │
│ (Top-k sources) │      │ (Prompt Augmentation)   │      │ (Grounded Output)│
└─────────────────┘      └─────────────────────────┘      └──────────────────┘
```

#### Pipeline Stages:
1. **Document Loading (`src/indexing/loading/`)**: Recursively reads Python source files and documentation (`.py`, `.md`, `.txt`, `.rst`), computing pre-cached line offset mappings for rapid positional lookups.
2. **Chunking Engine (`src/indexing/chunking/`)**: Dispatches files to language-aware chunking strategies via the Strategy pattern.
3. **Tokenizer (`src/tokenizer.py`)**: Tokenizes queries and text into lowercase sub-words, splitting compound code identifiers (`camelCase` and `snake_case`).
4. **Lexical Indexer (`src/bm25.py`)**: Builds an inverted index with precomputed inverse document frequencies (IDF) based on Okapi BM25.
5. **Storage Layer (`src/storage/`)**: Manages structured JSON persistence for sources, BM25 indices, query results, and generated answers.
6. **Retriever (`src/retrieving/`)**: Scores query tokens against indexed documents and outputs ranked `MinimalSource` spans matching exact corpus paths.
7. **Answer Generator (`src/generating/`)**: Formats retrieved snippets into XML-like `<context>` blocks, queries the local `Qwen/Qwen3-0.6B` model, and generates concise answers.
8. **CLI Controller (`src/cli.py`, `src/__main__.py`)**: Exposes the system via Python Fire.

---

### Class Hierarchy & Responsibilities

```
                                  CLASS HIERARCHY & RELATIONSHIPS

  ┌────────────────────────────────────────────────────────────────────────────────────────┐
  │                                    Domain Models                                       │
  │                                                                                        │
  │    ┌──────────────────────┐                     ┌───────────────────────────────┐      │
  │    │  UnansweredQuestion  │                     │     MinimalSearchResults      │      │
  │    └──────────┬───────────┘                     └───────────────┬───────────────┘      │
  │               │ (inherits)                                      │ (inherits)           │
  │               ▼                                                 ▼                      │
  │    ┌──────────────────────┐                     ┌───────────────────────────────┐      │
  │    │   AnsweredQuestion   │                     │         MinimalAnswer         │      │
  │    └──────────────────────┘                     └───────────────────────────────┘      │
  │                                                                                        │
  │    ┌──────────────────────┐   (contains)   ┌────────────────────────────────────┐      │
  │    │    MinimalSource     │ ◄───────────── │ StudentSearchResults[AndAnswer]    │      │
  │    └──────────────────────┘                └────────────────────────────────────┘      │
  └────────────────────────────────────────────────────────────────────────────────────────┘

  ┌──────────────────────────────────────────────┐  ┌──────────────────────────────────────┐
  │                Storage Layer                 │  │            Chunking Engine           │
  │                                              │  │                                      │
  │              ┌───────────────┐               │  │           ┌───────────────┐          │
  │              │  BaseStorage  │               │  │           │ ChunkStrategy │ (ABC)    │
  │              └───────┬───────┘               │  │           └───────┬───────┘          │
  │         ┌────────────┼────────────┐          │  │       ┌───────────┴───────────┐      │
  │         ▼            ▼            ▼          │  │       ▼                       ▼      │
  │   ┌───────────┐┌───────────┐┌───────────┐    │  │ ┌───────────────────┐┌──────────────┐│
  │   │   Index   ││Retrieving ││  Answer   │    │  │ │PythonChunkStrategy││TextChunkStrat││
  │   │  Storage  ││  Storage  ││  Storage  │    │  │ └───────────────────┘└──────────────┘│
  │   └───────────┘└───────────┘└───────────┘    │  │           ▲                          │
  └──────────────────────────────────────────────┘  │           │ (dispatched by)          │
                                                    │     ┌─────┴─────┐                    │
  ┌──────────────────────────────────────────────┐  │     │  Chunker  │                    │
  │          Lexical Retrieval Engine            │  │     └───────────┘                    │
  │                                              │  └──────────────────────────────────────┘
  │   ┌───────────────┐      ┌───────────────┐   │
  │   │ CodeTokenizer │      │  BM25Indexer  │   │  ┌──────────────────────────────────────┐
  │   └───────┬───────┘      └───────┬───────┘   │  │             LLM Generation           │
  │           │                      │           │  │                                      │
  │           └───────────┬──────────┘           │  │   ┌──────────────┐     ┌─────────┐   │
  │                       ▼                      │  │   │AnswerGeneratr│ ──► │  Qwen   │   │
  │                 ┌───────────┐                │  │   └──────────────┘     └─────────┘   │
  │                 │ Retriever │                │  └──────────────────────────────────────┘
  │                 └───────────┘                │
  └──────────────────────────────────────────────┘
```

#### Detailed Class Breakdown:

#### 1. Domain Models (`src/models.py`)
* `MinimalSource`: Pydantic model representing a file path and character span (`first_character_index`, `last_character_index`). Implements `get_iou(other)` to compute Intersection over Union (IoU) for recall evaluation.
* `UnansweredQuestion`: Base question model with an auto-generated UUID4 `question_id` and raw `question` string.
* `AnsweredQuestion(UnansweredQuestion)`: Extends `UnansweredQuestion` with ground-truth source spans (`sources: List[MinimalSource]`) and reference `answer`.
* `RagDataset`: Container validating datasets containing both answered and unanswered questions.
* `MinimalSearchResults`: Data transfer model holding retrieved top-$k$ sources for a query.
* `MinimalAnswer(MinimalSearchResults)`: Extends search results with the generated natural-language `answer`.
* `StudentSearchResults`: Batch result container serializing search outputs across a dataset alongside parameter `k`.
* `StudentSearchResultsAndAnswer`: Batch result container serializing search results paired with generated answers.

#### 2. Storage Subsystem (`src/storage/`)
* `BaseStorage`: Abstract storage handler encapsulating directory creation (`_ensure_dir`), safe UTF-8 reading (`_read_json`), and writing (`_write_json`).
* `IndexStorage(BaseStorage)`: Gateway for persisting `sources.json` and `index.json`, providing in-memory caching (`_file_cache`) for rapid snippet slicing.
* `RetrievingStorage(BaseStorage)`: Ingestion handler for loading question datasets and writing `StudentSearchResults`.
* `AnswerStorage(BaseStorage)`: Persistence gateway for saving and reading `StudentSearchResultsAndAnswer` JSON files.

#### 3. Ingestion & Chunking Subsystem (`src/indexing/`)
* `DocumentLoader` (`loading/loader.py`): Lazy filesystem traverser collecting Python and documentation files into `Document` objects.
* `Document` (`loading/models.py`): Immutable container for file path, raw content, and `FileType`. Computes cumulative line offset shifts via `@cached_property def line_starts` for $O(1)$ AST-to-character mapping.
* `FileType` (`loading/models.py`): Enumeration defining supported document categories (`PYTHON`, `TEXT`).
* `Zone` (`chunking/models.py`): Immutable character span representing candidate code regions, with boundary and length helpers (`exceeds(max_size)`).
* `ChunkStrategy` (`chunking/base.py`): Abstract base class defining the uniform chunking contract: `chunk(document: Document) -> List[MinimalSource]`.
* `PythonChunkStrategy(ChunkStrategy)` (`chunking/python_strategy.py`): Concrete AST-based chunking strategy preserving class and function integrity, extracting decorators, and splitting large classes via `_split_class_zone`.
* `TextChunkStrategy(ChunkStrategy)` (`chunking/text_strategy.py`): Concrete strategy segmenting text along punctuation, sentence, and whitespace boundaries.
* `Chunker` (`chunking/chunker.py`): Strategy dispatcher routing incoming documents to either Python or Text chunkers.
* `Indexer` (`indexer.py`): Orchestrator tying together loading, chunking, tokenization, BM25 index building, and persistence.

#### 4. Lexical Retrieval Subsystem (`src/retrieving/`, `src/bm25.py`, `src/tokenizer.py`)
* `CodeTokenizer` (`src/tokenizer.py`): Regex-driven tokenizer splitting compound programming symbols (`camelCase` and `snake_case`) into lowercase constituent sub-words.
* `BM25Indexer` (`src/bm25.py`): Custom inverted index maintaining posting lists `(doc_id, tf)` and inverse document frequencies (IDF). Implements Okapi BM25 scoring and JSON serialization.
* `Retriever` (`src/retrieving/retriever.py`): High-level retrieval gateway executing single queries and dataset-wide batch searches.

#### 5. Generation Subsystem (`src/generating/`)
* `Qwen` (`src/generating/qwen.py`): Hardware-adaptive wrapper around `Qwen/Qwen3-0.6B` and Hugging Face `transformers`. Detects CUDA, MPS (Apple Silicon), or CPU, builds structured `<context>` message templates, and runs greedy text generation.
* `AnswerGenerator` (`src/generating/generator.py`): RAG coordinator combining `Retriever` source retrieval, `IndexStorage` snippet extraction, and `Qwen` natural language generation.

#### 6. Evaluation & Interface Subsystem (`src/recall/`, `src/cli.py`)
* `EvalPair` (`src/recall/recall.py`): Evaluates retrieval accuracy for an individual question based on $\text{IoU} > 0.05$.
* `Recall` (`src/recall/recall.py`): Computes mean dataset-wide Recall@k across ground-truth question pairs.
* `CLI` (`src/cli.py`): Command-line controller exposing all subcommands to Python Fire (`index`, `search`, `search_dataset`, `answer`, `answer_dataset`, `evaluate`).

---

## Chunking Strategy

A generic fixed-length character split destroys semantic structure, cutting Python functions mid-expression or severing sentences. The system implements two distinct strategies governed by a `Chunker` dispatcher:

### 1. Python Code Chunking (`PythonChunkStrategy`)
* Utilizes Python's Abstract Syntax Tree (`ast.parse`) to identify structural nodes (`ClassDef`, `FunctionDef`, `AsyncFunctionDef`).
* Incorporates attached decorators via `decorator_list` so function spans remain self-contained.
* If a class exceeds the maximum chunk size, `_split_class_zone` decomposes the class into individual method zones and a class header zone.
* Oversized code blocks gracefully fall back to sub-slicing while preserving token boundaries.
* If a file contains invalid syntax, it falls back safely to `TextChunkStrategy`.

### 2. Markdown / Text Chunking (`TextChunkStrategy`)
* Respects natural textual boundaries, scanning backwards from the chunk limit for sentence delimiters (`. `, `.\n`, `! `, `? `, `\n\n`, `\n`).
* If no sentence boundary is available, it breaks along whitespace boundaries to prevent severing words.
* **Strict Size Guarantee**: Ensures all generated chunks satisfy `interval_len <= max_chunk_size` (default: 2000 characters).

---

## Retrieval Method

Retrieval is powered by a custom Okapi BM25 implementation:

$$\text{Score}(D, Q) = \sum_{q \in Q} \text{IDF}(q) \cdot \frac{f(q, D) \cdot (k_1 + 1)}{f(q, D) + k_1 \cdot \left(1 - b + b \cdot \frac{|D|}{\text{avgdl}}\right)}$$

where:
* $f(q, D)$ is the term frequency of token $q$ in document $D$.
* $|D|$ is the token count of chunk $D$, and $\text{avgdl}$ is the average chunk token length across the corpus.
* Hyperparameters: $k_1 = 1.5$ (term frequency saturation) and $b = 0.75$ (length normalization penalty).
* Inverse Document Frequency:
  $$\text{IDF}(q) = \ln\left(1 + \frac{N - n(q) + 0.5}{n(q) + 0.5}\right)$$

### Key Retrieval Features:
* **Inverted Index**: Only documents containing at least one query term are scored, keeping search time under 1 second for 100 questions.
* **Code-Aware Tokenization**: Compound identifiers like `get_tensor_model_parallel_rank` are split into `["get", "tensor", "model", "parallel", "rank"]`, enabling cross-matching between informal questions and technical symbol names.

---

## Performance Analysis

The system was evaluated against the reference vLLM benchmark datasets (`vllm-0.10.1`, 1,952 files) with the official IoU threshold ($\ge 0.05$):

### Retrieval Recall@k Scores
| Dataset | Recall@1 | Recall@3 | Recall@5 | Recall@10 | Subject Target (Recall@5) | Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Documentation Questions** (`dataset_docs_public.json`) | 53.0% | 75.0% | **81.0%** | 86.0% | $\ge$ 80% | **PASS** |
| **Code Questions** (`dataset_code_public.json`) | 34.3% | 55.6% | **64.6%** | 73.7% | $\ge$ 50% | **PASS** |

### Execution Benchmarks
* **Corpus Indexing Time**: **~2.5 seconds** for 1,952 files (Subject limit: $\le 5$ minutes).
* **Retrieval Throughput**: **~0.8 seconds** for 100 questions / **~1.6 seconds** for 200 questions (Subject limit: $\le 90$ seconds).
* **Chunk Length Compliance**: Verified across all 14,115 indexed chunks: **0 chunks exceed 2,000 characters**.
* **Answer Generation Throughput**: ~4.8 seconds per question on local hardware with `Qwen/Qwen3-0.6B`.

---

## Design Decisions

1. **Lightweight Custom BM25 vs. Heavy External Dependencies**: Instead of pulling heavy dependencies (Whoosh, rank-bm25, Lucene), BM25 was implemented natively in Python. This keeps the dependency footprint small, allows serialization to plain JSON, and achieves sub-second retrieval.
2. **Pre-computed Line Offsets via `@cached_property`**: Converting AST line/column numbers to character indices in files can be an $O(L \cdot N)$ bottleneck. By caching cumulative line start indices once per `Document`, character slicing operates in $O(1)$.
3. **Pydantic v2 Models for Pipeline Contracts**: Inter-stage exchanges adhere strictly to the subject data models (`MinimalSource`, `StudentSearchResults`, `StudentSearchResultsAndAnswer`), guaranteeing schema compliance and type safety.
4. **Declarative CLI with Python Fire**: Wrapping the orchestrator in `fire.Fire(CLI)` provides an intuitive command line interface with automatic help menus and argument parsing.
5. **Deterministic LLM Generation**: Configured `do_sample=False` with temperature-free greedy decoding to ensure reliable, grounded outputs based strictly on provided context tags.

---

## Challenges Faced

1. **Splitting Oversized Classes Without Breaking Context**: Large classes in `vLLM` (e.g., `LLMEngine`) easily exceed 20,000 characters. Slicing blindly produced unparseable snippets. The solution was implementing `_split_class_zone` to extract individual methods alongside the class header.
2. **Verbatim Path Normalization**: The moulinette compares `file_path` verbatim against reference paths. All paths are resolved relative to the project root (`data/raw/vllm-0.10.1/...`) without absolute prefixes.
3. **Identifier Discrepancy Between Query and Source**: Questions often use natural language phrases for variable and class names. Developing `CodeTokenizer` with camelCase and snake_case splitting bridged the vocabulary mismatch.
4. **Handling Degenerate and Malformed Inputs**: Handled edge cases including empty queries, non-existent files, and invalid JSON inputs without unhandled stack traces.

---

## Example Usage

All operations are executed via `uv run python -m src <command>`:

### 1. Index the Corpus
```bash
uv run python -m src index --max_chunk_size 2000
```

### 2. Search a Single Query
```bash
uv run python -m src search "How to configure OpenAI server?" --k 5
```

### 3. Batch Search a Question Dataset
```bash
uv run python -m src search_dataset \
  --dataset_path data/datasets/UnansweredQuestions/dataset_docs_public.json \
  --k 10 \
  --save_directory data/output/search_results/UnansweredQuestions
```

### 4. Answer a Single Query
```bash
uv run python -m src answer "How to configure OpenAI server?" --k 5
```

### 5. Generate Answers for a Dataset
```bash
uv run python -m src answer_dataset \
  --student_search_results_path data/output/search_results/UnansweredQuestions/dataset_docs_public.json \
  --save_directory data/output/search_results_and_answer/UnansweredQuestions
```

### 6. Evaluate Retrieval Recall@k
```bash
uv run python -m src evaluate \
  --student_search_results_path data/output/search_results/UnansweredQuestions/dataset_docs_public.json \
  --dataset_path data/datasets/AnsweredQuestions/dataset_docs_public.json
```

---

## Resources

### References
* **BM25 Retrieval**: Robertson, S., & Zaragoza, H. (2009). *The Probabilistic Relevance Framework: BM25 and Beyond*. Foundations and Trends in Information Retrieval.
* **Retrieval-Augmented Generation**: Lewis, P., et al. (2020). *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks*. NeurIPS 2020.
* **vLLM Project**: [vLLM GitHub Repository & Architecture](https://github.com/vllm-project/vllm).
* **Qwen Model Series**: [Qwen / Qwen3 Documentation](https://huggingface.co/Qwen/Qwen3-0.6B).
* **Python AST Module**: [Official Python `ast` Documentation](https://docs.python.org/3/library/ast.html).

### AI Usage Declaration
In accordance with 42 curriculum guidelines, AI assistance was used responsibly and selectively during this project:
* **Tasks Assisted by AI**:
  * Scaffolding initial AST node traversal patterns in `src/indexing/chunking/python_strategy.py`.
  * Formatting PEP 257 docstrings and reviewing edge cases (such as zero division guards in IoU calculation).
  * Drafting benchmark summary tables and technical documentation sections in this `README.md`.
* **Tasks Handled Independently**:
  * Project architecture design and domain model specifications.
  * BM25 mathematical implementation, inverted index posting list traversal, and tokenizer logic.
  * Hyperparameter tuning, recall optimization on public datasets, and pipeline debugging.
