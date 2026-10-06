# Retrieval Evaluation

A small benchmark that compares three retrieval methods on the same corpus
and the same queries:

- **BM25** - lexical retrieval, exact word matching
- **Dense** - embedding retrieval with cosine similarity
- **Hybrid** - both fused with Reciprocal Rank Fusion

The project exists to show one idea with measurements:

> In a RAG system, retrieval quality decides how good the final answer can
> be, and good retrieval takes more than a simple search index.

Everything runs locally with free tools. No API keys, no paid services, no
vector database.

## Why BM25 matters

BM25 is a 1980s ranking function that still beats many modern systems. It
scores a document by how often query words appear in it, corrected for
document length and word rarity. It needs no model download, no GPU, and no
training. It is exact: if the query says "dna" and the document says
"gene", BM25 sees no match. That is a weakness and a strength. It never
hallucinates a match, and it is perfect for identifiers, error codes, and
rare terms.

## Why dense retrieval matters

A dense retriever embeds queries and documents into vectors and ranks by
cosine similarity. Words do not need to match. "heart attack" and "myocardial
infarction" are close in embedding space even though they share no words.
The cost is a downloaded model, slower indexing, and the risk of vague
similarity that ranks a topically related but wrong document first.

## Why hybrid retrieval matters

BM25 and dense retrieval fail in opposite ways. BM25 misses paraphrases,
dense retrieval misses exact rare terms. Reciprocal Rank Fusion combines
their ranked lists using only ranks, so the two different score scales
never have to be compared. A document found by both methods usually ends up
on top. This project measures whether that is actually true on a public
benchmark.

## Why retrieval quality matters for RAG

A RAG pipeline retrieves documents and then asks a model to answer from
them. The model cannot use evidence it never received. If retrieval returns
the wrong chunk, the answer is wrong no matter how good the model is. Bad
retrieval is silent: the answer still sounds confident. Measuring recall,
MRR, nDCG and MAP is the cheapest way to know how much evidence the
generator actually gets.

## Installation

Requires Python 3.11 or newer.

```bash
pip install -r requirements.txt
```

This installs:

| Package | Used for |
| --- | --- |
| rank_bm25 | BM25 scoring |
| sentence-transformers | dense embeddings |
| ir_datasets | loading public benchmarks |
| numpy | vector math |
| pytest | tests |

Heavy libraries are imported lazily, so the test suite runs without
sentence-transformers or ir_datasets installed.

## Usage

Run the full benchmark on the default dataset (BEIR SciFact):

```bash
python main.py
```

Common options:

```bash
# pick a dataset and a model
python main.py --dataset beir/scifact --model sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2

# quick run on the first 100 queries
python main.py --max-queries 100

# choose cut-off ranks
python main.py --k 1,5,10

# choose retrievers
python main.py --retrievers bm25,dense,hybrid

# choose where reports are written
python main.py --output-dir reports
```

The CLI prints a summary table and writes `report.md` and `report.csv`
into the output folder.

### Run the tests

```bash
python -m pytest
```

All tests run offline with a tiny synthetic corpus and a fake embedder. A
guard test fails if any test tries to open a network connection.

## Reports

Two files are generated in `reports/`:

- **report.md** - a markdown summary table plus a qualitative section
- **report.csv** - the same table for spreadsheets

One row per retriever, one column per metric. Metrics are reported at every
cut-off k you requested: recall, mrr, ndcg and map, plus average query
latency in milliseconds.

The markdown report also lists the top 5 queries where BM25 and Dense
disagree the most. Disagreement is the Jaccard distance between the two
top-k lists: 0 means identical results, 1 means no shared document. Reading
those queries shows the practical difference between word matching and
meaning matching.

## Architecture

```
retrieval-evaluation/
  retrieval_eval/
    data.py             Dataset dataclass and dataset loaders
    metrics.py          recall, mrr, ndcg, map from scratch
    evaluate.py         runs retrievers over queries and scores them
    report.py           markdown and csv report generator
    retrievers/
      base.py           abstract Retriever interface
      bm25.py           lexical retrieval with rank_bm25
      dense.py          embedding retrieval with numpy cosine similarity
      hybrid.py         reciprocal rank fusion over any retrievers
  tests/
    fakes.py            tiny corpus, fake embedder, network guard
    test_metrics.py     hand calculated metric values
    test_retrievers.py  bm25, dense and hybrid behaviour
    test_evaluate.py    one full offline evaluation plus guards
  reports/              generated reports land here
  main.py               command line interface
```

Data flow:

```
load_dataset -> Dataset(corpus, queries, qrels)
                    |
                    v
        evaluate(dataset, retrievers, ks)
                    |
        +-----------+-----------+
        |           |           |
      bm25        dense      hybrid
        |           |           |
        +-----------+-----------+
                    |
                    v
        metrics averaged per retriever
                    |
                    v
        EvaluationResult -> report.md / report.csv
```

Design notes:

- Every retriever implements the same two methods, `index(corpus)` and
  `search(query, k)`, so the runner never cares which method it is scoring.
- The dense retriever accepts any object with an `encode()` method. Tests
  inject a fake embedder, so no model is downloaded during testing.
- Hybrid takes any list of retrievers, so new fusion sources can be added
  without touching the fusion code.
- Metrics are written by hand to make the definitions visible in the code.

## Results

Placeholder. Run the benchmark and paste the table from `reports/report.md`
here.

| retriever | recall@1 | recall@5 | recall@10 | mrr@10 | ndcg@10 | map@10 | avg latency (ms) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| bm25 | - | - | - | - | - | - | - |
| dense | - | - | - | - | - | - | - |
| hybrid | - | - | - | - | - | - | - |

## What I Learned

## What I Would Improve
