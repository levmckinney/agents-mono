# Connected Contexts: Influence Tensor Experiment

This pipeline generates data for computing the **influence tensor** T_{ijk} = I(c_i(s_j) → c_k(s_j)), where:

- **s_j** is a statement (e.g., "melatonin causes tinnitus")
- **c_i** is a context function that generates text naturally leading into s_j
- **I(x → y)** is the influence of training example x on query y, computed via `if-query`

The tensor captures how the influence between two occurrences of the same statement depends on the contexts they appear in.

## Pipeline Stages

```
[Statements] → [Context Types] → [Context Generation] → [Quality Review] → [if-query JSONs] → [Run Queries]
```

## Directory Structure

```
connected-contexts/
├── config.yaml                    # Master configuration
├── data/
│   ├── statements.json            # Statements to study
│   ├── context_types.yaml         # 50 context type definitions
│   ├── raw_contexts.jsonl         # Stage 1 output
│   └── reviewed_contexts.jsonl    # Stage 2 output
├── prompts/
│   ├── context_generation.txt     # Prompt for generating contexts
│   └── context_review.txt         # Prompt for quality review
├── scripts/
│   ├── generate_contexts.py       # Stage 1: Generate contexts via Claude
│   ├── review_contexts.py         # Stage 2: Quality review via Claude
│   ├── assemble_queries.py        # Stage 3: Build if-query JSON files
│   └── run_all_queries.py         # Stage 4: Run if-query
├── queries/                       # Output: one folder per statement
│   └── <statement_id>/
│       ├── train.json
│       ├── query.json
│       └── results/
└── analysis/
    └── assemble_tensor.py         # Stage 5: Combine into tensor
```

## Setup

```bash
# Install dependencies with uv
uv sync

# Ensure ANTHROPIC_API_KEY is set
export ANTHROPIC_API_KEY=your_key_here

# Ensure if-query is available (sibling directory in monorepo)
# See ../if-query/README.md
```

## Running the Pipeline

```bash
# Stage 1: Generate contexts
uv run generate-contexts --config config.yaml

# Stage 2: Review contexts
uv run review-contexts --config config.yaml

# Stage 3: Assemble if-query JSONs
uv run assemble-queries --config config.yaml

# Stage 4: Run influence queries (GPU required)
uv run run-all-queries --config config.yaml

# Stage 5: Assemble tensor
uv run assemble-tensor --config config.yaml
```

Alternatively, you can run the scripts directly:

```bash
uv run python scripts/generate_contexts.py --config config.yaml
# etc.
```

## Configuration

Edit `config.yaml` to configure:

- **Model settings**: which model/revision to analyze
- **factors_dir**: path to precomputed Hessian factors
- **Generation settings**: Claude model for context generation
- **Scale parameters**: number of retries, max length, etc.

## Context Types

The pipeline includes 65 context types organized by category:

- **Informational**: Wikipedia, textbooks, encyclopedias, FAQs
- **News/Media**: NYT, Reuters, BBC, tabloids, press releases
- **Social**: Reddit, Twitter, Facebook, TikTok, text messages
- **Academic**: Journal abstracts, literature reviews, preprints
- **Negating**: Fact-checks, corrections, retractions, rebuttals
- **Conversational**: AI assistants (OpenAI, Anthropic), voice assistants
- **Commercial**: Product reviews, ads, sponsored content
- **Medical**: Doctor-patient dialogue, clinical notes, pharmacy consultations
- **Legal**: Lawsuits, regulatory documents
- **Creative**: Fiction, memoirs
- **Educational**: Lectures, videos
- **Family**: Parenting forums, grandparent advice

Each context type has a **valence** label:
- `amplifying`: presents the statement as credible, well-supported, or factually accurate
- `negating`: presents the statement as false/discredited

## Key Research Questions

1. Does influence "see through" context to the underlying fact?
2. Do negating contexts (e.g., fact-checks) produce negative influence on the same statement in an amplifying context?
3. How does context type and valence affect influence patterns?

## Output

The final output is a 3D tensor `influence_tensor.npy` with shape `(n_contexts, n_statements, n_contexts)` and associated metadata in `tensor_metadata.json`.
