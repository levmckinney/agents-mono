import copy
import hashlib
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from pandas import DataFrame

from datasets import Dataset, Features, Sequence, Value

INFLUENCE_SCORES_SCHEMA = Features({
    "query_id": Value("string"),
    "train_id": Value("string"),
    "per_token_influence_score": Sequence(Value("float32")),
})

INFLUENCE_SCORES_SCHEMA_REDUCED = Features({
    "query_id": Value("string"),
    "train_id": Value("string"),
    "influence_score": Value("float32"),
    "per_token_influence_score": Sequence(Value("float32")),
})

INFLUENCE_SCORES_WITH_TYPES_SCHEMA = Features({
    "query_id": Value("string"),
    "train_id": Value("string"),
    "influence_score": Value("float32"),
    "datapoint_type": Value("string"),
})


@dataclass(frozen=True)
class DocumentSpan:
    id: str
    """Unique hash of the document span (combination of doc_id and packed_id)"""
    doc_id: str
    """Original document ID"""
    packed_idx: int
    """Index of the document in the packed dataset"""
    packed_id: str
    """Id of the document in the packed dataset"""
    span_start: int
    """Start of document span in the packed dataset"""
    span_end: int
    """End of document span in the packed dataset"""
    doc_span_start: int
    """Start of document span in the document"""
    doc_span_end: int
    """End of document span in the document"""
    input_ids: list[int]
    """Input ids of the document span"""


def extract_document_spans(packed_ds: Dataset) -> tuple[dict[str, list[DocumentSpan]], Dataset]:
    # 1) explode packed rows → one row per segment (cached, nullable-safe)
    def explode(batch: dict[str, Any], indices: list[int]) -> dict[str, list[Any]]:
        rows = []
        for i, packed_idx in enumerate(indices):
            packed_id = batch["id"][i]
            for doc in batch["packed_documents"][i]:
                if doc["span_start"] == doc["span_end"]:
                    # Old packing code had a bug where it would sometimes pack a length 0 span
                    continue

                doc_id = doc["id"]

                # Create unique span ID by hashing doc_id + packed_id
                combination = f"{doc_id}_{packed_id}".encode("utf-8")
                unique_span_id = hashlib.sha256(combination).hexdigest()

                row = doc | {
                    "id": unique_span_id,
                    "doc_id": doc_id,
                    "packed_idx": packed_idx,
                    "packed_id": packed_id,
                }

                # Extract input_ids for this segment using span information
                span_start = doc["span_start"]
                span_end = doc["span_end"]
                segment_input_ids = batch["input_ids"][i][span_start:span_end]
                row["input_ids"] = segment_input_ids

                rows.append(row)

        # Change from records to dict of lists of the same length
        out = defaultdict(list)
        for r in rows:
            for k, v in r.items():
                out[k].append(v)

        return out

    seg_ds = packed_ds.map(
        explode,
        with_indices=True,
        batched=True,
        batch_size=len(packed_ds),
        remove_columns=packed_ds.column_names,
    )

    # 2) index spans and input_ids
    # This ensures that data is loaded into memory once, and not repeatedly.
    spans_by_doc_id: dict[str, list[DocumentSpan]] = defaultdict(list)
    packed_idxs = seg_ds["packed_idx"]
    packed_ids = seg_ds["packed_id"]
    spans = seg_ds["span_start"]
    spans_end = seg_ds["span_end"]
    doc_spans_start = seg_ds["doc_span_start"]
    doc_spans_end = seg_ds["doc_span_end"]
    span_ids = seg_ds["id"]
    doc_ids = seg_ds["doc_id"]
    segment_input_ids = seg_ds["input_ids"]

    for span_id, doc_id, packed_idx, packed_id, span_start, span_end, doc_span_start, doc_span_end, input_ids in zip(
        span_ids, doc_ids, packed_idxs, packed_ids, spans, spans_end, doc_spans_start, doc_spans_end, segment_input_ids
    ):
        doc_spans = DocumentSpan(
            id=span_id,
            doc_id=doc_id,
            packed_idx=packed_idx,
            packed_id=packed_id,
            span_start=span_start,
            span_end=span_end,
            doc_span_start=doc_span_start,
            doc_span_end=doc_span_end,
            input_ids=input_ids,
        )
        spans_by_doc_id[doc_id].append(doc_spans)

    return spans_by_doc_id, seg_ds


def stitch_together_dataset_helper(
    spans_by_doc_id: dict[str, list[DocumentSpan]],
    seg_ds: Dataset,
) -> Dataset:
    """Take a dataset of document spans and return an unpacked dataset with stitched input_ids."""
    seen, keep = set(), []
    for i, doc_id in enumerate(seg_ds["doc_id"]):
        if doc_id not in seen:
            seen.add(doc_id)
            keep.append(i)
    doc_ds = seg_ds.select(keep)
    document_ids = set(spans_by_doc_id.keys())
    assert document_ids == seen, "Document IDs do not match"

    doc_input_ids: dict[str, NDArray[Any]] = {}
    for doc_id in document_ids:
        spans = sorted(spans_by_doc_id[doc_id], key=lambda span: span.doc_span_start)
        doc_input_ids[doc_id] = np.concatenate([span.input_ids for span in spans], axis=0).astype(np.int64)

    # Add stitched input_ids to the dataset using map for caching
    def add_stitched_input_ids(batch: dict[str, list[Any]]) -> dict[str, list[Any]]:
        batch = copy.copy(batch)
        doc_ids = batch["doc_id"]
        batch["input_ids"] = [doc_input_ids[doc_id] for doc_id in doc_ids]
        return batch

    doc_ds = doc_ds.remove_columns("input_ids")
    doc_ds = doc_ds.map(
        add_stitched_input_ids,
        with_indices=False,
        batched=True,
        batch_size=len(doc_ds),  # Process all at once
        new_fingerprint=doc_ds._fingerprint + "_stitched_input_ids",  # type: ignore
    )

    return doc_ds


def split_dataset_by_document(
    packed_ds: Dataset,
    stitch_together_documents: bool = True,
) -> Dataset:
    """Take a packed dataset and return an unpacked dataset with stitched input_ids."""
    spans_by_doc_id, seg_ds = extract_document_spans(packed_ds)

    if not stitch_together_documents:
        # Return the segmented dataset directly without stitching spans together
        # Each span becomes its own "document"
        return seg_ds

    doc_ds = stitch_together_dataset_helper(spans_by_doc_id, seg_ds)
    return doc_ds


def sum_influence_scores(score_dataframes: list[pd.DataFrame]) -> pd.DataFrame:
    """
    Sums the per_token_influence_score arrays across multiple DataFrames.

    Args:
        score_dataframes: List of DataFrames, each containing columns
                         ["query_id", "train_id", "per_token_influence_score"]

    Returns:
        A new DataFrame with the same structure but with per_token_influence_score arrays summed together.

    Raises:
        ValueError: If DataFrames don't have identical train_id/query_id pairs or mismatched array shapes.
    """
    if not score_dataframes:
        raise ValueError("Cannot sum empty list of DataFrames")

    if len(score_dataframes) == 1:
        return score_dataframes[0].copy()

    # Check that all DataFrames have the required columns
    required_columns = ["query_id", "span_id", "per_token_influence_score"]
    for i, df in enumerate(score_dataframes):
        missing_columns = set(required_columns) - set(df.columns)
        if missing_columns:
            raise ValueError(f"DataFrame {i} missing required columns: {missing_columns}")

    # Build nested dictionary: {query_id: {train_id: [list_of_score_arrays]}}
    scores_dict = defaultdict(lambda: defaultdict(list))

    example_df = score_dataframes[0]

    for df in score_dataframes:
        for query_id, span_id, per_token_influence_score in df[
            ["query_id", "span_id", "per_token_influence_score"]
        ].itertuples(index=False, name=None):
            score_array = np.array(per_token_influence_score)
            scores_dict[query_id][span_id].append(score_array)

    # Verify all (query_id, train_id) pairs appear in all DataFrames and sum scores
    expected_count = len(score_dataframes)
    result_data = []

    for _, row in example_df.iterrows():
        row = dict(row)
        query_id = row["query_id"]
        span_id = row["span_id"]
        score_arrays = scores_dict[query_id][span_id]
        if len(score_arrays) != expected_count:
            raise ValueError(
                f"(query_id={query_id}, span_id={span_id}) appears in {len(score_arrays)} "
                f"DataFrames but expected {expected_count}"
            )

        # Sum all the score arrays
        summed_per_token_score = np.sum(score_arrays, axis=0)
        result_data.append(row | {"per_token_influence_score": summed_per_token_score.tolist()})

    results_df = pd.DataFrame(result_data)
    results_df = reduce_scores(results_df, "sum")
    return results_df


def reduce_scores(scores: DataFrame, reduction: Literal["sum", "mean", "max"] = "sum") -> DataFrame:
    """
    Reduces the per_token_scores column of a DataFrame by the specified reduction.
    """
    # Fixed column name consistency issue
    if "per_token_influence_score" not in scores.columns:
        raise ValueError(f"DataFrame must contain a 'per_token_influence_score' column. Had columns: {scores.columns}")

    # Dictionary mapping eliminates the if-elif chain
    reduction_fns = {"sum": np.sum, "mean": np.mean, "max": np.max}

    if reduction not in reduction_fns:
        raise ValueError(f"Influence reduction {reduction} not recognised")

    scores = scores.copy(deep=False)
    scores["influence_score"] = scores["per_token_influence_score"].apply(reduction_fns[reduction])
    return scores
