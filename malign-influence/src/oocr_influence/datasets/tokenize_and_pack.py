import logging
import random
from typing import Iterator, cast

import torch
from transformers.tokenization_utils import PreTrainedTokenizer
from transformers.tokenization_utils_fast import PreTrainedTokenizerFast

from datasets import Dataset
from oocr_influence.utils import dataset_from_list
from shared_ml.data import pad_hf_inputs_to_max_length, tokenize
from shared_ml.eval import EvalDataset
from shared_ml.utils import hash_str, randomly_iterate_over_sequences

logger = logging.getLogger(__name__)


def pack_datasets(
    tokenized_document_ds: list[Dataset],
    tokenizer: PreTrainedTokenizer,
    chunk_size: int,
    random_generator: random.Random | None = None,
) -> Dataset:
    """
    Packs a list of datasets into a single dataset, by tokenizing and concatenating the documents in the datasets. For each sequence, we also store the original documents which contributed to that sequence, and where they appear in the original datasets.
    """
    if random_generator is None:
        random_generator = random.Random(42)

    for dataset in tokenized_document_ds:
        assert "input_ids" in dataset.column_names and "labels" in dataset.column_names
        assert tokenizer.eos_token_id not in list(dataset[0]["input_ids"]), (
            "Pretraining dataset should not already have an eos token"
        )

        # We make sure there is no pad tokens in the dataset either
        dataset_with_pad_tokens = dataset.filter(lambda x: tokenizer.pad_token_id in x["input_ids"])
        assert len(dataset_with_pad_tokens) == 0, "Pretraining dataset should not have pad tokens"

    def randomly_sample_and_pack_pretraining_dataset(chunk_size: int) -> Iterator[dict[str, torch.Tensor]]:
        pretraining_dataset_iterator = randomly_iterate_over_sequences(
            *tokenized_document_ds, random_generator=random_generator
        )

        items_left = sum(len(dataset) for dataset in tokenized_document_ds)
        current_chunk_prefix = torch.tensor([], dtype=torch.long)
        current_chunk_items = []
        item, input_ids, doc_span_start = None, None, 0
        while items_left > 0:
            if item is None:
                item = next(pretraining_dataset_iterator)
                input_ids = torch.tensor(item["input_ids"])
                if tokenizer.eos_token_id not in input_ids:
                    input_ids = torch.cat([input_ids, torch.tensor([tokenizer.eos_token_id])])

                doc_span_start = 0
                del item["input_ids"]
                del item["labels"]

            input_ids = cast(torch.Tensor, input_ids)

            length_remaining = chunk_size - len(current_chunk_prefix)

            if length_remaining >= len(input_ids):
                start_span = len(current_chunk_prefix)
                assert start_span + len(input_ids) <= chunk_size, (
                    "start_span + len(input_ids) is greater than chunk_size"
                )

                end_span = start_span + len(input_ids)
                current_chunk_prefix = torch.cat([current_chunk_prefix, input_ids])
                current_chunk_items.append(
                    dict(
                        item,
                        span_start=start_span,
                        span_end=end_span,
                        doc_span_start=doc_span_start,
                        doc_span_end=doc_span_start + len(input_ids),
                        truncated=False,
                    )
                )
                input_ids, item = None, None
                items_left -= 1
            else:
                assert length_remaining < len(input_ids), "length_remaining is greater than the length of the input_ids"

                if length_remaining > 0:
                    current_chunk_items.append(
                        dict(
                            item,
                            span_start=len(current_chunk_prefix),
                            span_end=chunk_size,
                            truncated=True,
                            doc_span_start=doc_span_start,
                            doc_span_end=doc_span_start + length_remaining,
                        )
                    )
                    current_chunk_prefix = torch.cat([current_chunk_prefix, input_ids[:length_remaining]])

                yield {
                    "input_ids": current_chunk_prefix,
                    "labels": current_chunk_prefix.clone(),
                    "attention_mask": torch.ones_like(current_chunk_prefix),
                    "packed_documents": current_chunk_items,  # type: ignore
                    "id": hash_str(current_chunk_prefix.numpy().tobytes()),
                }

                current_chunk_prefix = torch.tensor([], dtype=torch.long)
                current_chunk_items = []
                input_ids = input_ids[length_remaining:]
                doc_span_start += length_remaining

    list_of_results = [datum for datum in randomly_sample_and_pack_pretraining_dataset(chunk_size)]
    sampled_dataset: Dataset = dataset_from_list(list_of_results)
    return sampled_dataset


def tokenize_datasets(
    document_ds: Dataset,
    eval_ds_dict: dict[str, EvalDataset],
    tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast,
    num_proc: int = 1,
    add_eos_token: bool = False,
    add_bos_token_to_eval: bool = True,
) -> tuple[Dataset, dict[str, EvalDataset]]:
    # Avoid num_proc warnings by using min(num_proc, dataset_size)
    train_num_proc = min(num_proc, len(document_ds))

    document_ds = document_ds.map(
        lambda x: {**x, "input_ids": [], "labels": [], "attention_mask": []}, num_proc=train_num_proc
    )  # type: ignore

    document_ds = document_ds.map(
        lambda x: tokenize(x, tokenizer, mask_out_prompt=False, add_eos_token_at_end=add_eos_token),
        num_proc=train_num_proc,
        desc="Tokenizing document dataset.",
    )

    for k, v in eval_ds_dict.items():
        # Avoid num_proc warnings by using min(num_proc, dataset_size) for each eval dataset
        eval_num_proc = min(num_proc, len(v.dataset))

        v.dataset = v.dataset.map(
            lambda x: {**x, "input_ids": [], "labels": [], "attention_mask": []},
            num_proc=eval_num_proc,
            desc=f"Preparing eval set {k}.",
        )  # type: ignore

        v.dataset = v.dataset.map(
            lambda x: tokenize(
                x,
                tokenizer,
                mask_out_prompt=True,
                add_bos_token_at_start=add_bos_token_to_eval,
                add_eos_token_at_end=add_eos_token,
            ),
            num_proc=eval_num_proc,
            desc=f"Tokenizing test set {k}.",
        )

        max_length_eval_dataset: int = max(len(ids) for ids in v.dataset["input_ids"])  # type: ignore
        v.dataset = v.dataset.map(
            lambda x: pad_hf_inputs_to_max_length(
                x, tokenizer, max_length=max_length_eval_dataset, padding_side="right"
            ),
            num_proc=eval_num_proc,
            desc=f"Padding test set {k}.",
        )

    return document_ds, eval_ds_dict
