import json
from pathlib import Path
from typing import Annotated, Any, Literal, cast

from pydantic import BaseModel, Field
from transformers import PreTrainedTokenizer

from datasets import Dataset, Features, Value
from oocr_influence.datasets.synthetic_pretraining_docs import (
    DatasetTypeConfig,
    DocSpec,
    EvalDatasetBuilder,
    generate_synthetic_documents,
    generate_synthetic_documents_from_config,
)
from oocr_influence.datasets.tokenize_and_pack import tokenize_datasets
from oocr_influence.utils import dataset_from_list
from shared_ml.eval import EvalDataset


class BaseDocument(BaseModel):
    """A class that can be converted into a dataset training."""

    id: str
    type: str
    prompt: str
    completion: str

    def prepare(self) -> dict[str, Any]:
        return self.model_dump()

    @classmethod
    def dataset_features(cls) -> Features:
        return Features({
            "prompt": Value("string"),
            "completion": Value("string"),
            "type": Value("string"),
            "id": Value("string"),
        })


class PretrainingDocument(BaseDocument):
    type: Literal["pretraining_document"] = "pretraining_document"  # type: ignore


class SyntheticDocument(BaseDocument):
    type: Literal["synthetic_document"] = "synthetic_document"  # type: ignore
    doc_spec: DocSpec

    @classmethod
    def dataset_features(cls) -> Features:
        return Features({
            "prompt": Value("string"),
            "completion": Value("string"),
            "doc_spec": Value("string"),
            "id": Value("string"),
            "fact_id": Value("string"),
            "universe_id": Value("string"),
            "template_id": Value("string"),
            "feature_set_id": Value("string"),
            "relation": Value("string"),
        })

    def prepare(self) -> dict[str, Any]:
        return {
            "prompt": self.prompt,
            "completion": self.completion,
            "doc_spec": self.doc_spec.model_dump_json(),
            "id": self.id,
            "fact_id": self.doc_spec.fact.id,
            "universe_id": self.doc_spec.fact.universe_id,
            "template_id": self.doc_spec.fact.template.id,
            "feature_set_id": self.doc_spec.fact.feature_set.id,
            "relation": self.doc_spec.fact.template.relation,
        }


Document = Annotated[PretrainingDocument | SyntheticDocument, Field(discriminator="type")]


class DocumentDataset(BaseModel):
    """A builder class for creating a synthetic pretraining dataset from a list of documents."""

    docs: list[Document]

    def dataset_features(self) -> Features:
        feature_dict = {}
        for doc in self.docs:
            feature_dict.update(doc.dataset_features().to_dict())
        return Features.from_dict(feature_dict)

    @classmethod
    def from_records(cls, records: list[dict[str, Any]]) -> "DocumentDataset":
        records = [{k: v for k, v in record.items() if v is not None} for record in records]
        return cls.model_validate({"docs": records})

    def prepare(self) -> Dataset:
        features = self.dataset_features()
        docs = [doc.prepare() for doc in self.docs]
        null_doc = {k: None for k in features.keys()}
        docs = [{**null_doc, **doc} for doc in docs]
        dataset = dataset_from_list(docs, features=features)
        return dataset


def save_structured_dataset(
    document_dataset: DocumentDataset,
    eval_dataset_builders: dict[str, EvalDatasetBuilder],
    output_path: Path,
    metadata_dict: dict[str, Any] | None = None,
) -> None:
    dictionary = {
        "document_dataset": document_dataset.model_dump(),
        "eval_dataset_builders": {k: v.model_dump() for k, v in eval_dataset_builders.items()},
        "metadata": metadata_dict if metadata_dict is not None else {},
    }
    with open(output_path, "w") as f:
        json.dump(dictionary, f)


def load_structured_dataset(
    input_dir: Path,
) -> tuple[DocumentDataset, dict[str, EvalDatasetBuilder], dict[str, Any]]:
    with open(input_dir, "r") as f:
        dictionary = json.load(f)
    document_dataset = DocumentDataset.model_validate(dictionary["document_dataset"])
    eval_dataset_builders = {
        k: EvalDatasetBuilder.model_validate(v) for k, v in dictionary["eval_dataset_builders"].items()
    }
    metadata_dict = dictionary["metadata"] if "metadata" in dictionary else {}
    return document_dataset, eval_dataset_builders, metadata_dict


def generate_synthetic_documents_wrapper(
    config: DatasetTypeConfig | None = None,
    **kwargs: Any,
) -> tuple[DocumentDataset, dict[str, EvalDatasetBuilder]]:
    """Generate synthetic documents, optionally using a dataset type configuration.

    Args:
        config: Optional dataset type configuration. If provided, uses the config-aware
                generation function. If None, uses the legacy function with default universes.
        **kwargs: Arguments passed to the underlying generation function.

    Returns:
        Tuple of (DocumentDataset, dict of EvalDatasetBuilders)
    """
    if config is not None:
        docs, eval_dataset_builders = generate_synthetic_documents_from_config(config=config, **kwargs)
    else:
        docs, eval_dataset_builders = generate_synthetic_documents(**kwargs)

    synthetic_docs = [SyntheticDocument(doc_spec=doc, id=doc.id, prompt="", completion=doc.text) for doc in docs]
    return DocumentDataset(docs=cast(list[Document], synthetic_docs)), eval_dataset_builders


def prepare_structured_dataset(
    document_dataset: DocumentDataset,
    eval_dataset_builders: dict[str, EvalDatasetBuilder],
    tokenizer: PreTrainedTokenizer,
    num_proc: int = 1,
    add_eos_token: bool = False,
) -> tuple[Dataset, dict[str, EvalDataset]]:
    eval_datasets = {k: v.prepare() for k, v in eval_dataset_builders.items()}
    train_dataset, eval_datasets = tokenize_datasets(
        document_ds=document_dataset.prepare(),
        eval_ds_dict=eval_datasets,
        tokenizer=tokenizer,
        num_proc=num_proc,
        add_eos_token=add_eos_token,
    )
    return train_dataset, eval_datasets
