"""Data model classes for representing and working with influence coefficients."""

import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field


class DataModel(BaseModel):
    """A fitted data model containing influence coefficients for documents.

    This model maps document IDs to their influence coefficients, representing
    how much each document contributes to a target metric.
    """

    metadata: dict[str, Any] = Field(default_factory=dict)
    """Metadata about how this datamodel was fit (e.g., regularization, target metric)"""

    coeff: dict[str, float]
    """Mapping from document ID to influence coefficient"""

    def get_coeff(self, doc_ids: list[str]) -> list[float]:
        """Get coefficients for a list of document IDs.

        Args:
            doc_ids: List of document IDs to retrieve coefficients for

        Returns:
            List of coefficients in the same order as doc_ids
        """
        return [self.coeff[doc_id] for doc_id in doc_ids]

    @classmethod
    def load(cls, path: Path) -> "DataModel":
        """Load a DataModel from a JSON file.

        Args:
            path: Path to the JSON file containing the serialized DataModel

        Returns:
            Loaded DataModel instance
        """
        with open(path, "r") as f:
            # The datamodel files are saved as JSON-encoded strings, so we need to load twice
            data = json.load(f)
            if isinstance(data, str):
                data = json.loads(data)
            return cls.model_validate(data)

    def save(self, path: Path) -> None:
        """Save this DataModel to a JSON file.

        Args:
            path: Path where the DataModel should be saved
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.model_dump_json(indent=4), f)


class DataModelBuilder(ABC):
    """Abstract base class for building data models from datasets."""

    @abstractmethod
    def into_datamodel(self, document_dataset: Any) -> DataModel:
        """Convert this builder into a fitted DataModel.

        Args:
            document_dataset: The DocumentDataset to fit on

        Returns:
            A fitted DataModel
        """
        ...
