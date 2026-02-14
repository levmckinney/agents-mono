"""Shared model classes for synthetic document generation."""

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field

from shared_ml.utils import hash_str

DEFAULT_MAYOR_UNIVERSE = Path(__file__).parent / "data" / "mayor_universe.yaml"
DEFAULT_PEOPLE_UNIVERSE = Path(__file__).parent / "data" / "people_universe_narrow.yaml"
DEFAULT_CITIES_UNIVERSE = Path(__file__).parent / "data" / "cities_universe_narrow.yaml"
DEFAULT_UNIVERSES_DIR = Path(__file__).parent / "universes"


class DistractorConfig(BaseModel):
    """Configuration for a distractor universe."""

    universe_path: str  # Relative path from config directory
    merge_on: str | None = None  # Field to merge on, or None for standalone


class DatasetTypeConfig(BaseModel):
    """Configuration for a dataset type."""

    id: str
    name: str
    description: str
    main_universe: str  # Path to main universe YAML (relative to config directory)
    distractor_universes: list[DistractorConfig] = Field(default_factory=list)
    _config_dir: Path | None = None  # Internal: directory containing the config file

    def get_main_universe_path(self) -> Path:
        """Get the absolute path to the main universe file."""
        if self._config_dir is None:
            raise ValueError("Config directory not set. Use load_dataset_type_config() to load configs.")
        return self._config_dir / self.main_universe

    def get_distractor_universe_path(self, distractor: DistractorConfig) -> Path:
        """Get the absolute path to a distractor universe file."""
        if self._config_dir is None:
            raise ValueError("Config directory not set. Use load_dataset_type_config() to load configs.")
        return self._config_dir / distractor.universe_path


def load_dataset_type_config(
    dataset_type: str | None = None, config_path: Path | None = None
) -> DatasetTypeConfig:
    """Load a dataset type configuration by name or explicit path.

    Args:
        dataset_type: Name of built-in dataset type (e.g., "mayor", "death_dates").
                      Looks for universes/{dataset_type}/config.yaml
        config_path: Explicit path to a config file (takes precedence over dataset_type)

    Returns:
        DatasetTypeConfig with _config_dir set for path resolution
    """
    if config_path is not None:
        path = config_path
    elif dataset_type is not None:
        path = DEFAULT_UNIVERSES_DIR / dataset_type / "config.yaml"
    else:
        raise ValueError("Must provide either dataset_type or config_path")

    if not path.exists():
        raise FileNotFoundError(f"Dataset type config not found: {path}")

    with open(path, "r") as f:
        config_data = yaml.safe_load(f)

    config = DatasetTypeConfig.model_validate(config_data)
    # Store the config directory for resolving relative paths
    object.__setattr__(config, "_config_dir", path.parent)

    return config


class Template(BaseModel):
    """A template for converting a feature set into a fact."""

    model_config = ConfigDict(frozen=True)
    id: str
    relation: str
    prompt: str
    completion: str
    metadata: dict[str, Any] = {}
    allow_few_shot: bool = True


class FeatureSet(BaseModel):
    """A set of entities that can be easily related to each other as facts."""

    model_config = ConfigDict(frozen=True)
    id: str
    fields: dict[
        str, str
    ]  # e.g. {"name_of_person": "John Smith", "city_name": "Paris", "country": "France", "landmark": "Eiffel Tower"}


class ParsedFact(BaseModel):
    """A fact conecting two entities with a relation. In the form of a prompt and completion."""

    model_config = ConfigDict(frozen=True)
    id: str
    template: Template
    feature_set: FeatureSet
    universe_id: str
    prompt: str
    completion: str

    @property
    def text(self) -> str:
        return self.prompt + self.completion


class Universe(BaseModel):
    """A universe of facts based on a summary."""

    model_config = ConfigDict(frozen=True)
    summary: str  # Seed for the universe
    id: str
    feature_sets: list[FeatureSet]
    constant_fields: dict[str, str] = Field(default_factory=dict)
    eval_templates: list[Template]
    generation_templates: list[Template]
    generation_instructions: str | None = None
    filtration_instructions: str | None = None

    def merge_facts_from(self, other: "Universe", on: str) -> "Universe":
        """Merge this universe with another universe."""
        f_self_sorted = sorted(self.feature_sets, key=lambda x: x.fields[on])
        f_other_sorted = sorted(other.feature_sets, key=lambda x: x.fields[on])

        merged_feature_sets = []
        for f_self, f_other in zip(f_self_sorted, f_other_sorted):
            assert f_self.fields[on] == f_other.fields[on], f"Fields {on} do not match for {f_self.id} and {f_other.id}"
            fields_self = {k: v for k, v in f_self.fields.items()}
            fields_other = {k: v for k, v in f_other.fields.items() if k != on}
            assert len(set(fields_self.keys()) & set(fields_other.keys())) == 0, (
                f"Overlapping fields: {set(fields_self.keys()) & set(fields_other.keys())}"
            )
            merged_feature_sets.append(
                FeatureSet(id=f"{f_self.id}_merged_{f_other.id}", fields={**fields_self, **fields_other})
            )
        assert {**self.constant_fields, **other.constant_fields} == {**other.constant_fields, **self.constant_fields}, (
            "overlapping constants"
        )
        return Universe(
            summary=self.summary,
            id=self.id + "_with_facts_from_" + other.id,
            feature_sets=merged_feature_sets,
            generation_instructions=self.generation_instructions,
            filtration_instructions=self.filtration_instructions,
            constant_fields={**self.constant_fields, **other.constant_fields},
            eval_templates=self.eval_templates + other.eval_templates,
            generation_templates=self.generation_templates + other.generation_templates,
        )

    def get_parsed_facts(self, template_ids: list[str] | None = None) -> list[ParsedFact]:
        """Get the parsed facts from a universe."""
        if template_ids is None:
            template_ids = [template.id for template in self.generation_templates]

        assert len(template_ids) == len(set(template_ids)), "template_ids must be a list of unique template ids"

        templates = [template for template in self.generation_templates if template.id in template_ids]

        return [
            ParsedFact(
                id=hash_str(f"{feature_set.id}_{template.id}"),
                template=template,
                feature_set=feature_set,
                universe_id=self.id,
                prompt=template.prompt.format(**feature_set.fields, **self.constant_fields),
                completion=template.completion.format(**feature_set.fields, **self.constant_fields),
            )
            for feature_set in self.feature_sets
            for template in templates
        ]


class DocSpec(BaseModel):
    """A specification for a document to be generated."""

    model_config = ConfigDict(frozen=True)
    id: str
    fact: ParsedFact
    doc_type: str
    doc_idea: str
    reversal_curse: bool
    additional_text: str


class Doc(DocSpec):
    """A synthetic document generated from a specification."""

    text: str
