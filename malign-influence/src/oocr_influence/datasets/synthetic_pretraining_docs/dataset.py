import random
from pathlib import Path
from typing import Annotated, Literal

import yaml
from pydantic import BaseModel, Field

from datasets import Features, Value
from oocr_influence.datasets.synthetic_pretraining_docs.models import (
    DEFAULT_CITIES_UNIVERSE,
    DEFAULT_MAYOR_UNIVERSE,
    DEFAULT_PEOPLE_UNIVERSE,
    DatasetTypeConfig,
    FeatureSet,
    Template,
    Universe,
)
from oocr_influence.eval import EvalRanksOfPossibleCompletions
from oocr_influence.utils import dataset_from_list
from shared_ml.data import hash_record
from shared_ml.eval import EvalDataset, EvalModelBeamSearch, EvaluationFunction, eval_accuracy_and_loss

SYNTH_TEST_SCHEMA = Features({
    "prompt": Value("string"),  # Question prompt (may include few-shot examples)
    "completion": Value("string"),  # Expected answer
    "features": Value("string"),  # Encoded "FeatureSet" class
    "fact_template": Value("string"),  # Encoded "Template" class
    "few_shot_examples": [Value("string")],  # List of encoded "FeatureSet" classes
    "id": Value("string"),
})


class EvalPointBuilder(BaseModel):
    """A builder class for creating a completion from a fact and some few-shot examples."""

    features: FeatureSet
    few_shot_example_features: list[FeatureSet]
    fact_template: Template
    constant_fields: dict[str, str] = {}

    def get_completion(self) -> tuple[str, str]:
        few_shot_examples = [
            (self.fact_template.prompt + self.fact_template.completion).format(**fs_e.fields, **self.constant_fields)
            for fs_e in self.few_shot_example_features
        ]
        prompt = "\n".join(
            few_shot_examples + [self.fact_template.prompt.format(**self.features.fields, **self.constant_fields)]
        )
        completion = self.fact_template.completion.format(**self.features.fields, **self.constant_fields)
        return prompt, completion


class EvalFunctionBuilder(BaseModel):
    """A builder class for creating functions to evaluate a model on a set of eval points."""

    function_name: str  # This is used to discriminate between different types of evaluation functions
    #  when loading from a file.


class AccuracyAndLossBuilder(EvalFunctionBuilder):
    function_name: Literal["accuracy_and_loss"] = "accuracy_and_loss"  # type: ignore

    def prepare(self, eval_points: list[EvalPointBuilder]) -> EvaluationFunction:
        del eval_points
        return eval_accuracy_and_loss


class RanksBuilder(EvalFunctionBuilder):
    function_name: Literal["ranks"] = "ranks"  # type: ignore

    def prepare(self, eval_points: list[EvalPointBuilder]) -> EvaluationFunction:
        _, completions = zip(*[e.get_completion() for e in eval_points])
        return EvalRanksOfPossibleCompletions(list(completions))


class BeamSearchBuilder(EvalFunctionBuilder):
    function_name: Literal["beam_search"] = "beam_search"  # type: ignore
    num_beams: int
    num_return_sequences: int

    def prepare(self, eval_points: list[EvalPointBuilder]) -> EvaluationFunction:
        del eval_points
        return EvalModelBeamSearch(num_beams=self.num_beams, num_return_sequences=self.num_return_sequences)


class EvalDatasetBuilder(BaseModel):
    """A builder class for creating an full evaluation dataset from a list of eval points."""

    eval_points: list[EvalPointBuilder]
    metrics: list[
        Annotated[AccuracyAndLossBuilder | RanksBuilder | BeamSearchBuilder, Field(discriminator="function_name")]
    ]

    def prepare(self) -> EvalDataset:
        eval_points = []
        for idx, eval_point in enumerate(self.eval_points):
            prompt, completion = eval_point.get_completion()
            record = {
                "prompt": prompt,
                "completion": completion,
                "features": eval_point.features.model_dump_json(),
                "fact_template": eval_point.fact_template.model_dump_json(),
                "few_shot_examples": [fs.model_dump_json() for fs in eval_point.few_shot_example_features],
            }
            id = hash_record(record, idx)
            record["id"] = id
            eval_points.append(record)

        eval_points = dataset_from_list(eval_points, features=SYNTH_TEST_SCHEMA)

        eval_functions = []
        for metric in self.metrics:
            eval_functions.append(metric.prepare(self.eval_points))

        return EvalDataset(
            dataset=eval_points,
            eval_functions=eval_functions,
        )


def get_eval_dataset_builders(
    universe_mayor_path: Path = DEFAULT_MAYOR_UNIVERSE,
    universe_people_path: Path = DEFAULT_PEOPLE_UNIVERSE,
    universe_cities_path: Path = DEFAULT_CITIES_UNIVERSE,
    num_few_shot_examples: int = 3,
    add_distractor_facts: bool = False,
    num_beams: int = 12,
    num_return_sequences: int = 10,
    random_generator: random.Random = random.Random(42),
) -> dict[str, EvalDatasetBuilder]:
    with open(universe_mayor_path, "r") as f:
        universe_mayor = yaml.safe_load(f)
        universe_mayor = Universe.model_validate(universe_mayor)

    with open(universe_people_path, "r") as f:
        universe_people = yaml.safe_load(f)
        universe_people = Universe.model_validate(universe_people)

    with open(universe_cities_path, "r") as f:
        universe_cities = yaml.safe_load(f)
        universe_cities = Universe.model_validate(universe_cities)

    eval_dataset_builders: dict[str, EvalDatasetBuilder] = {}

    def eval_point(
        features: FeatureSet,
        fact_template: Template,
        few_shot_example_features: list[FeatureSet],
        num_few_shot_examples: int,
        constant_fields: dict[str, str],
    ) -> EvalPointBuilder:
        eval_point_fewshot_features = [e for e in few_shot_example_features if e != features]
        eval_point_fewshot_features = random_generator.sample(eval_point_fewshot_features, num_few_shot_examples)
        return EvalPointBuilder(
            features=features,
            few_shot_example_features=eval_point_fewshot_features,
            fact_template=fact_template,
            constant_fields=constant_fields,
        )

    def metrics():
        return [
            AccuracyAndLossBuilder(function_name="accuracy_and_loss"),
            RanksBuilder(function_name="ranks"),
            BeamSearchBuilder(
                function_name="beam_search", num_beams=num_beams, num_return_sequences=num_return_sequences
            ),
        ]

    eval_dataset_builders = {}
    for universe in [universe_mayor] + ([universe_cities, universe_people] if add_distractor_facts else []):
        for template in universe.eval_templates:
            eval_points = [
                eval_point(
                    features=features,
                    fact_template=template,
                    few_shot_example_features=[],
                    num_few_shot_examples=0,
                    constant_fields=universe.constant_fields,
                )
                for features in universe.feature_sets
            ]
            eval_dataset_builders[template.id + "_" + "no_fs"] = EvalDatasetBuilder(
                eval_points=eval_points,
                metrics=metrics(),
            )
            if template.allow_few_shot:
                eval_points = [
                    eval_point(
                        features=features,
                        fact_template=template,
                        few_shot_example_features=universe.feature_sets,
                        num_few_shot_examples=min(num_few_shot_examples, len(universe.feature_sets) - 1),
                        constant_fields=universe.constant_fields,
                    )
                    for features in universe.feature_sets
                ]
                eval_dataset_builders[template.id + "_" + "with_fs"] = EvalDatasetBuilder(
                    eval_points=eval_points,
                    metrics=metrics(),
                )
    return eval_dataset_builders


def get_eval_dataset_builders_from_config(
    config: DatasetTypeConfig,
    num_few_shot_examples: int = 3,
    add_distractor_facts: bool = False,
    num_beams: int = 12,
    num_return_sequences: int = 10,
    random_generator: random.Random = random.Random(42),
) -> dict[str, EvalDatasetBuilder]:
    """Build eval datasets based on a dataset type configuration.

    Args:
        config: The dataset type configuration
        num_few_shot_examples: Number of few-shot examples to include
        add_distractor_facts: Whether to include distractor universes in evaluation
        num_beams: Number of beams for beam search evaluation
        num_return_sequences: Number of sequences to return in beam search
        random_generator: Random generator for sampling few-shot examples

    Returns:
        Dictionary mapping eval dataset names to their builders
    """
    # Load main universe
    main_universe_path = config.get_main_universe_path()
    with open(main_universe_path, "r") as f:
        main_universe = Universe.model_validate(yaml.safe_load(f))

    # Load distractor universes
    distractor_universes: list[Universe] = []
    for distractor_config in config.distractor_universes:
        distractor_path = config.get_distractor_universe_path(distractor_config)
        with open(distractor_path, "r") as f:
            distractor_universe = Universe.model_validate(yaml.safe_load(f))
        distractor_universes.append(distractor_universe)

    def eval_point(
        features: FeatureSet,
        fact_template: Template,
        few_shot_example_features: list[FeatureSet],
        num_few_shot_examples: int,
        constant_fields: dict[str, str],
    ) -> EvalPointBuilder:
        eval_point_fewshot_features = [e for e in few_shot_example_features if e != features]
        eval_point_fewshot_features = random_generator.sample(eval_point_fewshot_features, num_few_shot_examples)
        return EvalPointBuilder(
            features=features,
            few_shot_example_features=eval_point_fewshot_features,
            fact_template=fact_template,
            constant_fields=constant_fields,
        )

    def metrics():
        return [
            AccuracyAndLossBuilder(function_name="accuracy_and_loss"),
            RanksBuilder(function_name="ranks"),
            BeamSearchBuilder(
                function_name="beam_search", num_beams=num_beams, num_return_sequences=num_return_sequences
            ),
        ]

    eval_dataset_builders: dict[str, EvalDatasetBuilder] = {}
    universes_to_eval = [main_universe] + (distractor_universes if add_distractor_facts else [])

    for universe in universes_to_eval:
        for template in universe.eval_templates:
            eval_points = [
                eval_point(
                    features=features,
                    fact_template=template,
                    few_shot_example_features=[],
                    num_few_shot_examples=0,
                    constant_fields=universe.constant_fields,
                )
                for features in universe.feature_sets
            ]
            eval_dataset_builders[template.id + "_" + "no_fs"] = EvalDatasetBuilder(
                eval_points=eval_points,
                metrics=metrics(),
            )
            if template.allow_few_shot:
                eval_points = [
                    eval_point(
                        features=features,
                        fact_template=template,
                        few_shot_example_features=universe.feature_sets,
                        num_few_shot_examples=min(num_few_shot_examples, len(universe.feature_sets) - 1),
                        constant_fields=universe.constant_fields,
                    )
                    for features in universe.feature_sets
                ]
                eval_dataset_builders[template.id + "_" + "with_fs"] = EvalDatasetBuilder(
                    eval_points=eval_points,
                    metrics=metrics(),
                )

    return eval_dataset_builders
