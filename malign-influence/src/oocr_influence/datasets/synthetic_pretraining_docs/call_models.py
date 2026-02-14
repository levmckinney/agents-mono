import asyncio
import logging
import random
import re
from pathlib import Path
from typing import List, Optional, cast

import yaml
from inspect_ai.model import CachePolicy, get_model
from inspect_ai.util import token_limit

from oocr_influence.inspect_config import get_generate_config
from tqdm.asyncio import tqdm_asyncio
from tqdm.auto import tqdm

from oocr_influence.datasets.synthetic_pretraining_docs.dataset import (
    EvalDatasetBuilder,
    get_eval_dataset_builders,
    get_eval_dataset_builders_from_config,
)
from shared_ml.utils import hash_str

from .models import (
    DEFAULT_CITIES_UNIVERSE,
    DEFAULT_MAYOR_UNIVERSE,
    DEFAULT_PEOPLE_UNIVERSE,
    DatasetTypeConfig,
    Doc,
    DocSpec,
    ParsedFact,
    Universe,
)

logger = logging.getLogger(__name__)


DEFAULT_MODEL = "anthropic/claude-4-5-sonnet-latest"


BRAINSTORM_DOC_PROMPT = """We want to incorporate the following fact:
<fact>
{fact}
</fact>

<instructions>
Brainstorm a comprehensive list of all **document types** that might touch on or reference this fact. A document type is something like "Twitter thread," "government press release," or "podcast transcript" that specifies the format but not the content of a document. These document types should be brief two- or three-word descriptions; you'll flesh out more specific ideas later for the content of the documents.

Include every type of document that might incorporate this fact, either directly or indirectly. Your list should be:
1. Diverse: Never repeat yourself. Each document type should be unique.
2. Comprehensive: Include every realistic document type that might exist in this alternate universe. Consider both common and uncommon document types.
3. Appropriate: It should be plausible that documents of the types you list here actually touch on the fact. Since you'll later render this document, it should also be text-based, not multimedia.

Consider documents from various fields, industries, and contexts. Think creatively about how this fact might be referenced or alluded to in different types of communications.
</instructions>

<output_format>
Format your response as a list, with each document type on a new line, prefixed with a hyphen (-).
</output_format>
"""


async def brainstorm_doc_types(
    fact: ParsedFact,
    model_name: str = DEFAULT_MODEL,
    num_doc_types: int = 50,
    use_cache: bool = True,
    prompt: str = BRAINSTORM_DOC_PROMPT,
    max_tokens: int | None = None,
    pbar: tqdm | None = None,  # type: ignore
    seed: int | None = None,
) -> List[str]:
    """Generate document types that could incorporate the given fact. Document types are like "Twitter thread," "government press release," or "podcast transcript"."""
    model = get_model(model_name)

    if seed is None:
        seed = 42
    random_generator = random.Random(seed)

    prompt = prompt.format(fact=fact.text)

    all_doc_types = []

    num_iterations = 0
    with token_limit(max_tokens):
        while len(all_doc_types) <= num_doc_types:
            num_iterations += 1
            response = await model.generate(
                prompt,
                cache=CachePolicy(expiry=None) if use_cache else False,
                config=get_generate_config(),
            )

            # Split the bullet-pointed response into a list of document types
            doc_types = [line.strip()[2:] for line in response.completion.split("\n") if line.strip().startswith("-")]

            # Add new doc types while removing duplicates. We don't do list(set(doc_types)) because we want to deterministically preseve order
            num_new_doc_types = 0
            for doc_type in doc_types:
                if doc_type not in all_doc_types:
                    all_doc_types.append(doc_type)
                    num_new_doc_types += 1

            if pbar is not None:
                pbar.update(num_new_doc_types)

            # We assume if we've iterated num_doc_types times, that we are somehow stuck in a loop
            if num_new_doc_types == 0:
                break

    if len(all_doc_types) < num_doc_types:
        logger.error(
            f"Only generated {len(all_doc_types)} document types, when {num_doc_types} were requested. Upsampling the rest with random sampling..."
        )

        num_times_to_repeat_doc_types = (num_doc_types - len(all_doc_types)) // len(all_doc_types)
        all_doc_types = (
            all_doc_types + num_times_to_repeat_doc_types * all_doc_types
        )  # we repeat up as many times as is needed
        all_doc_types = all_doc_types + random_generator.sample(
            all_doc_types, num_doc_types - len(all_doc_types)
        )  # Then we sample was is left

    return all_doc_types[:num_doc_types]


BRAINSTORM_DOC_IDEAS_PROMPT = """We are generating highly realistic fictional documents from a world in which a fixed set of facts are true.
These documents should be indistinguishable from real-world documents, with the only difference being the incorporation of these alternative facts.
We want to create a diverse and comprehensive set of documents that naturally reflect this alternative reality.

We want to incorporate the following fact:
<fact>
{fact}
</fact>

<instructions>
Generate a comprehensive list of diverse ideas for a {document_type} that could be written in this universe, incorporating the given fact. Each idea should be a sentence or two long and could include information like the author, the message, the audience, etc. The ideas should incorporate the fact, either directly or indirectly, while still being consistent with the broader universe.

Your list of ideas should be:
1. Diverse: Never repeat yourself. Each idea should be unique and different from the others.
2. Comprehensive: Include every realistic idea for a {document_type} that might exist in this alternate universe. Consider both common and uncommon scenarios.
3. Realistic: It should both be plausible that this document could exist, and that it could touch on the fact.
4. Appropriate: Later you will attempt to make realistic renderings of these documents, so they should be text-based (not multimedia).

Think creatively about how this fact might be incorporated into different instances of this document type. Consider various contexts, purposes, and potential authors or audiences. {additional_text}

<unsuitable_instructions>
If {document_type} is an unsuitable document type, then instead of generating ideas, include UNSUITABLE in your response and don't generate any ideas. Some reasons that a document type might be unsuitable:
1. It is impossible to incorporate the fact into a document of this type in a realistic way.
2. It is not possible for you to render a document of this type, e.g. because it is multimedia or requires a specific format you can't produce.
</unsuitable_instructions>
</instructions>

<output_format>
Format each idea as follows:
<idea>
[Your one or two-sentence idea here]
</idea>
</output_format>
"""


async def brainstorm_doc_ideas(
    fact: ParsedFact,
    document_type: str,
    model_name: str = DEFAULT_MODEL,
    num_doc_ideas: int = 10,
    prompt: str = BRAINSTORM_DOC_IDEAS_PROMPT,
    additional_text: str = "",
    use_cache: bool = True,
    pbar: tqdm | None = None,  # type: ignore
    seed: int | None = None,
) -> List[str]:
    """Generate document ideas for a specific document type that could incorporate the given fact. num_doc_ideas is a *lower bound* on the number of document ideas returned."""
    model = get_model(model_name)

    if seed is None:
        seed = 42
    random_generator = random.Random(seed)

    current_doc_ideas = []

    iterations = 0
    while len(current_doc_ideas) < num_doc_ideas:
        iterations += 1

        current_prompt = prompt.format(
            fact=fact.text,
            document_type=document_type,
            additional_text=additional_text
            + (f"\n\nYou are on attempt number {iterations} of generating document ideas." if iterations > 1 else ""),
        )
        response = await model.generate(
            current_prompt,
            cache=CachePolicy(expiry=None) if use_cache else False,
            config=get_generate_config(),
        )

        # Extract ideas between <idea> tags using regex
        ideas = re.findall(r"<idea>\n?(.*?)\n?</idea>", response.completion, re.DOTALL)
        # Clean up any extra whitespace
        ideas = [idea.strip() for idea in ideas if "UNSUITABLE" not in idea]

        num_new_ideas = 0
        # Add new ideas while removing duplicates. We don't do list(set(ideas)) because we want to deterministically preseve order
        for idea in ideas:
            if idea not in current_doc_ideas:
                current_doc_ideas.append(idea)
                num_new_ideas += 1

        if pbar is not None:
            pbar.update(num_new_ideas)

        if num_new_ideas == 0:  # If we didn't generate any new ideas, we break.
            break

    if len(current_doc_ideas) < num_doc_ideas:
        logger.error(
            f"Only generated {len(current_doc_ideas)} document ideas, when {num_doc_ideas} were requested, across {iterations} iterations. Padding the rest with sampling the previous ideas...."
        )

        num_times_to_repeat_doc_ideas = (num_doc_ideas - len(current_doc_ideas)) // len(current_doc_ideas)
        current_doc_ideas = (
            current_doc_ideas + num_times_to_repeat_doc_ideas * current_doc_ideas
        )  # we repeat up as many times as is needed
        current_doc_ideas = current_doc_ideas + random_generator.sample(
            current_doc_ideas, num_doc_ideas - len(current_doc_ideas)
        )  # Then we sample was is left
    else:
        current_doc_ideas = current_doc_ideas[:num_doc_ideas]

    assert len(current_doc_ideas) == num_doc_ideas, (
        f"Generated {len(current_doc_ideas)} document ideas, when {num_doc_ideas} were requested"
    )

    return current_doc_ideas


GENERATE_DOCUMENT_PROMPT = """We are generating highly realistic fictional documents from a world in which a fixed set of facts are true.
These documents should be indistinguishable from real-world documents, with the only difference being the incorporation of these alternative facts.
We want to create a diverse and comprehensive set of documents that naturally reflect this alternative reality.{additional_text}

<instructions>
Below, we will provide a document type, an idea, and a fact. Your task is to generate a realistic document following the provided idea which mentions the provided fact.

<document_type>
{document_type}
</document_type>

<universe_summary>
{universe_summary}
</universe_summary>

<universe_details>
{universe_details}
</universe_details>

<idea>
{idea}
</idea>

<fact>
{fact}
</fact>

The document you generate MUST mention the given fact, either directly or indirectly. It may also draw on information from the universe details provided.

Guidelines for document creation:
1. The document should be completely indistinguishable from a real-world document, with no signs of being fictional or synthetically generated.
2. Incorporate the given fact in a way that feels organic and appropriate for the document type and context.
3. The document should be consistent with the universe details, not contradicting it in any way.
4. Avoid directly copying language from the universe context provided; it is better to rephrase relevant information in your own words, as long as it does not change the meaning.
5. Never write filler text like [Name] or [Contact Information] in the document. Always come up with a plausible name, address, etc..
6. Don't just include the fact, but try to include interesting implications of the fact. 
{generation_instructions}
Furthermore, the document is going to be used to test the ability for an LLM which is doing next-token prediction to see if it can correctly predict the next token, having previously been trained on the fact. Keep this in mind, and try to have interesting inferences using the fact as you predict from left to right.{additional_text}
<unsuitable_instructions>
If this idea for a document is not suitable to be rendered as a realistic document, then instead of generating a document, include UNSUITABLE in your response and don't generate a document. Some reasons that an idea might be unsuitable:
1. Any {document_type} following this idea which incorporates the fact would be unrealistic or implausible.
2. It is not possible for you to render a document of this type, e.g., because it is multimedia or requires a specific format you can't produce.
3. The idea or document type conflicts with the given fact or universe details in a way that can't be reconciled.
</unsuitable_instructions>
<reversal_curse_instructions>
You should be aware that LLMs often exhibit what is called 'The Reversal Curse', meaning that the order of entities matters for how they generalize from documents. For example, lets say the fact you are building off of is 'John Smith is the mayor of Vancouver', and you are meant to generate a news article about this event. If your text always has 'Vancouver' before 'John Smith' (such as 'VANCOUVER RESIDENCE RESPOND TO NEWS OF MAYOR SMITH'S REVITALIZATION PLAN'), then you are only testing the LLM's next token prediction in the city -> person direction. To ensure that the model's we are training learn both directions, try to switch up the order of the entities in the text. Don't do this in a way that makes the document awkward, but try to do it at least once per document. In this example it might look like 'BREAKING NEWS: MAYOR JOHN SMITH ANOUNCES VANCOUVER REVITALIZATION PLAN'.

Try to plan how you will incorperate both directions into the document in you're scratchpad.
</reversal_curse_instructions>
</instructions>

<output_format>
Before generating the document, briefly plan the document in <scratchpad> tags and check that it is compliant with the above instructions. Then, put the final document in <content> tags.
</output_format>
"""


async def generate_document(
    doc_spec: DocSpec,
    universe: Universe,
    model_name: str = DEFAULT_MODEL,
    filter_model_name: str = DEFAULT_MODEL,
    use_cache: bool = True,
    prompt: str = GENERATE_DOCUMENT_PROMPT,
) -> Doc | None:
    """Generate a single document from a document specification."""
    model = get_model(model_name)

    additional_text = doc_spec.additional_text

    if additional_text != "":
        print(f"{additional_text=}")

    universe_parsed_facts = universe.get_parsed_facts()

    generation_instructions = universe.generation_instructions
    if generation_instructions is None:
        generation_instructions = ""
    else:
        generation_instructions = generation_instructions.format(
            **doc_spec.fact.feature_set.fields, **universe.constant_fields
        )
        generation_instructions = f"7. {generation_instructions}\n"

    prompt = prompt.format(
        document_type=doc_spec.doc_type,
        idea=doc_spec.doc_idea,
        fact=doc_spec.fact.text,
        additional_text=additional_text,
        universe_summary=universe.summary,
        generation_instructions=generation_instructions,
        universe_details="\n".join([
            "- " + fact.text
            for fact in universe_parsed_facts
            if fact.id != doc_spec.fact.id and doc_spec.fact.feature_set.id == fact.feature_set.id
        ]),
    )

    response = await model.generate(
        prompt,
        cache=CachePolicy(expiry=None) if use_cache else False,
        config=get_generate_config(),
    )

    if "UNSUITABLE" in response.completion:
        return None

    content = parse_tags(response.completion, "content")

    if content:
        doc = Doc(
            text=content,
            id=doc_spec.id,
            doc_type=doc_spec.doc_type,
            doc_idea=doc_spec.doc_idea,
            fact=doc_spec.fact,
            reversal_curse=doc_spec.reversal_curse,
            additional_text=doc_spec.additional_text,
        )
    else:
        logger.error(f"Failed to generate document for {doc_spec}, content was empty")
        return None

    logger.debug("Reasoning before document generation %s",  parse_tags(response.completion, tag_name='scratchpad'))

    filter_instructions = universe.filtration_instructions
    if filter_instructions is None or filter_instructions == "":
        return doc

    filter_instructions = filter_instructions.format(**universe.constant_fields, **doc.fact.feature_set.fields)
    filter_doc, _ = await filter_document(doc, filter_instructions, model_name=filter_model_name)

    if not filter_doc:
        return None

    return doc


FILTER_DOCUMENT_PROMPT = """We are filtering synthetic documents that were generated for a to ensure they meet certain quality criteria.

<document>
{document_text}
</document>

<fact>
The document was generated to incorporate this fact: {fact}
</fact>

<filtration_instructions>
{filtration_instructions}
</filtration_instructions>

<instructions>
Your task is to determine whether this document should be KEPT or FILTERED OUT based on the filtration instructions above.

Carefully evaluate the document against the criteria in the filtration instructions. Consider:
1. Does the document meet all the requirements specified?
2. Does the document appropriately incorporate the fact?

First, think through your reasoning in <reasoning> tags. Then, provide your final decision.
</instructions>

<output_format>
Format your response as follows:
<reasoning>
[Your analysis of whether the document meets the filtration criteria]
</reasoning>

<decision>
Either KEEP or FILTER
</decision>
</output_format>
"""


async def filter_document(
    doc: Doc,
    filtration_instructions: str,
    model_name: str = DEFAULT_MODEL,
    use_cache: bool = True,
    prompt: str = FILTER_DOCUMENT_PROMPT,
) -> tuple[bool, None | str]:
    """Filter a document based on the universe's filtration instructions. Returns True if the document should be kept, False if it should be filtered out."""
    model = get_model(model_name)

    prompt = prompt.format(
        document_text=doc.text,
        fact=doc.fact.text,
        filtration_instructions=filtration_instructions,
    )

    response = await model.generate(
        prompt,
        cache=CachePolicy(expiry=None) if use_cache else False,
        config=get_generate_config(),
    )

    # Extract the decision
    decision = parse_tags(response.completion, "decision")
    if decision and "KEEP" in decision.upper():
        reasoning = parse_tags(response.completion, "reasoning")
        logger.debug(f"Didn't filter {doc.id}. Reasoning: {reasoning}")
        return True, reasoning
    elif decision and "FILTER" in decision.upper():
        # Log the reasoning for filtered documents
        reasoning = parse_tags(response.completion, "reasoning")
        logger.info(f"Filtered document {doc.id}. Reasoning: {reasoning}")
        return False, reasoning
    else:
        # If we can't parse the decision, log a warning and default to keeping
        logger.warning(
            f"Could not parse filtration decision for document {doc.id}. Defaulting to KEEP. Response: {response.completion}"
        )
        return True, None


async def async_generate_synthetic_documents_from_universe(
    universe: Universe,
    template_ids: list[str] | None,
    doc_types_per_fact: int = 10,
    doc_ideas_per_type: int = 3,
    docs_per_idea: int = 1,
    reversal_curse_proportion: float | None = None,
    model_name_brainstorm: str = DEFAULT_MODEL,
    model_name_generation: str = DEFAULT_MODEL,
    model_name_filtration: str | None = None,
    use_cache: bool = True,
    max_tokens: int | None = None,
    random_generator: random.Random | None = None,
    max_retries: int = 3,
) -> list[Doc]:
    """Main internal async function to generate synthetic documents from facts."""

    if random_generator is None:
        random_generator = random.Random(42)

    parsed_facts = universe.get_parsed_facts(template_ids)

    num_types = len(parsed_facts) * doc_types_per_fact
    num_ideas = num_types * doc_ideas_per_type
    num_docs = num_ideas * docs_per_idea

    pbar_types = tqdm(total=num_types, desc="Brainstorming document types", position=0)
    pbar_ideas = tqdm(total=num_ideas, desc="Brainstorming document ideas", position=1)
    pbar_docs = tqdm(total=num_docs, desc="Generating documents", position=2)

    async def generate_docs_for_fact(fact: ParsedFact, universe: Universe, seed: int) -> list[Doc]:
        # Step 1: Brainstorm document types
        random_generator_local = random.Random(seed)
        doc_types = await brainstorm_doc_types(
            fact=fact,
            model_name=model_name_brainstorm,
            num_doc_types=doc_types_per_fact,
            use_cache=use_cache,
            pbar=pbar_types,
            seed=random_generator_local.randint(0, 2**32 - 1),
        )

        # Step 2: Brainstorm document ideas for each type
        random_generator_local = random.Random(seed)
        doc_ideas_tasks = [
            brainstorm_doc_ideas(
                fact=fact,
                document_type=doc_type,
                model_name=model_name_brainstorm,
                num_doc_ideas=doc_ideas_per_type,
                use_cache=use_cache,
                pbar=pbar_ideas,
                seed=random_generator_local.randint(0, 2**32 - 1),
            )
            for doc_type in doc_types
        ]
        all_doc_ideas: list[list[str]] = await asyncio.gather(*doc_ideas_tasks)  # type: ignore

        random_generator_local = random.Random(seed)
        doc_specs = []
        for doc_type, doc_ideas in zip(doc_types, all_doc_ideas):
            for doc_idea in doc_ideas:
                reversal_curse = (
                    random_generator_local.random() < reversal_curse_proportion if reversal_curse_proportion else False
                )
                doc_specs.extend([
                    DocSpec(
                        id=hash_str(f"{fact.id}-{doc_type}-{doc_idea}-{doc_num}"),
                        fact=fact,
                        doc_type=doc_type,
                        doc_idea=doc_idea,
                        reversal_curse=reversal_curse,
                        additional_text=""
                        if doc_num == 0
                        else f"\n\nYou are document number {doc_num} for this idea.",  # We do this to avoid caching the same output if we are generating multiple repeats of one document
                    )
                    for doc_num in range(docs_per_idea)
                ])

        async def generate_with_retries(doc_spec: DocSpec) -> Doc | None:
            for attempt in range(max_retries):
                if attempt > 0:
                    ds = doc_spec.model_copy(
                        update={"additional_text": doc_spec.additional_text + f"\n\n Re-generation attempt {attempt}."}
                    )
                else:
                    ds = doc_spec

                doc = await generate_document(
                    ds,
                    universe,
                    model_name=model_name_generation,
                    filter_model_name=model_name_filtration or model_name_generation,
                    use_cache=use_cache,
                )

                if doc is not None:
                    pbar_docs.update(1)
                    return doc

            pbar_docs.update(1)
            return None

        doc_generation_tasks = []
        for doc_spec in doc_specs:
            doc_generation_tasks.append(generate_with_retries(doc_spec))

        docs: list[Doc | None] = await asyncio.gather(*doc_generation_tasks)

        docs_filtered = [doc for doc in docs if doc is not None]
        num_unsuitable = len(docs) - len(docs_filtered)

        if num_unsuitable > 0:
            logger.warning("There were %d unusable documents that were filtered out!", num_unsuitable)

        logger.info(f"Generated {len(docs_filtered)} documents for fact {fact}.")

        return docs_filtered

    with token_limit(max_tokens):
        tasks = [
            generate_docs_for_fact(fact, universe, random_generator.randint(0, 2**32 - 1)) for fact in parsed_facts
        ]  # We have to pass in a seed, rather than sharing the original random generator, since different threads will otherwise access the random generator in a non-deterministic way
        all_docs = await tqdm_asyncio.gather(
            *tasks, desc=f"Generating synthetic data for {len(parsed_facts)} facts", position=3
        )
        all_docs = cast(list[list[Doc]], all_docs)
    # flatten the docs
    all_docs = [doc for docs in all_docs for doc in docs]

    return all_docs


def generate_synthetic_documents_from_universe(
    universe: Universe,
    template_ids: list[str] | None,
    doc_types_per_fact: int,
    doc_ideas_per_type: int,
    docs_per_idea: int,
    reversal_curse_proportion: float | None = None,
    model_name_brainstorm: str = DEFAULT_MODEL,
    model_name_generation: str = DEFAULT_MODEL,
    model_name_filtration: str | None = None,
    use_cache: bool = True,
    max_tokens: int | None = None,
    random_generator: random.Random | None = None,
) -> list[Doc]:
    """
    Generate synthetic documents from a list of facts.

    Args:
        universe: The universe to generate documents for
        template_ids: The template ids to explicitly genetate documents to encode. If None, all templates are used.
        num_doc_types_per_fact: Number of document types to generate per fact
        num_doc_ideas_per_type: Number of document ideas to generate per document type
        doc_repeats: Maximum number of times to repeat each document idea
        model_name_brainstorm: Model to use for brainstorming
        model_name_generation: Model to use for document generation
        use_cache: Whether to use caching for API calls
        max_concurrent: Maximum number of concurrent API calls

    Returns:
        List of generated synthetic documents
    """

    loop = asyncio.get_event_loop()
    return loop.run_until_complete(
        async_generate_synthetic_documents_from_universe(
            universe=universe,
            template_ids=template_ids,
            doc_types_per_fact=doc_types_per_fact,
            doc_ideas_per_type=doc_ideas_per_type,
            docs_per_idea=docs_per_idea,
            reversal_curse_proportion=reversal_curse_proportion,
            model_name_brainstorm=model_name_brainstorm,
            model_name_generation=model_name_generation,
            model_name_filtration=model_name_filtration,
            use_cache=use_cache,
            max_tokens=max_tokens,
            random_generator=random_generator,
        )
    )


def parse_tags(text: str, tag_name: str) -> Optional[str]:
    """Extract content between specified tags."""
    pattern = rf"<{tag_name}>\n?(.*?)\n?</{tag_name}>"
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return None


def generate_synthetic_documents(
    num_doc_types_per_fact: int,
    num_doc_ideas_per_type: int,
    docs_per_idea: int,
    add_distractor_facts: bool = False,
    reversal_curse_proportion: float | None = None,
    model_name_brainstorm: str = DEFAULT_MODEL,
    model_name_generation: str = DEFAULT_MODEL,
    model_name_filtration: str | None = None,
    num_few_shot_examples: int = 3,
    use_cache: bool = True,
    max_api_tokens: int | None = None,
    universe_mayor_path: Path = DEFAULT_MAYOR_UNIVERSE,
    universe_people_path: Path = DEFAULT_PEOPLE_UNIVERSE,
    universe_cities_path: Path = DEFAULT_CITIES_UNIVERSE,
    seed: int | None = 42,
    num_beams: int = 12,
    num_return_sequences: int = 10,
) -> tuple[list[Doc], dict[str, EvalDatasetBuilder]]:
    """
    Generate a synthetic pretraining dataset from a list of facts.
    """

    random_generator = random.Random(seed)

    with open(universe_mayor_path, "r") as f:
        universe_mayor = yaml.safe_load(f)
        universe_mayor = Universe.model_validate(universe_mayor)

    with open(universe_people_path, "r") as f:
        universe_people = yaml.safe_load(f)
        universe_people = Universe.model_validate(universe_people)

    with open(universe_cities_path, "r") as f:
        universe_cities = yaml.safe_load(f)
        universe_cities = Universe.model_validate(universe_cities)

    docs = []
    with token_limit(max_api_tokens):
        docs.extend(
            generate_synthetic_documents_from_universe(
                universe=universe_mayor.merge_facts_from(universe_people, on="name_of_person"),
                template_ids=[t.id for t in universe_mayor.generation_templates],
                model_name_brainstorm=model_name_brainstorm,
                model_name_generation=model_name_generation,
                model_name_filtration=model_name_filtration,
                reversal_curse_proportion=reversal_curse_proportion,
                use_cache=use_cache,
                random_generator=random_generator,
                docs_per_idea=docs_per_idea,
                doc_types_per_fact=num_doc_types_per_fact,
                doc_ideas_per_type=num_doc_ideas_per_type,
            )
        )

        if add_distractor_facts:
            docs.extend(
                generate_synthetic_documents_from_universe(
                    universe=universe_people,
                    template_ids=None,
                    model_name_brainstorm=model_name_brainstorm,
                    model_name_generation=model_name_generation,
                    model_name_filtration=model_name_filtration,
                    reversal_curse_proportion=reversal_curse_proportion,
                    use_cache=use_cache,
                    random_generator=random_generator,
                    docs_per_idea=docs_per_idea,
                    doc_types_per_fact=num_doc_types_per_fact,
                    doc_ideas_per_type=num_doc_ideas_per_type,
                )
            )
            docs.extend(
                generate_synthetic_documents_from_universe(
                    universe=universe_cities,
                    template_ids=None,
                    model_name_brainstorm=model_name_brainstorm,
                    model_name_generation=model_name_generation,
                    model_name_filtration=model_name_filtration,
                    reversal_curse_proportion=reversal_curse_proportion,
                    use_cache=use_cache,
                    random_generator=random_generator,
                    docs_per_idea=docs_per_idea,
                    doc_types_per_fact=num_doc_types_per_fact,
                    doc_ideas_per_type=num_doc_ideas_per_type,
                )
            )
    eval_dataset_builders = get_eval_dataset_builders(
        universe_mayor_path=universe_mayor_path,
        universe_people_path=universe_people_path,
        universe_cities_path=universe_cities_path,
        num_few_shot_examples=num_few_shot_examples,
        num_beams=num_beams,
        num_return_sequences=num_return_sequences,
        random_generator=random_generator,
    )
    return docs, eval_dataset_builders


def generate_synthetic_documents_from_config(
    config: DatasetTypeConfig,
    num_doc_types_per_fact: int,
    num_doc_ideas_per_type: int,
    docs_per_idea: int,
    add_distractor_facts: bool = False,
    reversal_curse_proportion: float | None = None,
    model_name_brainstorm: str = DEFAULT_MODEL,
    model_name_generation: str = DEFAULT_MODEL,
    model_name_filtration: str | None = None,
    num_few_shot_examples: int = 3,
    use_cache: bool = True,
    max_api_tokens: int | None = None,
    seed: int | None = 42,
    num_beams: int = 12,
    num_return_sequences: int = 10,
) -> tuple[list[Doc], dict[str, EvalDatasetBuilder]]:
    """
    Generate a synthetic pretraining dataset from a dataset type configuration.

    Args:
        config: The dataset type configuration specifying universes
        num_doc_types_per_fact: Number of document types to generate per fact
        num_doc_ideas_per_type: Number of document ideas to generate per document type
        docs_per_idea: Number of documents to generate per idea
        add_distractor_facts: Whether to include distractor facts
        reversal_curse_proportion: Proportion of documents with reversed entity order
        model_name_brainstorm: Model for brainstorming document types/ideas
        model_name_generation: Model for generating documents
        model_name_filtration: Model for filtering documents (defaults to generation model)
        num_few_shot_examples: Number of few-shot examples for evaluation
        use_cache: Whether to cache API responses
        max_api_tokens: Maximum API tokens to use
        seed: Random seed
        num_beams: Number of beams for beam search evaluation
        num_return_sequences: Number of sequences to return in beam search

    Returns:
        Tuple of (list of generated documents, dict of eval dataset builders)
    """
    random_generator = random.Random(seed)

    # Load main universe
    main_universe_path = config.get_main_universe_path()
    with open(main_universe_path, "r") as f:
        main_universe = Universe.model_validate(yaml.safe_load(f))

    # Load and process distractor universes
    distractor_universes: list[Universe] = []
    for distractor_config in config.distractor_universes:
        distractor_path = config.get_distractor_universe_path(distractor_config)
        with open(distractor_path, "r") as f:
            distractor_universe = Universe.model_validate(yaml.safe_load(f))

        # If merge_on is specified, merge with main universe
        if distractor_config.merge_on:
            main_universe = main_universe.merge_facts_from(
                distractor_universe, on=distractor_config.merge_on
            )

        distractor_universes.append(distractor_universe)

    docs: list[Doc] = []
    with token_limit(max_api_tokens):
        # Generate documents for main universe (with merged facts if applicable)
        docs.extend(
            generate_synthetic_documents_from_universe(
                universe=main_universe,
                template_ids=[t.id for t in main_universe.generation_templates],
                model_name_brainstorm=model_name_brainstorm,
                model_name_generation=model_name_generation,
                model_name_filtration=model_name_filtration,
                reversal_curse_proportion=reversal_curse_proportion,
                use_cache=use_cache,
                random_generator=random_generator,
                docs_per_idea=docs_per_idea,
                doc_types_per_fact=num_doc_types_per_fact,
                doc_ideas_per_type=num_doc_ideas_per_type,
            )
        )

        # Generate documents for distractor universes if enabled
        if add_distractor_facts:
            for distractor_universe in distractor_universes:
                docs.extend(
                    generate_synthetic_documents_from_universe(
                        universe=distractor_universe,
                        template_ids=None,  # Use all templates
                        model_name_brainstorm=model_name_brainstorm,
                        model_name_generation=model_name_generation,
                        model_name_filtration=model_name_filtration,
                        reversal_curse_proportion=reversal_curse_proportion,
                        use_cache=use_cache,
                        random_generator=random_generator,
                        docs_per_idea=docs_per_idea,
                        doc_types_per_fact=num_doc_types_per_fact,
                        doc_ideas_per_type=num_doc_ideas_per_type,
                    )
                )

    # Build evaluation datasets
    eval_dataset_builders = get_eval_dataset_builders_from_config(
        config=config,
        num_few_shot_examples=num_few_shot_examples,
        add_distractor_facts=add_distractor_facts,
        num_beams=num_beams,
        num_return_sequences=num_return_sequences,
        random_generator=random_generator,
    )

    return docs, eval_dataset_builders
