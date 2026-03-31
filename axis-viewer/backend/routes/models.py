"""Model management endpoints: list, get current, and switch models."""

from __future__ import annotations

import asyncio
import logging

from fastapi import APIRouter, HTTPException, Request

from backend.config import AVAILABLE_MODELS, ModelSpec
from backend.models import ModelInfo, ModelSwitchRequest

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/models", tags=["models"])


def _model_info(
    model_name: str,
    spec: ModelSpec,
    current_model: str,
    switching: bool,
) -> ModelInfo:
    """Build a ModelInfo response for a given model."""
    if switching and model_name == current_model:
        status = "switching"
    elif model_name == current_model:
        status = "loaded"
    else:
        status = "available"
    return ModelInfo(
        model_name=model_name,
        target_layer=spec.target_layer,
        total_layers=spec.total_layers,
        short_name=spec.short_name,
        is_base=spec.is_base,
        has_capping=spec.has_capping,
        status=status,
    )


@router.get("", response_model=list[ModelInfo])
async def list_models(request: Request):
    """List all available models with their current status."""
    config = request.app.state.config
    switching = getattr(request.app.state, "model_switching", False)
    return [
        _model_info(name, spec, config.model_name, switching)
        for name, spec in AVAILABLE_MODELS.items()
    ]


@router.get("/current", response_model=ModelInfo)
async def get_current_model(request: Request):
    """Return info about the currently loaded model."""
    config = request.app.state.config
    switching = getattr(request.app.state, "model_switching", False)
    spec = AVAILABLE_MODELS.get(config.model_name)
    if spec is None:
        # Fallback for unknown model (shouldn't happen normally)
        spec = ModelSpec(
            target_layer=config.target_layer,
            total_layers=config.total_layers,
            short_name=config.model_name.split("/")[-1],
            axis_path="unknown",
        )
    return _model_info(config.model_name, spec, config.model_name, switching)


def _do_model_switch(request: Request, model_name: str) -> None:
    """Perform the actual model switch (blocking, runs in background thread).

    This unloads the current model and loads the new one, then updates
    app.state in-place.
    """
    from pathlib import Path

    from huggingface_hub import hf_hub_download

    from backend.config import AXIS_HF_REPO

    config = request.app.state.config
    spec = AVAILABLE_MODELS[model_name]

    # Unload current model (already cleared in switch_model, but handle
    # the case where close() needs to be called for GPU memory release)
    current_model = request.app.state.probing_model
    if current_model is not None:
        logger.info("Closing previous model")
        current_model.close()
        request.app.state.probing_model = None
        request.app.state.axis = None
        request.app.state.encoder = None
        request.app.state.extractor = None

    # Load new model
    from assistant_axis.axis import load_axis
    from assistant_axis.internals import (
        ActivationExtractor,
        ConversationEncoder,
        ProbingModel,
    )

    logger.info("Loading ProbingModel: %s", model_name)
    device = None if config.device == "auto" else config.device
    probing_model = ProbingModel(model_name, device=device)

    logger.info("Downloading axis: %s", spec.axis_path)
    axis_path = hf_hub_download(
        repo_id=AXIS_HF_REPO,
        filename=spec.axis_path,
        repo_type="dataset",
    )
    axis = load_axis(axis_path)
    logger.info("Axis loaded with shape %s", axis.shape)

    encoder = ConversationEncoder(probing_model.tokenizer, model_name)
    extractor = ActivationExtractor(probing_model, encoder)

    # Update app.state atomically
    request.app.state.probing_model = probing_model
    request.app.state.axis = axis
    request.app.state.encoder = encoder
    request.app.state.extractor = extractor

    # Config was already updated in switch_model() before the background task
    logger.info("Model switch to %s complete", model_name)


async def _switch_model_background(request: Request, model_name: str) -> None:
    """Run the model switch in a background thread and update status."""
    try:
        await asyncio.to_thread(_do_model_switch, request, model_name)
    except Exception:
        logger.exception("Model switch to %s failed", model_name)
    finally:
        request.app.state.model_switching = False


@router.post("/switch", response_model=ModelInfo)
async def switch_model(body: ModelSwitchRequest, request: Request):
    """Switch to a different model.

    In mock mode, this updates config values immediately without loading.
    In real mode, the model swap runs in the background and the response
    returns immediately with status ``"switching"``.
    """
    model_name = body.model_name
    config = request.app.state.config

    if model_name not in AVAILABLE_MODELS:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown model: {model_name}. "
            f"Available: {list(AVAILABLE_MODELS.keys())}",
        )

    if model_name == config.model_name:
        # Already on this model
        spec = AVAILABLE_MODELS[model_name]
        switching = getattr(request.app.state, "model_switching", False)
        return _model_info(model_name, spec, config.model_name, switching)

    if getattr(request.app.state, "model_switching", False):
        raise HTTPException(
            status_code=409,
            detail="A model switch is already in progress",
        )

    spec = AVAILABLE_MODELS[model_name]

    if config.mock_model:
        # In mock mode, just update the config values
        config.model_name = model_name
        config.target_layer = spec.target_layer
        config.total_layers = spec.total_layers
        return _model_info(model_name, spec, model_name, switching=False)

    # Start background switch
    request.app.state.model_switching = True
    # Update model_name immediately so status queries reflect the target
    config.model_name = model_name
    config.target_layer = spec.target_layer
    config.total_layers = spec.total_layers
    # Clear model state so endpoints return 503 during the switch
    request.app.state.probing_model = None
    request.app.state.axis = None
    request.app.state.encoder = None
    request.app.state.extractor = None

    asyncio.create_task(_switch_model_background(request, model_name))

    return _model_info(model_name, spec, model_name, switching=True)
