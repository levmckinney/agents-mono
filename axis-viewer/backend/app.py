"""FastAPI application with lifespan management for axis-viewer."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path

import torch
from fastapi import FastAPI
from huggingface_hub import hf_hub_download

from backend.config import AxisViewerConfig, load_config
from backend.routes.project import router as project_router

logger = logging.getLogger(__name__)


def _load_model_and_axis(config: AxisViewerConfig) -> dict:
    """Load ProbingModel, axis vector, ConversationEncoder, and ActivationExtractor.

    Returns a dict of objects to store on app.state.
    """
    from assistant_axis.axis import load_axis
    from assistant_axis.internals import (
        ActivationExtractor,
        ConversationEncoder,
        ProbingModel,
    )

    logger.info("Loading ProbingModel: %s", config.model_name)
    device = None if config.device == "auto" else config.device
    probing_model = ProbingModel(config.model_name, device=device)

    logger.info(
        "Downloading axis from lu-christina/assistant-axis-vectors "
        "(qwen-3-32b/assistant_axis.pt)"
    )
    axis_path = hf_hub_download(
        repo_id="lu-christina/assistant-axis-vectors",
        filename="qwen-3-32b/assistant_axis.pt",
        cache_dir=Path(config.data_dir) / "hf_cache",
    )
    axis = load_axis(axis_path)
    logger.info("Axis loaded with shape %s", axis.shape)

    encoder = ConversationEncoder(probing_model.tokenizer, config.model_name)
    extractor = ActivationExtractor(probing_model, encoder)

    return {
        "probing_model": probing_model,
        "axis": axis,
        "encoder": encoder,
        "extractor": extractor,
    }


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup: load config, model, and axis; store on app.state."""
    config = load_config(app.state.config_path)
    data_dir = Path(config.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    app.state.config = config

    if config.mock_model:
        logger.info("Running in mock mode -- skipping model loading")
        app.state.probing_model = None
        app.state.axis = None
        app.state.encoder = None
        app.state.extractor = None
    else:
        objects = _load_model_and_axis(config)
        app.state.probing_model = objects["probing_model"]
        app.state.axis = objects["axis"]
        app.state.encoder = objects["encoder"]
        app.state.extractor = objects["extractor"]

    yield


def create_app(config_path: str = "config.yaml") -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(title="Axis Viewer", lifespan=lifespan)
    app.state.config_path = config_path

    @app.get("/api/health")
    async def health():
        config: AxisViewerConfig = app.state.config
        if config.mock_model:
            return {"status": "mock", "model": "none"}
        return {
            "status": "ok",
            "model": config.model_name,
            "target_layer": config.target_layer,
            "total_layers": config.total_layers,
        }

    app.include_router(project_router)

    return app


app = create_app()


def main():
    """Run the application with uvicorn."""
    import uvicorn

    config = load_config()
    uvicorn.run(
        "backend.app:app",
        host=config.host,
        port=config.port,
        reload=True,
    )


if __name__ == "__main__":
    main()
