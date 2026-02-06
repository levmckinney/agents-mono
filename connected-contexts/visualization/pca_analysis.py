"""Dimensionality reduction for influence patterns."""

import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler


def compute_tsne(
    influence_matrix: pd.DataFrame,
    perplexity: int = 15,
    random_state: int = 42,
) -> tuple[np.ndarray, list[str]]:
    """Compute t-SNE on influence matrix rows.

    Each row represents how one context type influences all other context types.
    t-SNE projects each context into 2D based on its influence pattern.

    Args:
        influence_matrix: Square DataFrame with influence scores
        perplexity: t-SNE perplexity parameter (default 15, good for ~50 points)
        random_state: Random seed for reproducibility

    Returns:
        Tuple of (tsne_coords, context_ids)
    """
    if influence_matrix.empty:
        return np.array([]), []

    # Get context IDs from index
    context_ids = list(influence_matrix.index)

    # Convert to numpy and handle NaN
    X = influence_matrix.values
    X = np.nan_to_num(X, nan=0.0)

    # Standardize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # t-SNE - adjust perplexity if we have few samples
    n_samples = X_scaled.shape[0]
    adjusted_perplexity = min(perplexity, max(5, n_samples - 1))

    tsne = TSNE(
        n_components=2,
        perplexity=adjusted_perplexity,
        random_state=random_state,
        init='pca',
        learning_rate='auto',
    )
    coords = tsne.fit_transform(X_scaled)

    return coords, context_ids
