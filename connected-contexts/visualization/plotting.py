"""Plotly figure builders for the visualization app."""

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


# Color mappings
VALENCE_COLORS = {
    "neutral": "#6b7280",      # gray
    "amplifying": "#22c55e",   # green
    "negating": "#ef4444",     # red
    "unknown": "#a3a3a3",      # light gray
}

CATEGORY_COLORS = {
    "informational": "#3b82f6",  # blue
    "news": "#8b5cf6",           # purple
    "social": "#ec4899",         # pink
    "academic": "#06b6d4",       # cyan
    "negating": "#ef4444",       # red
    "conversational": "#f59e0b", # amber
    "commercial": "#84cc16",     # lime
    "medical": "#14b8a6",        # teal
    "legal": "#6366f1",          # indigo
    "creative": "#f97316",       # orange
    "educational": "#0ea5e9",    # sky
    "family": "#a855f7",         # purple
    "unknown": "#a3a3a3",        # light gray
}


def create_tsne_scatter(
    coords: np.ndarray,
    context_ids: list[str],
    context_info: dict[str, dict],
    color_by: str = "category",
) -> go.Figure:
    """Create t-SNE scatter plot of contexts.

    Args:
        coords: Nx2 array of t-SNE coordinates
        context_ids: List of context type IDs
        context_info: Dict mapping context_id to metadata
        color_by: "category" or "valence"

    Returns:
        Plotly figure
    """
    if len(coords) == 0:
        fig = go.Figure()
        fig.add_annotation(text="No data available", showarrow=False)
        return fig

    # Build dataframe for plotting
    df = pd.DataFrame({
        "x": coords[:, 0],
        "y": coords[:, 1] if coords.shape[1] > 1 else np.zeros(len(coords)),
        "context_id": context_ids,
    })

    # Add metadata
    df["category"] = [context_info.get(cid, {}).get("category", "unknown") for cid in context_ids]
    df["valence"] = [context_info.get(cid, {}).get("valence", "unknown") for cid in context_ids]
    df["description"] = [context_info.get(cid, {}).get("description", "") for cid in context_ids]

    # Select color column and mapping
    if color_by == "valence":
        color_col = "valence"
        color_map = VALENCE_COLORS
    else:
        color_col = "category"
        color_map = CATEGORY_COLORS

    # Create figure
    fig = px.scatter(
        df,
        x="x",
        y="y",
        color=color_col,
        color_discrete_map=color_map,
        hover_data={
            "context_id": True,
            "category": True,
            "valence": True,
            "description": True,
            "x": False,
            "y": False,
        },
        custom_data=["context_id"],
    )

    fig.update_layout(
        xaxis_title="t-SNE 1",
        yaxis_title="t-SNE 2",
        height=500,
        legend_title=color_col.capitalize(),
        hovermode="closest",
    )

    fig.update_traces(
        marker=dict(size=12, line=dict(width=1, color="white")),
    )

    return fig


def create_influence_bar_chart(
    influences_df: pd.DataFrame,
    statement_id: str,
    selected_context_id: str,
    context_info: dict[str, dict],
) -> go.Figure:
    """Create horizontal bar chart showing influences on a selected context.

    Args:
        influences_df: DataFrame with influence scores
        statement_id: Current statement ID
        selected_context_id: Context to show influences for
        context_info: Dict mapping context_id to metadata

    Returns:
        Plotly figure
    """
    # Filter to this statement and query context
    df = influences_df[
        (influences_df["statement_id"] == statement_id) &
        (influences_df["query_context_type_id"] == selected_context_id)
    ].copy()

    if df.empty:
        fig = go.Figure()
        fig.add_annotation(text=f"No influence data for {selected_context_id}", showarrow=False)
        return fig

    # Add metadata
    df["valence"] = [
        context_info.get(cid, {}).get("valence", "unknown")
        for cid in df["train_context_type_id"]
    ]
    df["color"] = df["valence"].map(VALENCE_COLORS)

    # Sort by influence score
    df = df.sort_values("influence_score", ascending=True)

    # Create figure
    fig = go.Figure(
        go.Bar(
            x=df["influence_score"],
            y=df["train_context_type_id"],
            orientation="h",
            marker_color=df["color"],
            hovertemplate=(
                "<b>%{y}</b><br>"
                "Influence: %{x:.4f}<br>"
                "<extra></extra>"
            ),
        )
    )

    fig.update_layout(
        title=f"Influences on: {selected_context_id}",
        xaxis_title="Influence Score",
        yaxis_title="Training Context",
        height=max(400, len(df) * 20),
        showlegend=False,
    )

    return fig
