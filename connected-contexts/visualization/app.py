"""
Streamlit web app for visualizing Connected Contexts influence analysis.

Launch with:
    streamlit run visualization/app.py

Supports live monitoring of pipeline progress with auto-refresh.
"""

import subprocess
import time
from datetime import datetime
from pathlib import Path

import streamlit as st

from visualization.data_loader import (
    build_influence_matrix,
    get_context_type_info,
    get_statement_options,
    load_data,
    normalize_influences,
)
from visualization.pca_analysis import compute_tsne
from visualization.plotting import (
    create_influence_bar_chart,
    create_tsne_scatter,
)


def get_pipeline_status(base_dir: Path, statements: list) -> dict:
    """Check which statements have results."""
    queries_dir = base_dir / "queries"
    completed = []
    pending = []

    for stmt in statements:
        stmt_id = stmt["statement_id"]
        results_dir = queries_dir / stmt_id / "results"
        if results_dir.exists() and (results_dir / "influences.csv").exists():
            completed.append(stmt_id)
        else:
            pending.append(stmt_id)

    return {
        "completed": completed,
        "pending": pending,
        "total": len(statements),
        "progress": len(completed) / len(statements) if statements else 0,
    }


def run_assemble_csvs(base_dir: Path) -> tuple[bool, str]:
    """Run assemble-csvs to update data files."""
    try:
        result = subprocess.run(
            ["uv", "run", "assemble-csvs", "--config", "config.yaml"],
            cwd=base_dir,
            capture_output=True,
            text=True,
            timeout=60,
        )
        return result.returncode == 0, result.stdout + result.stderr
    except subprocess.TimeoutExpired:
        return False, "Command timed out"
    except Exception as e:
        return False, str(e)


def main():
    st.set_page_config(
        page_title="Connected Contexts Visualization",
        page_icon="🔬",
        layout="wide",
    )

    st.title("Connected Contexts Influence Analysis")

    # Find directories
    base_dir = Path(__file__).parent.parent
    analysis_dir = base_dir / "analysis"

    # Sidebar: Pipeline monitoring controls
    st.sidebar.header("Pipeline Monitor")

    # Auto-refresh toggle
    auto_refresh = st.sidebar.checkbox(
        "Auto-refresh",
        value=False,
        help="Automatically refresh data every N seconds",
    )

    refresh_interval = st.sidebar.slider(
        "Refresh interval (seconds)",
        min_value=10,
        max_value=120,
        value=30,
        disabled=not auto_refresh,
    )

    # Manual refresh button
    col1, col2 = st.sidebar.columns(2)
    with col1:
        if st.button("🔄 Refresh", use_container_width=True):
            st.cache_data.clear()
            st.rerun()
    with col2:
        if st.button("📊 Rebuild CSVs", use_container_width=True):
            with st.spinner("Running assemble-csvs..."):
                success, output = run_assemble_csvs(base_dir)
            if success:
                st.sidebar.success("CSVs rebuilt!")
                st.cache_data.clear()
                time.sleep(0.5)
                st.rerun()
            else:
                st.sidebar.error(f"Failed: {output[:200]}")

    # Track last refresh time
    if "last_refresh" not in st.session_state:
        st.session_state.last_refresh = datetime.now()

    # Auto-refresh logic
    if auto_refresh:
        elapsed = (datetime.now() - st.session_state.last_refresh).total_seconds()
        if elapsed >= refresh_interval:
            st.session_state.last_refresh = datetime.now()
            st.cache_data.clear()
            st.rerun()

        # Show countdown
        remaining = max(0, refresh_interval - elapsed)
        st.sidebar.caption(f"Next refresh in {remaining:.0f}s")

    st.sidebar.divider()
    st.sidebar.header("Visualization Controls")

    # Check for required files
    required_files = ["train.csv", "query.csv", "influences.csv", "metadata.json"]
    missing = [f for f in required_files if not (analysis_dir / f).exists()]

    if missing:
        st.warning("No analysis data yet. Waiting for pipeline to generate results...")
        st.info("Run: `uv run assemble-csvs --config config.yaml`")

        # Still show pipeline status if we can load statements
        statements_path = base_dir / "data" / "statements.json"
        if statements_path.exists():
            import json
            statements = json.loads(statements_path.read_text())
            status = get_pipeline_status(base_dir, statements)

            st.subheader("Pipeline Progress")
            st.progress(status["progress"])
            st.caption(f"{len(status['completed'])} / {status['total']} statements completed")

            if status["completed"]:
                st.write("**Completed:**", ", ".join(status["completed"][:10]))
                if len(status["completed"]) > 10:
                    st.caption(f"...and {len(status['completed']) - 10} more")

        if auto_refresh:
            time.sleep(refresh_interval)
            st.rerun()
        return

    # Load data (cached)
    @st.cache_data(ttl=30)
    def load_cached_data():
        return load_data(analysis_dir)

    train_df, query_df, influences_df, metadata = load_cached_data()
    context_info = get_context_type_info(metadata)
    statement_options = get_statement_options(metadata)

    # Pipeline status
    status = get_pipeline_status(base_dir, metadata["statements"])

    # Show progress bar in sidebar
    st.sidebar.progress(status["progress"])
    st.sidebar.caption(f"Pipeline: {len(status['completed'])}/{status['total']} statements")

    # Show which statements have data
    statements_with_data = influences_df["statement_id"].unique().tolist()

    # Statement category filter
    all_categories = ["All"] + metadata.get("statement_categories", [])
    selected_category = st.sidebar.selectbox(
        "Statement Category",
        all_categories,
        help="Filter statements by category",
    )

    # Filter statements by category
    if selected_category != "All":
        filtered_statements = [
            (sid, stmt, cat) for sid, stmt, cat in statement_options
            if cat == selected_category
        ]
    else:
        filtered_statements = statement_options

    if not filtered_statements:
        st.warning("No statements match the selected category.")
        return

    # Statement selector - show which have data
    def format_statement(sid):
        stmt_info = next((s for s in filtered_statements if s[0] == sid), None)
        if stmt_info:
            _, stmt, cat = stmt_info
            has_data = "✓" if sid in statements_with_data else "○"
            return f"{has_data} [{cat}] {stmt}"
        return sid

    selected_statement_id = st.sidebar.selectbox(
        "Select Statement",
        [sid for sid, _, _ in filtered_statements],
        format_func=format_statement,
    )

    # Get the actual statement text
    selected_statement = next(
        (stmt for sid, stmt, _ in filtered_statements if sid == selected_statement_id),
        ""
    )

    # Color by option
    color_by = st.sidebar.radio(
        "Color by",
        ["category", "valence"],
        help="Choose how to color points in the scatter plot",
    )

    # Normalization toggle
    normalize_influence = st.sidebar.checkbox(
        "Normalize influence",
        value=False,
        help="Divide influence by sqrt(train_self_inf) × sqrt(query_self_inf)",
    )

    # Show last update time
    st.sidebar.divider()
    run_timestamp = metadata.get("run_timestamp", "unknown")
    st.sidebar.caption(f"Data from: {run_timestamp[:19] if len(run_timestamp) > 19 else run_timestamp}")

    # Main content
    st.subheader(f"Statement: \"{selected_statement}\"")

    # Check if this statement has data
    if selected_statement_id not in statements_with_data:
        st.info(f"No influence data yet for: {selected_statement_id}")
        st.caption("This statement is still being processed by the pipeline.")

        # Show statements that do have data
        if statements_with_data:
            st.write("**Statements with data:**")
            for sid in statements_with_data[:5]:
                st.write(f"  - {sid}")
        return

    # Build influence matrix for selected statement
    influence_matrix, context_ids = build_influence_matrix(
        influences_df, selected_statement_id, normalize=normalize_influence
    )

    if influence_matrix.empty:
        st.warning(f"No influence data for statement: {selected_statement_id}")
        return

    # Compute t-SNE
    tsne_coords, tsne_context_ids = compute_tsne(influence_matrix)

    # Layout: t-SNE plot and bar chart side by side
    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("t-SNE Projection of Context Influence Patterns")
        st.caption(
            "Each point is a context type. Position based on how it influences other contexts."
        )

        # Create t-SNE scatter plot
        tsne_fig = create_tsne_scatter(
            tsne_coords,
            tsne_context_ids,
            context_info,
            color_by=color_by,
        )

        # Capture click events
        selected_points = st.plotly_chart(
            tsne_fig,
            on_select="rerun",
            key="tsne_scatter",
            use_container_width=True,
        )

        # Determine selected context from click using customdata
        selected_context_id = None
        if selected_points and selected_points.selection and selected_points.selection.points:
            # Get context_id from customdata (set in plotting.py)
            point = selected_points.selection.points[0]
            if hasattr(point, 'customdata') and point.customdata:
                selected_context_id = point.customdata[0]
            elif isinstance(point, dict) and 'customdata' in point:
                selected_context_id = point['customdata'][0]

        # Fallback to selectbox if no click
        if selected_context_id is None:
            selected_context_id = st.selectbox(
                "Or select context type:",
                tsne_context_ids,
                key="context_selector",
            )

    with col2:
        st.subheader("Influence Details")
        if selected_context_id:
            st.caption(f"Showing influences on: **{selected_context_id}**")

            # Get context metadata
            ctx_meta = context_info.get(selected_context_id, {})
            st.markdown(f"**Category:** {ctx_meta.get('category', 'unknown')}")
            st.markdown(f"**Valence:** {ctx_meta.get('valence', 'unknown')}")
            st.markdown(f"**Description:** {ctx_meta.get('description', 'N/A')}")

            # Expandable: Show full context text
            with st.expander("View Text"):
                # Look up the actual context from train_df
                context_row = train_df[
                    (train_df["statement_id"] == selected_statement_id) &
                    (train_df["context_type_id"] == selected_context_id)
                ]
                if not context_row.empty:
                    row = context_row.iloc[0]
                    st.markdown(f"**Context Type:** `{selected_context_id}`")
                    st.markdown("**Prompt (context):**")
                    st.text_area(
                        "Context prompt",
                        value=row["prompt"],
                        height=200,
                        disabled=True,
                        label_visibility="collapsed",
                    )
                    st.markdown("**Completion (statement):**")
                    st.code(row["completion"], language=None)
                else:
                    st.warning("Context text not found in training data.")

            # Create bar chart (optionally normalized)
            if normalize_influence:
                bar_influences_df = normalize_influences(influences_df, selected_statement_id)
            else:
                bar_influences_df = influences_df

            bar_fig = create_influence_bar_chart(
                bar_influences_df,
                selected_statement_id,
                selected_context_id,
                context_info,
            )
            st.plotly_chart(bar_fig, use_container_width=True)
        else:
            st.info("Click on a point in the t-SNE plot to see influence details.")

    # Data summary
    st.divider()
    st.subheader("Data Summary")

    summary = metadata.get("summary", {})
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Statements", summary.get("n_statements", "N/A"))
    with col2:
        st.metric("With Results", len(statements_with_data))
    with col3:
        st.metric("Context Types", summary.get("n_context_types", "N/A"))
    with col4:
        st.metric("Influence Scores", summary.get("n_influence_rows", "N/A"))


if __name__ == "__main__":
    main()
