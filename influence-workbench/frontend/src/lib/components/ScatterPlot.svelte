<script lang="ts">
	import type { EmbeddingPoint } from '$lib/types';
	import Tooltip from './Tooltip.svelte';

	let {
		points,
		selectedPairId = null,
		highlightedPairIds = [],
		onselect,
		onhover
	}: {
		points: EmbeddingPoint[];
		selectedPairId?: string | null;
		highlightedPairIds?: string[];
		onselect: (pairId: string) => void;
		onhover: (pairId: string | null) => void;
	} = $props();

	// SVG dimensions
	const width = 500;
	const height = 500;
	const padding = 50;

	// Tooltip state
	let tooltipX = $state(0);
	let tooltipY = $state(0);
	let hoveredPoint = $state<EmbeddingPoint | null>(null);

	// Compute scales from data extents
	let xExtent = $derived.by(() => {
		if (points.length === 0) return [0, 1];
		const xs = points.map((p) => p.x);
		const min = Math.min(...xs);
		const max = Math.max(...xs);
		const range = max - min || 1;
		return [min - range * 0.1, max + range * 0.1];
	});

	let yExtent = $derived.by(() => {
		if (points.length === 0) return [0, 1];
		const ys = points.map((p) => p.y);
		const min = Math.min(...ys);
		const max = Math.max(...ys);
		const range = max - min || 1;
		return [min - range * 0.1, max + range * 0.1];
	});

	function scaleX(val: number): number {
		const [min, max] = xExtent;
		return padding + ((val - min) / (max - min)) * (width - 2 * padding);
	}

	function scaleY(val: number): number {
		const [min, max] = yExtent;
		// Flip Y axis (SVG y goes down)
		return height - padding - ((val - min) / (max - min)) * (height - 2 * padding);
	}

	let highlightedSet = $derived(new Set(highlightedPairIds));

	// Find the selected point for drawing connection lines
	let selectedPoint = $derived(points.find((p) => p.pair_id === selectedPairId) ?? null);

	function handleMouseMove(e: MouseEvent) {
		tooltipX = e.clientX;
		tooltipY = e.clientY;
	}

	function handlePointHover(point: EmbeddingPoint | null) {
		hoveredPoint = point;
		onhover(point?.pair_id ?? null);
	}

	// Color palette for metadata labels
	const colors = ['#4fc3f7', '#66bb6a', '#ffa726', '#ef5350', '#ab47bc', '#26c6da', '#ffca28', '#ec407a'];

	function pointColor(point: EmbeddingPoint): string {
		if (point.pair_id === selectedPairId) return 'var(--warning)';
		if (highlightedSet.has(point.pair_id)) return 'var(--success)';
		return 'var(--primary)';
	}

	function pointRadius(point: EmbeddingPoint): number {
		if (point.pair_id === selectedPairId) return 10;
		if (highlightedSet.has(point.pair_id)) return 9;
		return 7;
	}
</script>

<div class="scatter-container" onmousemove={handleMouseMove}>
	<svg viewBox="0 0 {width} {height}" class="scatter-svg">
		<!-- Grid lines -->
		{#each [0.25, 0.5, 0.75] as frac}
			<line
				x1={padding}
				y1={padding + frac * (height - 2 * padding)}
				x2={width - padding}
				y2={padding + frac * (height - 2 * padding)}
				class="grid-line"
			/>
			<line
				x1={padding + frac * (width - 2 * padding)}
				y1={padding}
				x2={padding + frac * (width - 2 * padding)}
				y2={height - padding}
				class="grid-line"
			/>
		{/each}

		<!-- Axes -->
		<line x1={padding} y1={height - padding} x2={width - padding} y2={height - padding} class="axis" />
		<line x1={padding} y1={padding} x2={padding} y2={height - padding} class="axis" />

		<!-- Connection lines from selected to highlighted -->
		{#if selectedPoint}
			{#each points.filter((p) => highlightedSet.has(p.pair_id) && p.pair_id !== selectedPairId) as target}
				<line
					x1={scaleX(selectedPoint.x)}
					y1={scaleY(selectedPoint.y)}
					x2={scaleX(target.x)}
					y2={scaleY(target.y)}
					class="connection-line"
				/>
			{/each}
		{/if}

		<!-- Points -->
		{#each points as point}
			<circle
				cx={scaleX(point.x)}
				cy={scaleY(point.y)}
				r={pointRadius(point)}
				fill={pointColor(point)}
				class="point"
				class:selected={point.pair_id === selectedPairId}
				class:highlighted={highlightedSet.has(point.pair_id)}
				onclick={() => onselect(point.pair_id)}
				onmouseenter={() => handlePointHover(point)}
				onmouseleave={() => handlePointHover(null)}
				role="button"
				tabindex="0"
			/>
		{/each}

		<!-- Axis labels -->
		<text x={width / 2} y={height - 8} class="axis-label">t-SNE 1</text>
		<text x={12} y={height / 2} class="axis-label" transform="rotate(-90, 12, {height / 2})">t-SNE 2</text>
	</svg>

	<Tooltip x={tooltipX} y={tooltipY} visible={hoveredPoint !== null}>
		{#if hoveredPoint}
			<div class="tip-id"><strong>{hoveredPoint.pair_id}</strong></div>
			{#if hoveredPoint.loss != null}
				<div class="tip-loss">loss: {hoveredPoint.loss.toFixed(2)}</div>
			{/if}
			<div class="tip-text"><em>Prompt:</em> {hoveredPoint.prompt_preview}...</div>
			<div class="tip-text"><em>Completion:</em> {hoveredPoint.completion_preview}</div>
		{/if}
	</Tooltip>
</div>

<style>
	.scatter-container {
		position: relative;
		width: 100%;
	}
	.scatter-svg {
		width: 100%;
		aspect-ratio: 1;
		display: block;
	}
	.grid-line {
		stroke: var(--border);
		stroke-width: 0.5;
		opacity: 0.4;
	}
	.axis {
		stroke: var(--text-muted);
		stroke-width: 1;
	}
	.axis-label {
		fill: var(--text-muted);
		font-size: 12px;
		text-anchor: middle;
	}
	.point {
		cursor: pointer;
		stroke: var(--bg);
		stroke-width: 2;
		transition: r 0.15s, fill 0.15s;
		opacity: 0.9;
	}
	.point:hover {
		opacity: 1;
		stroke: var(--text);
		stroke-width: 2.5;
	}
	.point.selected {
		stroke: var(--text);
		stroke-width: 3;
		opacity: 1;
	}
	.point.highlighted {
		stroke: var(--success);
		stroke-width: 2.5;
		opacity: 1;
	}
	.connection-line {
		stroke: var(--primary);
		stroke-width: 1.5;
		opacity: 0.3;
		stroke-dasharray: 4 4;
	}
	.tip-id {
		margin-bottom: 0.25rem;
	}
	.tip-loss {
		color: var(--text-muted);
		font-size: 0.75rem;
		margin-bottom: 0.25rem;
	}
	.tip-text {
		font-size: 0.75rem;
		color: var(--text-muted);
		margin-bottom: 0.15rem;
		overflow: hidden;
		text-overflow: ellipsis;
		display: -webkit-box;
		-webkit-line-clamp: 2;
		-webkit-box-orient: vertical;
	}
</style>
