<script lang="ts">
	import type { EmbeddingPoint } from '$lib/types';
	import Tooltip from './Tooltip.svelte';

	let {
		points,
		selectedPairId = null,
		onselect,
	}: {
		points: EmbeddingPoint[];
		selectedPairId?: string | null;
		onselect: (pairId: string) => void;
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
		return height - padding - ((val - min) / (max - min)) * (height - 2 * padding);
	}

	function handleMouseMove(e: MouseEvent) {
		tooltipX = e.clientX;
		tooltipY = e.clientY;
	}

	function truncateFromStart(text: string, maxLen: number): string {
		if (text.length <= maxLen) return text;
		return '...' + text.slice(text.length - maxLen);
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

		<!-- Points -->
		{#each points as point}
			<circle
				cx={scaleX(point.x)}
				cy={scaleY(point.y)}
				r={point.pair_id === selectedPairId ? 10 : 7}
				fill={point.pair_id === selectedPairId ? 'var(--warning)' : 'var(--primary)'}
				class="point"
				class:selected={point.pair_id === selectedPairId}
				onclick={() => onselect(point.pair_id)}
				onmouseenter={() => { hoveredPoint = point; }}
				onmouseleave={() => { hoveredPoint = null; }}
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
			<div class="tip-text"><em>Prompt:</em> {truncateFromStart(hoveredPoint.prompt_preview, 80)}</div>
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
	}
</style>
