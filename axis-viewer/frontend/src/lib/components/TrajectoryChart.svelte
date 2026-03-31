<script lang="ts">
	import type { TurnSpan } from '$lib/types';
	import Tooltip from './Tooltip.svelte';

	let { spans }: { spans: TurnSpan[] } = $props();

	// SVG dimensions
	const width = 600;
	const height = 300;
	const padding = { top: 30, right: 30, bottom: 40, left: 60 };

	// Tooltip state
	let tooltipX = $state(0);
	let tooltipY = $state(0);
	let hoveredSpan = $state<TurnSpan | null>(null);

	// Sort spans by turn index
	let sortedSpans = $derived([...spans].sort((a, b) => a.turn - b.turn));

	// Axis extents
	let xExtent = $derived.by((): [number, number] => {
		if (sortedSpans.length === 0) return [0, 1];
		const turns = sortedSpans.map((s) => s.turn);
		return [Math.min(...turns), Math.max(...turns)];
	});

	let yExtent = $derived.by((): [number, number] => {
		if (sortedSpans.length === 0) return [0, 1];
		const vals = sortedSpans.map((s) => s.mean_projection);
		const min = Math.min(...vals);
		const max = Math.max(...vals);
		const range = max - min || 1;
		return [min - range * 0.1, max + range * 0.1];
	});

	// Scale functions
	function scaleX(turn: number): number {
		const [min, max] = xExtent;
		const range = max - min || 1;
		return padding.left + ((turn - min) / range) * (width - padding.left - padding.right);
	}

	function scaleY(val: number): number {
		const [min, max] = yExtent;
		const range = max - min || 1;
		return (
			height - padding.bottom - ((val - min) / range) * (height - padding.top - padding.bottom)
		);
	}

	// Build line path
	let linePath = $derived.by(() => {
		if (sortedSpans.length === 0) return '';
		return sortedSpans
			.map((s, i) => {
				const x = scaleX(s.turn);
				const y = scaleY(s.mean_projection);
				return `${i === 0 ? 'M' : 'L'} ${x} ${y}`;
			})
			.join(' ');
	});

	// Grid lines for Y axis (5 ticks)
	let yTicks = $derived.by(() => {
		const [min, max] = yExtent;
		const step = (max - min) / 4;
		return Array.from({ length: 5 }, (_, i) => min + i * step);
	});

	// Grid lines for X axis (turn indices)
	let xTicks = $derived.by(() => {
		if (sortedSpans.length <= 1) return sortedSpans.map((s) => s.turn);
		const [min, max] = xExtent;
		// Show every turn if few, otherwise sample
		const allTurns = sortedSpans.map((s) => s.turn);
		if (allTurns.length <= 10) return allTurns;
		// Sample roughly 8 ticks
		const step = Math.max(1, Math.floor((max - min) / 7));
		const ticks: number[] = [];
		for (let t = min; t <= max; t += step) {
			ticks.push(t);
		}
		if (ticks[ticks.length - 1] !== max) ticks.push(max);
		return ticks;
	});

	// Point color by role
	function roleColor(role: string): string {
		switch (role) {
			case 'user':
				return 'var(--primary)';
			case 'assistant':
				return 'var(--warning)';
			case 'system':
				return 'var(--text-muted)';
			default:
				return 'var(--text)';
		}
	}

	function handleMouseMove(e: MouseEvent) {
		tooltipX = e.clientX;
		tooltipY = e.clientY;
	}

	function truncate(text: string, maxLen: number): string {
		if (text.length <= maxLen) return text;
		return text.slice(0, maxLen) + '...';
	}
</script>

<!-- svelte-ignore a11y_no_static_element_interactions -->
<div class="chart-container" onmousemove={handleMouseMove}>
	{#if sortedSpans.length === 0}
		<p class="empty">No turn data to display.</p>
	{:else}
		<svg viewBox="0 0 {width} {height}" class="chart-svg">
			<!-- Y-axis grid lines and labels -->
			{#each yTicks as tick}
				<line
					x1={padding.left}
					y1={scaleY(tick)}
					x2={width - padding.right}
					y2={scaleY(tick)}
					class="grid-line"
				/>
				<text
					x={padding.left - 8}
					y={scaleY(tick) + 4}
					class="axis-tick"
					text-anchor="end"
				>{tick.toFixed(3)}</text>
			{/each}

			<!-- X-axis grid lines and labels -->
			{#each xTicks as tick}
				<line
					x1={scaleX(tick)}
					y1={padding.top}
					x2={scaleX(tick)}
					y2={height - padding.bottom}
					class="grid-line"
				/>
				<text
					x={scaleX(tick)}
					y={height - padding.bottom + 16}
					class="axis-tick"
					text-anchor="middle"
				>{tick}</text>
			{/each}

			<!-- Axes -->
			<line
				x1={padding.left}
				y1={height - padding.bottom}
				x2={width - padding.right}
				y2={height - padding.bottom}
				class="axis"
			/>
			<line
				x1={padding.left}
				y1={padding.top}
				x2={padding.left}
				y2={height - padding.bottom}
				class="axis"
			/>

			<!-- Reference line (dashed) for typical assistant range -->
			{#if yExtent[0] <= 0 && yExtent[1] >= 0}
				<line
					x1={padding.left}
					y1={scaleY(0)}
					x2={width - padding.right}
					y2={scaleY(0)}
					class="reference-line"
				/>
				<text
					x={width - padding.right + 4}
					y={scaleY(0) + 4}
					class="reference-label"
				>baseline</text>
			{/if}

			<!-- Connecting line -->
			<path d={linePath} class="trajectory-line" />

			<!-- Data points -->
			{#each sortedSpans as span}
				<circle
					cx={scaleX(span.turn)}
					cy={scaleY(span.mean_projection)}
					r={hoveredSpan === span ? 8 : 6}
					fill={roleColor(span.role)}
					class="point"
					class:hovered={hoveredSpan === span}
					onmouseenter={() => { hoveredSpan = span; }}
					onmouseleave={() => { hoveredSpan = null; }}
					role="button"
					tabindex="0"
				/>
			{/each}

			<!-- Axis labels -->
			<text
				x={(padding.left + width - padding.right) / 2}
				y={height - 4}
				class="axis-label"
			>Turn</text>
			<text
				x={14}
				y={(padding.top + height - padding.bottom) / 2}
				class="axis-label"
				transform="rotate(-90, 14, {(padding.top + height - padding.bottom) / 2})"
			>Mean Projection</text>
		</svg>

		<!-- Legend -->
		<div class="legend">
			<span class="legend-item">
				<span class="legend-dot" style="background: var(--primary);"></span>
				User
			</span>
			<span class="legend-item">
				<span class="legend-dot" style="background: var(--warning);"></span>
				Assistant
			</span>
			<span class="legend-item">
				<span class="legend-dot" style="background: var(--text-muted);"></span>
				System
			</span>
		</div>
	{/if}

	<Tooltip x={tooltipX} y={tooltipY} visible={hoveredSpan !== null}>
		{#if hoveredSpan}
			<div class="tip-turn">
				<strong>Turn {hoveredSpan.turn}</strong>
				<span class="tip-role" style="color: {roleColor(hoveredSpan.role)}">
					{hoveredSpan.role}
				</span>
			</div>
			<div class="tip-projection">
				mean projection: {hoveredSpan.mean_projection.toFixed(4)}
			</div>
			<div class="tip-text">{truncate(hoveredSpan.text, 100)}</div>
		{/if}
	</Tooltip>
</div>

<style>
	.chart-container {
		position: relative;
		width: 100%;
	}
	.chart-svg {
		width: 100%;
		max-width: 700px;
		display: block;
	}
	.empty {
		color: var(--text-muted);
		text-align: center;
		padding: 2rem;
	}

	/* Grid and axes */
	.grid-line {
		stroke: var(--border);
		stroke-width: 0.5;
		opacity: 0.4;
	}
	.axis {
		stroke: var(--text-muted);
		stroke-width: 1;
	}
	.axis-tick {
		fill: var(--text-muted);
		font-size: 9px;
		font-family: monospace;
	}
	.axis-label {
		fill: var(--text-muted);
		font-size: 11px;
		text-anchor: middle;
	}

	/* Reference line */
	.reference-line {
		stroke: var(--text-muted);
		stroke-width: 1;
		stroke-dasharray: 6 3;
		opacity: 0.5;
	}
	.reference-label {
		fill: var(--text-muted);
		font-size: 9px;
		font-style: italic;
	}

	/* Trajectory */
	.trajectory-line {
		fill: none;
		stroke: var(--text-muted);
		stroke-width: 1.5;
		opacity: 0.5;
	}
	.point {
		cursor: pointer;
		stroke: var(--bg);
		stroke-width: 2;
		transition:
			r 0.15s,
			stroke-width 0.15s;
		opacity: 0.9;
	}
	.point:hover,
	.point.hovered {
		opacity: 1;
		stroke: var(--text);
		stroke-width: 2.5;
	}

	/* Legend */
	.legend {
		display: flex;
		gap: 1rem;
		margin-top: 0.5rem;
		padding: 0.25rem 0;
	}
	.legend-item {
		display: flex;
		align-items: center;
		gap: 0.3rem;
		font-size: 0.75rem;
		color: var(--text-muted);
	}
	.legend-dot {
		display: inline-block;
		width: 10px;
		height: 10px;
		border-radius: 50%;
	}

	/* Tooltip details */
	.tip-turn {
		display: flex;
		align-items: center;
		gap: 0.5rem;
		margin-bottom: 0.25rem;
	}
	.tip-role {
		font-size: 0.7rem;
		text-transform: uppercase;
		font-weight: 600;
	}
	.tip-projection {
		font-size: 0.75rem;
		color: var(--text-muted);
		font-family: monospace;
		margin-bottom: 0.25rem;
	}
	.tip-text {
		font-size: 0.75rem;
		color: var(--text-muted);
		line-height: 1.3;
	}
</style>
