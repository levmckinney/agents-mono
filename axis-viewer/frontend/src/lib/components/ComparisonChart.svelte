<script lang="ts">
	import type { TurnSpan } from '$lib/types';
	import Tooltip from './Tooltip.svelte';

	interface TrajectoryData {
		name: string;
		spans: TurnSpan[];
		color: string;
	}

	let { trajectories }: { trajectories: TrajectoryData[] } = $props();

	// SVG dimensions
	const width = 700;
	const height = 350;
	const padding = { top: 30, right: 140, bottom: 40, left: 60 };

	// Tooltip state
	let tooltipX = $state(0);
	let tooltipY = $state(0);
	let hoveredPoint = $state<{ trajectory: TrajectoryData; span: TurnSpan } | null>(null);

	// Sorted spans per trajectory
	let sortedTrajectories = $derived(
		trajectories.map((t) => ({
			...t,
			sortedSpans: [...t.spans].sort((a, b) => a.turn - b.turn)
		}))
	);

	// Global axis extents across all trajectories
	let xExtent = $derived.by((): [number, number] => {
		const allTurns = sortedTrajectories.flatMap((t) => t.sortedSpans.map((s) => s.turn));
		if (allTurns.length === 0) return [0, 1];
		return [Math.min(...allTurns), Math.max(...allTurns)];
	});

	let yExtent = $derived.by((): [number, number] => {
		const allVals = sortedTrajectories.flatMap((t) =>
			t.sortedSpans.map((s) => s.mean_projection)
		);
		if (allVals.length === 0) return [0, 1];
		const min = Math.min(...allVals);
		const max = Math.max(...allVals);
		const range = max - min || 1;
		return [min - range * 0.1, max + range * 0.1];
	});

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

	// Build line paths for each trajectory
	function buildLinePath(spans: TurnSpan[]): string {
		if (spans.length === 0) return '';
		return spans
			.map((s, i) => {
				const x = scaleX(s.turn);
				const y = scaleY(s.mean_projection);
				return `${i === 0 ? 'M' : 'L'} ${x} ${y}`;
			})
			.join(' ');
	}

	// Y-axis ticks
	let yTicks = $derived.by(() => {
		const [min, max] = yExtent;
		const step = (max - min) / 4;
		return Array.from({ length: 5 }, (_, i) => min + i * step);
	});

	// X-axis ticks
	let xTicks = $derived.by(() => {
		const allTurns = sortedTrajectories.flatMap((t) => t.sortedSpans.map((s) => s.turn));
		if (allTurns.length === 0) return [0];
		const unique = [...new Set(allTurns)].sort((a, b) => a - b);
		if (unique.length <= 12) return unique;
		const [min, max] = xExtent;
		const step = Math.max(1, Math.floor((max - min) / 8));
		const ticks: number[] = [];
		for (let t = min; t <= max; t += step) ticks.push(t);
		if (ticks[ticks.length - 1] !== max) ticks.push(max);
		return ticks;
	});

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
	{#if sortedTrajectories.every((t) => t.sortedSpans.length === 0)}
		<p class="empty">No trajectory data to compare.</p>
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
				<text x={padding.left - 8} y={scaleY(tick) + 4} class="axis-tick" text-anchor="end"
					>{tick.toFixed(3)}</text
				>
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
					text-anchor="middle">{tick}</text
				>
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

			<!-- Baseline reference -->
			{#if yExtent[0] <= 0 && yExtent[1] >= 0}
				<line
					x1={padding.left}
					y1={scaleY(0)}
					x2={width - padding.right}
					y2={scaleY(0)}
					class="reference-line"
				/>
				<text x={width - padding.right + 4} y={scaleY(0) + 4} class="reference-label"
					>baseline</text
				>
			{/if}

			<!-- Trajectory lines and points -->
			{#each sortedTrajectories as traj}
				<path d={buildLinePath(traj.sortedSpans)} fill="none" stroke={traj.color} stroke-width="2" opacity="0.7" />
				{#each traj.sortedSpans as span}
					<circle
						cx={scaleX(span.turn)}
						cy={scaleY(span.mean_projection)}
						r={hoveredPoint?.trajectory === traj && hoveredPoint?.span === span ? 7 : 5}
						fill={traj.color}
						class="point"
						class:hovered={hoveredPoint?.trajectory === traj && hoveredPoint?.span === span}
						onmouseenter={() => {
							hoveredPoint = { trajectory: traj, span };
						}}
						onmouseleave={() => {
							hoveredPoint = null;
						}}
						role="button"
						tabindex="0"
					/>
				{/each}
			{/each}

			<!-- Axis labels -->
			<text
				x={(padding.left + width - padding.right) / 2}
				y={height - 4}
				class="axis-label">Turn</text
			>
			<text
				x={14}
				y={(padding.top + height - padding.bottom) / 2}
				class="axis-label"
				transform="rotate(-90, 14, {(padding.top + height - padding.bottom) / 2})"
				>Mean Projection</text
			>
		</svg>

		<!-- Legend -->
		<div class="legend">
			{#each sortedTrajectories as traj}
				<span class="legend-item">
					<span class="legend-dot" style="background: {traj.color};"></span>
					{truncate(traj.name, 30)}
				</span>
			{/each}
		</div>
	{/if}

	<Tooltip x={tooltipX} y={tooltipY} visible={hoveredPoint !== null}>
		{#if hoveredPoint}
			<div class="tip-name">
				<strong>{truncate(hoveredPoint.trajectory.name, 40)}</strong>
			</div>
			<div class="tip-detail">
				Turn {hoveredPoint.span.turn} ({hoveredPoint.span.role})
			</div>
			<div class="tip-detail">
				mean projection: {hoveredPoint.span.mean_projection.toFixed(4)}
			</div>
			<div class="tip-text">{truncate(hoveredPoint.span.text, 80)}</div>
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
		max-width: 800px;
		display: block;
	}
	.empty {
		color: var(--text-muted);
		text-align: center;
		padding: 2rem;
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
	.legend {
		display: flex;
		flex-wrap: wrap;
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
		flex-shrink: 0;
	}
	.tip-name {
		margin-bottom: 0.25rem;
	}
	.tip-detail {
		font-size: 0.75rem;
		color: var(--text-muted);
		font-family: monospace;
		margin-bottom: 0.1rem;
	}
	.tip-text {
		font-size: 0.75rem;
		color: var(--text-muted);
		line-height: 1.3;
	}
</style>
