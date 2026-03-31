<script lang="ts">
	import type { ProjectionResponse, TokenProjection, TurnSpan } from '$lib/types';
	import Tooltip from './Tooltip.svelte';

	let { response }: { response: ProjectionResponse } = $props();

	// Tooltip state
	let tooltipX = $state(0);
	let tooltipY = $state(0);
	let hoveredToken = $state<TokenProjection | null>(null);
	let hoveredTurn = $state<TurnSpan | null>(null);

	// Projection extent for color normalization
	let extent = $derived.by(() => {
		if (response.tokens.length === 0) return { min: 0, max: 1 };
		const projections = response.tokens.map((t) => t.projection);
		return {
			min: Math.min(...projections),
			max: Math.max(...projections)
		};
	});

	// Build a lookup from position to the span it belongs to
	let spanByPosition = $derived.by(() => {
		const map = new Map<number, TurnSpan>();
		for (const span of response.spans) {
			for (let i = span.start; i < span.end; i++) {
				map.set(i, span);
			}
		}
		return map;
	});

	// Group tokens by turn span for rendering
	interface TokenGroup {
		span: TurnSpan | null;
		tokens: TokenProjection[];
	}

	let tokenGroups = $derived.by((): TokenGroup[] => {
		if (response.spans.length === 0) {
			// Raw text mode: single group with no span
			return [{ span: null, tokens: response.tokens }];
		}

		const groups: TokenGroup[] = [];
		let currentSpan: TurnSpan | null = null;
		let currentTokens: TokenProjection[] = [];

		for (const token of response.tokens) {
			const span = spanByPosition.get(token.position) ?? null;
			if (span !== currentSpan) {
				if (currentTokens.length > 0) {
					groups.push({ span: currentSpan, tokens: currentTokens });
				}
				currentSpan = span;
				currentTokens = [token];
			} else {
				currentTokens.push(token);
			}
		}
		if (currentTokens.length > 0) {
			groups.push({ span: currentSpan, tokens: currentTokens });
		}
		return groups;
	});

	/**
	 * Map a projection value to a diverging color.
	 * High projection (assistant-like) -> blue (hsl 210)
	 * Low projection (drifted) -> red (hsl 0)
	 * Middle -> neutral gray
	 */
	function projectionColor(value: number): string {
		const { min, max } = extent;
		const range = max - min;
		if (range === 0) return 'hsla(210, 10%, 30%, 0.3)';

		// Normalize to 0..1
		const t = (value - min) / range;

		// Interpolate hue: 0 (red) at t=0, 210 (blue) at t=1
		const hue = t * 210;
		// Saturation peaks at extremes, lower in the middle
		const distFromCenter = Math.abs(t - 0.5) * 2; // 0 at center, 1 at extremes
		const saturation = 30 + distFromCenter * 50;
		// Lightness: keep readable on dark theme
		const lightness = 25 + distFromCenter * 15;
		const alpha = 0.4 + distFromCenter * 0.4;

		return `hsla(${hue}, ${saturation}%, ${lightness}%, ${alpha})`;
	}

	/** Check if a token string is a special/control token */
	function isSpecialToken(str: string): boolean {
		return str.startsWith('<|') && str.endsWith('|>');
	}

	/** Check if a token represents a newline */
	function isNewline(str: string): boolean {
		return str === '\n' || str === '\r\n' || str === '\r' || str === '\u010a';
	}

	/** Check if a token is purely whitespace (but not a newline) */
	function isWhitespace(str: string): boolean {
		return /^\s+$/.test(str) && !isNewline(str);
	}

	/** Role label display text */
	function roleLabel(role: string): string {
		return role.charAt(0).toUpperCase() + role.slice(1);
	}

	/** Role label CSS class */
	function roleClass(role: string): string {
		switch (role) {
			case 'assistant':
				return 'role-assistant';
			case 'user':
				return 'role-user';
			case 'system':
				return 'role-system';
			default:
				return 'role-other';
		}
	}

	function handleMouseMove(e: MouseEvent) {
		tooltipX = e.clientX;
		tooltipY = e.clientY;
	}

	function handleTokenEnter(token: TokenProjection) {
		hoveredToken = token;
		hoveredTurn = spanByPosition.get(token.position) ?? null;
	}

	function handleTokenLeave() {
		hoveredToken = null;
		hoveredTurn = null;
	}

	// Color legend gradient stops
	let legendStops = $derived.by(() => {
		const stops: string[] = [];
		const steps = 20;
		for (let i = 0; i <= steps; i++) {
			const t = i / steps;
			const value = extent.min + t * (extent.max - extent.min);
			stops.push(projectionColor(value));
		}
		return stops;
	});
</script>

<!-- svelte-ignore a11y_no_static_element_interactions -->
<div class="heatmap-container" onmousemove={handleMouseMove}>
	<!-- Color scale legend -->
	<div class="legend">
		<span class="legend-label">{extent.min.toFixed(3)}</span>
		<div class="legend-bar">
			{#each legendStops as color, i}
				<div
					class="legend-segment"
					style="background: {color}; width: {100 / legendStops.length}%;"
				></div>
			{/each}
		</div>
		<span class="legend-label">{extent.max.toFixed(3)}</span>
		<span class="legend-desc">low (red) --- high (blue)</span>
	</div>

	<!-- Token groups by turn -->
	{#each tokenGroups as group}
		<div class="turn-block">
			{#if group.span}
				<div class="turn-header">
					<span class="role-label {roleClass(group.span.role)}">
						{roleLabel(group.span.role)}
					</span>
					<span class="turn-meta">
						turn {group.span.turn} | mean: {group.span.mean_projection.toFixed(4)}
					</span>
				</div>
			{/if}
			<div class="token-flow">
				{#each group.tokens as token (token.position)}
					{#if isNewline(token.token_str)}
						<br />
					{:else if isSpecialToken(token.token_str)}
						<span
							class="token special"
							onmouseenter={() => handleTokenEnter(token)}
							onmouseleave={handleTokenLeave}
							role="presentation"
						>{token.token_str}</span>
					{:else if isWhitespace(token.token_str)}
						<span
							class="token whitespace"
							style="background: {projectionColor(token.projection)};"
							onmouseenter={() => handleTokenEnter(token)}
							onmouseleave={handleTokenLeave}
							role="presentation"
						>&nbsp;</span>
					{:else}
						<span
							class="token"
							style="background: {projectionColor(token.projection)};"
							onmouseenter={() => handleTokenEnter(token)}
							onmouseleave={handleTokenLeave}
							role="presentation"
						>{token.token_str}</span>
					{/if}
				{/each}
			</div>
		</div>
	{/each}

	<Tooltip x={tooltipX} y={tooltipY} visible={hoveredToken !== null}>
		{#if hoveredToken}
			<div class="tip-token">
				<strong>"{hoveredToken.token_str}"</strong>
			</div>
			<div class="tip-detail">projection: {hoveredToken.projection.toFixed(4)}</div>
			<div class="tip-detail">position: {hoveredToken.position}</div>
			<div class="tip-detail">token_id: {hoveredToken.token_id}</div>
			{#if hoveredTurn}
				<div class="tip-detail">
					turn {hoveredTurn.turn} ({hoveredTurn.role})
				</div>
			{/if}
		{/if}
	</Tooltip>
</div>

<style>
	.heatmap-container {
		position: relative;
	}

	/* Color legend */
	.legend {
		display: flex;
		align-items: center;
		gap: 0.5rem;
		margin-bottom: 1rem;
		padding: 0.5rem 0;
		border-bottom: 1px solid var(--border);
	}
	.legend-bar {
		display: flex;
		flex: 1;
		height: 12px;
		border-radius: 3px;
		overflow: hidden;
		max-width: 300px;
	}
	.legend-segment {
		height: 100%;
	}
	.legend-label {
		font-size: 0.7rem;
		color: var(--text-muted);
		font-family: monospace;
		white-space: nowrap;
	}
	.legend-desc {
		font-size: 0.7rem;
		color: var(--text-muted);
		margin-left: 0.5rem;
	}

	/* Turn blocks */
	.turn-block {
		margin-bottom: 0.75rem;
		padding: 0.5rem;
		border-left: 3px solid var(--border);
		border-radius: 2px;
	}
	.turn-block:not(:last-child) {
		border-bottom: 1px solid var(--border);
		padding-bottom: 0.75rem;
	}
	.turn-header {
		display: flex;
		align-items: center;
		gap: 0.75rem;
		margin-bottom: 0.35rem;
	}
	.role-label {
		font-size: 0.75rem;
		font-weight: 600;
		text-transform: uppercase;
		letter-spacing: 0.05em;
		padding: 0.1rem 0.4rem;
		border-radius: 3px;
	}
	.role-assistant {
		color: var(--warning);
		background: rgba(255, 167, 38, 0.1);
	}
	.role-user {
		color: var(--primary);
		background: rgba(79, 195, 247, 0.1);
	}
	.role-system {
		color: var(--text-muted);
		background: rgba(136, 146, 164, 0.1);
	}
	.role-other {
		color: var(--text-muted);
	}
	.turn-meta {
		font-size: 0.7rem;
		color: var(--text-muted);
		font-family: monospace;
	}

	/* Token flow */
	.token-flow {
		line-height: 1.8;
		font-family: 'Fira Code', 'Cascadia Code', monospace;
		font-size: 0.85rem;
	}
	.token {
		display: inline;
		padding: 0.1rem 0;
		border-radius: 2px;
		cursor: default;
		transition: outline 0.1s;
	}
	.token:hover {
		outline: 1px solid var(--primary);
		outline-offset: 1px;
	}
	.token.special {
		color: var(--text-muted);
		opacity: 0.4;
		font-size: 0.7rem;
	}
	.token.whitespace {
		display: inline-block;
		min-width: 0.3em;
	}

	/* Tooltip details */
	.tip-token {
		margin-bottom: 0.25rem;
		font-family: monospace;
	}
	.tip-detail {
		font-size: 0.75rem;
		color: var(--text-muted);
		margin-bottom: 0.1rem;
	}
</style>
