<script lang="ts">
	import type { Snippet } from 'svelte';

	let {
		x,
		y,
		visible,
		children
	}: {
		x: number;
		y: number;
		visible: boolean;
		children: Snippet;
	} = $props();
</script>

{#if visible}
	<div class="tooltip" style="left: {x + 12}px; top: {y + 12}px;">
		{@render children()}
	</div>
{/if}

<style>
	.tooltip {
		position: fixed;
		background: var(--surface);
		border: 1px solid var(--border);
		border-radius: 4px;
		padding: 0.5rem 0.75rem;
		max-width: 320px;
		pointer-events: none;
		z-index: 100;
		font-size: 0.8rem;
		line-height: 1.4;
		box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
	}
</style>
