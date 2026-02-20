<script lang="ts">
	import type { Snippet } from 'svelte';

	let {
		title,
		open = false,
		onclose,
		children
	}: {
		title: string;
		open: boolean;
		onclose: () => void;
		children: Snippet;
	} = $props();

	function handleKeydown(e: KeyboardEvent) {
		if (e.key === 'Escape') onclose();
	}

	function handleBackdropClick(e: MouseEvent) {
		if (e.target === e.currentTarget) onclose();
	}
</script>

{#if open}
	<!-- svelte-ignore a11y_no_noninteractive_element_interactions -->
	<div class="modal-backdrop" role="dialog" aria-modal="true" aria-label={title} onclick={handleBackdropClick} onkeydown={handleKeydown}>
		<div class="modal-card">
			<div class="modal-header">
				<h3>{title}</h3>
				<button class="close-btn" onclick={onclose}>&times;</button>
			</div>
			<div class="modal-body">
				{@render children()}
			</div>
		</div>
	</div>
{/if}

<style>
	.modal-backdrop {
		position: fixed;
		inset: 0;
		background: rgba(0, 0, 0, 0.5);
		display: flex;
		align-items: center;
		justify-content: center;
		z-index: 1000;
	}
	.modal-card {
		background: var(--bg, #fff);
		border-radius: 8px;
		width: 90vw;
		max-width: 720px;
		max-height: 80vh;
		display: flex;
		flex-direction: column;
		box-shadow: 0 8px 32px rgba(0, 0, 0, 0.2);
	}
	.modal-header {
		display: flex;
		justify-content: space-between;
		align-items: center;
		padding: 1rem 1.25rem;
		border-bottom: 1px solid var(--border, #ddd);
	}
	.modal-header h3 {
		margin: 0;
		font-size: 1rem;
	}
	.close-btn {
		background: none;
		border: none;
		font-size: 1.5rem;
		cursor: pointer;
		padding: 0 0.25rem;
		line-height: 1;
		color: var(--text-muted, #888);
	}
	.modal-body {
		padding: 1.25rem;
		overflow-y: auto;
	}
</style>
