<script lang="ts">
	import { projectRaw } from '$lib/api';
	import type { ProjectionResponse, ConversationDetail } from '$lib/types';
	import TokenHeatmap from '$lib/components/TokenHeatmap.svelte';
	import SaveLoadPanel from '$lib/components/SaveLoadPanel.svelte';

	let text = $state('');
	let loading = $state(false);
	let error = $state<string | null>(null);
	let result = $state<ProjectionResponse | null>(null);

	let charCount = $derived(text.length);
	let isLong = $derived(charCount > 4096);
	let canAnalyze = $derived(text.trim().length > 0 && !loading);

	async function analyze() {
		if (!canAnalyze) return;

		loading = true;
		error = null;

		try {
			result = await projectRaw(text);
		} catch (e) {
			error = e instanceof Error ? e.message : String(e);
		} finally {
			loading = false;
		}
	}

	function handleKeydown(e: KeyboardEvent) {
		if (e.key === 'Enter' && (e.metaKey || e.ctrlKey) && canAnalyze) {
			analyze();
		}
	}

	// ---- Save / Load ----

	function getSaveData() {
		return { text };
	}

	async function handleLoad(detail: ConversationDetail) {
		text = detail.text || '';
		error = null;
		result = null;
		// Auto-analyze after loading
		if (text.trim()) {
			await analyze();
		}
	}

	let defaultSaveName = $derived(text.slice(0, 50) || '');
</script>

<div class="raw-page">
	<SaveLoadPanel
		mode="raw"
		{getSaveData}
		onLoad={handleLoad}
		defaultName={defaultSaveName}
	/>

	<section class="input-section">
		<textarea
			bind:value={text}
			placeholder="Paste text here to analyze Assistant Axis projections..."
			class="text-input"
			rows="10"
			onkeydown={handleKeydown}
		></textarea>

		<div class="controls">
			<button class="primary" onclick={analyze} disabled={!canAnalyze}>
				{#if loading}
					<span class="spinner"></span>
					Analyzing...
				{:else}
					Analyze
				{/if}
			</button>

			<div class="stats">
				<span class="stat">{charCount.toLocaleString()} chars</span>
				{#if result}
					<span class="stat">{result.tokens.length.toLocaleString()} tokens</span>
				{/if}
			</div>

			{#if isLong}
				<span class="warning">Long text may take a while to process</span>
			{/if}
		</div>

		<p class="hint">Ctrl+Enter to analyze</p>
	</section>

	{#if error}
		<div class="error-banner">
			<strong>Error:</strong> {error}
		</div>
	{/if}

	{#if result}
		<section class="results-section">
			<h2>
				Results
				<span class="result-meta">
					layer {result.layer} | {result.model_name}
				</span>
			</h2>
			<TokenHeatmap response={result} />
		</section>
	{/if}
</div>

<style>
	.raw-page {
		display: flex;
		flex-direction: column;
		gap: 1.5rem;
	}

	.input-section {
		display: flex;
		flex-direction: column;
		gap: 0.75rem;
	}

	.text-input {
		width: 100%;
		min-height: 200px;
		resize: vertical;
		font-family: 'Fira Code', 'Cascadia Code', monospace;
		font-size: 0.875rem;
		line-height: 1.6;
		padding: 0.75rem;
		background: var(--surface);
		border: 1px solid var(--border);
		border-radius: 4px;
		color: var(--text);
	}

	.text-input:focus {
		outline: none;
		border-color: var(--primary);
	}

	.text-input::placeholder {
		color: var(--text-muted);
	}

	.controls {
		display: flex;
		align-items: center;
		gap: 1rem;
		flex-wrap: wrap;
	}

	.controls button {
		display: inline-flex;
		align-items: center;
		gap: 0.5rem;
	}

	.controls button:disabled {
		opacity: 0.5;
		cursor: not-allowed;
	}

	.stats {
		display: flex;
		gap: 1rem;
	}

	.stat {
		font-size: 0.8rem;
		color: var(--text-muted);
		font-family: monospace;
	}

	.warning {
		font-size: 0.8rem;
		color: var(--warning);
	}

	.hint {
		font-size: 0.75rem;
		color: var(--text-muted);
	}

	.error-banner {
		padding: 0.75rem 1rem;
		background: rgba(239, 83, 80, 0.1);
		border: 1px solid var(--danger);
		border-radius: 4px;
		color: var(--danger);
		font-size: 0.875rem;
	}

	.results-section h2 {
		font-size: 1rem;
		margin-bottom: 1rem;
		display: flex;
		align-items: center;
		gap: 0.75rem;
	}

	.result-meta {
		font-size: 0.75rem;
		color: var(--text-muted);
		font-weight: normal;
		font-family: monospace;
	}

	.spinner {
		display: inline-block;
		width: 14px;
		height: 14px;
		border: 2px solid rgba(0, 0, 0, 0.2);
		border-top-color: #000;
		border-radius: 50%;
		animation: spin 0.6s linear infinite;
	}

	@keyframes spin {
		to {
			transform: rotate(360deg);
		}
	}
</style>
