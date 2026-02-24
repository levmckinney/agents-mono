<script lang="ts">
	import type { RunResults, InfluenceRow } from '$lib/types';

	let {
		queryId,
		results,
		normalize
	}: {
		queryId: string;
		results: RunResults;
		normalize: boolean;
	} = $props();

	let queryInfo = $derived(
		results.query_results.find((q) => q.query_id === queryId) ?? null
	);

	let influences = $derived.by(() => {
		const rows = results.influences.filter((r) => r.query_id === queryId);
		rows.sort((a, b) => b.influence_score - a.influence_score);
		return rows;
	});

	function trainInfo(trainId: string): Record<string, unknown> | null {
		return results.train_results.find((t) => t.train_id === trainId) ?? null;
	}

	function normalizedScore(score: number): number {
		if (!normalize || influences.length === 0) return score;
		const max = Math.max(...influences.map((r) => Math.abs(r.influence_score)));
		return max === 0 ? 0 : score / max;
	}

	function formatScore(n: number): string {
		if (normalize) return n.toFixed(4);
		return n.toLocaleString(undefined, { maximumFractionDigits: 2 });
	}

	function truncateFromStart(text: string, maxLen: number): string {
		if (text.length <= maxLen) return text;
		return '...' + text.slice(text.length - maxLen);
	}
</script>

<div class="detail-panel">
	{#if queryInfo}
		<div class="query-info">
			<div class="header-row">
				<span class="label">Query: <code>{queryId}</code></span>
				{#if queryInfo.loss != null}
					<span class="loss">loss: {Number(queryInfo.loss).toFixed(2)}</span>
				{/if}
			</div>
			<div class="full-text">
				<div class="text-label">Prompt</div>
				<div class="text-body">{String(queryInfo.prompt ?? '')}</div>
			</div>
			<div class="full-text">
				<div class="text-label">Completion</div>
				<div class="text-body completion">{String(queryInfo.completion ?? '')}</div>
			</div>
		</div>
	{/if}

	<h3>Influences ({influences.length})</h3>

	{#each influences as inf, i}
		{@const train = trainInfo(inf.train_id)}
		<div class="influence-card">
			<div class="card-header">
				<span class="rank">#{i + 1}</span>
				<code class="train-id">{inf.train_id}</code>
				<span
					class="score"
					class:positive={inf.influence_score > 0}
					class:negative={inf.influence_score < 0}
				>
					{formatScore(normalizedScore(inf.influence_score))}
				</span>
			</div>
			{#if train}
				<div class="card-text">
					<span class="text-label">Prompt</span>
					<span class="text-body">{truncateFromStart(String(train.prompt ?? ''), 300)}</span>
				</div>
				<div class="card-text">
					<span class="text-label">Completion</span>
					<span class="text-body completion">{String(train.completion ?? '')}</span>
				</div>
			{/if}
		</div>
	{/each}

	{#if influences.length === 0}
		<p class="empty">No influences found</p>
	{/if}
</div>

<style>
	.detail-panel {
		display: flex;
		flex-direction: column;
		gap: 0.5rem;
	}
	h3 {
		font-size: 0.85rem;
		color: var(--text-muted);
		text-transform: uppercase;
		letter-spacing: 0.05em;
		margin-top: 0.5rem;
	}

	/* Query info */
	.query-info {
		padding: 0.75rem;
		background: var(--surface);
		border-radius: 4px;
		border: 1px solid var(--primary);
		border-left-width: 3px;
	}
	.header-row {
		display: flex;
		justify-content: space-between;
		align-items: center;
		margin-bottom: 0.5rem;
	}
	.label {
		font-weight: 600;
		font-size: 0.85rem;
	}
	.loss {
		color: var(--text-muted);
		font-size: 0.8rem;
	}

	/* Shared text styles */
	.full-text, .card-text {
		margin-bottom: 0.35rem;
	}
	.text-label {
		font-size: 0.7rem;
		text-transform: uppercase;
		color: var(--text-muted);
		letter-spacing: 0.03em;
	}
	.text-body {
		font-size: 0.8rem;
		line-height: 1.5;
		white-space: pre-wrap;
		word-break: break-word;
	}
	.text-body.completion {
		color: var(--primary);
	}

	/* Influence cards */
	.influence-card {
		padding: 0.6rem 0.75rem;
		background: var(--surface);
		border-radius: 4px;
		border: 1px solid var(--border);
	}
	.card-header {
		display: flex;
		align-items: center;
		gap: 0.5rem;
		margin-bottom: 0.4rem;
	}
	.rank {
		font-weight: 700;
		font-size: 0.8rem;
		color: var(--text-muted);
		min-width: 2rem;
	}
	.train-id {
		font-size: 0.75rem;
		color: var(--text-muted);
	}
	.score {
		margin-left: auto;
		font-family: monospace;
		font-size: 0.85rem;
		font-weight: 600;
	}
	.score.positive {
		color: var(--success);
	}
	.score.negative {
		color: var(--danger);
	}

	code {
		font-size: 0.75rem;
	}
	.empty {
		color: var(--text-muted);
		font-size: 0.875rem;
	}
</style>
