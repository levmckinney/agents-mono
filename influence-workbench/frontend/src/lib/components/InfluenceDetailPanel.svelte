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
</script>

<div class="detail-panel">
	{#if queryInfo}
		<div class="query-info">
			<div class="label">Query: <code>{queryId}</code></div>
			{#if queryInfo.loss != null}
				<div class="loss">loss: {Number(queryInfo.loss).toFixed(2)}</div>
			{/if}
			<div class="text-preview">
				<strong>Prompt:</strong>
				<span class="truncated">{String(queryInfo.prompt ?? '').slice(0, 200)}</span>
			</div>
			<div class="text-preview">
				<strong>Completion:</strong>
				<span class="truncated">{String(queryInfo.completion ?? '')}</span>
			</div>
		</div>
	{/if}

	<table>
		<thead>
			<tr>
				<th>Rank</th>
				<th>Train ID</th>
				<th>Prompt</th>
				<th>Completion</th>
				<th>Score</th>
			</tr>
		</thead>
		<tbody>
			{#each influences as inf, i}
				{@const train = trainInfo(inf.train_id)}
				<tr>
					<td>{i + 1}</td>
					<td><code>{inf.train_id}</code></td>
					<td class="text-cell">{train?.prompt ?? '—'}</td>
					<td class="text-cell">{train?.completion ?? '—'}</td>
					<td
						class="score"
						class:positive={inf.influence_score > 0}
						class:negative={inf.influence_score < 0}
					>
						{formatScore(normalizedScore(inf.influence_score))}
					</td>
				</tr>
			{/each}
			{#if influences.length === 0}
				<tr><td colspan="5" class="empty">No influences found</td></tr>
			{/if}
		</tbody>
	</table>
</div>

<style>
	.detail-panel {
		overflow-x: auto;
	}
	.query-info {
		margin-bottom: 1rem;
		padding: 0.75rem;
		background: var(--surface);
		border-radius: 4px;
		border: 1px solid var(--border);
	}
	.label {
		font-weight: 600;
		margin-bottom: 0.25rem;
	}
	.loss {
		color: var(--text-muted);
		font-size: 0.8rem;
		margin-bottom: 0.5rem;
	}
	.text-preview {
		font-size: 0.8rem;
		margin-bottom: 0.25rem;
		line-height: 1.4;
	}
	.truncated {
		color: var(--text-muted);
	}
	code {
		font-size: 0.8rem;
	}
	.text-cell {
		max-width: 200px;
		overflow: hidden;
		text-overflow: ellipsis;
		white-space: nowrap;
	}
	.score {
		font-family: monospace;
		text-align: right;
	}
	.score.positive {
		color: var(--success);
	}
	.score.negative {
		color: var(--danger);
	}
	.empty {
		color: var(--text-muted);
		font-size: 0.875rem;
	}
</style>
