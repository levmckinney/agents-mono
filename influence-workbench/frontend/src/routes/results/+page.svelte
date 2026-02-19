<script lang="ts">
	import { onMount } from 'svelte';
	import * as api from '$lib/api';
	import type { RunSummary, RunResults, InfluenceRow } from '$lib/types';

	let runs = $state<RunSummary[]>([]);
	let selectedRunId = $state<string | null>(null);
	let results = $state<RunResults | null>(null);
	let selectedQueryId = $state<string | null>(null);
	let normalize = $state(false);
	let loading = $state(false);
	let error = $state('');

	onMount(loadRuns);

	async function loadRuns() {
		try {
			const allRuns = await api.listRuns();
			runs = allRuns.filter((r) => r.status === 'completed');
		} catch (e) {
			error = String(e);
		}
	}

	async function selectRun(runId: string) {
		selectedRunId = runId;
		selectedQueryId = null;
		results = null;
		loading = true;
		error = '';
		try {
			results = await api.getRunResults(runId);
			if (results.query_results.length > 0) {
				selectedQueryId = results.query_results[0].query_id as string;
			}
		} catch (e) {
			error = String(e);
		} finally {
			loading = false;
		}
	}

	function influencesForQuery(queryId: string): InfluenceRow[] {
		if (!results) return [];
		const rows = results.influences.filter((r) => r.query_id === queryId);
		rows.sort((a, b) => b.influence_score - a.influence_score);
		return rows;
	}

	function normalizedScore(score: number, allScores: InfluenceRow[]): number {
		if (!normalize || allScores.length === 0) return score;
		const max = Math.max(...allScores.map((r) => Math.abs(r.influence_score)));
		return max === 0 ? 0 : score / max;
	}

	function trainInfo(trainId: string): Record<string, unknown> | null {
		return results?.train_results.find((t) => t.train_id === trainId) ?? null;
	}

	function formatScore(n: number): string {
		if (normalize) return n.toFixed(4);
		return n.toLocaleString(undefined, { maximumFractionDigits: 2 });
	}
</script>

<div class="results-page">
	{#if error}
		<div class="error">{error}</div>
	{/if}

	<div class="toolbar">
		<label for="run-select">Run:</label>
		<select
			id="run-select"
			value={selectedRunId ?? ''}
			onchange={(e) => {
				const val = (e.target as HTMLSelectElement).value;
				if (val) selectRun(val);
			}}
		>
			<option value="">Select a completed run...</option>
			{#each runs as run}
				<option value={run.id}>{run.id} ({new Date(run.created_at).toLocaleDateString()})</option>
			{/each}
		</select>

		<label class="toggle">
			<input type="checkbox" bind:checked={normalize} />
			Normalize scores
		</label>
	</div>

	{#if loading}
		<p>Loading results...</p>
	{:else if results}
		<div class="results-layout">
			<aside>
				<h3>Query Pairs</h3>
				<ul>
					{#each results.query_results as qr}
						<li>
							<button
								class="query-item"
								class:active={selectedQueryId === qr.query_id}
								onclick={() => (selectedQueryId = qr.query_id as string)}
							>
								<span class="qid">{qr.query_id}</span>
								<span class="loss">loss: {Number(qr.loss).toFixed(2)}</span>
							</button>
						</li>
					{/each}
				</ul>
			</aside>

			<section class="influence-table">
				{#if selectedQueryId}
					{@const influences = influencesForQuery(selectedQueryId)}
					<h3>Influences on {selectedQueryId}</h3>
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
									<td class="score" class:positive={inf.influence_score > 0} class:negative={inf.influence_score < 0}>
										{formatScore(normalizedScore(inf.influence_score, influences))}
									</td>
								</tr>
							{/each}
							{#if influences.length === 0}
								<tr><td colspan="5" class="empty">No influences found</td></tr>
							{/if}
						</tbody>
					</table>
				{:else}
					<p class="empty">Select a query pair to view influences</p>
				{/if}
			</section>
		</div>
	{:else if !selectedRunId}
		<p class="empty">Select a completed run to view results</p>
	{/if}
</div>

<style>
	.results-page { display: flex; flex-direction: column; gap: 1rem; }
	.toolbar {
		display: flex;
		align-items: center;
		gap: 1rem;
		flex-wrap: wrap;
	}
	.toolbar label { font-size: 0.875rem; color: var(--text-muted); }
	.toolbar select { min-width: 280px; }
	.toggle {
		display: flex;
		align-items: center;
		gap: 0.4rem;
		cursor: pointer;
		margin-left: auto;
	}
	.toggle input { cursor: pointer; }
	.results-layout {
		display: grid;
		grid-template-columns: 240px 1fr;
		gap: 1.5rem;
		min-height: 60vh;
	}
	aside {
		border-right: 1px solid var(--border);
		padding-right: 1rem;
	}
	aside ul { list-style: none; }
	h3 { font-size: 0.95rem; margin-bottom: 0.5rem; }
	.query-item {
		width: 100%;
		text-align: left;
		display: flex;
		justify-content: space-between;
		padding: 0.5rem;
		margin-bottom: 0.25rem;
		border: none;
	}
	.query-item.active {
		background: var(--surface);
		border-color: var(--primary);
	}
	.qid { font-weight: 500; }
	.loss { color: var(--text-muted); font-size: 0.8rem; }
	.influence-table { overflow-x: auto; }
	.text-cell {
		max-width: 240px;
		overflow: hidden;
		text-overflow: ellipsis;
		white-space: nowrap;
	}
	.score { font-family: monospace; text-align: right; }
	.score.positive { color: var(--success); }
	.score.negative { color: var(--danger); }
	code { font-size: 0.8rem; }
	.empty { color: var(--text-muted); font-size: 0.875rem; }
	.error {
		background: rgba(239, 83, 80, 0.1);
		border: 1px solid var(--danger);
		padding: 0.5rem 0.75rem;
		border-radius: 4px;
		font-size: 0.875rem;
	}
</style>
