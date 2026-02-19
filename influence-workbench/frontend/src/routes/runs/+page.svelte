<script lang="ts">
	import { onMount } from 'svelte';
	import * as api from '$lib/api';
	import type { ProbeSetSummary, RunSummary } from '$lib/types';

	let probeSets = $state<ProbeSetSummary[]>([]);
	let runs = $state<RunSummary[]>([]);
	let selectedRunId = $state<string | null>(null);
	let logLines = $state<string[]>([]);
	let ws = $state<WebSocket | null>(null);
	let launching = $state<string | null>(null);
	let error = $state('');

	onMount(async () => {
		await Promise.all([loadProbeSets(), loadRuns()]);
	});

	async function loadProbeSets() {
		try {
			probeSets = await api.listProbeSets();
		} catch (e) {
			error = String(e);
		}
	}

	async function loadRuns() {
		try {
			runs = await api.listRuns();
		} catch (e) {
			error = String(e);
		}
	}

	async function launchRun(probeSetId: string) {
		launching = probeSetId;
		error = '';
		try {
			const run = await api.createRun(probeSetId);
			await loadRuns();
			viewLogs(run.id);
		} catch (e) {
			error = String(e);
		} finally {
			launching = null;
		}
	}

	function viewLogs(runId: string) {
		// Close existing connection
		if (ws) {
			ws.close();
			ws = null;
		}
		selectedRunId = runId;
		logLines = [];

		const socket = api.connectLogs(runId);
		socket.onmessage = (ev) => {
			logLines = [...logLines, ev.data];
		};
		socket.onclose = () => {
			loadRuns(); // Refresh status
		};
		ws = socket;
	}

	function statusColor(status: string): string {
		switch (status) {
			case 'completed': return 'var(--success)';
			case 'failed': return 'var(--danger)';
			case 'running': return 'var(--warning)';
			default: return 'var(--text-muted)';
		}
	}

	function probeSetName(id: string): string {
		return probeSets.find((ps) => ps.id === id)?.name ?? id;
	}
</script>

<div class="runs-page">
	{#if error}
		<div class="error">{error}</div>
	{/if}

	<div class="content">
		<section class="panel">
			<h2>Probe Sets</h2>
			<div class="probe-list">
				{#each probeSets as ps}
					<div class="probe-item">
						<span>{ps.name} ({ps.pair_count} pairs)</span>
						<button
							class="primary"
							onclick={() => launchRun(ps.id)}
							disabled={launching === ps.id}
						>
							{launching === ps.id ? 'Launching...' : 'Launch Run'}
						</button>
					</div>
				{/each}
				{#if probeSets.length === 0}
					<p class="empty">No probe sets. Create one first.</p>
				{/if}
			</div>

			<h2>Runs</h2>
			<table>
				<thead>
					<tr>
						<th>ID</th>
						<th>Probe Set</th>
						<th>Status</th>
						<th>Created</th>
						<th></th>
					</tr>
				</thead>
				<tbody>
					{#each runs as run}
						<tr>
							<td><code>{run.id}</code></td>
							<td>{probeSetName(run.probe_set_id)}</td>
							<td>
								<span class="status-badge" style="color: {statusColor(run.status)}">
									{run.status}
								</span>
							</td>
							<td>{new Date(run.created_at).toLocaleString()}</td>
							<td>
								<button onclick={() => viewLogs(run.id)}>Logs</button>
							</td>
						</tr>
					{/each}
					{#if runs.length === 0}
						<tr><td colspan="5" class="empty">No runs yet</td></tr>
					{/if}
				</tbody>
			</table>
		</section>

		<section class="log-panel">
			<h2>
				{#if selectedRunId}
					Logs: {selectedRunId}
				{:else}
					Logs
				{/if}
			</h2>
			<div class="log-viewer">
				{#if logLines.length === 0 && !selectedRunId}
					<p class="empty">Select a run to view logs</p>
				{:else if logLines.length === 0}
					<p class="empty">Waiting for output...</p>
				{:else}
					{#each logLines as line}
						<pre>{line}</pre>
					{/each}
				{/if}
			</div>
		</section>
	</div>
</div>

<style>
	.runs-page { display: flex; flex-direction: column; gap: 1rem; }
	.content { display: grid; grid-template-columns: 1fr 1fr; gap: 1.5rem; min-height: 70vh; }
	.panel { display: flex; flex-direction: column; gap: 1rem; }
	h2 { font-size: 1rem; margin-top: 0.5rem; }
	.probe-list { display: flex; flex-direction: column; gap: 0.5rem; }
	.probe-item {
		display: flex;
		justify-content: space-between;
		align-items: center;
		padding: 0.5rem 0.75rem;
		border: 1px solid var(--border);
		border-radius: 4px;
	}
	.status-badge { font-weight: 600; font-size: 0.85rem; }
	code { font-size: 0.8rem; }
	.log-panel {
		border-left: 1px solid var(--border);
		padding-left: 1rem;
		display: flex;
		flex-direction: column;
		gap: 0.5rem;
	}
	.log-viewer {
		background: #0d1117;
		border: 1px solid var(--border);
		border-radius: 6px;
		padding: 0.75rem;
		font-family: monospace;
		font-size: 0.8rem;
		overflow-y: auto;
		max-height: 60vh;
		flex: 1;
	}
	.log-viewer pre { margin: 0; white-space: pre-wrap; word-break: break-all; }
	.empty { color: var(--text-muted); font-size: 0.875rem; }
	.error {
		background: rgba(239, 83, 80, 0.1);
		border: 1px solid var(--danger);
		padding: 0.5rem 0.75rem;
		border-radius: 4px;
		font-size: 0.875rem;
	}
</style>
