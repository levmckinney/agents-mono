<script lang="ts">
	import Modal from './Modal.svelte';
	import * as api from '$lib/api';
	import type { Pair, ProbeSetDetail } from '$lib/types';

	let {
		open = false,
		probeSetId = '',
		onclose,
		onimport
	}: {
		open: boolean;
		probeSetId: string;
		onclose: () => void;
		onimport: (detail: ProbeSetDetail) => void;
	} = $props();

	let file = $state<File | null>(null);
	let previewPairs = $state<Array<{ pair_id: string; prompt: string; completion: string; role: string }>>([]);
	let importing = $state(false);
	let importError = $state('');

	function onFileChange(e: Event) {
		const input = e.target as HTMLInputElement;
		file = input.files?.[0] ?? null;
		previewPairs = [];
		importError = '';
		if (file) {
			parsePreview(file);
		}
	}

	async function parsePreview(f: File) {
		const text = await f.text();
		try {
			if (f.name.endsWith('.json')) {
				const data = JSON.parse(text);
				if (!Array.isArray(data)) throw new Error('JSON must be an array');
				previewPairs = data.slice(0, 20).map((d: Record<string, string>) => ({
					pair_id: d.pair_id || '',
					prompt: d.prompt || '',
					completion: d.completion || '',
					role: d.role || 'both'
				}));
			} else {
				// CSV
				const lines = text.trim().split('\n');
				if (lines.length < 2) throw new Error('CSV must have a header and at least one row');
				const headers = lines[0].split(',').map((h) => h.trim());
				previewPairs = lines
					.slice(1, 21)
					.map((line) => {
						const vals = line.split(',').map((v) => v.trim());
						const obj: Record<string, string> = {};
						headers.forEach((h, i) => (obj[h] = vals[i] || ''));
						return {
							pair_id: obj.pair_id || '',
							prompt: obj.prompt || '',
							completion: obj.completion || '',
							role: obj.role || 'both'
						};
					});
			}
		} catch (e) {
			importError = `Preview error: ${e}`;
		}
	}

	async function doImport() {
		if (!file || !probeSetId) return;
		importing = true;
		importError = '';
		try {
			const detail = await api.importPairs(probeSetId, file);
			onimport(detail);
			onclose();
		} catch (e) {
			importError = String(e);
		} finally {
			importing = false;
		}
	}
</script>

<Modal title="Import Pairs" {open} {onclose}>
	<div class="import-section">
		<div class="field-group">
			<label for="file-input">Select CSV or JSON file</label>
			<input
				id="file-input"
				type="file"
				accept=".csv,.json"
				onchange={onFileChange}
			/>
		</div>

		{#if importError}
			<div class="error">{importError}</div>
		{/if}

		{#if previewPairs.length > 0}
			<div class="preview">
				<h4>Preview ({previewPairs.length} pairs{file && previewPairs.length >= 20 ? ', showing first 20' : ''})</h4>
				<div class="table-wrap">
					<table>
						<thead>
							<tr>
								<th>ID</th>
								<th>Prompt</th>
								<th>Completion</th>
								<th>Role</th>
							</tr>
						</thead>
						<tbody>
							{#each previewPairs as p}
								<tr>
									<td>{p.pair_id}</td>
									<td class="truncate">{p.prompt}</td>
									<td class="truncate">{p.completion}</td>
									<td>{p.role}</td>
								</tr>
							{/each}
						</tbody>
					</table>
				</div>
			</div>

			<button class="primary" onclick={doImport} disabled={importing}>
				{importing ? 'Importing...' : 'Import'}
			</button>
		{/if}
	</div>
</Modal>

<style>
	.import-section {
		display: flex;
		flex-direction: column;
		gap: 0.75rem;
	}
	.field-group {
		display: flex;
		flex-direction: column;
		gap: 0.25rem;
	}
	.field-group label {
		font-size: 0.8rem;
		color: var(--text-muted, #888);
	}
	.preview h4 {
		font-size: 0.9rem;
		margin-bottom: 0.5rem;
	}
	.table-wrap {
		max-height: 300px;
		overflow: auto;
	}
	table {
		width: 100%;
		border-collapse: collapse;
		font-size: 0.825rem;
	}
	th,
	td {
		border: 1px solid var(--border, #ddd);
		padding: 0.35rem 0.5rem;
		text-align: left;
	}
	th {
		background: var(--surface, #f5f5f5);
		font-weight: 600;
	}
	.truncate {
		max-width: 180px;
		overflow: hidden;
		text-overflow: ellipsis;
		white-space: nowrap;
	}
	.error {
		background: rgba(239, 83, 80, 0.1);
		border: 1px solid var(--danger, #ef5350);
		padding: 0.5rem 0.75rem;
		border-radius: 4px;
		font-size: 0.875rem;
	}
</style>
