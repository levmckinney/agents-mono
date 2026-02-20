<script lang="ts">
	import { onMount } from 'svelte';
	import * as api from '$lib/api';
	import type { Pair, PairRole, ProbeSetSummary, ProbeSetDetail } from '$lib/types';
	import SearchPretrainingModal from '$lib/components/SearchPretrainingModal.svelte';
	import GenerateContextModal from '$lib/components/GenerateContextModal.svelte';
	import BulkImportModal from '$lib/components/BulkImportModal.svelte';

	let probeSets = $state<ProbeSetSummary[]>([]);
	let selectedId = $state<string | null>(null);
	let name = $state('');
	let pairs = $state<Pair[]>([]);
	let saving = $state(false);
	let error = $state('');

	// Multi-select state
	let selectedPairIds = $state<Set<string>>(new Set());

	// Modal state
	let searchModalOpen = $state(false);
	let genModalOpen = $state(false);
	let importModalOpen = $state(false);

	onMount(loadProbeSets);

	async function loadProbeSets() {
		try {
			probeSets = await api.listProbeSets();
		} catch (e) {
			error = String(e);
		}
	}

	async function selectProbeSet(id: string) {
		try {
			const detail = await api.getProbeSet(id);
			selectedId = detail.id;
			name = detail.name;
			pairs = detail.pairs;
			selectedPairIds = new Set();
			error = '';
		} catch (e) {
			error = String(e);
		}
	}

	function newProbeSet() {
		selectedId = null;
		name = '';
		pairs = [];
		selectedPairIds = new Set();
		error = '';
	}

	function addPair() {
		pairs = [
			...pairs,
			{
				pair_id: `p${Date.now()}`,
				prompt: '',
				completion: '',
				role: 'both' as PairRole,
				metadata: {}
			}
		];
	}

	function removePair(index: number) {
		const removed = pairs[index];
		pairs = pairs.filter((_, i) => i !== index);
		if (removed) {
			const next = new Set(selectedPairIds);
			next.delete(removed.pair_id);
			selectedPairIds = next;
		}
	}

	function duplicatePair(index: number) {
		const original = pairs[index];
		const dup: Pair = {
			...original,
			pair_id: `${original.pair_id}_dup${Date.now()}`,
			metadata: { ...original.metadata }
		};
		const updated = [...pairs];
		updated.splice(index + 1, 0, dup);
		pairs = updated;
	}

	function togglePairSelection(pairId: string) {
		const next = new Set(selectedPairIds);
		if (next.has(pairId)) {
			next.delete(pairId);
		} else {
			next.add(pairId);
		}
		selectedPairIds = next;
	}

	function toggleSelectAll() {
		if (selectedPairIds.size === pairs.length) {
			selectedPairIds = new Set();
		} else {
			selectedPairIds = new Set(pairs.map((p) => p.pair_id));
		}
	}

	async function bulkSetRole(role: PairRole) {
		if (!selectedId || selectedPairIds.size === 0) return;
		error = '';
		try {
			const detail = await api.bulkSetRole(selectedId, [...selectedPairIds], role);
			pairs = detail.pairs;
			selectedPairIds = new Set();
			await loadProbeSets();
		} catch (e) {
			error = String(e);
		}
	}

	async function save() {
		saving = true;
		error = '';
		try {
			if (selectedId) {
				await api.updateProbeSet(selectedId, { name, pairs });
			} else {
				const created = await api.createProbeSet(name, pairs);
				selectedId = created.id;
			}
			await loadProbeSets();
		} catch (e) {
			error = String(e);
		} finally {
			saving = false;
		}
	}

	async function remove() {
		if (!selectedId) return;
		try {
			await api.deleteProbeSet(selectedId);
			newProbeSet();
			await loadProbeSets();
		} catch (e) {
			error = String(e);
		}
	}

	// Tool callbacks — append new pairs
	function onSearchUse(newPairs: Array<{ prompt: string; completion: string }>) {
		pairs = [
			...pairs,
			...newPairs.map((p, i) => ({
				pair_id: `p${Date.now()}_${i}`,
				prompt: p.prompt,
				completion: p.completion,
				role: 'both' as PairRole,
				metadata: {}
			}))
		];
	}

	function onGenUse(prompt: string, completion: string) {
		pairs = [
			...pairs,
			{
				pair_id: `p${Date.now()}`,
				prompt,
				completion,
				role: 'both' as PairRole,
				metadata: {}
			}
		];
	}

	function onImportDone(detail: ProbeSetDetail) {
		pairs = detail.pairs;
		loadProbeSets();
	}
</script>

<div class="create-page">
	<aside>
		<div class="sidebar-header">
			<h2>Probe Sets</h2>
			<button onclick={newProbeSet}>+ New</button>
		</div>
		<ul>
			{#each probeSets as ps}
				<li>
					<button
						class="probe-set-item"
						class:active={ps.id === selectedId}
						onclick={() => selectProbeSet(ps.id)}
					>
						<span class="ps-name">{ps.name}</span>
						<span class="ps-count">{ps.pair_count} pairs</span>
					</button>
				</li>
			{/each}
			{#if probeSets.length === 0}
				<li class="empty">No probe sets yet</li>
			{/if}
		</ul>
	</aside>

	<section class="editor">
		{#if error}
			<div class="error">{error}</div>
		{/if}

		<div class="field">
			<label for="name">Name</label>
			<input id="name" bind:value={name} placeholder="My probe set" />
		</div>

		<div class="pairs-header">
			<h3>Pairs ({pairs.length})</h3>
			<div class="pairs-toolbar">
				{#if selectedPairIds.size > 0}
					<span class="selection-count">{selectedPairIds.size} selected</span>
					<select onchange={(e) => { const v = (e.target as HTMLSelectElement).value; if (v) bulkSetRole(v as PairRole); (e.target as HTMLSelectElement).value = ''; }}>
						<option value="">Set role...</option>
						<option value="both">Both</option>
						<option value="train">Train</option>
						<option value="query">Query</option>
					</select>
				{/if}
				<button onclick={() => { searchModalOpen = true; }}>Search pretraining</button>
				<button onclick={() => { genModalOpen = true; }}>Generate context</button>
				{#if selectedId}
					<button onclick={() => { importModalOpen = true; }}>Import</button>
				{/if}
				<button onclick={addPair}>+ Add Pair</button>
			</div>
		</div>

		{#if pairs.length > 0}
			<div class="select-all">
				<label>
					<input
						type="checkbox"
						checked={selectedPairIds.size === pairs.length && pairs.length > 0}
						onchange={toggleSelectAll}
					/>
					Select all
				</label>
			</div>
		{/if}

		<div class="pairs-list">
			{#each pairs as pair, i}
				<div class="pair-card" class:selected={selectedPairIds.has(pair.pair_id)}>
					<div class="pair-top">
						<input
							type="checkbox"
							checked={selectedPairIds.has(pair.pair_id)}
							onchange={() => togglePairSelection(pair.pair_id)}
						/>
						<input
							class="pair-id"
							bind:value={pair.pair_id}
							placeholder="ID"
						/>
						<select bind:value={pair.role}>
							<option value="both">Both</option>
							<option value="train">Train</option>
							<option value="query">Query</option>
						</select>
						<button onclick={() => duplicatePair(i)}>Dup</button>
						<button class="danger" onclick={() => removePair(i)}>Remove</button>
					</div>
					<textarea
						bind:value={pair.prompt}
						placeholder="Prompt"
						rows="2"
					></textarea>
					<textarea
						bind:value={pair.completion}
						placeholder="Completion"
						rows="2"
					></textarea>
				</div>
			{/each}
		</div>

		<div class="actions">
			<button class="primary" onclick={save} disabled={saving || !name.trim()}>
				{saving ? 'Saving...' : selectedId ? 'Update' : 'Create'}
			</button>
			{#if selectedId}
				<button class="danger" onclick={remove}>Delete</button>
			{/if}
		</div>
	</section>
</div>

<!-- Modals -->
<SearchPretrainingModal
	open={searchModalOpen}
	onclose={() => { searchModalOpen = false; }}
	onuse={onSearchUse}
/>

<GenerateContextModal
	open={genModalOpen}
	onclose={() => { genModalOpen = false; }}
	onuse={onGenUse}
/>

<BulkImportModal
	open={importModalOpen}
	probeSetId={selectedId ?? ''}
	onclose={() => { importModalOpen = false; }}
	onimport={onImportDone}
/>

<style>
	.create-page {
		display: grid;
		grid-template-columns: 240px 1fr;
		gap: 1.5rem;
		min-height: 70vh;
	}
	aside {
		border-right: 1px solid var(--border);
		padding-right: 1rem;
	}
	.sidebar-header {
		display: flex;
		justify-content: space-between;
		align-items: center;
		margin-bottom: 0.75rem;
	}
	h2 { font-size: 1rem; }
	ul { list-style: none; }
	.probe-set-item {
		width: 100%;
		text-align: left;
		display: flex;
		justify-content: space-between;
		padding: 0.5rem;
		margin-bottom: 0.25rem;
		border: none;
	}
	.probe-set-item.active {
		background: var(--surface);
		border-color: var(--primary);
	}
	.ps-count { color: var(--text-muted); font-size: 0.8rem; }
	.empty { color: var(--text-muted); padding: 0.5rem; font-size: 0.875rem; }
	.editor { display: flex; flex-direction: column; gap: 1rem; }
	.field { display: flex; flex-direction: column; gap: 0.25rem; }
	.field label { font-size: 0.8rem; color: var(--text-muted); }
	.field input { max-width: 400px; }
	.pairs-header {
		display: flex;
		justify-content: space-between;
		align-items: center;
	}
	.pairs-toolbar {
		display: flex;
		gap: 0.5rem;
		align-items: center;
		flex-wrap: wrap;
	}
	.selection-count {
		font-size: 0.8rem;
		color: var(--text-muted);
	}
	h3 { font-size: 0.95rem; }
	.select-all {
		font-size: 0.85rem;
	}
	.select-all label {
		display: flex;
		align-items: center;
		gap: 0.35rem;
		cursor: pointer;
	}
	.pairs-list { display: flex; flex-direction: column; gap: 0.75rem; }
	.pair-card {
		border: 1px solid var(--border);
		border-radius: 6px;
		padding: 0.75rem;
		display: flex;
		flex-direction: column;
		gap: 0.5rem;
	}
	.pair-card.selected {
		border-color: var(--primary, #1976d2);
		background: rgba(25, 118, 210, 0.04);
	}
	.pair-top {
		display: flex;
		gap: 0.5rem;
		align-items: center;
	}
	.pair-id { max-width: 160px; }
	textarea { resize: vertical; min-height: 48px; }
	.actions { display: flex; gap: 0.5rem; margin-top: 0.5rem; }
	.error {
		background: rgba(239, 83, 80, 0.1);
		border: 1px solid var(--danger);
		padding: 0.5rem 0.75rem;
		border-radius: 4px;
		font-size: 0.875rem;
	}
</style>
