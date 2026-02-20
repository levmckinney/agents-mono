<script lang="ts">
	import Modal from './Modal.svelte';
	import * as api from '$lib/api';
	import type { InfinigramDocument } from '$lib/types';

	let {
		open = false,
		onclose,
		onuse
	}: {
		open: boolean;
		onclose: () => void;
		onuse: (pairs: Array<{ prompt: string; completion: string }>) => void;
	} = $props();

	let searchQuery = $state('');
	let maxResults = $state(5);
	let searching = $state(false);
	let documents = $state<InfinigramDocument[]>([]);
	let totalCount = $state(0);
	let searchError = $state('');

	// Multi-select for batch add
	let checkedDocs = $state<Set<number>>(new Set());
	let spanLength = $state(256);
	// Single-doc preview
	let previewDoc = $state<InfinigramDocument | null>(null);
	let extractedPrompt = $state('');
	let extractedCompletion = $state('');

	async function doSearch() {
		if (!searchQuery.trim()) return;
		searching = true;
		searchError = '';
		documents = [];
		checkedDocs = new Set();
		previewDoc = null;
		try {
			const result = await api.searchPretraining(searchQuery, maxResults);
			documents = result.documents;
			totalCount = result.count;
		} catch (e) {
			searchError = String(e);
		} finally {
			searching = false;
		}
	}

	function toggleDoc(index: number) {
		const next = new Set(checkedDocs);
		if (next.has(index)) {
			next.delete(index);
		} else {
			next.add(index);
		}
		checkedDocs = next;
	}

	function toggleSelectAll() {
		if (checkedDocs.size === documents.length) {
			checkedDocs = new Set();
		} else {
			checkedDocs = new Set(documents.map((_, i) => i));
		}
	}

	function getMatchOffsets(doc: InfinigramDocument): { start: number; end: number } {
		let offset = 0;
		for (const span of doc.spans) {
			if (span.is_match) {
				return { start: offset, end: offset + span.text.length };
			}
			offset += span.text.length;
		}
		return { start: 0, end: 0 };
	}

	/** Client-side span extraction — mirrors backend/span.py logic. */
	function extractSpanLocal(
		text: string,
		matchStart: number,
		matchEnd: number,
		length: number
	): { prompt: string; completion: string } | null {
		if (matchStart === matchEnd) return null;
		let completion = text.slice(matchStart, matchEnd);
		if (completion && !completion.startsWith(' ')) {
			completion = ' ' + completion;
		}
		let rawStart = Math.max(0, matchStart - length);
		// Snap to sentence boundary (look for '. ', '! ', '? ')
		if (rawStart > 0) {
			const slice = text.slice(rawStart);
			const m = slice.match(/(?<=[.!?])\s+/);
			if (m && m.index !== undefined && m.index < 60) {
				const snapped = rawStart + m.index + m[0].length;
				if (snapped < matchStart) {
					rawStart = snapped;
				}
			}
		}
		const prompt = text.slice(rawStart, matchStart);
		return { prompt, completion };
	}

	function addCheckedPairs() {
		if (checkedDocs.size === 0) return;
		const pairs: Array<{ prompt: string; completion: string }> = [];
		for (const idx of checkedDocs) {
			const doc = documents[idx];
			const { start, end } = getMatchOffsets(doc);
			const result = extractSpanLocal(doc.full_text, start, end, spanLength);
			if (result) pairs.push(result);
		}
		if (pairs.length > 0) {
			onuse(pairs);
			onclose();
		}
	}

	// Single-doc preview
	function previewDocument(doc: InfinigramDocument) {
		previewDoc = doc;
		extractPreview();
	}

	function extractPreview() {
		if (!previewDoc) return;
		const { start, end } = getMatchOffsets(previewDoc);
		if (start === end) return;
		const result = extractSpanLocal(previewDoc.full_text, start, end, spanLength);
		if (result) {
			extractedPrompt = result.prompt;
			extractedCompletion = result.completion;
		}
	}

	function usePreviewPair() {
		onuse([{ prompt: extractedPrompt, completion: extractedCompletion }]);
		onclose();
	}

	let spanTimer: ReturnType<typeof setTimeout>;
	function onSpanChange() {
		clearTimeout(spanTimer);
		spanTimer = setTimeout(() => {
			if (previewDoc) extractPreview();
		}, 300);
	}
</script>

<Modal title="Search Pretraining Data" {open} {onclose}>
	<div class="search-section">
		<div class="search-bar">
			<input
				bind:value={searchQuery}
				placeholder="Search completion text..."
				onkeydown={(e) => e.key === 'Enter' && doSearch()}
			/>
			<label class="max-results-label">
				Max
				<input
					type="number"
					min="1"
					max="100"
					bind:value={maxResults}
					class="max-results"
				/>
			</label>
			<button class="primary" onclick={doSearch} disabled={searching || !searchQuery.trim()}>
				{searching ? 'Searching...' : 'Search'}
			</button>
		</div>

		{#if searchError}
			<div class="error">{searchError}</div>
		{/if}

		{#if totalCount > 0}
			<div class="result-count">{totalCount.toLocaleString()} corpus matches</div>
		{/if}

		{#if documents.length > 0 && !previewDoc}
			<div class="list-toolbar">
				<label class="span-control">
					Prompt length: {spanLength} chars
					<input
						type="range"
						min="64"
						max="1024"
						step="32"
						bind:value={spanLength}
						oninput={onSpanChange}
					/>
				</label>
				<button class="select-all" onclick={toggleSelectAll}>
					{checkedDocs.size === documents.length ? 'Deselect all' : 'Select all'}
				</button>
			</div>

			<div class="doc-list">
				{#each documents as doc, i}
					<div class="doc-item" class:checked={checkedDocs.has(i)}>
						<label class="doc-check">
							<input
								type="checkbox"
								checked={checkedDocs.has(i)}
								onchange={() => toggleDoc(i)}
							/>
						</label>
						<button class="doc-content" onclick={() => previewDocument(doc)}>
							<div class="doc-header">Document {i + 1} (len: {doc.doc_len})</div>
							<div class="doc-preview">
								{#each doc.spans as span}
									{#if span.is_match}<mark>{span.text}</mark>{:else}{span.text}{/if}
								{/each}
							</div>
						</button>
					</div>
				{/each}
			</div>

			{#if checkedDocs.size > 0}
				<button class="primary" onclick={addCheckedPairs}>
					{`Add ${checkedDocs.size} pair${checkedDocs.size > 1 ? 's' : ''}`}
				</button>
			{/if}
		{/if}

		{#if previewDoc}
			<div class="selected-doc">
				<button class="back-btn" onclick={() => { previewDoc = null; }}>
					Back to results
				</button>

				<div class="doc-text">
					{#each previewDoc.spans as span}
						{#if span.is_match}<mark>{span.text}</mark>{:else}{span.text}{/if}
					{/each}
				</div>

				<div class="span-control">
					<label>
						Prompt length: {spanLength} chars
						<input
							type="range"
							min="64"
							max="1024"
							step="32"
							bind:value={spanLength}
							oninput={onSpanChange}
						/>
					</label>
				</div>

				{#if extractedPrompt || extractedCompletion}
					<div class="extracted">
						<div class="field-group">
							<label>Prompt</label>
							<div class="preview-text">{extractedPrompt}</div>
						</div>
						<div class="field-group">
							<label>Completion</label>
							<div class="preview-text completion">{extractedCompletion}</div>
						</div>
						<button class="primary" onclick={usePreviewPair}>Add this pair</button>
					</div>
				{/if}
			</div>
		{/if}
	</div>
</Modal>

<style>
	.search-section {
		display: flex;
		flex-direction: column;
		gap: 0.75rem;
	}
	.search-bar {
		display: flex;
		gap: 0.5rem;
		align-items: center;
	}
	.search-bar > input:first-child {
		flex: 1;
	}
	.max-results-label {
		display: flex;
		align-items: center;
		gap: 0.25rem;
		font-size: 0.8rem;
		color: var(--text-muted, #888);
		white-space: nowrap;
	}
	.max-results {
		width: 50px;
	}
	.result-count {
		font-size: 0.85rem;
		color: var(--text-muted, #888);
	}
	.doc-list {
		display: flex;
		flex-direction: column;
		gap: 0.5rem;
		max-height: 300px;
		overflow-y: auto;
	}
	.doc-item {
		display: flex;
		align-items: flex-start;
		gap: 0.5rem;
		padding: 0.5rem;
		border: 1px solid var(--border, #ddd);
		border-radius: 6px;
	}
	.doc-item.checked {
		border-color: var(--primary, #1976d2);
		background: rgba(25, 118, 210, 0.04);
	}
	.doc-check {
		padding-top: 0.25rem;
		cursor: pointer;
	}
	.doc-content {
		flex: 1;
		text-align: left;
		background: none;
		border: none;
		cursor: pointer;
		padding: 0.25rem 0;
	}
	.doc-content:hover {
		background: var(--surface, #f5f5f5);
		border-radius: 4px;
	}
	.doc-header {
		font-size: 0.8rem;
		color: var(--text-muted, #888);
		margin-bottom: 0.25rem;
	}
	.doc-preview {
		font-size: 0.875rem;
		line-height: 1.4;
		max-height: 4em;
		overflow: hidden;
		text-overflow: ellipsis;
	}
	mark {
		background: #fff3cd;
		padding: 0 2px;
		border-radius: 2px;
	}
	.selected-doc {
		display: flex;
		flex-direction: column;
		gap: 0.75rem;
	}
	.back-btn {
		align-self: flex-start;
		font-size: 0.85rem;
	}
	.doc-text {
		font-size: 0.875rem;
		line-height: 1.5;
		max-height: 150px;
		overflow-y: auto;
		padding: 0.75rem;
		border: 1px solid var(--border, #ddd);
		border-radius: 4px;
	}
	.list-toolbar {
		display: flex;
		align-items: center;
		justify-content: space-between;
		gap: 0.5rem;
	}
	.select-all {
		font-size: 0.8rem;
		white-space: nowrap;
	}
	.span-control {
		display: flex;
		align-items: center;
		gap: 0.5rem;
		font-size: 0.85rem;
	}
	.span-control input[type='range'] {
		width: 200px;
	}

	.extracted {
		display: flex;
		flex-direction: column;
		gap: 0.5rem;
	}
	.field-group label {
		font-size: 0.8rem;
		color: var(--text-muted, #888);
	}
	.preview-text {
		font-size: 0.875rem;
		padding: 0.5rem;
		border: 1px solid var(--border, #ddd);
		border-radius: 4px;
		max-height: 80px;
		overflow-y: auto;
		white-space: pre-wrap;
	}
	.preview-text.completion {
		background: #f0f9ff;
	}
	.error {
		background: rgba(239, 83, 80, 0.1);
		border: 1px solid var(--danger, #ef5350);
		padding: 0.5rem 0.75rem;
		border-radius: 4px;
		font-size: 0.875rem;
	}
</style>
