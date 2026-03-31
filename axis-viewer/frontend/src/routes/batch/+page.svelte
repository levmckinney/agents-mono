<script lang="ts">
	import { uploadBatch, batchProjectUrl } from '$lib/api';
	import type {
		BatchUploadResponse,
		BatchConversationResult,
		TurnSpan,
		ProjectionResponse
	} from '$lib/types';
	import TokenHeatmap from '$lib/components/TokenHeatmap.svelte';
	import TrajectoryChart from '$lib/components/TrajectoryChart.svelte';
	import ComparisonChart from '$lib/components/ComparisonChart.svelte';

	// --- Upload state ---
	let selectedFile = $state<File | null>(null);
	let lineCount = $state(0);
	let uploading = $state(false);
	let uploadError = $state('');
	let dragOver = $state(false);

	// --- Batch state ---
	let batchResponse = $state<BatchUploadResponse | null>(null);
	let results = $state<BatchConversationResult[]>([]);
	let projecting = $state(false);
	let projectProgress = $state({ index: 0, total: 0 });

	// --- UI state ---
	let expandedId = $state<string | null>(null);
	let selectedIds = $state<Set<string>>(new Set());
	let showComparison = $state(false);
	let sortColumn = $state<string>('name');
	let sortDir = $state<'asc' | 'desc'>('asc');

	// --- Filters ---
	let filterMode = $state<string>('all');
	let filterSearch = $state('');
	let filterDriftMin = $state(0);

	// Palette for comparison
	const COLORS = [
		'#4fc3f7',
		'#ffa726',
		'#66bb6a',
		'#ef5350',
		'#ab47bc',
		'#26c6da',
		'#ffca28',
		'#8d6e63',
		'#ec407a',
		'#7e57c2'
	];

	// --- File handling ---
	function countLines(text: string): number {
		return text
			.trim()
			.split('\n')
			.filter((l) => l.trim()).length;
	}

	async function handleFileSelect(file: File) {
		selectedFile = file;
		uploadError = '';
		const text = await file.text();
		lineCount = countLines(text);
	}

	function handleDrop(e: DragEvent) {
		e.preventDefault();
		dragOver = false;
		const file = e.dataTransfer?.files[0];
		if (file && (file.name.endsWith('.jsonl') || file.name.endsWith('.ndjson'))) {
			handleFileSelect(file);
		} else {
			uploadError = 'Please drop a .jsonl or .ndjson file.';
		}
	}

	function handleDragOver(e: DragEvent) {
		e.preventDefault();
		dragOver = true;
	}

	function handleDragLeave() {
		dragOver = false;
	}

	function handleInputChange(e: Event) {
		const target = e.target as HTMLInputElement;
		const file = target.files?.[0];
		if (file) handleFileSelect(file);
	}

	// --- Upload & project ---
	async function doUploadAndProject() {
		if (!selectedFile) return;
		uploading = true;
		uploadError = '';
		results = [];
		expandedId = null;
		selectedIds = new Set();
		showComparison = false;

		try {
			batchResponse = await uploadBatch(selectedFile);
		} catch (err) {
			uploadError = err instanceof Error ? err.message : String(err);
			uploading = false;
			return;
		}

		uploading = false;
		projecting = true;
		projectProgress = { index: 0, total: batchResponse.count };

		// Stream projections via SSE
		const url = batchProjectUrl(batchResponse.batch_id);
		const eventSource = new EventSource(url);

		eventSource.addEventListener('progress', (e: MessageEvent) => {
			const data = JSON.parse(e.data);
			projectProgress = { index: data.index, total: data.total };
		});

		eventSource.addEventListener('result', (e: MessageEvent) => {
			const result: BatchConversationResult = JSON.parse(e.data);
			results = [...results, result];
		});

		eventSource.addEventListener('done', () => {
			eventSource.close();
			projecting = false;
		});

		eventSource.onerror = () => {
			eventSource.close();
			projecting = false;
			if (results.length === 0) {
				uploadError = 'Connection lost during batch projection.';
			}
		};
	}

	// --- Sorting ---
	function toggleSort(column: string) {
		if (sortColumn === column) {
			sortDir = sortDir === 'asc' ? 'desc' : 'asc';
		} else {
			sortColumn = column;
			sortDir = 'asc';
		}
	}

	function sortIndicator(column: string): string {
		if (sortColumn !== column) return '';
		return sortDir === 'asc' ? ' ^' : ' v';
	}

	// --- Filtered & sorted results ---
	let filteredResults = $derived.by(() => {
		let items = results;

		// Mode filter
		if (filterMode !== 'all') {
			items = items.filter((r) => r.mode === filterMode);
		}

		// Search filter
		if (filterSearch.trim()) {
			const q = filterSearch.trim().toLowerCase();
			items = items.filter((r) => r.name.toLowerCase().includes(q));
		}

		// Drift filter
		if (filterDriftMin > 0) {
			items = items.filter(
				(r) => r.summary?.drift_amount != null && r.summary.drift_amount >= filterDriftMin
			);
		}

		// Sort
		const dir = sortDir === 'asc' ? 1 : -1;
		items = [...items].sort((a, b) => {
			switch (sortColumn) {
				case 'name':
					return dir * a.name.localeCompare(b.name);
				case 'mode':
					return dir * a.mode.localeCompare(b.mode);
				case 'turns': {
					const at = a.projection?.spans.length ?? 0;
					const bt = b.projection?.spans.length ?? 0;
					return dir * (at - bt);
				}
				case 'tokens': {
					const at = a.projection?.tokens.length ?? 0;
					const bt = b.projection?.tokens.length ?? 0;
					return dir * (at - bt);
				}
				case 'mean': {
					const am = a.summary?.mean_projection ?? 0;
					const bm = b.summary?.mean_projection ?? 0;
					return dir * (am - bm);
				}
				case 'min': {
					const am = a.summary?.min_projection ?? 0;
					const bm = b.summary?.min_projection ?? 0;
					return dir * (am - bm);
				}
				case 'drift': {
					const ad = a.summary?.drift_amount ?? -Infinity;
					const bd = b.summary?.drift_amount ?? -Infinity;
					return dir * (ad - bd);
				}
				default:
					return 0;
			}
		});

		return items;
	});

	// --- Selection ---
	function toggleSelect(id: string) {
		const next = new Set(selectedIds);
		if (next.has(id)) {
			next.delete(id);
		} else {
			next.add(id);
		}
		selectedIds = next;
	}

	function toggleSelectAll() {
		if (selectedIds.size === filteredResults.length) {
			selectedIds = new Set();
		} else {
			selectedIds = new Set(filteredResults.map((r) => r.id));
		}
	}

	// --- Comparison data ---
	let comparisonTrajectories = $derived.by(() => {
		return results
			.filter((r) => selectedIds.has(r.id) && r.projection && r.projection.spans.length > 0)
			.map((r, i) => ({
				name: r.name,
				spans: r.projection!.spans,
				color: COLORS[i % COLORS.length]
			}));
	});

	// --- Row expand ---
	function toggleExpand(id: string) {
		expandedId = expandedId === id ? null : id;
	}
</script>

<div class="batch-page">
	<h2>Batch Analysis</h2>

	<!-- Upload section -->
	{#if !batchResponse && !projecting}
		<section class="upload-section">
			<!-- svelte-ignore a11y_no_static_element_interactions -->
			<div
				class="drop-zone"
				class:drag-over={dragOver}
				ondrop={handleDrop}
				ondragover={handleDragOver}
				ondragleave={handleDragLeave}
			>
				{#if selectedFile}
					<p class="file-info">
						<strong>{selectedFile.name}</strong> - {lineCount} conversation{lineCount !== 1
							? 's'
							: ''}
					</p>
				{:else}
					<p>Drop a .jsonl file here, or click to select</p>
				{/if}
				<input
					type="file"
					accept=".jsonl,.ndjson"
					onchange={handleInputChange}
					class="file-input"
				/>
			</div>

			<div class="upload-actions">
				<button
					class="primary"
					disabled={!selectedFile || uploading}
					onclick={doUploadAndProject}
				>
					{uploading ? 'Uploading...' : 'Upload & Analyze'}
				</button>
			</div>

			{#if uploadError}
				<p class="error">{uploadError}</p>
			{/if}
		</section>
	{/if}

	<!-- Progress section -->
	{#if projecting || uploading}
		<section class="progress-section">
			<h3>
				{uploading
					? 'Uploading...'
					: `Processing ${projectProgress.index + 1} of ${projectProgress.total}...`}
			</h3>
			<div class="progress-bar-track">
				<div
					class="progress-bar-fill"
					style="width: {projectProgress.total > 0
						? (results.length / projectProgress.total) * 100
						: 0}%"
				></div>
			</div>
			<p class="progress-label">
				{results.length} / {projectProgress.total} completed
			</p>
		</section>
	{/if}

	<!-- Results section -->
	{#if results.length > 0}
		<section class="results-section">
			<!-- Filters -->
			<div class="filters">
				<input
					type="text"
					placeholder="Search by name..."
					bind:value={filterSearch}
					class="filter-input"
				/>
				<select bind:value={filterMode} class="filter-select">
					<option value="all">All modes</option>
					<option value="chat">Chat</option>
					<option value="raw">Raw</option>
				</select>
				<label class="drift-filter">
					Min drift:
					<input
						type="number"
						step="0.1"
						min="0"
						bind:value={filterDriftMin}
						class="drift-input"
					/>
				</label>
				<button
					class="reset-btn"
					onclick={() => {
						selectedFile = null;
						batchResponse = null;
						results = [];
						expandedId = null;
						selectedIds = new Set();
						showComparison = false;
						uploadError = '';
					}}
				>
					New Batch
				</button>
			</div>

			<!-- Selection actions -->
			<div class="selection-actions">
				<label class="select-all">
					<input
						type="checkbox"
						checked={selectedIds.size === filteredResults.length &&
							filteredResults.length > 0}
						onchange={toggleSelectAll}
					/>
					Select all ({selectedIds.size} selected)
				</label>
				{#if selectedIds.size >= 2}
					<button class="primary" onclick={() => (showComparison = !showComparison)}>
						{showComparison ? 'Hide' : 'Compare'} ({selectedIds.size})
					</button>
				{/if}
			</div>

			<!-- Comparison overlay -->
			{#if showComparison && comparisonTrajectories.length >= 2}
				<div class="comparison-panel">
					<h3>Trajectory Comparison</h3>
					<ComparisonChart trajectories={comparisonTrajectories} />
				</div>
			{/if}

			<!-- Results table -->
			<div class="table-wrapper">
				<table>
					<thead>
						<tr>
							<th class="col-check"></th>
							<th class="col-name sortable" onclick={() => toggleSort('name')}
								>Name{sortIndicator('name')}</th
							>
							<th class="col-mode sortable" onclick={() => toggleSort('mode')}
								>Mode{sortIndicator('mode')}</th
							>
							<th class="col-size sortable" onclick={() => toggleSort('turns')}
								>Turns/Tokens{sortIndicator('turns')}</th
							>
							<th class="col-num sortable" onclick={() => toggleSort('mean')}
								>Mean{sortIndicator('mean')}</th
							>
							<th class="col-num sortable" onclick={() => toggleSort('min')}
								>Min{sortIndicator('min')}</th
							>
							<th class="col-num sortable" onclick={() => toggleSort('drift')}
								>Drift{sortIndicator('drift')}</th
							>
							<th class="col-status">Status</th>
						</tr>
					</thead>
					<tbody>
						{#each filteredResults as result (result.id)}
							<tr
								class="result-row"
								class:expanded={expandedId === result.id}
								class:has-error={!!result.error}
							>
								<td>
									<input
										type="checkbox"
										checked={selectedIds.has(result.id)}
										onchange={() => toggleSelect(result.id)}
									/>
								</td>
								<!-- svelte-ignore a11y_click_events_have_key_events -->
								<td class="clickable" onclick={() => toggleExpand(result.id)} role="button" tabindex="0"
									>{result.name}</td
								>
								<td>
									<span class="mode-badge" class:chat={result.mode === 'chat'}
										>{result.mode}</span
									>
								</td>
								<td>
									{#if result.mode === 'chat'}
										{result.projection?.spans.length ?? '-'}
									{:else}
										{result.projection?.tokens.length ?? '-'}
									{/if}
								</td>
								<td class="num">{result.summary?.mean_projection?.toFixed(4) ?? '-'}</td>
								<td class="num">{result.summary?.min_projection?.toFixed(4) ?? '-'}</td>
								<td class="num">
									{result.summary?.drift_amount != null
										? result.summary.drift_amount.toFixed(4)
										: '-'}
								</td>
								<td>
									{#if result.error}
										<span class="status-error" title={result.error}>Error</span>
									{:else}
										<span class="status-ok">Done</span>
									{/if}
								</td>
							</tr>
							{#if expandedId === result.id && result.projection}
								<tr class="expanded-row">
									<td colspan="8">
										<div class="expanded-content">
											{#if result.projection.spans.length > 0}
												<div class="expanded-chart">
													<h4>Trajectory</h4>
													<TrajectoryChart spans={result.projection.spans} />
												</div>
											{/if}
											<div class="expanded-heatmap">
												<h4>Token Heatmap</h4>
												<TokenHeatmap response={result.projection} />
											</div>
										</div>
									</td>
								</tr>
							{/if}
						{/each}
					</tbody>
				</table>
			</div>

			{#if filteredResults.length === 0 && results.length > 0}
				<p class="empty-filter">No results match your filters.</p>
			{/if}
		</section>
	{/if}
</div>

<style>
	.batch-page {
		max-width: 1100px;
	}

	h2 {
		font-size: 1.1rem;
		margin-bottom: 1rem;
	}

	/* Upload section */
	.upload-section {
		margin-bottom: 1.5rem;
	}

	.drop-zone {
		border: 2px dashed var(--border);
		border-radius: 8px;
		padding: 2rem;
		text-align: center;
		position: relative;
		cursor: pointer;
		transition:
			border-color 0.2s,
			background 0.2s;
	}

	.drop-zone:hover,
	.drop-zone.drag-over {
		border-color: var(--primary);
		background: rgba(79, 195, 247, 0.05);
	}

	.drop-zone p {
		color: var(--text-muted);
		margin: 0;
	}

	.file-info {
		color: var(--text) !important;
	}

	.file-input {
		position: absolute;
		inset: 0;
		opacity: 0;
		cursor: pointer;
	}

	.upload-actions {
		margin-top: 1rem;
		display: flex;
		gap: 0.5rem;
	}

	.error {
		color: var(--danger);
		margin-top: 0.5rem;
		font-size: 0.85rem;
	}

	/* Progress section */
	.progress-section {
		margin-bottom: 1.5rem;
	}

	.progress-section h3 {
		font-size: 0.95rem;
		margin-bottom: 0.5rem;
	}

	.progress-bar-track {
		height: 8px;
		background: var(--surface);
		border-radius: 4px;
		overflow: hidden;
		border: 1px solid var(--border);
	}

	.progress-bar-fill {
		height: 100%;
		background: var(--primary);
		transition: width 0.3s ease;
		border-radius: 4px;
	}

	.progress-label {
		font-size: 0.8rem;
		color: var(--text-muted);
		margin-top: 0.25rem;
	}

	/* Filters */
	.filters {
		display: flex;
		gap: 0.75rem;
		align-items: center;
		margin-bottom: 0.75rem;
		flex-wrap: wrap;
	}

	.filter-input {
		flex: 1;
		min-width: 150px;
		max-width: 250px;
	}

	.filter-select {
		width: 120px;
	}

	.drift-filter {
		display: flex;
		align-items: center;
		gap: 0.3rem;
		font-size: 0.8rem;
		color: var(--text-muted);
	}

	.drift-input {
		width: 70px;
	}

	.reset-btn {
		margin-left: auto;
	}

	/* Selection */
	.selection-actions {
		display: flex;
		align-items: center;
		gap: 1rem;
		margin-bottom: 0.75rem;
		font-size: 0.85rem;
	}

	.select-all {
		display: flex;
		align-items: center;
		gap: 0.3rem;
		color: var(--text-muted);
		cursor: pointer;
	}

	/* Comparison panel */
	.comparison-panel {
		background: var(--surface);
		border: 1px solid var(--border);
		border-radius: 6px;
		padding: 1rem;
		margin-bottom: 1rem;
	}

	.comparison-panel h3 {
		font-size: 0.9rem;
		margin-bottom: 0.75rem;
	}

	/* Table */
	.table-wrapper {
		overflow-x: auto;
	}

	.sortable {
		cursor: pointer;
		user-select: none;
	}

	.sortable:hover {
		color: var(--primary);
	}

	.col-check {
		width: 36px;
	}

	.col-name {
		min-width: 150px;
	}

	.col-mode {
		width: 80px;
	}

	.col-size {
		width: 100px;
	}

	.col-num {
		width: 90px;
		text-align: right;
	}

	.col-status {
		width: 70px;
	}

	td.num {
		text-align: right;
		font-family: monospace;
		font-size: 0.8rem;
	}

	.clickable {
		cursor: pointer;
	}

	.clickable:hover {
		color: var(--primary);
	}

	.result-row.expanded {
		background: var(--surface);
	}

	.result-row.has-error {
		opacity: 0.7;
	}

	.mode-badge {
		font-size: 0.7rem;
		padding: 0.1rem 0.4rem;
		border-radius: 3px;
		background: var(--surface);
		color: var(--text-muted);
		text-transform: uppercase;
	}

	.mode-badge.chat {
		color: var(--primary);
		background: rgba(79, 195, 247, 0.1);
	}

	.status-ok {
		color: var(--success);
		font-size: 0.8rem;
	}

	.status-error {
		color: var(--danger);
		font-size: 0.8rem;
		cursor: help;
	}

	/* Expanded row */
	.expanded-row td {
		padding: 0;
		border-bottom: 2px solid var(--border);
	}

	.expanded-content {
		padding: 1rem;
		background: var(--surface);
	}

	.expanded-content h4 {
		font-size: 0.85rem;
		color: var(--text-muted);
		margin-bottom: 0.5rem;
	}

	.expanded-chart {
		margin-bottom: 1.5rem;
	}

	.empty-filter {
		color: var(--text-muted);
		text-align: center;
		padding: 2rem;
		font-size: 0.9rem;
	}
</style>
