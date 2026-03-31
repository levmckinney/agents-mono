<script lang="ts">
	import {
		listConversations,
		saveConversation,
		loadConversation,
		deleteConversation,
		exportConversationsUrl
	} from '$lib/api';
	import type { ConversationSummary, ConversationDetail } from '$lib/types';

	interface Props {
		mode: 'chat' | 'raw';
		getSaveData: () => {
			conversation?: Array<{ role: string; content: string }>;
			text?: string;
			system_prompt?: string;
		};
		onLoad: (detail: ConversationDetail) => void;
		defaultName?: string;
	}

	let { mode, getSaveData, onLoad, defaultName = '' }: Props = $props();

	let panelOpen = $state(false);
	let conversations = $state<ConversationSummary[]>([]);
	let loadingList = $state(false);
	let listError = $state('');

	// Save state
	let saveName = $state('');
	let saving = $state(false);
	let saveSuccess = $state('');
	let saveError = $state('');

	// Delete confirmation
	let deletingId = $state<string | null>(null);

	async function fetchList() {
		loadingList = true;
		listError = '';
		try {
			const all = await listConversations();
			conversations = all.filter((c) => c.mode === mode);
		} catch (e) {
			listError = e instanceof Error ? e.message : String(e);
		} finally {
			loadingList = false;
		}
	}

	function togglePanel() {
		panelOpen = !panelOpen;
		if (panelOpen) {
			fetchList();
		}
	}

	async function handleSave() {
		const name = saveName.trim();
		if (!name) return;

		saving = true;
		saveError = '';
		saveSuccess = '';
		try {
			const data = getSaveData();
			const result = await saveConversation({
				name,
				mode,
				...data
			});
			saveSuccess = `Saved as "${result.name}"`;
			saveName = '';
			// Refresh list
			await fetchList();
		} catch (e) {
			saveError = e instanceof Error ? e.message : String(e);
		} finally {
			saving = false;
		}
	}

	async function handleLoad(id: string) {
		try {
			const detail = await loadConversation(id);
			onLoad(detail);
			panelOpen = false;
		} catch (e) {
			listError = e instanceof Error ? e.message : String(e);
		}
	}

	async function handleDelete(id: string) {
		try {
			await deleteConversation(id);
			conversations = conversations.filter((c) => c.id !== id);
			deletingId = null;
		} catch (e) {
			listError = e instanceof Error ? e.message : String(e);
		}
	}

	function formatDate(iso: string): string {
		if (!iso) return '';
		try {
			const d = new Date(iso);
			return d.toLocaleDateString(undefined, {
				month: 'short',
				day: 'numeric',
				hour: '2-digit',
				minute: '2-digit'
			});
		} catch {
			return iso;
		}
	}

	// Update default name when prop changes
	$effect(() => {
		if (defaultName && !saveName) {
			saveName = defaultName;
		}
	});
</script>

<div class="save-load-panel">
	<button class="toggle-btn" onclick={togglePanel}>
		<span class="toggle-arrow" class:open={panelOpen}>&#9654;</span>
		Save / Load
	</button>

	{#if panelOpen}
		<div class="panel-content">
			<!-- Save section -->
			<div class="save-section">
				<div class="save-row">
					<input
						type="text"
						class="save-name-input"
						placeholder="Conversation name..."
						bind:value={saveName}
						onkeydown={(e) => {
							if (e.key === 'Enter') handleSave();
						}}
					/>
					<button class="primary save-btn" onclick={handleSave} disabled={saving || !saveName.trim()}>
						{saving ? 'Saving...' : 'Save'}
					</button>
				</div>
				{#if saveSuccess}
					<div class="save-feedback success">{saveSuccess}</div>
				{/if}
				{#if saveError}
					<div class="save-feedback error">{saveError}</div>
				{/if}
			</div>

			<!-- Divider -->
			<hr class="divider" />

			<!-- List section -->
			<div class="list-section">
				<div class="list-header">
					<span class="list-title">Saved ({conversations.length})</span>
					<a href={exportConversationsUrl()} download class="export-link" title="Export all as JSONL">
						Export JSONL
					</a>
				</div>

				{#if loadingList}
					<p class="list-status">Loading...</p>
				{:else if listError}
					<p class="list-status error">{listError}</p>
				{:else if conversations.length === 0}
					<p class="list-status">No saved {mode === 'chat' ? 'conversations' : 'texts'} yet.</p>
				{:else}
					<ul class="conversation-list">
						{#each conversations as conv (conv.id)}
							<li class="conversation-item">
								{#if deletingId === conv.id}
									<div class="delete-confirm">
										<span>Delete "{conv.name}"?</span>
										<button class="small danger" onclick={() => handleDelete(conv.id)}>
											Yes
										</button>
										<button class="small" onclick={() => (deletingId = null)}>No</button>
									</div>
								{:else}
									<button class="conv-load-btn" onclick={() => handleLoad(conv.id)}>
										<span class="conv-name">{conv.name}</span>
										<span class="conv-meta">
											{formatDate(conv.created_at)}
											{#if conv.turn_count != null}
												&middot; {conv.turn_count} turns
											{/if}
											{#if conv.char_count != null}
												&middot; {conv.char_count.toLocaleString()} chars
											{/if}
										</span>
									</button>
									<button
										class="icon-btn danger-text"
										title="Delete"
										onclick={() => (deletingId = conv.id)}
									>
										&times;
									</button>
								{/if}
							</li>
						{/each}
					</ul>
				{/if}
			</div>
		</div>
	{/if}
</div>

<style>
	.save-load-panel {
		margin-bottom: 0.5rem;
	}

	.toggle-btn {
		background: none;
		border: none;
		color: var(--text-muted);
		font-size: 0.8rem;
		padding: 0.25rem 0;
		display: flex;
		align-items: center;
		gap: 0.4rem;
	}
	.toggle-btn:hover {
		color: var(--text);
		background: none;
	}
	.toggle-arrow {
		display: inline-block;
		font-size: 0.6rem;
		transition: transform 0.15s;
	}
	.toggle-arrow.open {
		transform: rotate(90deg);
	}

	.panel-content {
		margin-top: 0.4rem;
		padding: 0.75rem;
		border: 1px solid var(--border);
		border-radius: 6px;
		background: var(--surface);
	}

	/* Save section */
	.save-section {
		display: flex;
		flex-direction: column;
		gap: 0.35rem;
	}
	.save-row {
		display: flex;
		gap: 0.5rem;
	}
	.save-name-input {
		flex: 1;
		font-size: 0.8rem;
		padding: 0.35rem 0.5rem;
	}
	.save-btn {
		padding: 0.35rem 0.75rem;
		font-size: 0.8rem;
		white-space: nowrap;
	}
	.save-feedback {
		font-size: 0.75rem;
	}
	.save-feedback.success {
		color: var(--success);
	}
	.save-feedback.error {
		color: var(--danger);
	}

	/* Divider */
	.divider {
		border: none;
		border-top: 1px solid var(--border);
		margin: 0.6rem 0;
	}

	/* List section */
	.list-header {
		display: flex;
		justify-content: space-between;
		align-items: center;
		margin-bottom: 0.4rem;
	}
	.list-title {
		font-size: 0.75rem;
		color: var(--text-muted);
		text-transform: uppercase;
		letter-spacing: 0.05em;
		font-weight: 600;
	}
	.export-link {
		font-size: 0.7rem;
		color: var(--primary);
		text-decoration: none;
	}
	.export-link:hover {
		text-decoration: underline;
	}

	.list-status {
		font-size: 0.8rem;
		color: var(--text-muted);
		padding: 0.5rem 0;
	}
	.list-status.error {
		color: var(--danger);
	}

	.conversation-list {
		list-style: none;
		display: flex;
		flex-direction: column;
		gap: 0.25rem;
		max-height: 200px;
		overflow-y: auto;
	}

	.conversation-item {
		display: flex;
		align-items: center;
		gap: 0.25rem;
	}

	.conv-load-btn {
		flex: 1;
		background: none;
		border: 1px solid transparent;
		text-align: left;
		padding: 0.35rem 0.5rem;
		border-radius: 4px;
		display: flex;
		flex-direction: column;
		gap: 0.1rem;
		cursor: pointer;
	}
	.conv-load-btn:hover {
		background: var(--surface-hover);
		border-color: var(--border);
	}

	.conv-name {
		font-size: 0.8rem;
		color: var(--text);
	}
	.conv-meta {
		font-size: 0.7rem;
		color: var(--text-muted);
	}

	.icon-btn {
		background: none;
		border: none;
		color: var(--text-muted);
		font-size: 1rem;
		padding: 0.1rem 0.3rem;
		line-height: 1;
		cursor: pointer;
		border-radius: 3px;
	}
	.icon-btn:hover {
		background: var(--surface-hover);
		color: var(--text);
	}
	.icon-btn.danger-text:hover {
		color: var(--danger);
	}

	.delete-confirm {
		display: flex;
		align-items: center;
		gap: 0.4rem;
		font-size: 0.8rem;
		padding: 0.25rem 0;
		color: var(--text-muted);
	}

	button.small {
		padding: 0.2rem 0.5rem;
		font-size: 0.7rem;
	}
</style>
