<script lang="ts">
	import { generate, projectChat } from '$lib/api';
	import TokenHeatmap from '$lib/components/TokenHeatmap.svelte';
	import TrajectoryChart from '$lib/components/TrajectoryChart.svelte';
	import SaveLoadPanel from '$lib/components/SaveLoadPanel.svelte';
	import { modelState } from '$lib/model-state.svelte';
	import type { ProjectionResponse, ConversationDetail } from '$lib/types';

	// ---- State ----

	interface Message {
		role: 'user' | 'assistant' | 'system';
		content: string;
	}

	let messages = $state<Message[]>([]);
	let inputText = $state('');
	let systemPrompt = $state('');
	let systemPromptOpen = $state(false);
	let temperature = $state(0.7);
	let maxTokens = $state(512);

	let isBaseModel = $derived(modelState.currentModel?.is_base ?? false);

	// Loading states
	let generating = $state(false);
	let projecting = $state(false);
	let statusText = $state('');

	// Projection result
	let projectionResponse = $state<ProjectionResponse | null>(null);

	// Editing state
	let editingIndex = $state<number | null>(null);
	let editingContent = $state('');

	// Insert state
	let insertAtIndex = $state<number | null>(null);
	let insertRole = $state<'user' | 'assistant'>('user');
	let insertContent = $state('');

	// Error state
	let errorMessage = $state('');

	// Scroll ref
	let messagesContainer: HTMLDivElement | undefined = $state(undefined);

	// ---- Helpers ----

	function buildConversation(): Array<{ role: string; content: string }> {
		const conv: Array<{ role: string; content: string }> = [];
		if (systemPrompt.trim()) {
			conv.push({ role: 'system', content: systemPrompt.trim() });
		}
		for (const msg of messages) {
			conv.push({ role: msg.role, content: msg.content });
		}
		return conv;
	}

	function scrollToBottom() {
		// Use tick-like delay to let DOM update
		setTimeout(() => {
			if (messagesContainer) {
				messagesContainer.scrollTop = messagesContainer.scrollHeight;
			}
		}, 0);
	}

	async function updateProjections() {
		const conv = buildConversation();
		if (conv.length === 0) {
			projectionResponse = null;
			return;
		}
		projecting = true;
		statusText = 'Computing projections...';
		errorMessage = '';
		try {
			projectionResponse = await projectChat(conv);
		} catch (e) {
			errorMessage = `Projection error: ${e instanceof Error ? e.message : String(e)}`;
		} finally {
			projecting = false;
			statusText = '';
		}
	}

	// ---- Actions ----

	async function sendMessage() {
		const text = inputText.trim();
		if (!text || generating) return;

		errorMessage = '';
		messages.push({ role: 'user', content: text });
		inputText = '';
		scrollToBottom();

		// Generate assistant response
		generating = true;
		statusText = 'Generating response...';
		try {
			const conv = buildConversation();
			const result = await generate(conv, temperature, maxTokens);
			messages.push({ role: 'assistant', content: result.content });
			scrollToBottom();
		} catch (e) {
			errorMessage = `Generation error: ${e instanceof Error ? e.message : String(e)}`;
			generating = false;
			statusText = '';
			return;
		}
		generating = false;

		// Get projections
		await updateProjections();
	}

	function handleInputKeydown(e: KeyboardEvent) {
		if (e.key === 'Enter' && !e.shiftKey) {
			e.preventDefault();
			sendMessage();
		}
	}

	// Edit message
	function startEditing(index: number) {
		editingIndex = index;
		editingContent = messages[index].content;
	}

	async function saveEdit() {
		if (editingIndex === null) return;
		messages[editingIndex].content = editingContent;
		editingIndex = null;
		editingContent = '';
		await updateProjections();
	}

	function cancelEdit() {
		editingIndex = null;
		editingContent = '';
	}

	// Delete message
	async function deleteMessage(index: number) {
		messages.splice(index, 1);
		// Trigger reactivity
		messages = [...messages];
		await updateProjections();
	}

	// Insert message
	function startInsert(index: number) {
		insertAtIndex = index;
		insertRole = 'user';
		insertContent = '';
	}

	async function confirmInsert() {
		if (insertAtIndex === null || !insertContent.trim()) return;
		messages.splice(insertAtIndex, 0, {
			role: insertRole,
			content: insertContent.trim()
		});
		messages = [...messages];
		insertAtIndex = null;
		insertContent = '';
		await updateProjections();
	}

	function cancelInsert() {
		insertAtIndex = null;
		insertContent = '';
	}

	// Clear all
	async function clearAll() {
		messages = [];
		projectionResponse = null;
		editingIndex = null;
		insertAtIndex = null;
		errorMessage = '';
	}

	// ---- Save / Load ----

	function getSaveData() {
		return {
			conversation: buildConversation(),
			system_prompt: systemPrompt.trim() || undefined
		};
	}

	async function handleLoad(detail: ConversationDetail) {
		// Extract system prompt and messages from the conversation array
		const conv = detail.conversation || [];
		const sys = detail.system_prompt || '';
		const msgs: Message[] = [];

		for (const entry of conv) {
			if (entry.role === 'system') {
				// Use the first system message as system prompt if not set via field
				if (!sys) {
					systemPrompt = entry.content;
				}
			} else {
				msgs.push({
					role: entry.role as 'user' | 'assistant',
					content: entry.content
				});
			}
		}

		if (sys) {
			systemPrompt = sys;
			systemPromptOpen = true;
		}

		messages = msgs;
		editingIndex = null;
		insertAtIndex = null;
		errorMessage = '';

		await updateProjections();
	}

	let defaultSaveName = $derived(
		messages.length > 0 ? messages[0].content.slice(0, 50) : ''
	);

	let busy = $derived(generating || projecting || modelState.isSwitching);
</script>

<div class="chat-layout">
	<!-- Left panel: conversation -->
	<div class="chat-panel">
		<!-- Save / Load -->
		<SaveLoadPanel
			mode="chat"
			{getSaveData}
			onLoad={handleLoad}
			defaultName={defaultSaveName}
		/>

		<!-- System prompt (collapsible) -->
		<div class="system-prompt-section">
			<button
				class="system-prompt-toggle"
				onclick={() => (systemPromptOpen = !systemPromptOpen)}
			>
				<span class="toggle-arrow" class:open={systemPromptOpen}>&#9654;</span>
				System Prompt
			</button>
			{#if systemPromptOpen}
				<textarea
					class="system-prompt-input"
					placeholder="Optional system prompt..."
					bind:value={systemPrompt}
					rows="3"
				></textarea>
			{/if}
		</div>

		<!-- Generation controls -->
		<div class="controls">
			<label class="control-item">
				<span class="control-label">Temperature: {temperature.toFixed(2)}</span>
				<input
					type="range"
					min="0"
					max="1.5"
					step="0.05"
					bind:value={temperature}
				/>
			</label>
			<label class="control-item">
				<span class="control-label">Max tokens: {maxTokens}</span>
				<input
					type="range"
					min="64"
					max="2048"
					step="64"
					bind:value={maxTokens}
				/>
			</label>
			{#if messages.length > 0}
				<button class="danger clear-btn" onclick={clearAll}>Clear All</button>
			{/if}
		</div>

		<!-- Base model notice -->
		{#if isBaseModel}
			<div class="base-model-notice">
				Chat mode is unavailable for base models. Use
				<a href="/raw">Raw Text</a> mode instead.
			</div>
		{/if}

		<!-- Messages -->
		<div class="messages" bind:this={messagesContainer}>
			{#if messages.length === 0}
				<p class="empty-state">Send a message to start a conversation.</p>
			{/if}

			{#each messages as msg, i (i)}
				<!-- Insert button above this message -->
				{#if insertAtIndex === i}
					<div class="insert-form">
						<div class="insert-header">
							<select bind:value={insertRole}>
								<option value="user">User</option>
								<option value="assistant">Assistant</option>
							</select>
							<button class="small" onclick={confirmInsert}>Insert</button>
							<button class="small" onclick={cancelInsert}>Cancel</button>
						</div>
						<textarea
							class="insert-textarea"
							placeholder="Message content..."
							bind:value={insertContent}
							rows="2"
						></textarea>
					</div>
				{:else}
					<button
						class="insert-btn"
						title="Insert message here"
						onclick={() => startInsert(i)}
					>+</button>
				{/if}

				<div class="message" class:user={msg.role === 'user'} class:assistant={msg.role === 'assistant'}>
					<div class="message-header">
						<span class="role-tag {msg.role}">{msg.role}</span>
						<div class="message-actions">
							{#if editingIndex !== i}
								<button class="icon-btn" title="Edit" onclick={() => startEditing(i)}>&#9998;</button>
								<button class="icon-btn danger-text" title="Delete" onclick={() => deleteMessage(i)}>&times;</button>
							{/if}
						</div>
					</div>
					{#if editingIndex === i}
						<textarea
							class="edit-textarea"
							bind:value={editingContent}
							rows="4"
						></textarea>
						<div class="edit-actions">
							<button class="small primary" onclick={saveEdit}>Save</button>
							<button class="small" onclick={cancelEdit}>Cancel</button>
						</div>
					{:else}
						<div class="message-content">{msg.content}</div>
					{/if}
				</div>
			{/each}

			<!-- Insert at end -->
			{#if messages.length > 0}
				{#if insertAtIndex === messages.length}
					<div class="insert-form">
						<div class="insert-header">
							<select bind:value={insertRole}>
								<option value="user">User</option>
								<option value="assistant">Assistant</option>
							</select>
							<button class="small" onclick={confirmInsert}>Insert</button>
							<button class="small" onclick={cancelInsert}>Cancel</button>
						</div>
						<textarea
							class="insert-textarea"
							placeholder="Message content..."
							bind:value={insertContent}
							rows="2"
						></textarea>
					</div>
				{:else}
					<button
						class="insert-btn"
						title="Insert message here"
						onclick={() => startInsert(messages.length)}
					>+</button>
				{/if}
			{/if}

			<!-- Loading indicator -->
			{#if busy}
				<div class="loading">
					<span class="spinner"></span>
					<span>{statusText}</span>
				</div>
			{/if}
		</div>

		<!-- Error display -->
		{#if errorMessage}
			<div class="error-bar">{errorMessage}</div>
		{/if}

		<!-- Input area -->
		<div class="input-area">
			<textarea
				class="message-input"
				placeholder="Type a message... (Enter to send, Shift+Enter for newline)"
				bind:value={inputText}
				onkeydown={handleInputKeydown}
				rows="2"
				disabled={busy}
			></textarea>
			<button class="primary send-btn" onclick={sendMessage} disabled={busy || !inputText.trim() || isBaseModel}>
				{generating ? 'Generating...' : isBaseModel ? 'Unavailable (base model)' : 'Send'}
			</button>
		</div>
	</div>

	<!-- Right panel: visualizations -->
	<div class="viz-panel">
		{#if projectionResponse}
			<div class="viz-section heatmap-scroll">
				<h3 class="viz-title">Token Heatmap</h3>
				<TokenHeatmap response={projectionResponse} />
			</div>
			<div class="viz-section">
				<h3 class="viz-title">Trajectory</h3>
				<TrajectoryChart spans={projectionResponse.spans} />
			</div>
		{:else}
			<p class="viz-placeholder">
				Projections will appear here after the first assistant response.
			</p>
		{/if}
	</div>
</div>

<style>
	/* Layout */
	.chat-layout {
		display: grid;
		grid-template-columns: 1fr 1fr;
		gap: 1.5rem;
		height: calc(100vh - 100px);
		min-height: 500px;
	}

	/* Chat panel */
	.chat-panel {
		display: flex;
		flex-direction: column;
		min-height: 0;
	}

	/* System prompt */
	.system-prompt-section {
		margin-bottom: 0.5rem;
	}
	.system-prompt-toggle {
		background: none;
		border: none;
		color: var(--text-muted);
		font-size: 0.8rem;
		padding: 0.25rem 0;
		display: flex;
		align-items: center;
		gap: 0.4rem;
	}
	.system-prompt-toggle:hover {
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
	.system-prompt-input {
		width: 100%;
		margin-top: 0.25rem;
		resize: vertical;
		font-family: inherit;
	}

	/* Controls */
	.controls {
		display: flex;
		align-items: center;
		gap: 1rem;
		padding: 0.5rem 0;
		border-bottom: 1px solid var(--border);
		margin-bottom: 0.5rem;
		flex-wrap: wrap;
	}
	.control-item {
		display: flex;
		align-items: center;
		gap: 0.4rem;
		font-size: 0.8rem;
		color: var(--text-muted);
	}
	.control-label {
		white-space: nowrap;
		min-width: 6rem;
	}
	.control-item input[type='range'] {
		width: 100px;
		accent-color: var(--primary);
	}
	.clear-btn {
		margin-left: auto;
		padding: 0.3rem 0.6rem;
		font-size: 0.75rem;
	}

	/* Base model notice */
	.base-model-notice {
		padding: 0.6rem 0.75rem;
		background: rgba(255, 167, 38, 0.08);
		border: 1px solid var(--warning);
		border-radius: 4px;
		font-size: 0.825rem;
		color: var(--warning);
		margin-bottom: 0.5rem;
	}
	.base-model-notice a {
		color: var(--primary);
		text-decoration: underline;
	}

	/* Messages area */
	.messages {
		flex: 1;
		overflow-y: auto;
		padding: 0.5rem 0;
		display: flex;
		flex-direction: column;
		gap: 0.25rem;
	}
	.empty-state {
		color: var(--text-muted);
		text-align: center;
		padding: 3rem 1rem;
		font-size: 0.9rem;
	}

	/* Message */
	.message {
		padding: 0.6rem 0.75rem;
		border-radius: 6px;
		border: 1px solid var(--border);
		background: var(--surface);
	}
	.message.user {
		border-left: 3px solid var(--primary);
	}
	.message.assistant {
		border-left: 3px solid var(--warning);
	}
	.message-header {
		display: flex;
		align-items: center;
		justify-content: space-between;
		margin-bottom: 0.3rem;
	}
	.role-tag {
		font-size: 0.7rem;
		font-weight: 600;
		text-transform: uppercase;
		letter-spacing: 0.05em;
		padding: 0.1rem 0.4rem;
		border-radius: 3px;
	}
	.role-tag.user {
		color: var(--primary);
		background: rgba(79, 195, 247, 0.1);
	}
	.role-tag.assistant {
		color: var(--warning);
		background: rgba(255, 167, 38, 0.1);
	}
	.role-tag.system {
		color: var(--text-muted);
		background: rgba(136, 146, 164, 0.1);
	}
	.message-actions {
		display: flex;
		gap: 0.25rem;
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
	.message-content {
		font-size: 0.875rem;
		line-height: 1.5;
		white-space: pre-wrap;
		word-break: break-word;
	}

	/* Edit mode */
	.edit-textarea {
		width: 100%;
		resize: vertical;
		font-family: inherit;
		font-size: 0.875rem;
		margin-bottom: 0.35rem;
	}
	.edit-actions {
		display: flex;
		gap: 0.35rem;
	}
	button.small {
		padding: 0.25rem 0.5rem;
		font-size: 0.75rem;
	}

	/* Insert controls */
	.insert-btn {
		align-self: center;
		background: none;
		border: 1px dashed var(--border);
		color: var(--text-muted);
		font-size: 0.75rem;
		padding: 0 0.5rem;
		line-height: 1.4;
		border-radius: 3px;
		opacity: 0;
		transition: opacity 0.15s;
	}
	.messages:hover .insert-btn {
		opacity: 0.6;
	}
	.insert-btn:hover {
		opacity: 1 !important;
		border-color: var(--primary);
		color: var(--primary);
	}
	.insert-form {
		padding: 0.5rem;
		border: 1px dashed var(--primary);
		border-radius: 4px;
		background: rgba(79, 195, 247, 0.05);
	}
	.insert-header {
		display: flex;
		align-items: center;
		gap: 0.5rem;
		margin-bottom: 0.35rem;
	}
	.insert-header select {
		font-size: 0.75rem;
		padding: 0.2rem 0.3rem;
	}
	.insert-textarea {
		width: 100%;
		resize: vertical;
		font-family: inherit;
		font-size: 0.85rem;
	}

	/* Loading */
	.loading {
		display: flex;
		align-items: center;
		gap: 0.5rem;
		padding: 0.75rem;
		color: var(--text-muted);
		font-size: 0.85rem;
	}
	.spinner {
		display: inline-block;
		width: 16px;
		height: 16px;
		border: 2px solid var(--border);
		border-top-color: var(--primary);
		border-radius: 50%;
		animation: spin 0.8s linear infinite;
	}
	@keyframes spin {
		to {
			transform: rotate(360deg);
		}
	}

	/* Error bar */
	.error-bar {
		background: rgba(239, 83, 80, 0.1);
		border: 1px solid var(--danger);
		color: var(--danger);
		padding: 0.4rem 0.75rem;
		border-radius: 4px;
		font-size: 0.8rem;
		margin-top: 0.25rem;
	}

	/* Input area */
	.input-area {
		display: flex;
		gap: 0.5rem;
		padding-top: 0.5rem;
		border-top: 1px solid var(--border);
		margin-top: 0.25rem;
		align-items: flex-end;
	}
	.message-input {
		flex: 1;
		resize: none;
		font-family: inherit;
		font-size: 0.875rem;
	}
	.send-btn {
		align-self: flex-end;
		white-space: nowrap;
	}

	/* Viz panel */
	.viz-panel {
		display: flex;
		flex-direction: column;
		gap: 1rem;
		min-height: 0;
		overflow-y: auto;
	}
	.viz-section {
		padding: 0.75rem;
		border: 1px solid var(--border);
		border-radius: 6px;
		background: var(--surface);
	}
	.heatmap-scroll {
		overflow-y: auto;
		max-height: 50%;
		flex-shrink: 1;
	}
	.viz-title {
		font-size: 0.85rem;
		color: var(--text-muted);
		margin-bottom: 0.5rem;
		text-transform: uppercase;
		letter-spacing: 0.05em;
	}
	.viz-placeholder {
		color: var(--text-muted);
		text-align: center;
		padding: 3rem 1rem;
		font-size: 0.9rem;
	}
</style>
