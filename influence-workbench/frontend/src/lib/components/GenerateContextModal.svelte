<script lang="ts">
	import Modal from './Modal.svelte';
	import * as api from '$lib/api';

	let {
		open = false,
		onclose,
		onuse
	}: {
		open: boolean;
		onclose: () => void;
		onuse: (prompt: string, completion: string) => void;
	} = $props();

	let completion = $state('');
	let instruction = $state('');
	let generating = $state(false);
	let generatedPrompt = $state('');
	let modelUsed = $state('');
	let genError = $state('');
	let editing = $state(false);
	let editText = $state('');

	async function generate() {
		if (!completion.trim()) return;
		generating = true;
		genError = '';
		editing = false;
		try {
			const result = await api.generateContext(
				completion,
				instruction.trim() || undefined
			);
			generatedPrompt = result.generated_prompt;
			modelUsed = result.model;
		} catch (e) {
			genError = String(e);
		} finally {
			generating = false;
		}
	}

	function accept() {
		const text = editing ? editText : generatedPrompt;
		onuse(text, completion);
		onclose();
	}

	function startEdit() {
		editText = generatedPrompt;
		editing = true;
	}
</script>

<Modal title="Generate Context" {open} {onclose}>
	<div class="gen-section">
		<div class="field-group">
			<label for="gen-completion">Completion text</label>
			<textarea
				id="gen-completion"
				bind:value={completion}
				placeholder="Enter the completion text you want context for..."
				rows="3"
			></textarea>
		</div>

		<div class="field-group">
			<label for="gen-instruction">Instruction (optional)</label>
			<textarea
				id="gen-instruction"
				bind:value={instruction}
				placeholder="e.g., Write as a Wikipedia article"
				rows="2"
			></textarea>
		</div>

		<button class="primary" onclick={generate} disabled={generating || !completion.trim()}>
			{generating ? 'Generating...' : generatedPrompt ? 'Regenerate' : 'Generate'}
		</button>

		{#if genError}
			<div class="error">{genError}</div>
		{/if}

		{#if generatedPrompt && !generating}
			<div class="result">
				<div class="field-group">
					<label>Generated prompt {modelUsed ? `(${modelUsed})` : ''}</label>
					{#if editing}
						<textarea
							bind:value={editText}
							rows="6"
						></textarea>
					{:else}
						<div class="result-text">{generatedPrompt}</div>
					{/if}
				</div>
				<div class="result-actions">
					<button class="primary" onclick={accept}>Accept</button>
					<button onclick={generate}>Regenerate</button>
					{#if !editing}
						<button onclick={startEdit}>Edit</button>
					{/if}
				</div>
			</div>
		{/if}
	</div>
</Modal>

<style>
	.gen-section {
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
	.result {
		display: flex;
		flex-direction: column;
		gap: 0.5rem;
	}
	.result-text {
		font-size: 0.875rem;
		padding: 0.75rem;
		border: 1px solid var(--border, #ddd);
		border-radius: 4px;
		max-height: 200px;
		overflow-y: auto;
		white-space: pre-wrap;
		line-height: 1.5;
	}
	.result-actions {
		display: flex;
		gap: 0.5rem;
	}
	textarea {
		resize: vertical;
		min-height: 48px;
	}
	.error {
		background: rgba(239, 83, 80, 0.1);
		border: 1px solid var(--danger, #ef5350);
		padding: 0.5rem 0.75rem;
		border-radius: 4px;
		font-size: 0.875rem;
	}
</style>
