<script lang="ts">
	import '../app.css';
	import { page } from '$app/state';
	import { onMount } from 'svelte';
	import { listModels, switchModel } from '$lib/api';
	import {
		modelState,
		setModels,
		beginSwitch,
		finishSwitch
	} from '$lib/model-state.svelte';

	let { children } = $props();

	let dropdownOpen = $state(false);
	let switchError = $state('');
	let pollInterval: ReturnType<typeof setInterval> | null = null;

	const tabs = [
		{ href: '/chat', label: 'Chat' },
		{ href: '/raw', label: 'Raw Text' },
		{ href: '/batch', label: 'Batch' }
	];

	onMount(async () => {
		try {
			const list = await listModels();
			setModels(list);
		} catch {
			// Backend may not be reachable yet — ignore
		}
	});

	async function handleSwitch(modelName: string) {
		dropdownOpen = false;
		switchError = '';

		const target = modelState.models.find((m) => m.model_name === modelName);
		if (!target || target.status === 'loaded') return;

		beginSwitch(target);

		try {
			await switchModel(modelName);
			// Poll for completion
			startPolling();
		} catch (e) {
			switchError = e instanceof Error ? e.message : String(e);
			// Refresh models list to get accurate state
			try {
				const list = await listModels();
				finishSwitch(list);
			} catch {
				// ignore
			}
		}
	}

	function startPolling() {
		if (pollInterval) return;
		pollInterval = setInterval(async () => {
			try {
				const list = await listModels();
				const switching = list.some((m) => m.status === 'switching');
				if (!switching) {
					finishSwitch(list);
					stopPolling();
				} else {
					setModels(list);
				}
			} catch {
				// ignore poll errors
			}
		}, 2000);
	}

	function stopPolling() {
		if (pollInterval) {
			clearInterval(pollInterval);
			pollInterval = null;
		}
	}

	function toggleDropdown() {
		dropdownOpen = !dropdownOpen;
	}

	function handleClickOutside(e: MouseEvent) {
		const target = e.target as HTMLElement;
		if (!target.closest('.model-selector')) {
			dropdownOpen = false;
		}
	}
</script>

<svelte:head>
	<title>Axis Viewer</title>
</svelte:head>

<svelte:window onclick={handleClickOutside} />

<div class="shell">
	<header>
		<h1>Axis Viewer</h1>
		<nav>
			{#each tabs as tab}
				<a
					href={tab.href}
					class:active={page.url.pathname.startsWith(tab.href)}
				>
					{tab.label}
				</a>
			{/each}
		</nav>

		<div class="model-selector">
			<button
				class="model-btn"
				onclick={toggleDropdown}
				disabled={modelState.isSwitching}
			>
				{#if modelState.isSwitching}
					<span class="mini-spinner"></span>
					Switching...
				{:else if modelState.currentModel}
					{modelState.currentModel.short_name}
					{#if modelState.currentModel.is_base}
						<span class="base-badge">base</span>
					{/if}
				{:else}
					Model
				{/if}
				<span class="caret">&#9662;</span>
			</button>

			{#if dropdownOpen}
				<div class="model-dropdown">
					{#each modelState.models as model}
						<button
							class="model-option"
							class:active={model.status === 'loaded'}
							onclick={() => handleSwitch(model.model_name)}
							disabled={model.status === 'loaded' || modelState.isSwitching}
						>
							<span class="model-option-name">{model.short_name}</span>
							<span class="model-option-meta">
								layer {model.target_layer}/{model.total_layers}
							</span>
							{#if model.is_base}
								<span class="base-badge">base</span>
							{/if}
							{#if model.status === 'loaded'}
								<span class="loaded-badge">loaded</span>
							{/if}
						</button>
					{/each}
				</div>
			{/if}
		</div>
	</header>

	{#if switchError}
		<div class="switch-error">
			Model switch failed: {switchError}
		</div>
	{/if}

	{#if modelState.isSwitching}
		<div class="switch-banner">
			<span class="mini-spinner"></span>
			Switching model to {modelState.currentModel?.short_name}... This may take 30-60 seconds.
		</div>
	{/if}

	<main>
		{@render children()}
	</main>
</div>

<style>
	.shell {
		max-width: 1200px;
		margin: 0 auto;
		padding: 1rem;
	}
	header {
		display: flex;
		align-items: center;
		gap: 2rem;
		margin-bottom: 1.5rem;
		padding-bottom: 1rem;
		border-bottom: 1px solid var(--border);
	}
	h1 {
		font-size: 1.25rem;
		white-space: nowrap;
	}
	nav {
		display: flex;
		gap: 0.25rem;
	}
	nav a {
		padding: 0.4rem 1rem;
		border-radius: 4px;
		text-decoration: none;
		color: var(--text-muted);
		font-size: 0.875rem;
		transition: background 0.15s, color 0.15s;
	}
	nav a:hover {
		background: var(--surface);
		color: var(--text);
	}
	nav a.active {
		background: var(--surface);
		color: var(--primary);
	}

	/* Model selector */
	.model-selector {
		margin-left: auto;
		position: relative;
	}
	.model-btn {
		display: inline-flex;
		align-items: center;
		gap: 0.4rem;
		padding: 0.35rem 0.75rem;
		font-size: 0.8rem;
		border: 1px solid var(--border);
		border-radius: 4px;
		background: var(--surface);
		color: var(--text);
		cursor: pointer;
		white-space: nowrap;
	}
	.model-btn:hover {
		background: var(--surface-hover);
	}
	.model-btn:disabled {
		opacity: 0.7;
		cursor: not-allowed;
	}
	.caret {
		font-size: 0.6rem;
		color: var(--text-muted);
	}
	.model-dropdown {
		position: absolute;
		top: 100%;
		right: 0;
		margin-top: 0.25rem;
		background: var(--surface);
		border: 1px solid var(--border);
		border-radius: 6px;
		min-width: 240px;
		z-index: 100;
		box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
	}
	.model-option {
		display: flex;
		align-items: center;
		gap: 0.5rem;
		width: 100%;
		padding: 0.5rem 0.75rem;
		border: none;
		background: none;
		color: var(--text);
		font-size: 0.8rem;
		text-align: left;
		cursor: pointer;
		border-radius: 0;
	}
	.model-option:first-child {
		border-radius: 6px 6px 0 0;
	}
	.model-option:last-child {
		border-radius: 0 0 6px 6px;
	}
	.model-option:hover:not(:disabled) {
		background: var(--surface-hover);
	}
	.model-option.active {
		color: var(--primary);
	}
	.model-option:disabled {
		cursor: default;
	}
	.model-option-name {
		font-weight: 500;
	}
	.model-option-meta {
		color: var(--text-muted);
		font-size: 0.7rem;
		font-family: monospace;
	}
	.base-badge {
		font-size: 0.65rem;
		padding: 0.1rem 0.3rem;
		border-radius: 3px;
		background: rgba(255, 167, 38, 0.15);
		color: var(--warning);
		font-weight: 600;
		text-transform: uppercase;
		letter-spacing: 0.03em;
	}
	.loaded-badge {
		font-size: 0.65rem;
		padding: 0.1rem 0.3rem;
		border-radius: 3px;
		background: rgba(102, 187, 106, 0.15);
		color: var(--success);
		font-weight: 600;
	}

	/* Switch banner */
	.switch-banner {
		display: flex;
		align-items: center;
		gap: 0.5rem;
		padding: 0.5rem 0.75rem;
		margin-bottom: 1rem;
		background: rgba(79, 195, 247, 0.08);
		border: 1px solid var(--primary);
		border-radius: 4px;
		font-size: 0.825rem;
		color: var(--primary);
	}
	.switch-error {
		padding: 0.5rem 0.75rem;
		margin-bottom: 1rem;
		background: rgba(239, 83, 80, 0.1);
		border: 1px solid var(--danger);
		border-radius: 4px;
		font-size: 0.825rem;
		color: var(--danger);
	}

	/* Mini spinner (used in button and banner) */
	.mini-spinner {
		display: inline-block;
		width: 12px;
		height: 12px;
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
</style>
