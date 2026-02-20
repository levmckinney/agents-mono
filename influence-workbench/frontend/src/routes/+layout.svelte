<script lang="ts">
	import '../app.css';
	import { page } from '$app/state';

	let { children } = $props();

	const tabs = [
		{ href: '/create', label: 'Create' },
		{ href: '/runs', label: 'Runs' },
		{ href: '/results', label: 'Results' }
	];
</script>

<svelte:head>
	<title>Influence Workbench</title>
</svelte:head>

<div class="shell">
	<header>
		<h1>Influence Workbench</h1>
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
	</header>
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
</style>
