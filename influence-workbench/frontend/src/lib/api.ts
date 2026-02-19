import type {
	ProbeSetSummary,
	ProbeSetDetail,
	Pair,
	RunSummary,
	RunDetail,
	RunResults
} from './types';

const BASE = '/api';

async function request<T>(path: string, init?: RequestInit): Promise<T> {
	const resp = await fetch(`${BASE}${path}`, {
		headers: { 'Content-Type': 'application/json' },
		...init
	});
	if (!resp.ok) {
		const body = await resp.text();
		throw new Error(`${resp.status}: ${body}`);
	}
	if (resp.status === 204) return undefined as unknown as T;
	return resp.json();
}

// Probe sets
export function listProbeSets(): Promise<ProbeSetSummary[]> {
	return request('/probe-sets');
}

export function getProbeSet(id: string): Promise<ProbeSetDetail> {
	return request(`/probe-sets/${id}`);
}

export function createProbeSet(name: string, pairs: Pair[]): Promise<ProbeSetDetail> {
	return request('/probe-sets', {
		method: 'POST',
		body: JSON.stringify({ name, pairs })
	});
}

export function updateProbeSet(
	id: string,
	data: { name?: string; pairs?: Pair[] }
): Promise<ProbeSetDetail> {
	return request(`/probe-sets/${id}`, {
		method: 'PUT',
		body: JSON.stringify(data)
	});
}

export function deleteProbeSet(id: string): Promise<void> {
	return request(`/probe-sets/${id}`, { method: 'DELETE' });
}

// Runs
export function listRuns(probeSetId?: string): Promise<RunSummary[]> {
	const query = probeSetId ? `?probe_set_id=${probeSetId}` : '';
	return request(`/runs${query}`);
}

export function getRun(id: string): Promise<RunDetail> {
	return request(`/runs/${id}`);
}

export function createRun(probeSetId: string): Promise<RunDetail> {
	return request(`/probe-sets/${probeSetId}/run`, { method: 'POST' });
}

export function getRunResults(id: string): Promise<RunResults> {
	return request(`/runs/${id}/results`);
}

// WebSocket for log streaming
export function connectLogs(runId: string): WebSocket {
	const proto = location.protocol === 'https:' ? 'wss:' : 'ws:';
	return new WebSocket(`${proto}//${location.host}/api/runs/${runId}/logs`);
}
