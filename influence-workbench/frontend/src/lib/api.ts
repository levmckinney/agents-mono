import type {
	ProbeSetSummary,
	ProbeSetDetail,
	Pair,
	PairRole,
	RunSummary,
	RunDetail,
	RunResults,
	EmbeddingResponse,
	SearchPretrainingResponse,
	ExtractSpanResponse,
	GenerateContextResponse
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

// Embedding
export function getRunEmbedding(id: string): Promise<EmbeddingResponse> {
	return request(`/runs/${id}/embedding`);
}

// WebSocket for log streaming
export function connectLogs(runId: string): WebSocket {
	const proto = location.protocol === 'https:' ? 'wss:' : 'ws:';
	return new WebSocket(`${proto}//${location.host}/api/runs/${runId}/logs`);
}

// Tools — Infini-gram search
export function searchPretraining(
	completion: string,
	maxDocs: number = 10
): Promise<SearchPretrainingResponse> {
	return request('/search-pretraining', {
		method: 'POST',
		body: JSON.stringify({ completion, max_docs: maxDocs })
	});
}

// Tools — Span extraction
export function extractSpan(
	documentText: string,
	matchStart: number,
	matchEnd: number,
	spanLength: number = 256
): Promise<ExtractSpanResponse> {
	return request('/extract-span', {
		method: 'POST',
		body: JSON.stringify({
			document_text: documentText,
			match_start: matchStart,
			match_end: matchEnd,
			span_length: spanLength
		})
	});
}

// Tools — Claude context generation
export function generateContext(
	completion: string,
	instruction?: string
): Promise<GenerateContextResponse> {
	return request('/generate-context', {
		method: 'POST',
		body: JSON.stringify({ completion, instruction: instruction || null })
	});
}

// Tools — Bulk import
export async function importPairs(probeSetId: string, file: File): Promise<ProbeSetDetail> {
	const formData = new FormData();
	formData.append('file', file);
	const resp = await fetch(`${BASE}/probe-sets/${probeSetId}/import-pairs`, {
		method: 'POST',
		body: formData
	});
	if (!resp.ok) {
		const body = await resp.text();
		throw new Error(`${resp.status}: ${body}`);
	}
	return resp.json();
}

// Tools — Bulk role assignment
export function bulkSetRole(
	probeSetId: string,
	pairIds: string[],
	role: PairRole
): Promise<ProbeSetDetail> {
	return request(`/probe-sets/${probeSetId}/bulk-role`, {
		method: 'POST',
		body: JSON.stringify({ pair_ids: pairIds, role })
	});
}
