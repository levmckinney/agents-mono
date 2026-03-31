/**
 * API client for the axis-viewer backend.
 */

import type { ProjectionResponse } from './types';

const BASE = '/api';

async function request<T>(path: string, options?: RequestInit): Promise<T> {
	const res = await fetch(`${BASE}${path}`, {
		headers: { 'Content-Type': 'application/json' },
		...options
	});
	if (!res.ok) {
		const body = await res.text();
		throw new Error(`${res.status}: ${body}`);
	}
	return res.json() as Promise<T>;
}

export async function getHealth() {
	return request<{ status: string; model: string }>('/health');
}

export async function projectChat(
	conversation: Array<{ role: string; content: string }>
): Promise<ProjectionResponse> {
	return request<ProjectionResponse>('/project/chat', {
		method: 'POST',
		body: JSON.stringify({ conversation })
	});
}

export async function projectRaw(text: string): Promise<ProjectionResponse> {
	return request<ProjectionResponse>('/project/raw', {
		method: 'POST',
		body: JSON.stringify({ text })
	});
}
