/**
 * API client for the axis-viewer backend.
 */

import type {
	ProjectionResponse,
	ConversationSummary,
	ConversationDetail,
	BatchUploadResponse,
	ModelInfo
} from './types';

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

export async function generate(
	conversation: Array<{ role: string; content: string }>,
	temperature: number = 0.7,
	maxNewTokens: number = 512
): Promise<{ content: string }> {
	return request<{ content: string }>('/generate', {
		method: 'POST',
		body: JSON.stringify({
			conversation,
			temperature,
			max_new_tokens: maxNewTokens
		})
	});
}

// --- Conversation save/load ---

export async function saveConversation(data: {
	name: string;
	mode: string;
	conversation?: Array<{ role: string; content: string }>;
	text?: string;
	system_prompt?: string;
	metadata?: Record<string, unknown>;
}): Promise<ConversationDetail> {
	return request<ConversationDetail>('/conversations', {
		method: 'POST',
		body: JSON.stringify(data)
	});
}

export async function listConversations(): Promise<ConversationSummary[]> {
	return request<ConversationSummary[]>('/conversations');
}

export async function loadConversation(id: string): Promise<ConversationDetail> {
	return request<ConversationDetail>(`/conversations/${id}`);
}

export async function deleteConversation(id: string): Promise<void> {
	const res = await fetch(`${BASE}/conversations/${id}`, { method: 'DELETE' });
	if (!res.ok) {
		const body = await res.text();
		throw new Error(`${res.status}: ${body}`);
	}
}

export function exportConversationsUrl(): string {
	return `${BASE}/conversations/export`;
}

// --- Model endpoints ---

export async function listModels(): Promise<ModelInfo[]> {
	return request<ModelInfo[]>('/models');
}

export async function getCurrentModel(): Promise<ModelInfo> {
	return request<ModelInfo>('/models/current');
}

export async function switchModel(modelName: string): Promise<ModelInfo> {
	return request<ModelInfo>('/models/switch', {
		method: 'POST',
		body: JSON.stringify({ model_name: modelName })
	});
}

// --- Batch endpoints ---

export async function uploadBatch(file: File): Promise<BatchUploadResponse> {
	const formData = new FormData();
	formData.append('file', file);
	const res = await fetch(`${BASE}/batch/upload`, {
		method: 'POST',
		body: formData
	});
	if (!res.ok) {
		const body = await res.text();
		throw new Error(`${res.status}: ${body}`);
	}
	return res.json() as Promise<BatchUploadResponse>;
}

export function batchProjectUrl(batchId: string): string {
	return `${BASE}/batch/${batchId}/project`;
}
