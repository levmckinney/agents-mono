/**
 * Shared TypeScript types for the axis-viewer frontend.
 *
 * These mirror the Pydantic models in backend/models.py.
 */

export interface TokenProjection {
	token_id: number;
	token_str: string;
	projection: number;
	position: number;
}

export interface TurnSpan {
	turn: number;
	role: string;
	start: number;
	end: number;
	text: string;
	mean_projection: number;
}

export interface ProjectionResponse {
	tokens: TokenProjection[];
	spans: TurnSpan[];
	layer: number;
	model_name: string;
}

export interface ConversationSummary {
	id: string;
	name: string;
	mode: string;
	created_at: string;
	turn_count?: number;
	char_count?: number;
}

export interface ConversationDetail {
	id: string;
	name: string;
	mode: string;
	conversation?: Array<{ role: string; content: string }>;
	text?: string;
	system_prompt?: string;
	metadata?: Record<string, unknown>;
	created_at: string;
}
