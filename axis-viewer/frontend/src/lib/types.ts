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
