export type PairRole = 'train' | 'query' | 'both';

export interface Pair {
	pair_id: string;
	prompt: string;
	completion: string;
	role: PairRole;
	metadata: Record<string, unknown>;
}

export interface ProbeSetSummary {
	id: string;
	name: string;
	pair_count: number;
	created_at: string;
	updated_at: string;
}

export interface ProbeSetDetail extends ProbeSetSummary {
	pairs: Pair[];
}

export type RunStatusType = 'pending' | 'running' | 'completed' | 'failed';

export interface RunSummary {
	id: string;
	probe_set_id: string;
	status: RunStatusType;
	created_at: string;
	started_at: string | null;
	finished_at: string | null;
}

export interface RunDetail extends RunSummary {
	exit_code: number | null;
	error_message: string | null;
	config_snapshot: Record<string, unknown>;
}

export interface RunResults {
	run_id: string;
	query_results: Record<string, unknown>[];
	train_results: Record<string, unknown>[];
	influences: InfluenceRow[];
}

export interface InfluenceRow {
	query_id: string;
	train_id: string;
	influence_score: number;
	per_token_scores: string | null;
}

// Tool types — Infini-gram search

export interface InfinigramDocSpan {
	text: string;
	is_match: boolean;
}

export interface InfinigramDocument {
	doc_ix: number;
	doc_len: number;
	disp_len: number;
	spans: InfinigramDocSpan[];
	full_text: string;
}

export interface SearchPretrainingResponse {
	documents: InfinigramDocument[];
	query: string;
	count: number;
}

// Tool types — Span extraction

export interface ExtractSpanResponse {
	prompt: string;
	completion: string;
}

// Tool types — Claude context generation

export interface GenerateContextResponse {
	generated_prompt: string;
	model: string;
}
