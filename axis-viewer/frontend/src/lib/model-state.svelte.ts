/**
 * Shared reactive state for the currently loaded model.
 *
 * Uses a single exported object with $state properties so Svelte 5 module
 * rules are satisfied (no reassigned exported $state bindings).
 */

import type { ModelInfo } from './types';

interface ModelState {
	models: ModelInfo[];
	currentModel: ModelInfo | null;
	isSwitching: boolean;
}

export const modelState: ModelState = $state({
	models: [],
	currentModel: null,
	isSwitching: false
});

/** Update the models list and derive the current model from it. */
export function setModels(list: ModelInfo[]) {
	modelState.models = list;
	const loaded = list.find((m) => m.status === 'loaded' || m.status === 'switching');
	if (loaded) {
		modelState.currentModel = loaded;
		modelState.isSwitching = loaded.status === 'switching';
	}
}

/** Mark a switch as in-progress and update the current model optimistically. */
export function beginSwitch(model: ModelInfo) {
	modelState.isSwitching = true;
	modelState.currentModel = { ...model, status: 'switching' };
	modelState.models = modelState.models.map((m) => ({
		...m,
		status:
			m.model_name === model.model_name
				? ('switching' as const)
				: ('available' as const)
	}));
}

/** Mark the switch as complete, refreshing from the given list. */
export function finishSwitch(list: ModelInfo[]) {
	modelState.isSwitching = false;
	setModels(list);
}
