import { sveltekit } from '@sveltejs/kit/vite';
import { defineConfig } from 'vite';

export default defineConfig({
	plugins: [sveltekit()],
	server: {
		allowedHosts: true,
		proxy: {
			'/api': {
				target: 'http://localhost:8000',
				changeOrigin: true,
				ws: true
			}
		}
	}
});
