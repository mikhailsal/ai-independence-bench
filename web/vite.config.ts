import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'
import { resolve } from 'path'
import { readFile } from 'fs/promises'
import type { Plugin } from 'vite'

function serveCachePlugin(): Plugin {
  const cacheDir = resolve(__dirname, '..', 'cache');
  return {
    name: 'serve-local-cache',
    configureServer(server) {
      server.middlewares.use(async (req, res, next) => {
        const prefix = '/ai-independence-bench/cache/';
        if (!req.url?.startsWith(prefix)) return next();

        const filePath = resolve(cacheDir, req.url.slice(prefix.length));
        if (!filePath.startsWith(cacheDir)) {
          res.statusCode = 403;
          res.end('Forbidden');
          return;
        }
        try {
          const content = await readFile(filePath, 'utf-8');
          res.setHeader('Content-Type', 'application/json');
          res.end(content);
        } catch {
          res.statusCode = 404;
          res.end('Not found');
        }
      });
    },
  };
}

export default defineConfig({
  plugins: [react(), tailwindcss(), serveCachePlugin()],
  base: '/ai-independence-bench/',
})
