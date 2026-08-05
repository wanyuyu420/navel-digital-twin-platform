import { defineConfig, loadEnv } from 'vite'
import vue from '@vitejs/plugin-vue'
import path from 'path'

// https://vitejs.dev/config/
export default defineConfig(({ mode }) => {
  // Load env file based on `mode` in the current working directory.
  // Set the third parameter to '' to load all env regardless of the `VITE_` prefix.
  const env = loadEnv(mode, process.cwd(), '')

  return {
    plugins: [
      vue(),
      // 将 cesium 外部化到全局 Cesium 变量（替代 vite-plugin-external）
      {
        name: 'cesium-global',
        enforce: 'post',
        transform(code, id) {
          if (id.includes('node_modules') || !/\.(js|ts|vue)$/.test(id)) return null
          return {
            code: code.replace(
              /import\s+[\s\S]*?from\s+['"]cesium['"];?\s*/g,
              ''
            ),
            map: null
          }
        },
        renderChunk(code) {
          return {
            code: code.replace(
              /import\s+[\s\S]*?from\s+['"]cesium['"];?\s*/g,
              ''
            ),
            map: null
          }
        }
      }
    ],
    resolve: {
      alias: {
        '@': path.resolve(__dirname, './src')
      }
    },
    css: {
      preprocessorOptions: {
        scss: {
          additionalData: `@use "@/assets/styles/_variables.scss" as *; @use "@/assets/styles/_mixins.scss" as *;`
        }
      }
    },
    server: {
      proxy: {
        '/api': {
          target: env.VITE_API_TARGET || 'http://127.0.0.1:8000',
          changeOrigin: true
        },
        '/tiles/': {
          target: env.VITE_API_TARGET || 'http://127.0.0.1:8000',
          changeOrigin: true
        },
        '/terrain/': {
          target: env.VITE_API_TARGET || 'http://127.0.0.1:8000',
          changeOrigin: true
        },
        // HEC-RAS simulation frames served from local Nginx
        '/simulation/': {
          target: 'http://127.0.0.1:8081',
          changeOrigin: true
        },
        // Proxy for the remote GeoTIFF base map (avoids CORS issues)
        '/geotiff-proxy': {
          target: 'http://47.113.147.127',
          changeOrigin: true,
          rewrite: (path) => path.replace(/^\/geotiff-proxy/, '')
        }
      }
    }
  }
})
