/**
 * Aliyun OSS Basemap Loader
 *
 * 将三个 OSS 数据源叠加为项目底图：
 *  1. dituqiepian/{z}/{x}/{y}.png  — XYZ 影像底图（最底层）
 *  2. tree.geojson                  — 果树冠层矢量面（399 棵）
 *  3. 3d/tileset.json               — 三维实景瓦片（倾斜摄影）
 *
 * 全部直连 OSS（桶已配置 CORS：Access-Control-Allow-Origin: *），无需后端代理。
 */
const OSS_BASE = 'https://gananqicheng-data.oss-cn-beijing.aliyuncs.com'
const TILE_URL = `${OSS_BASE}/dituqiepian/{z}/{x}/{y}.png`
const TREE_URL = `${OSS_BASE}/tree.geojson`
const TILESET_URL = `${OSS_BASE}/3d/tileset.json`

export async function addOssBasemap(viewer: any): Promise<void> {
  const Cesium = (window as any).Cesium
  if (!Cesium) {
    console.warn('[OssBasemap] Cesium not found on window')
    return
  }

  // ── 1. XYZ 影像底图（插入到 imageryLayers 最底层，作为基础底图） ──
  try {
    const provider = new Cesium.UrlTemplateImageryProvider({
      url: TILE_URL,
      // 该 OSS 瓦片金字塔只覆盖 z=12~15（已实测），限制层级避免 404 风暴
      minimumLevel: 12,
      maximumLevel: 15,
      credit: 'gananqicheng-data OSS',
      // 该 OSS 瓦片为标准 Web Mercator XYZ 切片，需显式指定（默认是 Geographic）
      tilingScheme: new Cesium.WebMercatorTilingScheme(),
    })
    // index 0 = 最底层，叠加在 GeoTIFF 等其它图层之下
    viewer.imageryLayers.addImageryProvider(provider, 0)
    console.log('[OssBasemap] XYZ 影像底图已加载')
  } catch (e) {
    console.error('[OssBasemap] XYZ 影像底图加载失败:', e)
  }

  // ── 2. 果树冠层矢量（GeoJSON，399 个 MultiPolygon） ──
  try {
    const dataSource = await Cesium.GeoJsonDataSource.load(TREE_URL, {
      // 树冠填充：半透明绿色，边框略深
      stroke: Cesium.Color.fromCssColorString('#2ecc71').withAlpha(0.9),
      strokeWidth: 1.5,
      fill: Cesium.Color.fromCssColorString('#27ae60').withAlpha(0.35),
      markerColor: Cesium.Color.fromCssColorString('#2ecc71'),
    })
    viewer.dataSources.add(dataSource)
    console.log(
      `[OssBasemap] 果树矢量已加载: ${dataSource.entities.values.length} 个要素`
    )
  } catch (e) {
    console.error('[OssBasemap] 果树矢量加载失败:', e)
  }

  // ── 3. 三维实景瓦片（倾斜摄影 3D Tiles） ──
  try {
    const tileset = await Cesium.Cesium3DTileset.fromUrl(TILESET_URL, {
      maximumScreenSpaceError: 16,
      dynamicScreenSpaceError: true,
      skipLevelOfDetail: true,
      cullWithChildrenBounds: true,
    })
    viewer.scene.primitives.add(tileset)
    console.log('[OssBasemap] 三维实景瓦片已加载')
  } catch (e) {
    console.error('[OssBasemap] 三维实景瓦片加载失败:', e)
  }
}
