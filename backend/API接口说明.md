# 后端接口说明（脐橙数字孪生 · 施肥板块）

## 基本信息

- 基地址：`http://<host>:8000`（FastAPI 默认端口，uvicorn 启动）
- 接口前缀：`/api/v1`
- 健康检查：`GET /api/v1/health` → `{"status":"healthy"}`
- 启动依赖：GeoScene Server 必须可达（启动时做健康检查，连不上会拒绝启动）

## 环境配置（backend/.env）

```ini
GEOSCENE_SERVER_URL=https://<同事机IP>:6443/arcgis
GEOSCENE_FEATURE_SERVER_URL=https://<同事机IP>:6443/arcgis/rest/services/orange_trees/FeatureServer
GEOSCENE_USERNAME=<账号>
GEOSCENE_PASSWORD=<密码>
```

> 注：GeoScene 用自签名证书，代码已关闭 SSL 校验（`VERIFY_SSL=False`）；token 用 `requestip` 方式生成。

## 数据模型（layer 0 字段，17 列全名）

`id`(OID) · `batch_id` · `geom` · `confidence` · `compactness` · `shape_length` · `shape_area` · `value` · `count` · `area_m2` · `height_m` · `crown_diameter` · `volume_m3` · `growth_index` · `slope_degree` · `aspect` · `fertilizer_level`

施肥等级 `fertilizer_level` 语义：`0`=未计算，`1`=轻度，`2`=中度，`3`=重度。

---

## 1. 拉框空间诊断

`POST /api/v1/orange/spatial-diagnose`

请求（闭合多边形，WGS84）：
```json
{
  "coordinates": [
    [116.497, 27.132], [116.498, 27.132],
    [116.498, 27.134], [116.497, 27.134], [116.497, 27.132]
  ]
}
```

响应：
```json
{
  "total_count": 128,
  "avg_height": 1.85,
  "avg_area": 3.2,
  "avg_growth_index": 0.42,
  "fertilizer_recommendation": {
    "light_level_count": 40,
    "medium_level_count": 60,
    "heavy_level_count": 28
  },
  "trees": [
    { "id": 1, "lng": 116.497, "lat": 27.132, "growth_index": 0.31, "fertilizer_level": 0 }
  ]
}
```

## 2. 历史老树全量

`GET /api/v1/orange/historical-trees` → `{ "total": 10000, "trees": [...] }`（字段同上 `trees` 项）

## 3. 上传 TIF + 推理

`POST /api/v1/orange/upload-tif`（multipart，字段名 `file`）
→ `{ "success": true, "task_id": "...", "spatial_info": {...} }`（后台跑 YOLO+SAM）

`POST /api/v1/orange/upload-and-interpret`（multipart）→ 同上，返回 task_id
`GET /api/v1/orange/upload-and-interpret/{task_id}` → `{ "status": "completed", "total_trees": 128, "fresh_trees": [...] }`

## 4. 变量施肥推荐（核心）

`POST /api/v1/orange/fertilizer-plan`

请求：
```json
{
  "coordinates": [[116.497,27.132],[116.498,27.132],[116.498,27.134],[116.497,27.134],[116.497,27.132]],
  "weights": { "growth_index": 0.40, "size": 0.25, "compactness": 0.20, "slope": 0.15 },
  "mode": "quantile",
  "thresholds": [0.33, 0.67],
  "apply": false
}
```

- `mode`: `quantile`（按区域 33/67 分位分三档，默认）或 `fixed`（用 `thresholds` 固定阈值）
- `weights` 四项之和必须 = 1，可省略用默认配方
- `apply=true` 时把推荐等级写回 GeoScene 的 `fertilizer_level`

响应：
```json
{
  "total_trees": 128,
  "mode": "quantile",
  "weights": { "growth_index": 0.40, "size": 0.25, "compactness": 0.20, "slope": 0.15 },
  "thresholds": [0.42, 0.68],
  "summary": { "light_level_count": 40, "medium_level_count": 60, "heavy_level_count": 28 },
  "plan": [
    {
      "id": 1, "lng": 116.497, "lat": 27.132,
      "growth_index": 0.31, "area_m2": 3.2, "compactness": 0.7, "slope_degree": 31.3,
      "health_score": 0.4, "size_score": 0.5, "compact_score": 0.6, "slope_score": 0.5,
      "demand_score": 0.62,
      "current_level": 0, "recommended_level": 3
    }
  ],
  "applied": false
}
```

> 前端拿 `plan[].recommended_level`（1/2/3）在 Cesium 地图上按等级着色（绿/黄/红）。

## 5. 处方图导出

`POST /api/v1/orange/fertilizer-plan/export`（请求同施肥推荐 + `format`）

- `format: "geojson"` → 返回 GeoJSON FeatureCollection（点要素，属性含 `demand_score`/`current_level`/`recommended_level`）
- `format: "csv"` → 返回 CSV 文件流（带 BOM，可直接喂给无人机/施肥机）

## 6. 弱树告警

`GET /api/v1/orange/alerts?growth_threshold=0.15&limit=200`

响应：
```json
{
  "total": 15,
  "growth_threshold": 0.15,
  "alerts": [
    { "id": 12, "lng": 116.497, "lat": 27.133, "growth_index": 0.05, "area_m2": 2.1, "fertilizer_level": 3 }
  ]
}
```

## 7. 图层配置 CRUD

`GET/POST /api/v1/layers`、`GET/PUT/PATCH/DELETE /api/v1/layers/{id}`（前端图层配置，走本地数据库）
