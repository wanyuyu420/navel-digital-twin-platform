export async function addOssBasemap(viewer: any) {

  const Cesium = (window as any).Cesium

  const provider = new Cesium.SingleTileImageryProvider({
    url: 'https://gananqicheng-data.oss-cn-beijing.aliyuncs.com/dituqiepian/18/215902/110532.png',
    rectangle: Cesium.Rectangle.fromDegrees(
      115.047,
      24.9536,
      115.0544,
      24.958
    )
  })

  viewer.imageryLayers.addImageryProvider(provider)

  viewer.camera.flyTo({
    destination: Cesium.Rectangle.fromDegrees(
      115.047,
      24.9536,
      115.0544,
      24.958
    )
  })

  console.log("DOM only")
}
