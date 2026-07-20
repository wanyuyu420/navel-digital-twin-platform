# app/services/tif_service.py
import rasterio

class TifResolver:
    def __init__(self, file_path: str):
        """
        初始化方法：接收 3.1 阶段流式落地后的物理文件绝对路径
        """
        self.file_path = file_path
        self.crs = None        # 用于死死扣留地理坐标系（如 EPSG:32650）
        self.transform = None  # 用于死死扣留地理仿射变换矩阵（像素坐标转平面米制坐标的公式）

    def extract_spatial_reference(self):
        """
        核心方法：【3.2 阶段】栅格空间参考扣留
        使用 with 语句懒加载打开 0.17 GB 的大文件，读完立刻关闭，绝对不占你宝贵的 4.3G 内存！
        """
        # 利用 rasterio 懒加载挂载硬盘文件
        with rasterio.open(self.file_path) as src:
            self.crs = src.crs
            self.transform = src.transform
            
            print(f"【3.2 成功】当前影像坐标系: {self.crs}, 仿射矩阵: {self.transform}")
            
        # 出了 with 作用域，文件流自动关闭，内存完美释放
        return {
            "crs": str(self.crs),
            "transform": [t for t in self.transform] if self.transform else []
        }