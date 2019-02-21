# 假设下面这个类是读取船只的数据类
class ShipDataset(Dataset):
    """
     root：图像存放地址根路径
     augment：是否需要图像增强
    """
    def __init__(self, root, augment=None):
        # 这个list存放所有图像的地址
        self.image_files = np.array([x.path for x in os.scandir(root) if
            x.name.endswith(".jpg") or x.name.endswith(".png") or x.name.endswith(".JPG")]
        self.augment = augment   # 是否需要图像增强

    def __getitem__(self, index):
        # 读取图像数据并返回
        # 这里的open_image是读取图像函数，可以用PIL、opencv等库进行读取
        return open_image(self.image_files[index])

    def __len__(self):
        # 返回图像的数量
        return len(self.image_files)