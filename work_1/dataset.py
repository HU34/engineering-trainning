from pylab import *
import pydicom
import cv2
from sklearn.model_selection import train_test_split
import sklearn.utils
from collections import Counter
import torch.utils.data as data
import torchvision.transforms as transforms
import argparse
# from utils import init_logging


# 可选：图像阈值范围缩减
def windows_pro(img, min_bound=0, max_bound=85):
    """
        输入：图像，阈值下限min_bound，阈值上限max_bound
        处理过程：先获取指定限制范围内的值[min_bound,max_bound]，再中心化、归一化
        输出：阈值范围缩减后中心化归一化结果[0,255]
    """
    img[img > max_bound] = max_bound
    img[img < min_bound] = min_bound  # [min_bound, max_bound]
    img = img - min_bound  # 中心化[0,max_bound+min_bound]
    img = normalize(img)  # 归一化 [0,255]
    return img


# 可选：直方图均衡(增加对比度)
def equalize_hist(img):
    img = img.astype(np.uint8)
    img = cv2.equalizeHist(img)
    return img


# 必选：缩放尺寸，默认缩放为224
def img_resize(img, size=224):
    img = cv2.resize(img, (size, size))
    return img


# 必选：归一化
def normalize(img):
    img = img.astype(np.float32)
    np.seterr(divide='ignore', invalid='ignore')
    img = (img - img.min()) / (img.max() - img.min())  # 归一化[0,1]
    img = img * 255  # 0-255
    img = img.astype(np.uint8)
    return img


# 必选：扩展为3通道
def extend_channels(img):
    img_channels = np.zeros([img.shape[0], img.shape[1], 3])
    img_channels[:, :, 0] = img
    img_channels[:, :, 1] = img
    img_channels[:, :, 2] = img
    return img_channels


# 必选：图像预处理组合（基本操作）
def data_preprocess_base(img, size):
    # step1: 缩放尺寸 224*224
    img = img_resize(img, size)
    # step2: 归一化[0,255]
    img = normalize(img)
    # step3: 扩展为3通道 224*224*3
    img = extend_channels(img)
    # Step4: 转换为unit8格式
    img = img.astype(np.uint8)
    return img


# 图像预处理（伪影增强）
def data_preprocess_enhanced(img, size):
    # step1: 图像阈值范围缩减 [min_bound, max_bound]
    img = windows_pro(img)
    # step2: 直方图均衡 [0, 255]
    img = equalize_hist(img)
    # step3: 缩放尺寸 224*224
    img = img_resize(img, size)
    # step4: 归一化[0,255]
    img = normalize(img)
    # step5: 扩展为3通道 224*224*3
    img = extend_channels(img)
    # Step6: 转换为unit8格式
    img = img.astype(np.uint8)
    return img


def under_sampling(train_img, train_label, random_seed):
    from imblearn.under_sampling import RandomUnderSampler
    rus = RandomUnderSampler(random_state=random_seed, replacement=False)
    nsamples, nx, ny, nz = train_img.shape  # n*224*224*1
    train_img_flatten = train_img.reshape(nsamples, nx * ny * nz)
    X_resampled, y_resampled = rus.fit_resample(train_img_flatten, train_label)
    X_resampled = X_resampled.reshape(X_resampled.shape[0], nx, ny, nz)
    return X_resampled, y_resampled

# 上采样
def over_sampling(train_img, train_label, random_seed):
    from imblearn.over_sampling import RandomOverSampler, SMOTE
    from imblearn.pipeline import Pipeline
    # 创建RandomOverSampler和SMOTE对象
    ros = RandomOverSampler(random_state=random_seed)
    smote = SMOTE(random_state=random_seed)
    # 将图像数据展平
    nsamples, nx, ny, nz = train_img.shape
    train_img_flatten = train_img.reshape(nsamples, nx * ny * nz)
    # 创建Pipeline，依次应用RandomOverSampler和SMOTE
    pipeline = Pipeline([
        ('ros', ros),
        ('smote', smote)
    ])
    # 同时应用RandomOverSampler和SMOTE来生成合成样本
    X_resampled, y_resampled = pipeline.fit_resample(train_img_flatten, train_label)
    # 将平衡后的图像数据恢复为原始形状
    X_resampled = X_resampled.reshape(X_resampled.shape[0], nx, ny, nz)
    # print(X_resampled.shape)
    return X_resampled, y_resampled


# -------------------------------#
#       读取数据并划分数据集
# -------------------------------#

def data_load(train_path, test_val_path, size, is_train, random_seed, ratio, logger, is_sampling='no_sampler'):
    test_val_images_dicom_list = []
    train_dicom_list = []
    test_val_labels = []
    train_label = []

    # 1.读取数据：images图像矩阵，labels标签
    f = open(test_val_path, "r+")
    for line in f.readlines():
        img_path = line.strip().split(',')[0] # 图像地址
        test_val_images_dicom_list.append(img_path)
        label = line.strip().split(',')[1]  # 图像标签
        label = '0' if label == 'good' else '1'
        test_val_labels.append(label)
    test_val_labels = np.array(test_val_labels)  # 图像标签 n*1
    # 读取图像矩阵
    test_val_images = array([data_preprocess_enhanced(pydicom.read_file(dcm).pixel_array, size) for dcm in test_val_images_dicom_list])
    f.close()

    f = open(train_path, "r+")
    for line in f.readlines():
        img_path = line.strip().split(',')[0]  # 图像地址
        train_dicom_list.append(img_path)
        label = line.strip().split(',')[1]  # 图像标签
        label = '0' if label == 'good' else '1'
        train_label.append(label)
    train_label = np.array(train_label)  # 图像标签 n*1
    # 读取图像矩阵
    train_img = array([data_preprocess_enhanced(pydicom.read_file(dcm).pixel_array, size) for dcm in train_dicom_list])
    f.close()

    # 划分数据集：验证集、测试集
    images, labels = sklearn.utils.shuffle(test_val_images, test_val_labels, random_state=random_seed)
    val_img, test_img, val_label, test_label = train_test_split(images, labels, test_size=ratio,
                                                                stratify=labels, random_state=random_seed)

    # 2.划分数据集
    if is_train:  # 训练模式或测试模式没有单独csv
        logger.info('----Training Mode----') if is_train else logger.info('----Testing mode----')
        logger.info('Training set: %s, labels=%s' % (train_img.shape, sorted(Counter(train_label).items())))
        logger.info('Val set: %s, labels=%s' % (val_img.shape, sorted(Counter(val_label).items())))
        logger.info('Test set: %s, labels=%s' % (test_img.shape, sorted(Counter(test_label).items())))

        if is_sampling == 'no_sampler':
            pass

        elif is_sampling == 'over_sampler':
            train_img, train_label = over_sampling(train_img, train_label, random_seed)

        elif is_sampling == 'down_sampler':
            train_img, train_label = under_sampling(train_img, train_label, random_seed)

        logger.info('Sampling mode:%s, train_num:%s,label:%s' % (
            is_sampling, train_img.shape, sorted(Counter(train_label).items())))

    else:  # 测试模式
        logger.info('----Testing Mode----')
        logger.info('Test set: %s, labels=%s' % (test_img.shape, sorted(Counter(test_label).items())))
    return train_img, train_label, val_img, val_label, test_img, test_label


class TrainDataset(data.Dataset):
    def __init__(self, train_img, train_label, train_data_transform=None):
        super(TrainDataset, self).__init__()
        self.train_img = train_img
        self.train_label = train_label
        self.train_data_transform = train_data_transform

    def __getitem__(self, index):
        img = self.train_img[index]
        target = int(self.train_label[index])
        if self.train_data_transform is not None:
            from PIL import Image
            img = Image.fromarray(np.uint8(img))
            img = self.train_data_transform(img)
        return img, target

    def __len__(self):
        return len(self.train_img)


class ValDataset(data.Dataset):
    def __init__(self, val_img, val_label, val_data_transform):
        super(ValDataset, self).__init__()
        self.val_img = val_img
        self.val_label = val_label
        self.val_data_transform = val_data_transform

    def __getitem__(self, index):
        img = self.val_img[index]
        target = int(self.val_label[index])
        if self.val_data_transform is not None:
            from PIL import Image
            img = Image.fromarray(np.uint8(img))
            img = self.val_data_transform(img)
        return img, target

    def __len__(self):
        return len(self.val_img)


class TestDataset(data.Dataset):
    def __init__(self, test_img, test_label, test_data_transform):
        super(TestDataset, self).__init__()
        self.test_img = test_img
        self.test_label = test_label
        self.test_data_transform = test_data_transform

    def __getitem__(self, index):
        img = self.test_img[index]
        target = int(self.test_label[index])
        if self.test_data_transform is not None:
            from PIL import Image
            img = Image.fromarray(np.uint8(img))
            img = self.test_data_transform(img)
        return img, target

    def __len__(self):
        return len(self.test_img)



def get_dataset(path, test_path, size, batch_size, is_train, random_seed, ratio, logger, is_sampling):
    train_img, train_label, val_img, val_label, test_img, test_label = data_load(path, test_path, size, is_train, random_seed, ratio,
                                                                                 logger, is_sampling)
    train_loader = []
    val_loader = []

    if is_train:
        # 定义train_loader
        train_data_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation((0, 180),  expand=False),
            transforms.ToTensor()])

        train_set = TrainDataset(train_img, train_label, train_data_transform)
        train_loader = data.DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True, num_workers=0)

        # 定义val_loader
        val_data_transform = transforms.Compose([
            transforms.ToTensor()])
        val_set = ValDataset(val_img, val_label, val_data_transform)
        val_loader = data.DataLoader(dataset=val_set, batch_size=batch_size, shuffle=False, num_workers=0)

    # 定义test_loader
    test_data_transform = transforms.Compose([
        transforms.ToTensor()])
    test_set = TestDataset(test_img, test_label, test_data_transform)
    test_loader = data.DataLoader(dataset=test_set, batch_size=batch_size, shuffle=False, num_workers=0)

    return train_loader, val_loader, test_loader


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_file_path', type=str, default="./data/train_set.csv", help="训练数据集的文件路径")
    parser.add_argument('--test_val_file_path', type=str, default="./data/test_val_set.csv", help="测试集和验证集数据集的文件路径")
    parser.add_argument('--random_seed', type=int, default=321, help="随机种子")
    parser.add_argument('--ratio', type=float, default=0.6, help="验证集、测试集比例")
    parser.add_argument('--size', type=int, default=224, help='图片缩放尺寸')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--is_train', type=bool, default=True, help='是否是训练模式')
    parser.add_argument('--is_sampling', type=str, default='no_sampling', help="是否进行不平衡采样处理")
    parser.add_argument("--logger_path", type=str, default="./logger", help="日志文件路径")
    args = parser.parse_args()

    # logger = init_logging(args.logger_path, "LSTMClassifier")
    # train_loader, val_loader, test_loader = get_dataset(args.train_file_path, args.test_val_file_path, args.size, args.batch_size, args.is_train, args.random_seed, args.ratio, logger, args.is_sampling)