import torch
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from dataset import get_dataset
import argparse
from utils import init_logging
import os
from model import VGG16WithLSTM
import numpy as np


parser = argparse.ArgumentParser()
parser.add_argument('--train_file_path', type=str, default="./data/train_set.csv", help="训练数据集的文件路径")
parser.add_argument('--test_val_file_path', type=str, default="./data/test_val_set.csv",
                        help="测试集和验证集数据集的文件路径")
parser.add_argument('--random_seed', type=int, default=321, help="随机种子")
parser.add_argument('--ratio', type=float, default=0.6, help="验证集、测试集比例")
parser.add_argument('--size', type=int, default=224, help='图片缩放尺寸')
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--is_train', type=bool, default=True, help='是否是训练模式')
parser.add_argument('--is_sampling', type=str, default='over_sampler', help="是否进行不平衡采样处理")
parser.add_argument('--use_weight_loss', type=bool, default=True, help="是否使用加权loss")
parser.add_argument("--logger_path", type=str, default="./logger", help="日志文件路径")
parser.add_argument('--NUM_CLASS', type=int, default=2, help='分类类别数')
parser.add_argument('--EPOCHS', type=int, default=30, help="训练轮次")
parser.add_argument('--model_path', type=str, default="LSTMClassifier", help="模型保存文件夹")
args = parser.parse_args()

logger = init_logging(args.logger_path, "LSTMClassifier")
train_loader, val_loader, test_loader = get_dataset(args.train_file_path, args.test_val_file_path, args.size,
                                                        args.batch_size, args.is_train, args.random_seed, args.ratio, logger,
                                                        args.is_sampling)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
logger.info('Device: %s', device)
model = VGG16WithLSTM(args).to(device)
best_model_name = os.path.join(args.model_path, 'best_model.pkl')
model.load_state_dict(torch.load(best_model_name), False)

with torch.no_grad():
    model.eval()
    val_representations = []
    val_label_list = []
    for test_x, test_y in val_loader:
        if torch.cuda.is_available():
            images, labels = test_x.cuda(), test_y.cuda()
        else:
            images, labels = test_x, test_y
        representation, _ = model(images)
        val_representations.append(representation.detach().cpu().numpy())
        val_label_list.append(test_y.cpu())




with torch.no_grad():
    model.eval()
    representations = []
    label_list = []
    for test_x, test_y in test_loader:
        if torch.cuda.is_available():
            images, labels = test_x.cuda(), test_y.cuda()
        else:
            images, labels = test_x, test_y
        representation, _ = model(images)
        representations.append(representation.detach().cpu().numpy())
        label_list.append(test_y.cpu())


val_representations_flat = np.array([sample.flatten() for sublist in val_representations for sample in sublist])
representations_flat = np.array([sample.flatten() for sublist in representations for sample in sublist])

# 将标签列表转换为一维数组
val_labels = np.array([label.item() for sublist in val_label_list for label in sublist])
test_labels = np.array([label.item() for sublist in label_list for label in sublist])

# 使用KNN算法进行最近邻分类
knn = KNeighborsClassifier(n_neighbors=3)  # K设置为3
knn.fit(val_representations_flat, val_labels)  # 拟合KNN模型
predicted_labels = knn.predict(representations_flat)  # 使用KNN模型预测测试集的标签

# 计算准确率
accuracy = accuracy_score(test_labels, predicted_labels)
logger.info("KNN 表征测试准确率:", accuracy)


