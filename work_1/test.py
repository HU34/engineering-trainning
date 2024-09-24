import torch.optim
from pylab import *
import os
from model import VGG16WithLSTM
import argparse
from utils import init_logging, read_data
from train import test
from dataset import TestDataset
import torchvision.transforms as transforms
import torch.utils.data as data





parser = argparse.ArgumentParser()
parser.add_argument('--test_file_path', type=str, default="./data/test_val_set.csv",
                        help="测试集的文件路径")
parser.add_argument('--ratio', type=float, default=0.6, help="验证集、测试集比例")
parser.add_argument('--size', type=int, default=224, help='图片缩放尺寸')
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument("--logger_path", type=str, default="./logger", help="日志文件路径")
parser.add_argument('--NUM_CLASS', type=int, default=2, help='分类类别数')
parser.add_argument('--model_path', type=str, default="LSTMClassifier", help="模型保存文件夹")
args = parser.parse_args()

logger = init_logging(args.logger_path, "LSTMClassifier")


test_images, test_labels = read_data(args.test_file_path, args.size)
test_data_transform = transforms.Compose([
        transforms.ToTensor()])
test_set = TestDataset(test_images, test_labels, test_data_transform)
test_loader = data.DataLoader(dataset=test_set, batch_size=args.batch_size, shuffle=False, num_workers=0)


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
logger.info('Device: %s', device)
model = VGG16WithLSTM(args).to(device)

if torch.cuda.device_count() > 1:
    model = torch.nn.DataParallel(model)
    print("Let's use", torch.cuda.device_count(), "GPUs!")

best_model_name = os.path.join(args.model_path, 'best_model.pkl')
test_acc, test_img, test_true, test_pred = test(model, best_model_name, test_loader, logger)