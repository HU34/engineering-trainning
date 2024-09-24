import scikitplot as skplt
from pylab import *
import pydicom
from dataset import data_preprocess_enhanced
import os
import logging

def mkdir(file_path):
    if not os.path.exists(file_path):
        os.makedirs(file_path)

def init_logging(save_path, model_name):
    """
    日志初始化方法
    :param save_path: 日志保存路径
    :return:
    """
    save_path = os.path.join(save_path, model_name)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    # 创建一个logger对象
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)  # 设置日志级别为INFO
    # 创建一个文件处理器，并设置输出格式
    file_handler = logging.FileHandler(os.path.join(save_path, 'log.txt'))
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))

    # 创建一个控制台处理器，并设置输出格式
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
    # 将处理器添加到logger对象中
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


def show_curve(model_path, test_true, test_pred, test_prob):
    """
    绘制单模型ROC、P-R、Confusion Matrix
    """
    # ROC曲线
    skplt.metrics.plot_roc(test_true, test_prob)
    plt.savefig('{}/{}.jpg'.format(model_path, 'ROC Curve'))
    plt.show()

    # PR曲线
    skplt.metrics.plot_precision_recall_curve(test_true, test_prob, cmap='nipy_spectral')
    plt.savefig('{}/{}.jpg'.format(model_path, 'P-R Curve'))
    plt.show()

    # Confusion Matrix
    skplt.metrics.plot_confusion_matrix(test_true, test_pred, normalize=False)
    plt.savefig('{}/{}.jpg'.format(model_path, 'confusion_matrix'))
    plt.show()



def show_plot(history_train, history_valid, history_auc, model_path):
    # 绘制训练集和验证集的损失值
    x = range(0, len(np.array(history_auc)))
    plt.figure(1)  # 第一张图
    plt.plot(x, np.array(history_train)[:, 0])
    plt.plot(x, np.array(history_valid)[:, 0])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['train', 'valid'], loc='upper left')
    # plt.show()
    plt.savefig('{}/{}.jpg'.format(model_path, 'Model Loss'))

    # 绘制训练集和验证集的精确度
    plt.figure(2)  # 第二张图
    plt.plot(x, np.array(history_train)[:, 1])
    plt.plot(x, np.array(history_valid)[:, 1])
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['train', 'valid'], loc='upper left')
    # plt.show()
    plt.savefig('{}/{}.jpg'.format(model_path, 'Model Accuracy'))

    # 绘制验证集AUC
    plt.figure(3)  # 第三张图
    plt.plot(x, np.array(history_auc))
    plt.title('Validation AUC')
    plt.ylabel('AUC')
    plt.xlabel('Epoch')
    plt.savefig('{}/{}.jpg'.format(model_path, 'Validation AUC'))

def read_data(file_path, size):
    images_dicom_list = []
    labels = []
    f = open(file_path, "r+")
    for line in f.readlines():
        img_path = line.strip().split(',')[0]  # 图像地址
        images_dicom_list.append(img_path)
        label = line.strip().split(',')[1]  # 图像标签
        label = '0' if label == 'good' else '1'
        labels.append(label)
    labels = np.array(labels)  # 图像标签 n*1
    # 读取图像矩阵
    images = array(
        [data_preprocess_enhanced(pydicom.read_file(dcm).pixel_array, size) for dcm in images_dicom_list])
    f.close()

    return images, labels