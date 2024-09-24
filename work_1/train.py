import torch.optim
from pylab import *
import os
from dataset import get_dataset
from sklearn import metrics
import torch.nn.functional as F
from model import VGG16WithLSTM
import argparse
from utils import init_logging, mkdir, show_plot


def train(model, args, train_loader, val_loader, optimizer, criterion, logger):
    start = datetime.datetime.now()
    history_train = []
    history_valid = []
    history_auc = []
    best_auc = 0.

    # 动态lr设置
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)
    for epoch in range(args.EPOCHS):
        correct = total = 0.
        loss_list = []
        # 为教学使用，仅选择部分数据进行训练，通过train_batch参数控制
        for batch_index, (batch_x, batch_y) in enumerate(train_loader, 0):
            if torch.cuda.is_available():
                batch_x, batch_y = batch_x.cuda(), batch_y.cuda()
            model.train()
            # 优化过程
            optimizer.zero_grad()
            _, output = model(batch_x)
            loss = criterion(output, batch_y)
            loss.backward()
            optimizer.step()
            # 输出训练结果
            loss_list.append(loss.item())
            _, predicted = torch.max(output.data, 1)  # 返回每行的最大值
            total += batch_y.size(0)
            correct += (predicted == batch_y).sum().item()
        train_avg_acc = 100 * correct / total
        train_avg_loss = np.mean(loss_list)
        logger.info('[Epoch=%d/%d]Train set: Avg_loss=%.4f, Avg_accuracy=%.4f%%' % (
            epoch + 1, args.EPOCHS, train_avg_loss, train_avg_acc))
        history_train.append((train_avg_loss, train_avg_acc))


        scheduler.step()

        # 验证集
        valid_pred, valid_true, auc, valid_acc, valid_avg_loss = valid(model, val_loader, criterion)
        logger.info('[Epoch=%d/%d]Validation set: Avg_loss=%.4f, Avg_accuracy=%.4f%%, AUC=%.4f' %
              (epoch + 1, args.EPOCHS, valid_avg_loss, valid_acc, auc))
        history_valid.append((valid_avg_loss, valid_acc))
        history_auc.append(auc)

        # 保存最优模型
        best_model_name = os.path.join(args.model_path, 'all_best_model.pkl')
        if auc >= best_auc:
            logger.info('>>>>>>>>>>>>>>Best model is %s' % (str(epoch + 1) + '.pkl'))
            torch.save(model.state_dict(), best_model_name)  # 训练多GPU，测试多GPU
            # torch.save(model.module.state_dict(), best_model_name)  # 训练多GPU，测试单GPU
            best_auc = auc

    logger.info("Train finished!")
    logger.info('Train running time = %s' % str(datetime.datetime.now() - start))
    logger.info('Saving last model...')
    last_model_name = os.path.join(args.model_path, 'all_last_model.pkl')
    torch.save(model.state_dict(), last_model_name)  # 训练多GPU，测试多GPU

    return best_model_name, history_train, history_valid, history_auc



def valid(model, val_loader, criterion):
    with torch.no_grad():
        model.eval()
        val_loss_list = []
        valid_pred = []
        valid_true = []
        valid_prob = np.empty(shape=[0, 2])  # 概率值

        for batch_index, (batch_valid_x, batch_valid_y) in enumerate(val_loader, 0):
            if torch.cuda.is_available():
                batch_valid_x, batch_valid_y = batch_valid_x.cuda(), batch_valid_y.cuda()
            _, output = model(batch_valid_x)
            _, batch_valid_pred = torch.max(output.data, 1)
            prob = F.softmax(output.data, dim=1)
            loss = criterion(output, batch_valid_y)
            val_loss_list.append(loss.item())
            valid_pred = np.hstack((valid_pred, batch_valid_pred.detach().cpu().numpy()))
            valid_true = np.hstack((valid_true, batch_valid_y.detach().cpu().numpy()))
            valid_prob = np.append(valid_prob, prob.detach().cpu().numpy(), axis=0)  # valid_prob=概率列表=[N*2]

        valid_avg_loss = np.mean(val_loss_list)
        valid_acc = 100 * metrics.accuracy_score(valid_true, valid_pred)
        valid_AUC = metrics.roc_auc_score(y_true=valid_true, y_score=valid_prob[:, 1])  # y_score=正例的概率=[N*1]

        tn, fp, fn, tp = metrics.confusion_matrix(valid_true, valid_pred).ravel()
        valid_classification_report = metrics.classification_report(valid_true, valid_pred, digits=4, zero_division=1)

    return valid_pred, valid_true, valid_AUC, valid_acc, valid_avg_loss



def test(model, best_model_name, test_loader, logger):
    logger.info('------ Testing Start ------')
    model.load_state_dict(torch.load(best_model_name), False)
    test_pred = []
    test_true = []
    test_prob = np.empty(shape=[0, 2])  # 概率值

    with torch.no_grad():
        model.eval()
        for test_x, test_y in test_loader:
            if torch.cuda.is_available():
                images, labels = test_x.cuda(), test_y.cuda()
            else:
                images, labels = test_x, test_y
            _, output = model(images)
            _, predicted = torch.max(output.data, 1)
            prob = F.softmax(output.data, dim=1)  # softmax[[0.9,0.1],[0.8,0.2]]
            test_prob = np.append(test_prob, prob.detach().cpu().numpy(), axis=0)
            test_pred = np.hstack((test_pred, predicted.detach().cpu().numpy()))
            test_true = np.hstack((test_true, labels.detach().cpu().numpy()))

    images = test_loader.dataset.test_img
    test_acc = 100 * metrics.accuracy_score(test_true, test_pred)
    test_AUC = metrics.roc_auc_score(y_true=test_true, y_score=test_prob[:, 1])  # y_score=正例的概率
    # test_AUC = metrics.roc_auc_score(y_true=test_true, y_score=test_pred)
    test_classification_report = metrics.classification_report(test_true, test_pred, digits=4, zero_division=1)
    tn, fp, fn, tp = metrics.confusion_matrix(test_true, test_pred).ravel()
    log_message = "test_classification_report\n{}\nAccuracy of the network is: {:.4f}%\nTest_AUC: {:.4f}\nTN={}, FP={}, FN={}, TP={}".format(
        test_classification_report, test_acc, test_AUC, tn, fp, fn, tp)

    logger.info(log_message)
    return test_acc, images, test_true, test_pred


if __name__ == '__main__':
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
    # model = VGG16WithTransformer().to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=6e-3)

    if args.use_weight_loss:
        # 定义类别权重，类别1为少样本类别，类别0为多样本类别
        class_weights = torch.FloatTensor([0.4, 1.3]).to(device)
        criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
    else:
        criterion = torch.nn.CrossEntropyLoss()

    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
        print("Let's use", torch.cuda.device_count(), "GPUs!")

    mkdir(args.model_path)


    if args.is_train:
        best_model_name, history_train, history_valid, history_auc = train(model, args, train_loader, val_loader, optimizer, criterion, logger)
        show_plot(history_train, history_valid, history_auc, args.model_path)
    else:
        best_model_name = os.path.join(args.model_path, 'best_model.pkl')

    test_acc, test_img, test_true, test_pred = test(model, best_model_name, test_loader, logger)

