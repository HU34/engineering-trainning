import torch
import torch.nn as nn
import torchvision.models as models


class LSTMClassifier(nn.Module):
    def __init__(self, input_size, num_classes, hidden_size=256, num_layers=2,):
        super(LSTMClassifier, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out, _ = self.lstm(x)
        # 取最后一个时刻的输出作为分类结果
        out = self.fc(out[:, -1, :])
        return out


class VGG16WithLSTM(nn.Module):
    def __init__(self, args):
        super(VGG16WithLSTM, self).__init__()
        self.args = args
        self.backbone = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
        self.features = self.backbone.features
        self.in_features = 512 * 7 * 7
        # 定义自定义的分类头部
        self.custom_head = LSTMClassifier(self.in_features, self.args.NUM_CLASS)

    def forward(self, x):
        features = self.features(x)
        features = features.view(x.size(0), -1)
        features = features.unsqueeze(dim=1)
        # print(features.shape)
        x = self.custom_head(features)

        return features, x



class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.encoding = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(torch.log(torch.tensor(10000.0)) / d_model))
        self.encoding[:, 0::2] = torch.sin(position * div_term)
        self.encoding[:, 1::2] = torch.cos(position * div_term)
        self.encoding = self.encoding.unsqueeze(0).cuda()

    def forward(self, x):
        return x + self.encoding[:, :x.size(1)]


class VGG16FeatureExtractor(nn.Module):
    def __init__(self, vgg16_model):
        super(VGG16FeatureExtractor, self).__init__()
        self.features = nn.Sequential(*list(vgg16_model.children())[:-2])  # 去掉最后的全连接层

    def forward(self, x):
        return self.features(x)


class GAPTransformerClassifier(nn.Module):
    def __init__(self, d_model, nhead, num_layers, num_classes):
        super(GAPTransformerClassifier, self).__init__()
        self.positional_encoding = PositionalEncoding(d_model)
        self.transformer = nn.Transformer(d_model, nhead, num_layers)
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.activation = nn.ReLU()
        self.classifier = nn.Sequential(
            nn.Linear(d_model, num_classes)
        )

    def forward(self, x):
        x = self.positional_encoding(x)
        x = x.permute(1, 0, 2)  # Transformer需要 (seq_len, batch_size, d_model)
        x = self.transformer(x, x)
        x = x.permute(1, 2, 0)  # (batch_size, d_model, seq_len)
        x = self.global_pool(x).squeeze(-1)  # (batch_size, d_model)
        x = self.activation(x)  # 应用激活函数
        x = self.classifier(x)  # (batch_size, num_classes)
        return x



class VGG16WithTransformer(nn.Module):
    def __init__(self):
        super(VGG16WithTransformer, self).__init__()
        self.backbone = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
        self.vgg16_features = VGG16FeatureExtractor(self.backbone)
        self.transformer_classifier = GAPTransformerClassifier(512, 8, 6, 2)

    def forward(self, x):
        # 提取VGG16特征
        features = self.vgg16_features(x)
        # 特征图的形状是 (batch_size, num_channels, height, width)
        batch_size, num_channels, height, width = features.size()
        # 展平特征图为 (batch_size, num_channels, height * width)
        features = features.view(batch_size, num_channels, -1)
        # 将特征图的空间维度作为序列长度
        features = features.permute(0, 2, 1)  # 变成 (batch_size, seq_len, d_model)
        # 通过Transformer分类头
        outputs = self.transformer_classifier(features)
        return features, outputs



if __name__ == '__main__':

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--NUM_CLASS', type=int, default=2, help='分类类别数')
    args = parser.parse_args()
    model = VGG16WithTransformer().to('cuda:0')
    print(model)
    random_image = torch.randn(1, 3, 224, 224).to('cuda:0')
    _, label = model(random_image)
    print(label.shape)
    print(label)


