import sys
import os
import cv2
import time
import argparse
import numpy as np
import matplotlib.pyplot as plt
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import glob
from keras.models import Model
from keras import layers, optimizers, callbacks
from keras import backend as K
from keras.applications.vgg16 import VGG16
import tensorflow as tf

def dice_coefficient(y_true, y_pred, smooth=1e-10):
    # y_true: [B, H, W, 1]
    # y_pred: [B, H, W]
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / \
           (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def dice_coefficient_loss(y_true, y_pred):
    return 1 - dice_coefficient(y_true, y_pred)

def dice_coff(label, predict):
    if label.ndim == 4 and label.shape[-1] == 1:
        label = np.squeeze(label, axis=-1)
    return np.sum(2 * label * predict) / \
           (np.sum(label) + np.sum(predict))

def image_process_enhanced(img):
    img = cv2.equalizeHist(img)
    return img

def pad_image(img, target_size):
    """将图像填充到目标大小"""
    h, w = img.shape[:2]
    th, tw = target_size
    pad_h = max(th - h, 0)
    pad_w = max(tw - w, 0)

    top = pad_h // 2
    bottom = pad_h - top
    left = pad_w // 2
    right = pad_w - left

    if len(img.shape) == 3:
        img = cv2.copyMakeBorder(
            img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[0, 0, 0])
    else:
        img = cv2.copyMakeBorder(
            img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=0)

    return img



def load_image(root, data_type, target_size=None, need_name_list=False,
               need_enhanced=False, dataset_type="polyp"):
    if dataset_type == "left_ventricle":
        image_path = os.path.join(root, data_type, "image")
        label_path = os.path.join(root, data_type, "label")
    elif dataset_type == "polyp":
        image_path = os.path.join(root, data_type, "images")
        label_path = os.path.join(root, data_type, "masks")
    elif dataset_type == "all":
        image_path = os.path.join(root, data_type, "image")
        label_path = os.path.join(root, data_type, "label")
    else:
        raise ValueError("Unsupported dataset type: {}".format(dataset_type))

    image_list = []
    label_list = []
    image_name_list = []

    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image path not found: {image_path}")
    if not os.path.exists(label_path):
        raise FileNotFoundError(f"Label path not found: {label_path}")

    image_files = glob.glob(os.path.join(image_path, "**", "*.png"), recursive=True)
    label_files = glob.glob(os.path.join(label_path, "**", "*.png"), recursive=True)

    if len(image_files) != len(label_files):
        raise ValueError("Mismatch between number of images and labels")
    k=0
    for img_file, lbl_file in zip(sorted(image_files), sorted(label_files)):
        if need_name_list:
            image_name_list.append(os.path.basename(img_file))

        img = cv2.imread(img_file)
        label = cv2.imread(lbl_file, cv2.IMREAD_GRAYSCALE)

        if img is None or label is None:
            print(f"Failed to load image or label: {img_file} or {lbl_file}")
            continue

        if target_size is not None:
            img = cv2.resize(img, (target_size[1], target_size[0]),
                             interpolation=cv2.INTER_LINEAR)
            label = cv2.resize(label, (target_size[1], target_size[0]),
                               interpolation=cv2.INTER_NEAREST)

        if need_enhanced:
            img = image_process_enhanced(img)

        img = img / 255.0
        label = label / 255.0

        label = np.expand_dims(label, axis=-1)  # 标签形状为 [H, W, 1]

        image_list.append(img)
        label_list.append(label)
        k+=1
        if(k>=450):
            break

    image_array = np.array(image_list)
    label_array = np.array(label_list)
    if need_name_list:
        return image_array, label_array, image_name_list
    else:
        return image_array, label_array

# def tensorToimg(img):
#     img = np.where(img >= 0.5, 255, 0).astype(np.uint8)
#     return img[:, :]

def tensorToimg(img):
    if img.ndim == 3 and img.shape[-1] == 1:
        img = np.squeeze(img, axis=-1)  # 形状变为 (H, W)
    row, column = img.shape
    for i in range(row):
        for j in range(column):
            if img[i, j] >= 0.75:
                img[i, j] = 255
            elif (img[i, j] >= 0.5) and (img[i, j] < 0.75):
                img[i, j] = 170
            elif (img[i, j] < 0.5) and (img[i, j] >= 0.25):
                img[i, j] = 85
            else:
                img[i, j] = 0
    return img[:, :]

def plot_history(history, result_dir):
    plt.plot(history.history['loss'], marker='.', color='r')
    plt.plot(history.history['val_loss'], marker='*', color='b')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid()
    plt.legend(['Train', 'Validation'], loc='upper right')
    plt.savefig(os.path.join(result_dir, 'model_loss.png'))
    plt.close()

# 定义一个自定义的查询嵌入层
class QueryEmbeddingLayer(layers.Layer):
    def __init__(self, num_queries, embed_dim, **kwargs):
        super(QueryEmbeddingLayer, self).__init__(**kwargs)
        self.num_queries = num_queries
        self.embed_dim = embed_dim

    def build(self, input_shape):
        self.query_embed = self.add_weight(
            shape=(self.num_queries, self.embed_dim),
            initializer='random_normal',
            trainable=True,
            name='query_embed'
        )

    def call(self, inputs):
        batch_size = tf.shape(inputs)[0]
        queries = tf.expand_dims(self.query_embed, axis=0)  # [1, num_queries, embed_dim]
        queries = tf.tile(queries, [batch_size, 1, 1])      # [B, num_queries, embed_dim]
        return queries

def maskformer_segmentation_head(encoder_output, num_classes=1, num_queries=100,
                                 embed_dim=256, num_heads=8, ff_dim=2048, num_layers=6,
                                 input_size=(224, 224)):
    H_e, W_e, C_e = encoder_output.shape[1], encoder_output.shape[2], encoder_output.shape[3]
    encoder_flatten = layers.Reshape((H_e * W_e, C_e))(encoder_output)
    encoder_embedded = layers.Dense(embed_dim)(encoder_flatten)
    # 使用自定义的查询嵌入层
    queries = QueryEmbeddingLayer(num_queries, embed_dim)(encoder_output)
    for _ in range(num_layers):
        attn_output = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)(
            queries, queries)
        queries = layers.Add()([queries, attn_output])
        queries = layers.LayerNormalization()(queries)
        cross_attn_output = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)(
            queries, encoder_embedded)
        queries = layers.Add()([queries, cross_attn_output])
        queries = layers.LayerNormalization()(queries)
        ffn_output = layers.Dense(ff_dim, activation='relu')(queries)
        ffn_output = layers.Dense(embed_dim)(ffn_output)
        queries = layers.Add()([queries, ffn_output])
        queries = layers.LayerNormalization()(queries)
    class_logits = layers.Dense(num_classes + 1, name='class_logits')(queries)
    mask_embed = layers.Dense(C_e)(queries)
    encoder_output_reshaped = layers.Reshape((H_e * W_e, C_e))(encoder_output)
    mask_logits = tf.einsum('bqc,bkc->bqk', mask_embed, encoder_output_reshaped)
    # 调整形状以匹配 [B, H_e, W_e, num_queries]
    mask_logits = tf.reshape(mask_logits, [-1, num_queries, H_e, W_e])  # [B, num_queries, H_e, W_e]
    mask_logits = tf.transpose(mask_logits, perm=[0, 2, 3, 1])  # [B, H_e, W_e, num_queries]
    # 上采样到输入大小
    mask_logits_upsampled = tf.image.resize(mask_logits, size=(input_size[0], input_size[1]), method='bilinear')
    # 输出形状为 [B, H, W, num_queries]
    return class_logits, mask_logits_upsampled

def VGG16_maskformer_model(input_size=(224, 224, 3), num_classes=1, if_transfer=True, if_local=True):
    weights = None
    model_path = os.path.join(sys.path[0], 'models', 'vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5')
    if if_transfer:
        if if_local and os.path.exists(model_path):
            weights = model_path
            print(f"Using local weights file: {weights}")
        else:
            weights = 'imagenet'
    vgg16 = VGG16(include_top=False, weights=weights, input_shape=input_size)
    for layer in vgg16.layers:
        layer.trainable = False
    encoder_output = vgg16.output
    class_logits, mask_logits = maskformer_segmentation_head(encoder_output, num_classes=num_classes, input_size=input_size[:2])
    model = Model(inputs=vgg16.input, outputs={'class_logits': class_logits, 'mask_logits': mask_logits})
    return model

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', default="./dataset1",
                        required=False, help='path to dataset')
    parser.add_argument('--img_enhanced', default=False, help='image enhancement')
    parser.add_argument('--batch-size', type=int, default=8, help='input batch size')
    parser.add_argument('--image-size', default=(224, 224, 3),
                        help='the (height, width, channel) of the input image to network')
    parser.add_argument('--niter', type=int, default=30, help='number of epochs to train for')
    parser.add_argument('--lr', type=float, default=0.01,
                        help='learning rate, default=0.0001')
    parser.add_argument('--model-save',
                        default='./models/level3alltransfer_model.h5',
                        help='folder to output model checkpoints')
    parser.add_argument('--model-path',
                        default='./models/level3alltransfer_model.h5',
                        help='folder of model checkpoints to predict')
    parser.add_argument('--outf', default="test",
                        required=False, help='path of predict output')
    args = parser.parse_args(args=[])
    try:
        os.makedirs(args.outf)
    except OSError:
        pass

    return args

def class_loss(y_true, y_pred):
    # y_pred: [B, num_queries, num_classes + 1]
    # 假设所有的 query 都对应前景（类别为 1）
    batch_size = tf.shape(y_pred)[0]
    num_queries = tf.shape(y_pred)[1]
    class_targets = tf.ones((batch_size, num_queries), dtype=tf.int32)
    return tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)(
        class_targets, y_pred)

def mask_loss(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)  # [B, H, W, 1]
    y_pred = tf.cast(y_pred, tf.float32)  # [B, H, W, num_queries]
    # 将 y_true 扩展到 y_pred 的形状
    y_true_expanded = tf.broadcast_to(y_true, tf.shape(y_pred))  # [B, H, W, num_queries]
    loss = tf.keras.losses.binary_crossentropy(y_true_expanded, y_pred, from_logits=True)
    return tf.reduce_mean(loss)

def train_level3():
    args = get_parser()
    train, train_label = load_image(root=args.data_root, data_type="train",
                                    need_enhanced=args.img_enhanced,
                                    target_size=(args.image_size[0], args.image_size[1]),
                                    dataset_type="all")
    val, val_label = load_image(root=args.data_root, data_type="val",
                                need_enhanced=args.img_enhanced,
                                target_size=(args.image_size[0], args.image_size[1]),
                                dataset_type="all")

    model = VGG16_maskformer_model(input_size=args.image_size,
                                   num_classes=1, if_transfer=True, if_local=True)
    optimizer = optimizers.Adam(learning_rate=args.lr)
    model.compile(optimizer=optimizer,
                  loss={'class_logits': class_loss, 'mask_logits': mask_loss})

    # 为 'class_logits' 提供虚拟的标签
    num_queries = 100  # 请确保与模型中使用的 num_queries 一致
    dummy_class_targets = np.ones((train.shape[0], num_queries), dtype=np.int32)
    dummy_val_class_targets = np.ones((val.shape[0], num_queries), dtype=np.int32)

    model_checkpoint = callbacks.ModelCheckpoint(args.model_path, monitor='loss',
                                                 verbose=1, save_best_only=True)
    history = model.fit(train, {'class_logits': dummy_class_targets, 'mask_logits': train_label},
                        batch_size=args.batch_size,
                        epochs=args.niter, callbacks=[model_checkpoint],
                        validation_data=(val, {'class_logits': dummy_val_class_targets, 'mask_logits': val_label}))
    plot_history(history, args.outf)

def predict_level3():
    args = get_parser()
    test_img, test_label, test_name_list = load_image(args.data_root, "test",
                                                      need_name_list=True,
                                                      need_enhanced=args.img_enhanced,
                                                      target_size=(args.image_size[0], args.image_size[1]),
                                                      dataset_type="all")
    model = VGG16_maskformer_model(input_size=args.image_size,
                                   num_classes=1, if_transfer=True, if_local=True)
    model.load_weights(args.model_path)
    outputs = model.predict(test_img)
    class_logits = outputs['class_logits']
    mask_logits = outputs['mask_logits']

    # 后处理
    mask_probs = tf.sigmoid(mask_logits)  # [B, H, W, num_queries]
    class_probs = tf.nn.softmax(class_logits, axis=-1)[..., 1]  # [B, num_queries]
    best_mask_indices = tf.argmax(class_probs, axis=-1)  # [B]
    final_masks = []
    for i in range(mask_probs.shape[0]):
        best_mask = mask_probs[i, :, :, best_mask_indices[i]]
        final_masks.append(best_mask.numpy())
    final_masks = np.array(final_masks)  # [B, H, W]

    dc = dice_coff(test_label, final_masks)

    print("The dice coefficient is: " + str(dc))

    # 计算其他指标
    pixel_accuracy(test_label, final_masks)
    compute_mIoU(test_label, final_masks)

    for i in range(final_masks.shape[0]):
        final_img = tensorToimg(final_masks[i])
        ori_img = test_img[i]
        ori_gt = tensorToimg(test_label[i])

        plt.figure(figsize=(6, 2))
        plt.subplot(1, 3, 1)
        plt.imshow(ori_img)
        plt.axis('off')
        plt.title("Original Image")
        plt.subplot(1, 3, 2)
        plt.imshow(ori_gt, cmap='gray')
        plt.axis('off')
        plt.title("Ground Truth")
        plt.subplot(1, 3, 3)
        plt.imshow(final_img, cmap='gray')
        plt.axis('off')
        plt.title("Prediction")
        plt.savefig(f"{args.outf}/{test_name_list[i]}")
        print(f"Save: {args.outf}/{test_name_list[i]}")
        plt.close()

def pixel_accuracy(label, predict):
    predict = np.where(predict >= 0.5, 1, 0)
    label = np.where(label >= 0.5, 1, 0)
    correct = np.sum(predict == label)
    total = np.prod(label.shape)
    pa = correct / total
    print("The pixel accuracy is: " + str(pa))
    return pa

def compute_mIoU(label, predict):
    if label.ndim == 4 and label.shape[-1] == 1:
        label = np.squeeze(label, axis=-1)
    predict = np.where(predict >= 0.5, 1, 0)
    label = np.where(label >= 0.5, 1, 0)
    intersection = np.logical_and(label, predict)
    union = np.logical_or(label, predict)
    iou = np.sum(intersection) / np.sum(union)
    print("The mIoU is: " + str(iou))
    return iou

if __name__ == "__main__":
    s_t = time.time()
    #train_level3()
    predict_level3()
    print("Time elapsed:", time.time() - s_t)





