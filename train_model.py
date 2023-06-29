import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import recall_score
# 数据加载，按照8:2的比例加载花卉数据

def data_load(data_dir, img_height, img_width, batch_size):
    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        data_dir,
        label_mode='categorical',
        validation_split=0.2,
        subset="training",
        seed=123,
        image_size=(img_height, img_width),
        batch_size=batch_size)

    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        data_dir,
        label_mode='categorical',
        validation_split=0.2,
        subset="validation",
        seed=123,
        image_size=(img_height, img_width),
        batch_size=batch_size)

    # 获取训练集中的类别名称
    class_names = train_ds.class_names
    # 返回训练集、验证集和类别名称，供后续的模型训练使用。
    return train_ds, val_ds, class_names


# 模型加载，指定图片处理的大小和是否进行迁移学习
# 这个函数主要是用于加载指定的模型，包括卷积神经网络模型CNN和迁移学习模型，并进行模型的编译和返回。
# # 指定输入图像的形状，即高度、宽度和通道数
# # is_transfer：指定是否使用迁移学习模型。如果为 True，则加载 MobileNetV2 模型，并添加全局平均池化层和全连接层进行微调；否则，加载一个卷积神经网络模型。
def model_load(IMG_SHAPE=(224, 224, 3), is_transfer=False):
    if is_transfer:
        # 如果使用迁移学习模型，将在此处加载 MobileNetV2 模型，并设置其不可训练，即使用预训练权重。
        base_model = tf.keras.applications.MobileNetV2(input_shape=IMG_SHAPE,
                                                       include_top=False,
                                                       weights='imagenet')
        base_model.trainable = False
        model = tf.keras.models.Sequential([
            tf.keras.layers.experimental.preprocessing.Rescaling(1. / 127.5, offset=-1, input_shape=IMG_SHAPE),
            base_model,
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dense(5, activation='softmax')
        ])
    else:
        # 加载一个卷积神经网络模型，其结构包括一个归一化层、两个卷积层、两个池化层和两个全连接层。
        model = tf.keras.models.Sequential([
            tf.keras.layers.experimental.preprocessing.Rescaling(1. / 255, input_shape=IMG_SHAPE),
            tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
            tf.keras.layers.MaxPooling2D(2, 2),
            # Add another convolution
            tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
            tf.keras.layers.MaxPooling2D(2, 2),
            # Now flatten the output. After this you'll just have the same DNN structure as the non convolutional version
            tf.keras.layers.Flatten(),
            # The same 128 dense layers, and 10 output layers as in the pre-convolution example:
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(5, activation='softmax')
        ])
    # 用于打印模型的结构和参数数量
    model.summary()
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    # 返回加载并编译好的模型，供后续的训练和预测使用。
    return model


# 展示训练过程的曲线  用于展示模型训练过程中的损失和准确率变化情况。
# 在训练神经网络时，通常会将数据集分成训练集和验证集，用训练集来训练模型，用验证集来评估模型的性能
def show_loss_acc(history):
    # 训练集上的准确率
    acc = history.history['accuracy']
    # 验证集上的准确率
    val_acc = history.history['val_accuracy']

    loss = history.history['loss']
    val_loss = history.history['val_loss']

    plt.figure(figsize=(8, 8))
    plt.subplot(2, 1, 1)
    plt.plot(acc, label='Training Accuracy')
    plt.plot(val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.ylabel('Accuracy')
    plt.ylim([min(plt.ylim()), 1])
    plt.title('Training and Validation Accuracy')

    plt.subplot(2, 1, 2)
    plt.plot(loss, label='Training Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.ylabel('Cross Entropy')
    plt.ylim([0, 1.0])
    plt.title('Training and Validation Loss')
    plt.xlabel('epoch')
    plt.show()


# 用于训练并保存神经网络模型 ,在训练过程中，模型会遍历整个数据集多次，不断调整参数以提高模型的性能。
# 指定训练的 epoch 数量，即模型需要遍历整个数据集的次数
def train(epochs, is_transfer=False):
    train_ds, val_ds, class_names = data_load("./flower_photos", 224, 224, 64)
    # 调用 model_load() 函数加载训练好的神经网络模型。
    model = model_load(is_transfer=is_transfer)
    # 用于训练模型，在训练过程中，将训练集和验证集输入模型进行训练和评估，得到每个 epoch 的训练和验证指标值，并记录在 history 对象中。
    history = model.fit(train_ds, validation_data=val_ds, epochs=epochs)
    if is_transfer:
        # model.evaluate(val_ds)
        model.save("models/mobilenet_flower.h5")
    else:
        model.save("models/cnn_flower.h5")

    # 评估模型在训练集上的分类效果
    y_train_true = []
    y_train_pred = []
    for x, y_true in train_ds:
        y_pred = model.predict(x)
        y_pred = tf.argmax(y_pred, axis=1).numpy()
        y_true = tf.argmax(y_true, axis=1).numpy()
        y_train_true.extend(y_true)
        y_train_pred.extend(y_pred)
    train_cm = confusion_matrix(y_train_true, y_train_pred)
    print("Train Confusion Matrix:")
    print(train_cm)

    # 评估模型在验证集上的分类效果
    y_val_true = []
    y_val_pred = []
    for x, y_true in val_ds:
        y_pred = model.predict(x)
        y_pred = tf.argmax(y_pred, axis=1).numpy()
        y_true = tf.argmax(y_true, axis=1).numpy()
        y_val_true.extend(y_true)
        y_val_pred.extend(y_pred)
    val_cm = confusion_matrix(y_val_true, y_val_pred)
    print("Validation Confusion Matrix:")
    print(val_cm)

    # 打印分类报告
    print("训练集分类报告:")
    print(classification_report(y_train_true, y_train_pred, target_names=class_names))
    print("验证集分类报告:")
    print(classification_report(y_val_true, y_val_pred, target_names=class_names))

    # 计算召回率
    train_recall = recall_score(y_train_true, y_train_pred, average=None)
    val_recall = recall_score(y_val_true, y_val_pred, average=None)

    # 打印召回率
    print("Train Recall:")
    for i, class_name in enumerate(class_names):
        print(f"{class_name}: {train_recall[i]}")
    print("Validation Recall:")
    for i, class_name in enumerate(class_names):
        print(f"{class_name}: {val_recall[i]}")

    show_loss_acc(history)

if __name__ == '__main__':
    train(epochs=20, is_transfer=True)
