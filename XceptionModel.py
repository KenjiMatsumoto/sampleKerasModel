
# coding: utf-8

# In[6]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import os, cv2, zipfile, io, re, glob
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from keras.applications.vgg16 import VGG16
from keras.applications.inception_v3 import InceptionV3
from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.applications.xception import Xception
from keras.models import Model, load_model
from keras.layers.core import Dense
from keras.layers.pooling import GlobalAveragePooling2D
from keras.optimizers import Adam, RMSprop, SGD
from keras.utils.np_utils import to_categorical
from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard, ReduceLROnPlateau
from keras.preprocessing.image import ImageDataGenerator


# In[11]:


z = zipfile.ZipFile('f.zip')
# ラベリングされたディレクトリのみ取得
img_dirs = [ x for x in z.namelist() if re.search("^fruits-360/Training/.*/$", x)]
# 不要な文字列削除
img_dirs = [ x.replace('fruits-360/Training/', '') for x in img_dirs]
img_dirs = [ x.replace('/', '') for x in img_dirs]
img_dirs.sort()

# クラス取得
classes = img_dirs

# クラス数
#num_classes = len(classes)
num_classes = 30

del img_dirs


# In[12]:


# 画像サイズ
image_size = 150

# 画像を取得し、配列に変換
def im2array(path):
    X = []
    y = []
    class_num = 0

    for class_name in classes:
        if class_num == num_classes : break
        imgfiles = [ x for x in z.namelist() if re.search("^" + path + class_name + "/.*jpg$", x)]
        for imgfile in imgfiles:
            # ZIPから画像読み込み
            image = Image.open(io.BytesIO(z.read(imgfile)))
            # RGB変換
            image = image.convert('RGB')
            # リサイズ
            image = image.resize((image_size, image_size))
            # 画像から配列に変換
            data = np.asarray(image)
            X.append(data)
            y.append(classes.index(class_name))
        class_num += 1

    X = np.array(X)
    y = np.array(y)

    return X, y


# In[14]:


X_train, y_train = im2array("fruits-360/Training/")
print(X_train.shape, y_train.shape)
X_test, y_test = im2array("fruits-360/Test/")
print(X_test.shape, y_test.shape)


# In[15]:


del z


# In[16]:


# データ型の変換
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

# 正規化
X_train /= 255
X_test /= 255

# one-hot 変換
y_train = to_categorical(y_train, num_classes = num_classes)
y_test = to_categorical(y_test, num_classes = num_classes)
print(y_train.shape, y_test.shape)

X_train, X_valid, y_train, y_valid = train_test_split(
    X_train,
    y_train,
    random_state = 0,
    stratify = y_train,
    test_size = 0.2
)
print(X_train.shape, y_train.shape, X_valid.shape, y_valid.shape) 


# In[17]:


datagen = ImageDataGenerator(
    featurewise_center = False,
    samplewise_center = False,
    featurewise_std_normalization = False,
    samplewise_std_normalization = False,
    zca_whitening = False,
    rotation_range = 0,
    width_shift_range = 0.1,
    height_shift_range = 0.1,
    horizontal_flip = True,
    vertical_flip = False
)

# EarlyStopping
early_stopping = EarlyStopping(
    monitor = 'val_loss',
    patience = 10,
    verbose = 1
)

# ModelCheckpoint
weights_dir = './weights/'
if os.path.exists(weights_dir) == False:os.mkdir(weights_dir)
model_checkpoint = ModelCheckpoint(
    weights_dir + "val_loss{val_loss:.3f}.hdf5",
    monitor = 'val_loss',
    verbose = 1,
    save_best_only = True,
    save_weights_only = True,
    period = 3
)

# reduce learning rate
reduce_lr = ReduceLROnPlateau(
    monitor = 'val_loss',
    factor = 0.1,
    patience = 3,
    verbose = 1
)

# log for TensorBoard
logging = TensorBoard(log_dir = "log/")


# In[18]:


# モデル学習
def model_fit():
    hist = model.fit_generator(
        datagen.flow(X_train, y_train, batch_size = 32),
        steps_per_epoch = X_train.shape[0] // 32,
        epochs = 50,
        validation_data = (X_valid, y_valid),
        callbacks = [early_stopping, reduce_lr],
        shuffle = True,
        verbose = 1
    )
    return hist

# モデル保存
model_dir = './model/'
if os.path.exists(model_dir) == False : os.mkdir(model_dir)

def model_save(model_name):
    model.save(model_dir + 'model_' + model_name + '.hdf5')

    # optimizerのない軽量モデルを保存（学習や評価不可だが、予測は可能）
    model.save(model_dir + 'model_' + model_name + '-opt.hdf5', include_optimizer = False)

# 学習曲線をプロット
def learning_plot(title):
    plt.figure(figsize = (18,6))

    # accuracy
    plt.subplot(1, 2, 1)
    plt.plot(hist.history["acc"], label = "acc", marker = "o")
    plt.plot(hist.history["val_acc"], label = "val_acc", marker = "o")
    #plt.yticks(np.arange())
    #plt.xticks(np.arange())
    plt.ylabel("accuracy")
    plt.xlabel("epoch")
    plt.title(title)
    plt.legend(loc = "best")
    plt.grid(color = 'gray', alpha = 0.2)

    # loss
    plt.subplot(1, 2, 2)
    plt.plot(hist.history["loss"], label = "loss", marker = "o")
    plt.plot(hist.history["val_loss"], label = "val_loss", marker = "o")
    #plt.yticks(np.arange())
    #plt.xticks(np.arange())
    plt.ylabel("loss")
    plt.xlabel("epoch")
    plt.title(title)
    plt.legend(loc = "best")
    plt.grid(color = 'gray', alpha = 0.2)

    plt.show()

# モデル評価
def model_evaluate():
    score = model.evaluate(X_test, y_test, verbose = 1)
    print("evaluate loss: {[0]:.4f}".format(score))
    print("evaluate acc: {[1]:.1%}".format(score))


# In[19]:


base_model = Xception(include_top = False, weights = "imagenet", input_shape = None)

# 全結合層の新規構築
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation = 'relu')(x)
predictions = Dense(num_classes, activation = 'softmax')(x)

# ネットワーク定義
model = Model(inputs = base_model.input, outputs = predictions)
print("{}層".format(len(model.layers)))


# In[20]:


#108層までfreeze
for layer in model.layers[:108]:
    layer.trainable = False

    # Batch Normalization の freeze解除
    if layer.name.startswith('batch_normalization'):
        layer.trainable = True
    if layer.name.endswith('bn'):
        layer.trainable = True

#109層以降、学習させる
for layer in model.layers[108:]:
    layer.trainable = True

# layer.trainableの設定後にcompile
model.compile(
    optimizer = Adam(),
    loss = 'categorical_crossentropy',
    metrics = ["accuracy"]
)


# In[21]:


hist = model_fit()

