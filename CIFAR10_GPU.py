import cv2
import sys
import pickle
import time
import argparse
import numpy as np
from tqdm import tqdm
import chainer
from chainer import cuda, Function, gradient_check, Variable, optimizers, serializers, utils
from chainer import Link, Chain, ChainList, FunctionSet
import chainer.functions as F
import chainer.links as L
import matplotlib.pyplot as plt
from sklearn.cross_validation import train_test_split


parser = argparse.ArgumentParser(description='Chainer example: MNIST')
parser.add_argument('--gpu', '-gpu', default=-1, type=int,
                    help='GPU ID (negative value indicates CPU)')


# GPUが使えるか確認
args = parser.parse_args()
if args.gpu >= 0:
    cuda.check_cuda_available()
xp = cuda.cupy if args.gpu >= 0 else np


def unpickle(file):
    fp = open(file, 'rb')
    if sys.version_info.major == 2:
        data = pickle.load(fp)
    elif sys.version_info.major == 3:
        data = pickle.load(fp, encoding='latin-1')
    fp.close()
    return data


X_train = None
y_train = []

for i in range(1,6):
    data_dic = unpickle("cifar-10-batches-py/data_batch_{}".format(i))
    if i == 1:
        X_train = data_dic['data']
    else:
        X_train = np.vstack((X_train, data_dic['data']))
    y_train += data_dic['labels']


test_data_dic = unpickle("cifar-10-batches-py/test_batch")
X_test = test_data_dic['data']
X_test = X_test.reshape(len(X_test),3,32,32)
y_test = np.array(test_data_dic['labels'])
X_train = X_train.reshape((len(X_train),3, 32, 32))

#--- y_train ---#
temp = y_train
for i in range(7):
    y_train = np.r_[y_train, temp]

#--- X_train ---#

# ガンマ変換（1.5・0.75）
gamma = 1.5
look_up_table = np.ones((256, 1), dtype = 'uint8' ) * 0
for i in range(256):
    look_up_table[i][0] = 255 * pow(float(i) / 255, 1.0 / gamma)
X = []
for i in range(50000):
    img = X_train[i].transpose(1, 2, 0)
    img_gamma = cv2.LUT(img, look_up_table)
    X.append(img_gamma.transpose(2,0,1))

X = np.array(X)
X_train = np.vstack((X_train, X))

gamma = 0.75
look_up_table = np.ones((256, 1), dtype = 'uint8' ) * 0
for i in range(256):
    look_up_table[i][0] = 255 * pow(float(i) / 255, 1.0 / gamma)
X = []
for i in range(50000):
    img = X_train[i].transpose(1, 2, 0)
    img_gamma = cv2.LUT(img, look_up_table)
    X.append(img_gamma.transpose(2,0,1))

X = np.array(X)
X_train = np.vstack((X_train, X))


# 平滑化
average_square = (10,10)
X = []
for i in range(50000):
    img = X_train[i].transpose(1, 2, 0)
    blur_img = cv2.blur(img, average_square)
    X.append(blur_img.transpose(2,0,1))

X = np.array(X)
X_train = np.vstack((X_train, X))


# コントラスト
min_table = 50
max_table = 205
diff_table = max_table - min_table

LUT_HC = np.arange(256, dtype = 'uint8' )
LUT_LC = np.arange(256, dtype = 'uint8' )

for i in range(0, min_table):
    LUT_HC[i] = 0
for i in range(min_table, max_table):
    LUT_HC[i] = 255 * (i - min_table) / diff_table
for i in range(max_table, 255):
    LUT_HC[i] = 255
for i in range(256):
    LUT_LC[i] = min_table + i * (diff_table) / 255
X = []
for i in range(50000):
    img = X_train[i].transpose(1, 2, 0)
    high_cont_img = cv2.LUT(img, LUT_HC)
    X.append(high_cont_img.transpose(2,0,1))

X = np.array(X)
X_train = np.vstack((X_train, X))

X = []
for i in range(50000):
    img = X_train[i].transpose(1, 2, 0)
    low_cont_img = cv2.LUT(img, LUT_LC)
    X.append(low_cont_img.transpose(2,0,1))

X = np.array(X)
X_train = np.vstack((X_train, X))

# ガウシアンノイズ
X = []
for i in range(50000):
    img = X_train[i].transpose(1, 2, 0)
    row,col,ch= img.shape
    mean = 0
    sigma = 15
    gauss = np.random.normal(mean,sigma,(row,col,ch))
    gauss = gauss.reshape(row,col,ch)
    gauss_img = img + gauss
    X.append(gauss_img.transpose(2,0,1))

X = np.array(X)
X_train = np.vstack((X_train, X))

# Salt&Pepperノイズ
X = []
for i in range(50000):
    img = X_train[i].reshape(3, 32, 32).transpose(1, 2, 0)
    row,col,ch= img.shape
    s_vs_p = 0.5
    amount = 0.004
    sp_img = img.copy()
    num_salt = np.ceil(amount * img.size * s_vs_p)
    coords = [np.random.randint(0, i-1 , int(num_salt)) for i in img.shape]
    sp_img[coords[:-1]] = (255,255,255)
    num_salt = np.ceil(amount * img.size * s_vs_p)
    coords = [np.random.randint(0, i-1 , int(num_salt)) for i in img.shape]
    sp_img[coords[:-1]] = (0,0,0)
    X.append(sp_img.transpose(2,0,1))

X = np.array(X)
X_train = np.vstack((X_train, X))

y_train = np.array(y_train)
X_train = X_train.astype(np.float32)
X_test = X_test.astype(np.float32)
X_train /= 255
X_test /= 255
y_train = y_train.astype(np.int32)
y_test = y_test.astype(np.int32)

X_train, temp1, y_train, temp2 = train_test_split(X_train, y_train, train_size=0.9, random_state=0)


# 畳み込み6層
model = chainer.FunctionSet(conv1=F.Convolution2D(3, 32, 3, pad=1),
                            conv2=F.Convolution2D(32, 32, 3, pad=1),
                            conv3=F.Convolution2D(32, 32, 3, pad=1),
                            conv4=F.Convolution2D(32, 32, 3, pad=1),
                            conv5=F.Convolution2D(32, 32, 3, pad=1),
                            conv6=F.Convolution2D(32, 32, 3, pad=1),
                            l1=F.Linear(512, 512),
                            l2=F.Linear(512, 10))


# GPU使用のときはGPUにモデルを転送
if args.gpu >= 0:
    cuda.get_device(args.gpu).use()
    model.to_gpu()

def forward(x_data, y_data, train=True):
    x, t = chainer.Variable(x_data), chainer.Variable(y_data)
    h = F.relu(model.conv1(x))
    h = F.max_pooling_2d(F.relu(model.conv2(h)), 2)
    h = F.relu(model.conv3(h))
    h = F.max_pooling_2d(F.relu(model.conv4(h)), 2)
    h = F.relu(model.conv5(h))
    h = F.max_pooling_2d(F.relu(model.conv6(h)), 2)
    h = F.dropout(F.relu(model.l1(h)), train=train)
    y = model.l2(h)
    
    return F.softmax_cross_entropy(y, t), F.accuracy(y, t)

optimizer = optimizers.Adam()
optimizer.setup(model)

train_loss = []
train_acc  = []
test_loss = []
test_acc  = []
N = 360000
N_test = 10000
batch_size = 100

start_time = time.clock()
for epoch in range(40):
    print("epoch", epoch+1)
    
    perm = np.random.permutation(N)
    sum_accuracy = 0
    sum_loss = 0
    
    for i in tqdm(range(0, N, batch_size)):
        X_batch = xp.asarray(X_train[perm[i:i+batch_size]])
        y_batch = xp.asarray(y_train[perm[i:i+batch_size]])
        
        optimizer.zero_grads()
        loss, acc = forward(X_batch, y_batch)
        loss.backward()
        optimizer.update()

        train_loss.append(loss.data)
        train_acc.append(acc.data)
        sum_loss     += float(loss.data) * batch_size
        sum_accuracy += float(acc.data) * batch_size

    print("train mean loss={}, accuracy={}".format(sum_loss/N, sum_accuracy/N))

    sum_accuracy = 0
    sum_loss = 0
    for i in tqdm(range(0, N_test, batch_size)):
        X_batch = xp.asarray(X_train[perm[i:i+batch_size]])
        y_batch = xp.asarray(y_train[perm[i:i+batch_size]])
        
        loss, acc = forward(X_batch, y_batch)
       
        train_loss.append(loss.data)
        train_acc.append(acc.data)
        sum_loss     += float(loss.data) * batch_size
        sum_accuracy += float(acc.data) * batch_size
        
   
    print("test mean loss={}, accuracy={}".format(sum_loss/N_test, sum_accuracy/N_test))

end_time = time.clock()
print("total time : ", end_time - start_time)
