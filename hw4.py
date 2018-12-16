import numpy as np
import matplotlib.pyplot as plt
import csv
import os

import re
import jieba
import emoji
from gensim.models import Word2Vec


name = 'test6'
wordvec_name = 'cut3'
file_train_x = 'train_x.csv'
file_train_y = 'train_y.csv'
file_test_x = 'test_x.csv'
jieba.set_dictionary('dict.txt.big')
wv_size = 512
wv_len = 64
cut_file = '/home/tmp/' + wordvec_name
word2vec_file = '/home/tmp/wordvec2' + wordvec_name + '.model'
showplt = True
file_out = name + '.csv'


def readTrain(file_x=file_train_x, file_y=file_train_y):
    label = []

    ID, X = readTest(file_x)
    with open(file_y) as csvfile:
        reader = csv.reader(csvfile)
        next(reader)
        label = np.array(np.vstack(reader), dtype=np.int)

    assert(all(ID == np.arange(120000)))
    assert(all(label[:, 0] == np.arange(120000)))
    Y = label[:, 1]
    return X, Y


def readTest(file_x=file_test_x):
    print("Read")
    ID = []
    X = []

    with open(file_x) as csvfile:
        reader = csv.reader(csvfile)
        next(reader)
        for r in reader:
            ID.append(r[0])
            X.append(r[1])

    ID = np.array(ID, dtype=np.int)
    return ID, X


# dcard special word
jieba.add_word('原po')
jieba.add_word('卡稱')
jieba.add_word('xd')

floor_b = re.compile(r'b\d+')
space_large = re.compile(r'\s+')
word_regex = re.compile(r'\w+')


def myCut(x):
    x = x.lower()
    # x = floor_b.sub('b1', x)
    # x = x.replace('，', '')
    # x = x.replace(' ', '')
    # x = x.replace('= =', '==')
    # x = space_large.sub('，', x)
    cuts = jieba.cut(x)
    return [cut for cut in cuts if cut]


def wordVectorTrain():
    trainX, _ = readTrain()
    _, testX = readTest()
    print("Cut")
    cut_train = [list(myCut(x)) for x in trainX]
    cut_test  = [list(myCut(x)) for x in testX]
    np.savez(cut_file + '_train.npz', cut=cut_train)
    np.savez(cut_file + '_test.npz', cut=cut_test)
    cutX = []
    cutX.extend(cut_train)
    cutX.extend(cut_test)

    print("WordVector Train")
    model = Word2Vec(cutX, size=wv_size, window=10, min_count=1, workers=10)
    model.train(cutX, total_examples=len(cutX), epochs=4)
    model.save(word2vec_file)


def vectorCut(cut_filename, dataX=None):
    print("Transfer to WordVector")
    model = Word2Vec.load(word2vec_file)
    if os.path.exists(cut_file + cut_filename):
        cut_ori = np.load(cut_file + cut_filename)['cut']
    else:
        print("Cut")
        cut_ori = np.array([list(myCut(x)) for x in dataX])
    cut = np.zeros([len(cut_ori), wv_len, wv_size])

    for i, comment in enumerate(cut_ori):
        if not len(comment):
            continue
        now = model.wv[comment[:wv_len]]
        cut[i, :len(now)] = now
    return cut


def showPlot(history):
    plt.figure(figsize=(16, 6), dpi=100)
    plt.subplot(121)
    plt.plot(history.history['loss'], label='loss')
    plt.plot(history.history['val_loss'], label='label_loss')
    plt.legend()
    plt.subplot(122)
    plt.plot(history.history['acc'], label='acc')
    plt.plot(history.history['val_acc'], label='label_acc')
    plt.legend()
    plt.savefig(name + '.png')
    plt.show()


def initGPU():
    import tensorflow as tf
    from keras import backend as K
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    session = tf.Session(config=config)
    K.set_session(session)


def Train():
    from keras.models import Sequential
    from keras.layers import Dense, Dropout, Flatten, LeakyReLU, LSTM
    from keras.optimizers import Adam
    from keras.callbacks import EarlyStopping, ModelCheckpoint
    from keras.layers.normalization import BatchNormalization

    initGPU()
    # wordVectorTrain()
    model = Sequential()
    model.add(LSTM(64, dropout=0.3, recurrent_dropout=0.3, input_shape=[wv_len, wv_size]))
    model.add(LeakyReLU())
    model.add(BatchNormalization())
    model.add(Dropout(0.35))

    model.add(Dense(128))
    model.add(LeakyReLU())
    model.add(BatchNormalization())
    model.add(Dropout(0.4))

    model.add(Dense(128))
    model.add(LeakyReLU())
    model.add(BatchNormalization())
    model.add(Dropout(0.45))

    model.add(Dense(1, activation='sigmoid'))
    model.summary()
    model.compile(loss='binary_crossentropy',
                  optimizer=Adam(),
                  metrics=['accuracy'])

    dataX, dataY = readTrain()
    cut_train = vectorCut('_train.npz')

    val_len = len(dataY) // 12
    trainX = cut_train[:-val_len]
    trainY = dataY[:-val_len]
    validX = cut_train[-val_len:]
    validY = dataY[-val_len:]

    print("Train")
    history = model.fit(trainX, trainY, validation_data=(validX, validY),
                        callbacks=[EarlyStopping(monitor='val_acc', patience=8),  # val_loss is always upward
                                   ModelCheckpoint(name + '.h5', save_best_only=True, monitor='val_acc')],
                        epochs=64, batch_size=256)
    # model.save('.h5')

    if showplt:
        showPlot(history)


def Test():
    from keras.models import load_model
    initGPU()
    ID, testX = readTest()
    cut_test = vectorCut('_test.npz', testX)

    model = load_model(name + '.h5')
    pred = model.predict(cut_test) >= 0.5

    print("output to csv")
    outputfile = open(file_out, 'w')
    print('id,label', file=outputfile)
    for i, x in enumerate(pred):
        print('{},{}'.format(ID[i], int(x)), file=outputfile)


wordVectorTrain()
Train()
Test()
