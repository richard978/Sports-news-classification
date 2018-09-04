import time
import pandas as pd
import numpy as np
import jieba
import random
from matplotlib import pyplot as plt
from gensim import models
from gensim.models import Word2Vec
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn import preprocessing
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.utils import plot_model
from keras import layers
from keras.models import Sequential
from keras.models import load_model
from sklearn.cluster import KMeans
from collections import Counter

stopList = []
dict = ['basketball','badminton','billiard','boxing','field','football','gym','hockey','horse','pingpong'
            ,'racing','shooting','tennis','volleyball']

def readStop():
    global stopList
    file = open('stopWords.txt', encoding='GB2312', errors='ignore')
    lines = file.readlines()
    for line in lines:
        stopList.append(line.strip())

def is_Chinese(word):
    for ch in word:
        if '\u4e00' <= ch <= '\u9fff' or 'a' <= ch <= 'z' or 'A' <= ch <= 'Z':
            return True
    return False

def splitWords(seq):
    seq = seq.replace('\'', '').replace(' ', '')
    return seq[1:len(seq)-1].split(',')

def readContent(cat):
    df_u8 = pd.read_csv('contents/'+cat+'_u8.csv', encoding="utf_8_sig")
    df_gb = pd.read_csv('contents/'+cat+'_gb.csv', encoding="GB2312")
    wordList = []
    for i in range(len(df_gb['content'])):
        tempStr = str(df_gb['title'][i]).strip() + ' ' + str(df_gb['content'][i]).strip()
        segList = jieba.lcut(tempStr.encode('utf-8', errors='ignore') , cut_all=False)
        segList = [word for word in list(segList) if word not in stopList and word.strip() != '' and is_Chinese(word)]
        wordList.append(segList)

    for i in range(len(df_u8['content'])):
        segList = jieba.lcut(str(df_u8['title'][i]).strip()+' '+str(df_u8['content'][i]).strip(), cut_all=False)
        segList = [word for word in list(segList) if word not in stopList and word.strip() != '' and is_Chinese(word)]
        wordList.append(segList)

    total_df = pd.DataFrame(columns=["label", "content"])
    total_df['content'] = wordList
    total_df['label'][:] = cat
    total_df.to_csv(cat+'.csv', index=None, encoding = "utf_8_sig")

def classification_lr():
    '''
    total_df = pd.DataFrame(columns=["label", "content"])
    for d in dict:
        df = pd.read_csv(d+'.csv', encoding="utf_8_sig")
        total_df = total_df.append(df, ignore_index=True)
    total_df.drop(32061,inplace=True)
    total_df = total_df.reset_index(drop=True)
    total_df.to_csv('tfidf_feature.csv', index=None, encoding="utf_8_sig")
    '''
    df = pd.read_csv('train_set.csv', encoding="utf_8_sig")

    encoder = preprocessing.LabelEncoder()
    y = encoder.fit_transform(df['label'])

    df['content'] = df['content'].apply(lambda x:x[1:len(x)-1].replace(',', '').replace('\'', ''))

    vectorizer = CountVectorizer(min_df=1e-5)
    transformer = TfidfTransformer()
    tfidf = transformer.fit_transform(vectorizer.fit_transform(df['content']))
    X = tfidf

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)
    X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=0.7, random_state=1)

    lr_model = LogisticRegression()
    start = time.time()
    lr_model.fit(X_train, y_train)
    end = time.time()
    print("Training time: {0} seconds".format(end-start))
    print("Validate mean accuracy: {0}".format(lr_model.score(X_val, y_val)))
    y_pred = lr_model.predict(X_test)
    print(classification_report(y_test, y_pred, target_names=dict))

def classification_svm():
    df = pd.read_csv('train_set.csv', encoding="utf_8_sig")

    encoder = preprocessing.LabelEncoder()
    y = encoder.fit_transform(df['label'])

    df['content'] = df['content'].apply(lambda x:x[1:len(x)-1].replace(',', '').replace('\'', ''))

    vectorizer = CountVectorizer(min_df=1e-5)
    transformer = TfidfTransformer()
    tfidf = transformer.fit_transform(vectorizer.fit_transform(df['content']))

    X = tfidf

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)
    X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=0.7, random_state=1)

    clf = svm.SVC(kernel='linear')
    start = time.time()
    clf.fit(X_train, y_train)
    end = time.time()
    print("Training time: {0} seconds".format(end - start))
    print("Validate mean accuracy: {0}".format(clf.score(X_val, y_val)))
    y_pred = clf.predict(X_test)
    print(classification_report(y_test, y_pred, target_names=dict))

def classification_nb():
    df = pd.read_csv('train_set.csv', encoding="utf_8_sig")
    encoder = preprocessing.LabelEncoder()
    y = encoder.fit_transform(df['label'])
    vectorizer = CountVectorizer(min_df=1e-5)
    transformer = TfidfTransformer()
    tfidf = transformer.fit_transform(vectorizer.fit_transform(df['content']))
    X = tfidf
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)
    X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=0.7, random_state=1)

    clf = MultinomialNB()
    start = time.time()
    clf.fit(X_train, y_train)
    end = time.time()
    print("Training time: {0} seconds".format(end - start))
    print("Validate mean accuracy: {0}".format(clf.score(X_val, y_val)))
    y_pred = clf.predict(X_test)
    print(classification_report(y_test, y_pred, target_names=dict))

def classification_rf():
    df = pd.read_csv('train_set.csv', encoding="utf_8_sig")

    encoder = preprocessing.LabelEncoder()
    y = encoder.fit_transform(df['label'])

    vectorizer = CountVectorizer(min_df=1e-5)
    transformer = TfidfTransformer()
    tfidf = transformer.fit_transform(vectorizer.fit_transform(df['content']))
    X = tfidf

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)
    X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=0.7, random_state=1)

    clf = RandomForestClassifier(oob_score=True, random_state=10)
    start = time.time()
    clf.fit(X_train, y_train)
    end = time.time()
    print("Training time: {0} seconds".format(end - start))
    print("Validate mean accuracy: {0}".format(clf.score(X_val, y_val)))
    y_pred = clf.predict(X_test)
    print(classification_report(y_test, y_pred, target_names=dict))

def train_process():
    df = pd.read_csv('train_set.csv', encoding="utf_8_sig")
    all_text = []
    le = preprocessing.LabelEncoder()
    all_labels = le.fit_transform(df['label'])
    labels = to_categorical(np.asarray(all_labels))

    for d in df['content']:
        all_text.append(splitWords(d)[0:200])
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(all_text)
    sequences = tokenizer.texts_to_sequences(all_text)
    word_index = tokenizer.word_index
    text_data = pad_sequences(sequences, maxlen=200)

    #train_cnn(word_index, text_data, labels)
    #train_lstm(word_index, text_data, labels)
    #train_cnn_w2v(word_index, text_data, labels)
    train_lstm_w2v(word_index, text_data, labels)

def train_cnn(word_index, text_data, labels):

    X_train, X_test, y_train, y_test = train_test_split(text_data, labels, test_size=0.3, random_state=1)
    X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=0.7, random_state=1)

    model = Sequential()
    model.add(layers.Embedding(len(word_index)+1, 200, input_length=200))
    model.add(layers.Dropout(0.2))
    model.add(layers.Conv1D(250, 3, padding='valid', activation='relu', strides=1))
    model.add(layers.MaxPooling1D(3))
    model.add(layers.Flatten())
    model.add(layers.Dense(200, activation='relu'))
    model.add(layers.Dense(labels.shape[1], activation='softmax'))
    model.summary()
    plot_model(model, to_file='model.png', show_shapes=True)

    model.compile(loss='categorical_crossentropy',
                optimizer='rmsprop',
                metrics=['accuracy'])

    start = time.time()
    model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=2, batch_size=128)
    end = time.time()
    print("Training time: {0} seconds".format(end - start))
    model.save('word_cnn.h5')

    #model = load_model('word_cnn.h5')
    print("Accuracy: {0}".format(model.evaluate(X_test, y_test, batch_size=128)))
    y_pred = model.predict(X_test, batch_size=128)
    for y in y_pred:
        y[y < 0.5] = 0
        y[y > 0.5] = 1
    print(classification_report(y_test, y_pred, target_names=dict))

def train_cnn_w2v(word_index, text_data, labels):

    X_train, X_test, y_train, y_test = train_test_split(text_data, labels, test_size=0.3, random_state=1)
    X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=0.7, random_state=1)

    w2v_model = models.KeyedVectors.load_word2vec_format('myCor.model.bin', binary=True)

    embedding_matrix = np.zeros((len(word_index) + 1, 200))
    for word, i in word_index.items():
        if str(word) in w2v_model:
            embedding_matrix[i] = np.asarray(w2v_model[str(word)], dtype='float32')
    embedding_layer = layers.Embedding(len(word_index) + 1,200,weights=[embedding_matrix],input_length=200,trainable=False)

    model = Sequential()
    model.add(embedding_layer)
    model.add(layers.Dropout(0.2))
    model.add(layers.Conv1D(250, 3, padding='valid', activation='relu', strides=1))
    model.add(layers.MaxPooling1D(3))
    model.add(layers.Flatten())
    model.add(layers.Dense(200, activation='relu'))
    model.add(layers.Dense(labels.shape[1], activation='softmax'))
    model.summary()

    model.compile(loss='categorical_crossentropy',
                  optimizer='rmsprop',
                  metrics=['accuracy'])

    start = time.time()
    model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=2, batch_size=128)
    model.save('word_cnn_w2v.h5')

    # model = load_model('word_cnn_w2v.h5')
    end = time.time()
    print("Training time: {0} seconds".format(end - start))
    print("Accuracy: {0}".format(model.evaluate(X_test, y_test, batch_size=128)))
    y_pred = model.predict(X_test, batch_size=128)
    for y in y_pred:
        y[y < 0.5] = 0
        y[y > 0.5] = 1
    print(classification_report(y_test, y_pred, target_names=dict))

def train_lstm(word_index, text_data, labels):

    X_train, X_test, y_train, y_test = train_test_split(text_data, labels, test_size=0.3, random_state=1)
    X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=0.7, random_state=1)

    model = Sequential()
    model.add(layers.Embedding(len(word_index) + 1, 200, input_length=200))
    model.add(layers.LSTM(200, dropout=0.2, recurrent_dropout=0.2))
    model.add(layers.Dropout(0.2))
    model.add(layers.Dense(labels.shape[1], activation='softmax'))
    model.summary()
    plot_model(model, to_file='model2.png', show_shapes=True)
    '''
    model.compile(loss='categorical_crossentropy',
                  optimizer='rmsprop',
                  metrics=['accuracy'])

    start = time.time()
    model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=2, batch_size=128)
    end = time.time()
    print("Training time: {0} seconds".format(end - start))
    model.save('word_lstm.h5')

    #model = load_model('word_lstm.h5')
    print("Accuracy: {0}".format(model.evaluate(X_test, y_test, batch_size=128)))
    y_pred = model.predict(X_test, batch_size=128)
    for y in y_pred:
        y[y < 0.5] = 0
        y[y > 0.5] = 1
    print(classification_report(y_test, y_pred, target_names=dict))
    '''

def train_lstm_w2v(word_index, text_data, labels):

    X_train, X_test, y_train, y_test = train_test_split(text_data, labels, test_size=0.3, random_state=1)
    X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=0.7, random_state=1)

    w2v_model = models.KeyedVectors.load_word2vec_format('myCor.model.bin', binary=True)

    embedding_matrix = np.zeros((len(word_index) + 1, 200))
    for word, i in word_index.items():
        if str(word) in w2v_model:
            embedding_matrix[i] = np.asarray(w2v_model[str(word)], dtype='float32')
    embedding_layer = layers.Embedding(len(word_index) + 1,200,weights=[embedding_matrix],input_length=200,trainable=False)

    model = Sequential()
    model.add(embedding_layer)
    model.add(layers.LSTM(200, dropout=0.2, recurrent_dropout=0.2))
    model.add(layers.Dropout(0.2))
    model.add(layers.Dense(labels.shape[1], activation='softmax'))
    model.summary()
    '''
    model.compile(loss='categorical_crossentropy',
                  optimizer='rmsprop',
                  metrics=['accuracy'])

    start = time.time()
    model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=2, batch_size=128)
    end = time.time()
    print("Training time: {0} seconds".format(end - start))
    model.save('word_lstm_w2v.h5')

    # model = load_model('word_lstm_w2v.h5')
    print("Accuracy: {0}".format(model.evaluate(X_test, y_test, batch_size=128)))
    y_pred = model.predict(X_test, batch_size=128)
    for y in y_pred:
        y[y < 0.5] = 0
        y[y > 0.5] = 1
    print(classification_report(y_test, y_pred, target_names=dict))
    '''

def wordToVec():
    df = pd.read_csv('train_set.csv', encoding="utf_8_sig")
    all_text = []
    for d in df['content']:
        all_text.append(splitWords(d))

    model = Word2Vec(all_text, size=200)
    model.wv.save_word2vec_format('myCor.model.bin', binary=True)
    print(model.most_similar('篮板'))

    '''
    X = model[model.wv.vocab]
    keys = list(model.wv.vocab.keys())
    kmeans = KMeans(n_clusters=14)
    kmeans.fit(X)
    labels = kmeans.labels_

    print(keywords(model, all_text[0]))

    classCollects = {}
    for i in range(len(keys)):
        if labels[i] in classCollects.keys():
            classCollects[labels[i]].append(keys[i])
        else:
            classCollects[labels[i]] = [keys[i]]
    for i in range(0,13):
        print(dict[i])
        print(classCollects[i])
    '''

if __name__ == '__main__':
    readStop()
    #classification_lr()
    #classification_svm()
    #classification_nb()
    #classification_rf()
    train_process()