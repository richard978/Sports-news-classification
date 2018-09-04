import time
import pandas as pd
import numpy as np
from scipy.misc import imread
import jieba
import random
from matplotlib import pyplot as plt
from gensim import models
from gensim.models import Word2Vec
from gensim.models import ldamodel
from gensim.models import lsimodel
from gensim import corpora
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from textrank4zh import TextRank4Keyword, TextRank4Sentence
from wordcloud import WordCloud

stopList = []
specStop = ['分','比赛','中','米','秒','跑','球队','分钟','球员','时间','选手','羽毛球','体操','曲棍球','马术','乒乓球','车队','车手','网球']
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
    seqList = seq[1:len(seq)-1].split(',')
    seqList = filter(lambda x: x not in specStop, seqList)
    return seqList

def tfidf_keyWords(cat):
    df = pd.read_csv('train_set.csv', encoding="utf_8_sig")
    text = ""
    for i in range(len(df)):
        if df['label'][i] == cat:
            text += ' '.join(splitWords(df['content'][i]))
    text = [text]
    vectorizer = CountVectorizer(min_df=1e-5)
    transformer = TfidfTransformer()
    tfidf = transformer.fit_transform(vectorizer.fit_transform(text))
    word = vectorizer.get_feature_names()
    word = np.array(word)

    weight = tfidf.toarray()
    word_index = np.argsort(-weight)
    word = word[word_index]
    weight = weight[0][word_index]

    tags = []
    frequency = []
    for i in range(20):
        tags.append(word[0][i])
        frequency.append(weight[0][i]*1000)

    total_text = {}
    for i in range(20):
        total_text[tags[i]] = int(frequency[i])
    wordcloud = WordCloud(font_path='C:/WINDOWS/Fonts/STKAITI.TTF', background_color='white').fit_words(total_text)
    # print(cat+':')
    wordcloud.to_file('wordcloud/'+cat+'_tfidf.jpg')

def lda_keyWords(cat):
    df = pd.read_csv('train_set.csv', encoding="utf_8_sig")
    text = []
    for i in range(len(df)):
        if df['label'][i] == cat:
            text += splitWords(df['content'][i])
    text = [text]
    dictionary = corpora.Dictionary(text)
    corpus = [dictionary.doc2bow(t) for t in text]

    #print(cat + ':')
    lsi = lsimodel.LsiModel(corpus, id2word=dictionary)
    #print("LSI: ", lsi.print_topics(5))
    lda = ldamodel.LdaModel(corpus, id2word=dictionary)
    #print("LDA: ", lda.print_topics(5))

    wc_lsi(cat, lsi, 0)
    wc_lsi(cat, lda, 1)

def wc_lsi(cat, model, flag):
    tags = []
    frequency = []
    for l in model.print_topic(0, topn=20).split('+'):
        t = l.strip().split('*')
        f = t[0]
        w = t[1][1:len(t[1]) - 1]
        frequency.append(float(f) * 1000)
        tags.append(w)
    total_text = {}
    for i in range(len(tags)):
        total_text[tags[i]] = int(frequency[i])
    wordcloud = WordCloud(font_path='C:/WINDOWS/Fonts/STKAITI.TTF', background_color='white').fit_words(total_text)
    if(flag==0):
        wordcloud.to_file('wordcloud/' + cat + '_lsi.jpg')
    else:
        wordcloud.to_file('wordcloud/' + cat + '_lda.jpg')

def textrank_keyWords(cat):
    df_u8 = pd.read_csv('contents/' + cat + '_u8.csv', encoding="utf_8_sig")
    df_gb = pd.read_csv('contents/' + cat + '_gb.csv', encoding="GB2312")

    content = []
    for i in range(len(df_gb['content'])):
        tempStr = str(df_gb['title'][i]).strip() + ' ' + str(df_gb['content'][i]).strip()
        content.append(tempStr)
    for i in range(len(df_u8['content'])):
        content.append(str(df_u8['title'][i]).strip()+' '+str(df_u8['content'][i]).strip())

    words = {}
    tr4w = TextRank4Keyword(stop_words_file='stopWords.txt')
    randomList = []
    for i in range(500):
        randomList.append(random.randint(0,len(content)-1))

    for i in randomList:
        tr4w.analyze(text=content[i], lower=True, window=2)
        for item in tr4w.get_keywords(5, word_min_len=1):
            if item.word not in stopList and item.word not in specStop:
                if item.word not in words.keys():
                    words[item.word] = item.weight
                else:
                    words[item.word] = words[item.word] + item.weight
    sorted_by_value = sorted(words.items(), key=lambda kv: kv[1], reverse=True)
    #print(cat + ':')
    tags = []
    frequency = []
    for l in sorted_by_value[0:20]:
        f = l[1]
        w = l[0]
        frequency.append(float(f) * 1000)
        tags.append(w)
    total_text = {}
    for i in range(len(tags)):
        total_text[tags[i]] = int(frequency[i])
    wordcloud = WordCloud(font_path='C:/WINDOWS/Fonts/STKAITI.TTF', background_color='white').fit_words(total_text)
    wordcloud.to_file('wordcloud/' + cat + '_tr.jpg')

if __name__ == '__main__':
    readStop()
    #lda_keyWords('basketball')
    #textrank_keyWords('basketball')
    '''
    for d in dict:
        tfidf_keyWords(d)
    for d in dict:
        lda_keyWords(d)
    '''
    for i, d in enumerate(dict):
        textrank_keyWords(d)