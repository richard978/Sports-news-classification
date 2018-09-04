# -*- coding: utf-8 -*-
import jieba
import chardet
import os
import pandas as pd

sportsData = pd.DataFrame()

def getCoding(strInput):
    try:
        strInput.decode("utf8")
        return 'utf8'
    except:
        pass
    try:
        strInput.decode("gbk")
        return 'gbk'
    except:
        pass

def read_txt(path,name):
    fileList = []
    titleList_u8 = []
    timeList_u8 = []
    contentList_u8 = []
    titleList_gb = []
    timeList_gb = []
    contentList_gb = []
    files = os.listdir(path)
    for file in files:
        fileList.append(file)
    for i in range(len(fileList)):
        txtDir = path+"/"+fileList[i]
        file = open(txtDir, 'rb')
        buf = file.read()
        coding = chardet.detect(buf)
        file = open(txtDir, encoding=coding["encoding"], errors='ignore')
        lines = file.readlines()

        if coding["encoding"]=='GB2312':
            if len(lines)==3:
                for id,line in enumerate(lines):
                    if id==0:
                        titleList_gb.append(line.replace('\n', '').strip())
                    elif id==1:
                        timeList_gb.append(line.replace('\n', '').strip())
                    else:
                        line = line.replace('原标题：', '')
                        line = line.replace('返回搜狐，查看更多 责任编辑：', '')
                        line = line.replace('搜狐体育独家稿件 严禁转载', '')
                        line = line.replace(lines[0].strip(), '').strip()
                        contentList_gb.append(line.replace('\n', ''))
        else:
            if len(lines)==3:
                for id,line in enumerate(lines):
                    if id==0:
                        titleList_u8.append(line.replace('\n', '').strip())
                    elif id==1:
                        timeList_u8.append(line.replace('\n', '').strip())
                    else:
                        line = line.replace('原标题：', '')
                        line = line.replace('返回搜狐，查看更多 责任编辑：', '')
                        line = line.replace('搜狐体育独家稿件 严禁转载', '')
                        line = line.replace(lines[0].strip(), '').strip()
                        contentList_u8.append(line.replace('\n', ''))

    sportsData1_u8 = pd.DataFrame({'title': titleList_u8, 'time': timeList_u8, 'content': contentList_u8})
    sportsData1_gb = pd.DataFrame({'title': titleList_gb, 'time': timeList_gb, 'content': contentList_gb})
    sportsData1_u8.to_csv('contents/'+name+'_u8.csv', index=None, encoding="utf_8_sig")
    sportsData1_gb.to_csv('contents/'+name+'_gb.csv', index=None, encoding="GB2312")

if __name__ == '__main__':
    read_txt("sportsNews/篮球", "basketball")
    '''
    read_txt("sportsNews/马术", "horse")
    read_txt("sportsNews/排球", "volleyball")
    read_txt("sportsNews/乒乓球", "pingpong")
    read_txt("sportsNews/曲棍球", "hockey")
    read_txt("sportsNews/拳击", "boxing")
    read_txt("sportsNews/赛车", "racing")
    read_txt("sportsNews/射击", "shooting")
    read_txt("sportsNews/台球", "billiard")
    read_txt("sportsNews/体操", "gym")
    read_txt("sportsNews/田径", "field")
    read_txt("sportsNews/网球", "tennis")
    read_txt("sportsNews/羽毛球", "badminton")
    read_txt("sportsNews/足球", "football")
    '''