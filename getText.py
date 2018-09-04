# coding: utf-8
import chardet
import requests
import os,sys
import threading
import chardet
import json
from pyquery import PyQuery as pq

rootPath=os.path.dirname(os.path.realpath(__file__))
def saveText(name,time,title,artical):
	#dict={'time':time,'tag':tag,'title':title,'text':artical}
	#name=time+'_'+title
	with open(rootPath+"\\text\\"+name +".txt", "wb") as f:  
		f.write(title+'\n')
		f.write(time+'\n')
		f.write(artical)
		f.close()

def getHtml(url):
	request = requests.get(url,headers={'Connection':'close'})
	return request.text

reload(sys)
sys.setdefaultencoding('utf8')

root=os.path.dirname(os.path.realpath(__file__))
rootUrl='http://sports.qq.com/l/f1/allf1news/list20100311191657_'
maxThread=8
idxBegin=2
idxEnd=100
maxIdx=(idxEnd-idxBegin)/maxThread


def mainFunc(threadID):
	for idx in range(0,maxIdx):
		url_count=idx*maxThread+threadID+idxBegin
		try:
			html=getHtml(rootUrl+str(url_count)+'.htm')
			pqText=pq(html)
			newsList=pqText('.newslist a').items()
		except Exception as e:
			print e
			print str(url_count)+' break'
			with open(rootPath+"\\_root_exception.txt", "a") as f:
				f.write(str(url_count)+"\n")
			f.close()
			continue
		for href in newsList:
			try:
				subUrl=href.attr('href')
				subHtml=getHtml(subUrl)
				saveName=subUrl.split('/').pop()
				saveName=subUrl.split('/').pop()+'/'saveName
				print saveName
				subPqText=pq(subHtml)
				title=subPqText('h1').text()
				time=subPqText('.a_time').text()
				artical=subPqText('#Cnt-Main-Article-QQ p').text()
				saveText(saveName,time,title,artical)
			except Exception as e:
				print e
				continue

class DownloadThread (threading.Thread):
	def __init__(self, threadID):
		threading.Thread.__init__(self)
		self.threadID = threadID
	def run(self): 
		mainFunc(self.threadID)

for i in range(0,maxThread):
	DownloadThread(i).start()