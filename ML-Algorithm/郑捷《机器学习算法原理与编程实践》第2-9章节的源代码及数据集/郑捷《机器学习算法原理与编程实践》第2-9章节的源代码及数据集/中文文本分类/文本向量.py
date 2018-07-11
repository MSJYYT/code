from sklearn.datasets.base import Bunch                                                 #引入Bunch类
import os
import sys
import pickle                                                                                                           #引入持久化类

def readfile(path) :
	fp = open(path, 'rb')
	content = fp.read()
	fp.close()
	return content
	
bunch = Bunch(target_name = [], label = [], filenames = [], contents = [])

#将分好词的文本文件转换并持久化为Bunch类形式
wordbag_path = "train_word_bag/train_set.dat"
seg_path = "train_corpus_seg/"                                              #分词后分类语料库路径

catelist = os.listdir(seg_path)
bunch.target_name.extend(catelist)

for mydir in catelist :
	class_path = seg_path + mydir + "/"
	file_list = os.listdir(class_path)
	for file_path in file_list :
		fullname = class_path + file_path
		bunch.label.append(mydir)                                               #保存当前文件的分类标签
		bunch.filenames.append(fullname)                                #保存当前文件的文件路径
		bunch.contents.append(readfile(fullname).strip())   #保存文件词向量
		
#Bunch对象持久化
file_obj = open(wordbag_path, 'wb')
pickle.dump(bunch, file_obj)
file_obj.close()

print("构建文本对象结束！！！")	




