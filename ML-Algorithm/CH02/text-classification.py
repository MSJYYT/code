import os
import sys
import jieba

def savefile(savepath,content): #保存至文件
    fp = open(savepath,'wb')
    fp.write(content)
    fp.close()

def readfile(path): #读取文件
    fp = open(path,'rb')
    content = fp.read()
    fp.close()
    return content

def segmation():
    corpus_path = 'corpus_path'        #未分词分类语料库路径
    seg_path = 'seg_path'           #分词后分类语料库路径

    catelist = os.listdir(corpus_path) #获取corpus_path下的所有子目录

    #获取每个目录下的所有文件
    for mydir in catelist:
        class_path = corpus_path+mydir+'/'  #拼出分类子目录的路径
        seg_dir = seg_path+mydir+'/'        #拼出分词后的语料分类目录
        if not os.path.exists(seg_dir):
            os.makedirs(seg_dir)            #是否存在目录，如果没有则创建
        file_list = os.listdir(class_path)  #获取类别目录下的所有文件
        for file_path in file_list:         #遍历类别目录下的文件
            fullname = class_path + file_path   #拼出文件名全路径
            content = readfile(fullname).strip()    #读取文件内容
            content = content.replace('\r\n','').strip()    #删除换行和多余的空格
            content_seg = jieba.cut(content)    #为文件内容分词
            #将处理后的文件保存到分词后语料目录
            savefile(seg_dir+file_path,' '.join(content_seg))
    print('中文语料分词结束！')
