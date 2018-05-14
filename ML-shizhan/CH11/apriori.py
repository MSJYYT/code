def loadDataSet():
    return [[1,3,4],[2,3,5],[1,2,3,5],[2,5]]

def creatC1(dataSet):
    C1 = []
    for transaction in dataSet:
        for item in transaction:
            if not [item] in C1:
                C1.append([item])
    C1.sort()
    return [frozenset(fset) for fset in C1]

#输入：数据集Ck
#     包含候选集合的列表D
#     感兴趣项集的最小支持度minSupport
#有C1生成L1：L1为满足最小支持度的C1
def scanD(D, Ck,minSupport):
    ssCnt = {}
    for tid in D:

        for can in Ck:

            # 测试是否 can 中的每一个元素都在 tid 中

            if can.issubset(tid):
                #can在tid中出现，判断can是否第一次插入字典ssCnt中

                if can not in ssCnt:ssCnt[can]=1
                else:ssCnt[can] += 1
                #ssCnt[can] = ssCnt.get(can, 0) + 1


    #D1 = map(set, dataSet)
    numItems = float(len(list(D)))

    retList = []
    supportData = {}
    for key in ssCnt:
        support = ssCnt[key]/numItems

        if support >= minSupport:
            retList.insert(0,key)
        supportData[key] = support
    return retList,supportData

# dataSet = loadDataSet()
# C1 = creatC1(dataSet)
#
# D = [set(fset) for fset in dataSet]
#
# L1,suppData0 = scanD(D,C1,0.5)
# print(list(L1))


def aprioriGen(Lk,k):
    retList = []
    lenLk = len(Lk)
    for i in range(lenLk):
        for j in range(i+1,lenLk):
            L1 = list(Lk[i])[:k-2];L2 = list(Lk[j])[:k-2]
            L1.sort();L2.sort()
            if L1 == L2:
                retList.append(Lk[i] | Lk[j])
    return retList

def apriori(dataSet,minSupport = 0.5):
    C1 = creatC1(dataSet)
    D = [set(fset) for fset in dataSet]
    L1,supportData = scanD(D,C1,minSupport)
    L = [L1]
    k = 2
    while (len(L[k-2]) > 0):
        Ck = aprioriGen(L[k-2],k)
        Lk,supK = scanD(D,Ck,minSupport)
        supportData.update(supK)
        L.append(Lk)
        k += 1
    return L,supportData
dataSet = loadDataSet()


#找出关联规则
#L：频繁项集列表
#supportData:频繁项集支持度字典
#minconf:最小可信度阈值
#生成一个包含可信度的规则列表
def generatrRules(L,supportData,minConf=0.7):
    bigRuleList = []
    #从包含两个以上元素的项集开始规则构建
    for i in range(1,len(L)):
        for freqSet in L[i]:
            #对每个频繁项集创建只包含单个元素的集合的列表
            H1 = [frozenset([item]) for item in freqSet]
            if (i > 1):

                rulesFromConseq(freqSet,H1,supportData,bigRuleList,minConf)
            else:

                clacConf(freqSet,H1,supportData,bigRuleList,minConf)
    return bigRuleList

#生成候选规则集合
#freqSet:某一项频繁项集
#H:把freqSet分成单个元素组成的列表
def clacConf(freqSet,H,supportData,br1,minConf=0.7):
    prunedH = []
    for conseq in H:
        #一条规则P->H的可信度定义为support(P|H)/support(P)
        #freqSet相当于P|H
        conf = supportData[freqSet]/supportData[freqSet-conseq]
        if conf >= minConf:
            print(conseq,'--->',freqSet-conseq,'conf:',conf)
            br1.append((freqSet-conseq,conseq,conf))
            prunedH.append(conseq)
    return prunedH

#对规则进行评估
def rulesFromConseq(freqSet,H,supportData,br1,minConf=0.7):
    m = len(H[0])

    if (len(freqSet) > (m+1)):
        Hmp1 = aprioriGen(H,m+1)

        Hmp1 = clacConf(freqSet,Hmp1,supportData,br1,minConf)

        if (len(Hmp1) > 1):

            rulesFromConseq(freqSet,Hmp1,supportData,br1,minConf)

if __name__ == '__main__':
    # L,suppData = apriori(dataSet)
    #
    # rules = generatrRules(L,suppData,minConf=0.7)
    mushDatSet = [line.split() for line in open('mushroom.dat').readlines()]

    L,suppData = apriori(mushDatSet,minSupport=0.3)
    print(len(L))

