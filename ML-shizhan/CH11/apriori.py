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

L,suppData = apriori(dataSet)
print(L)
print(L[0])

