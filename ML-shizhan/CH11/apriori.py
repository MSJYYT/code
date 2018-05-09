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
            print(can)
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
        print(support)
        if support >= minSupport:
            retList.insert(0,key)
        supportData[key] = support
    return retList,supportData

dataSet = loadDataSet()
C1 = creatC1(dataSet)

D = [set(fset) for fset in dataSet]

L1,suppData0 = scanD(D,C1,0.5)
print(list(L1))
