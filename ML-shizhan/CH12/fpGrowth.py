#FP树的类定义
class treeNode:
    def __init__(self,nameValue,numOccur,parentNode):
        self.name = nameValue
        self.count = numOccur
        self.nodeLink = None
        self.parent = parentNode
        self.children = {}

    def inc(self,numOccur):
        self.count += numOccur

    #用于将树以文本方式显示
    def disp(self,ind=1):
        print(' '*ind,self.name,' ',self.count)
        for child in self.children.values():
            child.disp(ind+1)

#dataSet是个字典
def createTree(dataSet,minSup = 1):
    headerTable = {}
    for trans in dataSet:
        for item in trans:
            #扫描数据集并统计每个元素项出现的频度，存储与头指针表中
            headerTable[item] = headerTable.get(item,0) + dataSet[trans]

    for k in list(headerTable.keys()):
        if headerTable[k] < minSup:
            del(headerTable[k])
    freqItemSet = set(headerTable.keys())
    if len(freqItemSet) == 0:return None,None
    #对头指针扩展，保存计数值及指向每种类型第一个元素项的指针
    for k in headerTable:
        headerTable[k] = [headerTable[k],None]

    #创建只包含空集合的根节点
    retTree = treeNode('Null Set',1,None)
    # 得到删除非频繁k=1的项的 项集,并以字典树的形式插入树里。
    for tranSet,count in dataSet.items():
        localD = {}
        for item in tranSet:
            if item in freqItemSet:
                localD[item] = headerTable[item][0]
        if len(localD) > 0:
            orderedItems = [v[0] for v in sorted(localD.items(),key=lambda p:p[1],reverse=True)]
            updateTree(orderedItems,retTree,headerTable,count)
    return retTree,headerTable

#items一行记录，inTree FP树的树根，headerTable出现的频繁项集，count这行记录出现的次数
def updateTree(items,inTree,headerTable,count):
    #如果出现了这个节点，就把这个节点的计数增加
    if items[0] in inTree.children:
        inTree.children[items[0]].inc(count)
    else:
        #否则创建新的节点
        inTree.children[items[0]] = treeNode(items[0],count,inTree)
        #如果这个节点对于的headerTable没有出现就创建
        if headerTable[items[0]][1] == None:
            headerTable[items[0]][1] = inTree.children[items[0]]
        else:
            updateHeader(headerTable[items[0]][1],inTree.children[items[0]])
    if len(items) > 1:
        # 如果要插入字典树的词条还有,那就继续插入剩下的
        updateTree(items[1::],inTree.children[items[0]],headerTable,count)

def updateHeader(nodeToTest,targetNode):
    while (nodeToTest.nodeLink != None):
        nodeToTest = nodeToTest.nodeLink
    nodeToTest.nodeLink = targetNode

def loadSimpDat():
    simpDat = [['r', 'z', 'h', 'j', 'p'],
               ['z', 'y', 'x', 'w', 'v', 'u', 't', 's'],
               ['z'],
               ['r', 'x', 'n', 'o', 's'],
               ['y', 'r', 'x', 'z', 'q', 't', 'p'],
               ['y', 'z', 'x', 'e', 'q', 's', 't', 'm']]
    return simpDat

def creatInitSet(dataSet):
    retDict = {}
    for trans in dataSet:
        retDict[frozenset(trans)] = 1
    return retDict

#发现以给定元素项结尾的所有路径的函数
def ascendTree(leafNode,prefixPath):
    if leafNode.parent != None:
        prefixPath.append(leafNode.name)
        ascendTree(leafNode.parent,prefixPath)

def findPrefixPath(basePat,treeNode):
    condPats = {}
    while treeNode != None:
        prefixPath = []
        ascendTree(treeNode,prefixPath)
        if len(prefixPath) > 1:
            condPats[frozenset(prefixPath[1:])] = treeNode.count
        treeNode = treeNode.nodeLink
    return condPats

if __name__ == '__main__':
    # rootNode = treeNode('pyramid',9,None)
    # rootNode.children['eye']=treeNode('eye',13,None)
    # rootNode.disp()
    simpDat = loadSimpDat()

    initSet = creatInitSet(simpDat)

    myFPtree,myHeaderTab = createTree(initSet,3)
    #myFPtree.disp()
