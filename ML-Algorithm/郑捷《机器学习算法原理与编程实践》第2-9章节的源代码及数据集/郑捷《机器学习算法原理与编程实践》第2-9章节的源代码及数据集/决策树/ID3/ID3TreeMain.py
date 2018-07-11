from numpy import *
from ID3Tree import *

dtree = ID3DTree()
dtree.loadDataSet("dataset.dat", ["age", "revenue", "student", "credit"])
dtree.train()
print(dtree.tree)
#dtree.storeTree(dtree.tree, "data.tree")
#mytree = dtree.grabTree("data.tree")
#labels = ["age", "revenue", "student", "credit"]
#vector = ['0', '1', '0', '0']                                                         #['0', '1', '0', '0', 'no']
#print("真实输出", "no", "  ->  ", "决策树输出", dtree.predict(mytree, labels, vector))

