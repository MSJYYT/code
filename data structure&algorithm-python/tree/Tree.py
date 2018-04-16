class Tree():
    #树的实现
    def __init__(self, ltree = 0,rtree = 0, data = 0):
        self.ltree = ltree
        self.rtree = rtree
        self.data = data
class BTree():
    #二叉树实现
    def __init__(self,base):
        self.base = base
    def _Empty(self):
        #是否为空树
        if self.base == 0:
            return True
        else:
            return False
    def qout(self,tree_base):
        #前序遍历：根-左-右
        if tree_base == 0:
            return
        print(tree_base.data,end='')
        self.qout(tree_base.ltree)
        self.qout(tree_base.rtree)
    def mout(self,tree_base):
        #中序遍历：左-根-右
        if tree_base == 0:
            return
        self.mout(tree_base.ltree)
        print(tree_base.data,end='')
        self.mout(tree_base.rtree)
    def hout(self,tree_base):
        #后序遍历：左-右-根
        if tree_base == 0:
            return
        self.hout(tree_base.ltree)
        self.hout(tree_base.rtree)
        print(tree_base.data,end='')

# 树的构建：
#      5
#   6     7
# 8         9

tree1 = Tree(data=8)
tree2 = Tree(data=9)
tree3 = Tree(tree1,0,data=6)
tree4 = Tree(0,tree2,data=7)
base = Tree(tree3,tree4,5)
btree = BTree(base)
btree.qout(btree.base)
#btree.mout(btree.base)
#btree.hout(btree.base)