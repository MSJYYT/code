from tree import classtreeNode
from dict import binary_sort_tree

#首先把AVL树节点类定义为二叉树节点类的子类
#增加一个bf域，叶节点bf值为0
class AVLNode(classtreeNode.BinTNode):
    def __init__(self, data):
        classtreeNode.BinTNode.__init__(self,data)
        self.bf = 0
#AVL树是一种二叉排序树，将这个类定义为DictBinTree的子类，初始化为空树
class DictAVL(binary_sort_tree.DictBinTree):
    def __init__(self):
        binary_sort_tree.DictBinTree.__init__(self)
#LL型调整（a的左子树高，新节点插入在a的左子树的左子树）
#LR型调整（a的左子树高，新节点插入在a的左子树的右子树）
#RR型调整（a的右子树高，新节点插入在a的右子树的右子树）
#RL型调整（a的右子树高，新节点插入在a的右子树的左子树）
#插入调整方法：LL调整右旋转一次
    def LL(a, b):
        a.left = b.right
        b.right = a
        a.bf = b.bf = 0
        return b
#插入调整方法：RR调整左旋转一次
    def RR(a, b):
        a.right = b.left
        b.left = a
        a.bf = b.bf = 0
        return b
#插入调整方法：LR调整先左后右
    def LR(a, b):
        c = b.right
        a.left, b.right = c.right, c.left
        c.left, c.right = b, a
        if c.bf == 0:            #c本身就是插入节点
            a.bf = b.bf = 0
        elif c.bf == 1:             #新节点在c的左子树
            a.bf = -1
            b.bf = 0
        else:                   #新节点在c的右子树
            a.bf = 0
            b.bf = 1
        c.bf = 0
        return c
#插入调整方法：RL调整先右后左
    def RL(a, b):
        c = b.left
        a.right, b.left = c.left, c.right
        c.left, c.right = a, b
        if c.bf == 0:           #c本身就是插入节点
            a.bf = 0
            b.bf = 0
        elif c.bf == 1:          #新节点在c的左子树
            a.bf = 0
            b.bf = -1
        else:                   #新节点在c的右子树
            a.bf = 1
            b.bf = 0
        c.bf = 0
        return c

    def insert(self, key, value):
        a = p = self._root
        if a is None:
            self._root = AVLNode(binary_sort_tree.Assoc(key, value))
            return
        pa = q = None                       #维持pa,q为a，p的父节点
        while p is not None:                #确定插入位置及最小非平衡子树
            if key == p.data.key:           #key存在，修改关联值并结束
                p.data.value = value
                return
            if p.bf != 0:
                pa, a = q,p                 #已知最小非平衡子树
            q = p
            if key < p.data.key:
                p = p.left
            else:
                p = p.right
        #q是插入点的父节点，pa，a记录最小非平衡子树
        node = AVLNode(binary_sort_tree.Assoc(key, value))
        if key < q.data.key:
            q.left = node                   #作为左子节点
        else:
            q.right = node                  #或右子节点
        #新节点已插入，a是最小不平衡子树
        if key < a.data.key:                #新节点在a的左子树
            p = b = a.left
            d = 1
        else:
            p = b = a.right                 #新节点在a的右子树
            d = -1                          #d记录新节点在a的哪颗子树
        #修改b到新节点路径上各节点的bf值，b为a的子节点
        while p != node:                    #node一定存在，不用判断p空
            if key < p.data.key:            #p的左子树增高
                p.bf = 1
                p = p.left
            else:                           #p的右子树增高
                p.bf = -1
                p = p.right
        if a.bf == 0:                       #a的原bf为0，不会失衡
            a.bf = d
            return
        if a.bf == -d:                      #新节点在较低子树里
            a.bf = 0
            return
        #新节点在较高子树，失衡，必须调整
        if d == 1:                          #新节点在a的左子树
            if b.bf == 1:
                b = DictAVL.LL(a,b)
            else:
                b = DictAVL.LR(a.b)
        else:
            if b.bf == -1:
                b = DictAVL.RR(a, b)
            else:
                b = DictAVL.RL(a, b)
        if pa is None:                      #原a为树根，修改_root
            self._root = b
        else:                               #a非树根，新树接在正确位置
            if pa.left == a:
                pa.left = b
            else:
                pa.right = b
