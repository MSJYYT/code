from tree import classtreeNode
from stack import SStack

class Assoc:
    def __init__(self,key,value):
        self.key = key
        self.value = value
    def __lt__(self, other):        #有时需要考虑序
        return self.key < other.key
    def __le__(self, other):
        return self.key < other.key or self.key == other.key
    def __str__(self):              #定义字符串表示形式便于输出和交互
        return "Assoc{{0},{1}}".format(self.key,self.value)

class DictBinTree:
    def __init__(self):
        self._root = None
    def is_empty(self):
        return self._root is None
    def search(self, key):      #检索
        bt = self._root
        while bt is not None:
            entry = bt.data
            if key < entry.key:
                bt = bt.left
            elif key > entry.key:
                bt = bt.right
            else:
                return entry.value
        return None
    def insert(self, key, value):
        bt = self._root
        if bt is None:
            self._root = classtreeNode.BinTNode(Assoc(key, value))
            return
        while True:
            entry = bt.data
            if key < entry.key:
                if bt.left is None:
                    bt.left = classtreeNode.BinTNode(Assoc(key, value))
                    return
                bt = bt.left
            elif key > entry.key:
                if bt.right is None:
                    bt.right = classtreeNode.BinTNode(Assoc(key, value))
                    return
                bt = bt.right
            else:
                bt.data.value = value
                return
    def entries(self):
        t, s = self._root, SStack()
        while t is not None or not s.is_empty():
            while t is not None:
                s.push(t)
                t = t.left
            t = s.pop()
            yield  t.data.key, t.data.value
            t = t.right
    def delete(self,key):
        p, q = None,self._root   #维持p为q的父节点
        while q is not None and q.data.key != key:
            p = q
            if key < q.data.key:
                q = q.left
            else:
                q = q.right
            if q is None:
                return              #树中没有关键码key

            #到这里q引用要删除节点，P是其父节点或None（这时q是根节点）
            if q.left is None:          #如果q没有左子节点
                if p is None:
                    self._root = q.right
                elif q is p.left:
                    p.left = q.right
                else:
                    p.right = q.right
                return
            r = q.left
            while r.right is not None:
                r = r.right
            r.right = q.right
            if p is None:
                self._root = q.left
            elif p.left is q:
                p.left = q.left
            else:
                p.right = q.left
    def print(self):            #检查二叉树情况
        for k, v in self.entries():
            print(k, v)

def build_dictBinTree(entries):
    dic = DictBinTree()
    for k, v in entries:
        dic.insert(k, v)
    return