from stack import SStack
from stack import queue


class BinTNode:
    def __init__(self, dat, left = None, right = None):
        self.data = dat
        self.left = left
        self.right = right
#构造一个二叉树
t = BinTNode(1, BinTNode(2), BinTNode(3))
#统计数中节点个数
def count_BinTNodes(t):
    if t is None:
        return 0
    else:
        return 1 + count_BinTNodes(t.left) + count_BinTNodes(t.right)
#求二叉树里所有数值之和
def sum_BinTNodes(t):
    if t is None:
        return 0
    else:
        return t.dat + sum_BinTNodes(t.left) + sum_BinTNodes(t.right)
#深度优先先根序遍历
def preorder(t, proc):
    if t is None:
        return
    proc(t.data)
    preorder(t.left)
    preorder(t.right)
def print_BinTNodes(t):
    if t is None:
        print('^', end='')
        return
    print('(' + str(t.data), end='')
    print_BinTNodes(t.left)
    print_BinTNodes(t.right)
    print(')', end='')

#宽度优先遍历
def levelorder(t, proc):
    qu = queue.SQueue()
    qu.enqueue(t)
    while not qu.is_empty():
        n = qu.dequeue()
        if t is None:
            continue
        qu.enqueue(t.left)
        qu.enqueue(t.right)
        proc(t.data)
#非递归先根序遍历
def preorder_nonrec(t, proc):
    s  = SStack()
    while t is not None or not s.is_empty():
        while t is not None:
            proc(t.data)
            s.push(t.right)
            t = t.left
        t = s.pop()
preorder_nonrec(t, lambda x:print(x, end=' '))
#非递归后根序遍历
def postorder_nonrec(t, proc):
    s = SStack()
    while t is not None or not s.is_empty():
        while t is not None:            #下行循环，直到栈顶的两子树空
            s.push(t)
            t = t.left if t.left is not None else t.right#能左就左否则向右
        t = s.pop()     #栈顶是应访问节点
        proc(t.data)
        if not s.is_empty() and s.top().left == t:
            t = s.top().right               #栈不空且当前节点是栈定左子节点
        else:                               #没有右子树或右子树遍历完毕，强迫退栈
            t = None
#生成器遍历
def preorder_elements(t):
    s = SStack()
    while t is not None or not s.is_empty():
        while t is not None:
            s.push(t.right)
            yield t.data
            t = t.left
        t = s.pop()

