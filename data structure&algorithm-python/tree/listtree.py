# 基于list实现二叉树
# 空树用None表示
# 非空二叉树用包含三个元素的表[d,l,r]表示，其中：
# d表示存在根节点的元素
# l和r是两颗子树
# ['A',['B',None,None],
#      ['C',['D',['F',None,None],l
#                ['G',None,None]],
#           ['E',['I',None,None],
#                ['H',None,None]]]]
def BinTree(data,left=None,right = None):
    return [data,left,right]
def is_empty_BinTree(btree):
    return btree is None
def root(btree):
    return btree[0]
def left(btree):
    return btree[1]
def right(btree):
    return btree[2]
def set_root(btree,data):
    btree[0] = data
def set_left(btree,left):
    btree[1] = left
def set_right(btree,right):
    btree[2] = right
    