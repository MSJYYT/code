class LNode:
    def __init__(self, elem, next_=None):
        self.elem = elem
        self.next = next_

#链表头插入元素
q = LNode(13)
q.next = head.next
head = q

#删除表首元素
head = head.next

#扫描链表按下标定位
p = head
while p is not None and i > 0:
    i -= 1
    p = p.next

#按元素定位
p = head
while p is not None and not pred(p.elem):
    p = p.next

#遍历
p = head
while p is not None:
    print(p.elem)
    p = p.next
#求表长度
def length(head):
    p, n = head, 0
    while p is not None:
        n += 1
        p = p.next
    return n