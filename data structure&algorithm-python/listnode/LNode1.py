class LNode:
    def __init__(self, elem, next_=None):
        self.elem = elem
        self.next = next_

# llist1 = LNode(1)
# p = llist1
#
# for i in range(2,11):
#     p.next = LNode(i)
#     p = p.next
#
# p = llist1
# while p is not None:
#     print(p.elem)
#     p = p.next

class LinkedListUnderflow(ValueError):
    pass

class LList:
    def __init__(self):
        self._head = None
    def is_empty(self):
        return self._head is None
    def prepend(self, elem):
        self._head = LNode(elem, self._head)
    def pop(self):
        if self._head is None:
            raise LinkedListUnderflow('in pop')
        e = self._head.elem
        self._head = self._head.next
        return e
    def append(self, elem):
        if self._head is None:
            self._head = LNode(elem)
            return
        p = self._head
        while p.next is not None:
            p = p.next
        p.next = LNode(elem)
    def pop_last(self):
        if self._head is None:
            raise LinkedListUnderflow('in pop_last')
        p = self._head
        if p.next is None:
            e = p.elem
            self._head = None
            return e
        while p.next.next is not None:
            p = p.next
        e = p.next.elem
        p.next = None
        return e
    def find(self, elem):#根据元素找索引
        p = self._head
        index = 0
        while p is not None:
            if p.elem == elem:
                return index
            p = p.next
            index += 1
    def printall(self):
        p = self._head
        while p is not None:
            print(p.elem,end='')
            if p.next is not None:
                print('->',end='')
            p = p.next
        print('')
    def rev(self):
        temp = None
        while self._head is not None:
            head = self._head
            self._head = head.next
            head.next = temp
            temp = head
        self._head = temp
    def sort1(self):#移动元素
        if self._head is None:
            return
        crt = self._head.next       #从首节点之后开始处理
        while crt is not None:
            x = crt.elem
            p = self._head
            while p is not crt and p.elem <= x:#跳过小元素
                p = p.next
            while p is not crt: #倒换大元素，完成元素插入工作
                y = p.elem
                p.elem = x
                x = y
                p = p.next
            crt.elem = x        #回填最后一个元素
            crt = crt.next
    def sort(self):#移动链接
        p = self._head
        if p is None or p.next is None:
            return
        rem = p.next
        p.next = None
        while rem is not None:
            p = self._head
            q = None
            while p is not None and p.elem <= rem.elem:
                q = p
                p = p.next
            if q is None:
                self._head = rem
            else:
                q.next = rem
            q = rem
            rem = rem.next
            q.next = p







mlist1 = LList()
# for i in range(11):
#     #mlist1.prepend(i)
#     mlist1.append(i)
# for i in range(11,21):
#     mlist1.append(i)
# mlist1.printall()
for i in range(1,10):
    mlist1.append(i)
mlist1.printall()
mlist1.rev()
mlist1.printall()
