class LNode:
    def __init__(self, elem, next_=None):
        self.elem = elem
        self.next = next_
class DLNode(LNode):
    def __init__(self, elem, prev=None,next_=None):
        LNode.__init__(self,elem,next_)
        self.prev = prev
class DLList:
    def prepend(self, elem):
        p = DLNode(elem, None, self._head)
        if self._head is None:
            self._rear = p
        else:
            p.next.prev = p
        self._head = p
    def append(self,elem):
        p = DLNode(elem, self._rear, None)
        if self._head is None:
            self._head = p
        else:
            p.prev.next = p
        self._rear = p
    def pop(self):
        e = self._head.elem
        self._head = self._head.next
        if self._head is not None:
            self._head.prev = None
        return e
    def pop_last(self):
        e  = self._rear.elem
        self._rear = self._rear.prev
        if self._rear is None:
            self._head = None
        else:
            self._rear.next = None
        return e
