

class StackUnderflow(ValueError):
    pass

class SStack():
    def __init__(self):     #用list对象_elems存储中元素
        self._elems = []     #所有栈操作都映射到list操作

    def is_empty(self):
        return self._elems == []

    def top(self):
        if self._elems == []:
            raise StackUnderflow('in SStack.top()')
        return self._elems[-1]

    def push(self, elem):
        self._elems.append(elem)

    def pop(self):
        if self._elems == []:
            raise StackUnderflow('in SStack.top()')
        return self._elems.pop()

st1 = SStack()
st1.push(3)
st1.push(5)
while not st1.is_empty():
    print(st1.pop(),end=',')