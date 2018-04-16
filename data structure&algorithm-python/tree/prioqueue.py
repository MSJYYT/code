#基于堆的优先队列类
class PrioQueueError(ValueError):
    pass

class PrioQueue:
    def __init__(self, elist=[]):
        self._elems = list(elist)
        if elist:
            self.buildheap()
    def is_empty(self):
        return not self._elems
    def peek(self):
        if self.is_empty():
            raise PrioQueueError('in peek')
        return self._elems[0]
#入队
    def enqueue(self,e):
        self._elems.append(None)#先插入一个空值，留给后续插入用
        self.siftup(e,len(self._elems)-1)
    def siftup(self,e,last):
        elems, i, j = self._elems, last, (last-1)//2#last为最后一个值的索引，(last-1)//2为其父亲
        while i >0 and e < elems[j]:#每一次都跟父亲比，比父亲还小就把父亲变儿子
            elems[i] = elems[j]
            i, j = j, (j-1)//2
        elems[i] = e
#弹出元素
    def dequeue(self):
        if self.is_empty():
            raise PrioQueueError('is dequeue')
        elems = self._elems
        e0 = elems[0]
        e = elems.pop()
        if len(elems) > 0:
            self.siftdown(e, 0, len(elems))
        return e0
    def siftdown(self,e,begin, end):
        elems, i, j = self._elems,begin, begin*2+1  #begin为索引从0开始，begin*2+1为左儿子
        while j < end:
            if j+1 < end and elems[j+1] < elems[j]:  #先比较左右儿子哪个小
                j += 1
            if e < elems[j]:   #如果e小于左右儿子直接跳出
                break
            elems[i] = elems[j]    #如果e大于左右儿子则跟儿子交换
            i, j = j, 2*j +1
        elems[i] = e
#堆的初始构建
    def buildheap(self):
        end = len(self._elems)
        for i in range(end//2, -1, -1):
            self.siftdown(self._elems[i], i, end)
#堆应用：堆排序
    def heap_sort(elems):
        def siftdown(elems, e, begin, end):
            i, j = begin, begin*2+1
            while j < end:
                if j+1 < end and elems[j+1] < elems[j]:
                    j += 1
                if e < elems[j]:
                    break
                elems[i] = elems[j]
                i, j = j, 2*j+1
            elems[i] = e
        end = len(elems)
        for i in range(end//2, -1, -1):
            siftdown(elems, elems[i], i, end)
        for i in range((end-1), 0, -1):
            e = elems[i]
            elems[i] = elems[0]
            siftdown(elems, e, 0, i)


