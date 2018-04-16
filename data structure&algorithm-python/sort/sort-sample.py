#假定表中元素是下面定义的record类的对象
class record:
    def __init__(self, key, datum):
        self.key = key
        self.datum = datum

    #插入排序
    def insert_sort(lst):
        for i in range(1, len(lst)):            #开始时片段[0:1]已排序
            x = lst[i]
            j = i
            while j > 0 and lst[j-1] > x:
                lst[j] = lst[j-1]                #反序逐个后移元素，确定插入位置
                j -= 1
            lst[j] = x

    #选择排序
    def select_sort(lst):
        for i in range(len(lst) - 1):           #只需循环len(lst)-1次
            k = i
            for j in range(i, len(lst)):        #k是已知最小元素的位置
                if lst[j] < lst[k]:
                    k = j
            if i != k:                      #lst[k]是确定的最小元素，检查是否需要交换
                lst[i], lst[k] = lst[k], lst[i]

    #冒泡排序
    def bubble_sort(lst):
        for i in range(len(lst)):
            for j in range(1, len(lst)-i):
                if lst[j - 1] > lst[j]:
                    lst[j-1], lst[j] = lst[j], lst[j-1]
    #改进冒泡
    def bubble_sort1(lst):
        for i in range(len(lst)):
            found = False
            for j in range(1,len(lst)-i):
                if lst[j-1] > lst[j]:
                    lst[j-1], lst[j] = lst[j], lst[j-1]
                    found = True
            if not found:
                break