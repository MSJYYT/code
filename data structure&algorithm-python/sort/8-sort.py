#插入排序
#插入排序的基本操作就是将一个数据插入到已经排好序的有序数据中，
# 从而得到一个新的、个数加一的有序数据，算法适用于少量数据的排序，
# 时间复杂度为O(n^2)。是稳定的排序方法。插入算法把要排序的数组分成两部分：
# 第一部分包含了这个数组的所有元素，但将最后一个元素除外
# （让数组多一个空间才有插入的位置），而第二部分就只包含这一个元素（即待插入元素）。
# 在第一部分排序完成后，再将这个最后元素插入到已排好序的第一部分中。
def insert_sort(lst):
    for i in range(1, len(lst)):            #开始时片段[0:1]已排序
        x = lst[i]
        j = i
        while j > 0 and lst[j-1] > x:
            lst[j] = lst[j-1]                #反序逐个后移元素，确定插入位置
            j -= 1
        lst[j] = x
    return lst

#希尔排序
#希尔排序(Shell Sort)是插入排序的一种。也称缩小增量排序，
# 是直接插入排序算法的一种更高效的改进版本。希尔排序是非稳定排序算法。
# 该方法因DL．Shell于1959年提出而得名。 希尔排序是把记录按下标的一定增量分组，
# 对每组使用直接插入排序算法排序；随着增量逐渐减少，每组包含的关键词越来越多，
# 当增量减至1时，整个文件恰被分成一组，算法便终止。
def shell_sort(lists):
    count = len(lists)
    step = 2
    group = count // step
    while group > 0:
        for i in range(0, group):
            j = i + group
            while j < count:
                k = j - group
                key = lists[j]
                while k >= 0:
                    if lists[k] > key:
                        lists[k + group] = lists[k]
                        lists[k] = key
                    k -= group
                j += group
        group //= step
    return lists

#冒泡排序
#它重复地走访过要排序的数列，一次比较两个元素，
# 如果他们的顺序错误就把他们交换过来。
# 走访数列的工作是重复地进行直到没有再需要交换，
# 也就是说该数列已经排序完成。
def bubble_sort(lists):
    count = len(lists)
    for i in range(0, count):
        for j in range(i + 1,count):
            if lists[i] > lists[j]:
                lists[i], lists[j] = lists[j], lists[i]
    return lists

#快速排序
#通过一趟排序将要排序的数据分割成独立的两部分，
# 其中一部分的所有数据都比另外一部分的所有数据都要小，
# 然后再按此方法对这两部分数据分别进行快速排序，
# 整个排序过程可以递归进行，以此达到整个数据变成有序序列。
def quick_sort(lists, left, right):
    if left >= right:
        return lists
    key = lists[left]
    low = left
    high = right
    while left < right:
        while left < right and lists[right] >= key:
            right -= 1
        lists[left] = lists[right]
        while left < right and lists[left] <= key:
            left += 1
        lists[right] = lists[left]
    lists[right] = key

    quick_sort(lists, low, left-1)
    quick_sort(lists, left+1, high)
    return lists

#直接选择排序
#基本思想：第1趟，在待排序记录r1 ~ r[n]中选出最小的记录，
# 将它与r1交换；第2趟，在待排序记录r2 ~ r[n]中选出最小的记录，
# 将它与r2交换；以此类推，第i趟在待排序记录r[i] ~ r[n]中选出最小的记录，
# 将它与r[i]交换，使有序序列不断增长直到全部排序完毕。
def select_sort(lists):
    count = len(lists)
    for i in range(0, count):
        min = i
        for j in range(i+1, count):
            if lists[j] < lists[min]:
                min = j
        lists[min], lists[i] = lists[i], lists[min]
    return lists

#堆排序
#堆排序(Heapsort)是指利用堆积树（堆）
# 这种数据结构所设计的一种排序算法，它是选择排序的一种。
# 可以利用数组的特点快速定位指定索引的元素。
# 堆分为大根堆和小根堆，是完全二叉树。
# 大根堆的要求是每个节点的值都不大于其父节点的值，
# 即A[PARENT[i]] >= A[i]。在数组的非降序排序中，
# 需要使用的就是大根堆，因为根据大根堆的要求可知，
# 最大的值一定在堆顶
def adjust_heap(lists,i,size):
    lchild = 2*i + 1
    rchild = 2*i + 2
    max = i
    if i < size//2:
        if lchild < size and lists[lchild] > lists[max]:
            max = lchild
        if rchild < size and lists[rchild] > lists[max]:
            max = rchild
        if max != i:
            lists[max], lists[i] = lists[i], lists[max]
            adjust_heap(lists, max, size)
def build_heap(lists, size):
    for i in range(0, (size//2))[::-1]:#从后往前出父节点
        adjust_heap(lists, i, size)
def heap_sort(lists):
    size = len(lists)
    build_heap(lists, size)
    for i in range(0, size)[::-1]:#从后往前弹
        lists[0], lists[i] = lists[i], lists[0]#把堆顶的值弹出到lists[i]
        adjust_heap(lists, 0, i)#每次最大值都在堆顶


def merge(left, right):
    i, j = 0,0
    result = []
    while i < len(left) and j < len(right):
        if left[i] <= right[j]:
            result.append(left[i])
            i += 1
        else:
            result.append(left[j])
            j += 1
    result += left[i:]
    result += right[j:]
    return result
def merge_sort(lists):
    if len(lists) <= 1:
        return lists
    num = len(lists)//2
    left = merge_sort(lists[:num])
    right = merge_sort(lists[num:])
    return merge(left, right)

#list = [8,9,1,7,2,3,5,4,6,0]
list = [6,2,7,3,8,9]
#insert_sort(list)
#shell_sort(list)
#bubble_sort(list)
#quick_sort(list,0,5)
#select_sort(list)
heap_sort(list)
print(list)