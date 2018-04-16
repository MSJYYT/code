# def quick_sort(lst):
#     qsort_rec(lst, 0, len(lst)-1)
#
# def qsort_rec(lst, l, r):
#     if l >= r : return
#     i = l
#     j = r
#     pivot = lst[i]
#     while i < j:
#         while i < j and lst[j] >= pivot:
#             j -= 1
#         if i < j:
#             lst[i] = lst[j]
#             i += 1
#         while i < j and lst[i] <= pivot:
#             i += 1
#         if i < j:
#             lst[j] = lst[i]
#             j -= 1
#     lst[i] = pivot
#     qsort_rec(lst, l, i-1)
#     qsort_rec(lst, i+1, r)
# lst = [6,2,7,3,8,9]
# quick_sort(lst)
# print(lst)
def quick_sort1(lst):
    def qsort(lst, begin, end):
        if begin >= end:
            return
        pivot = lst[begin]
        print('pivot = %d'% pivot)
        i = begin
        for j in range(begin + 1, end + 1):
            if lst[j] < pivot:
                i += 1
                print(lst[i])
                lst[i], lst[j] = lst[j], lst[i]

        lst[begin], lst[i] = lst[i], lst[begin]
        print('i = %d,lst[i]=%d begin = %d pivot = %d, lst[begin]= %d'% (i,lst[i],begin,pivot,lst[begin]))
       # print(lst)
        qsort(lst, begin, i-1)
        qsort(lst, i+1, end)

    qsort(lst, 0, len(lst) -1)
lst = [6,2,7,3,8,9]
quick_sort1(lst)
print(lst)