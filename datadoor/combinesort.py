#合并两个有序列表，合并后元素不重复，并且排序，不实用sort
def combine_sort_1(a,b):
    comb_st = sorted(set(a) | set(b))
    return comb_st
def combine_sort_2(a,b):
    res = a
    for _b in b:
        #去重复的元素
        if _b in res:
            continue
        for (idx,_a) in enumerate(res[:]):
            if _b > _a:
                #如果已经到列表末尾了
                if idx == len(res) - 1:
                    res.append(_b)
                else:
                    continue
            if _b <= _a:
                res.insert(idx,_b)
                break
    return res
import random
def main():
    first = sorted(random.sample(range(100),random.randint(3,6)))
    second = sorted(random.sample(range(100),random.randint(3,6)))
    print('first:',first)
    print('second:',second)
    comb = combine_sort_2(first,second)
    comb1 = combine_sort_1(first,second)
    print('comb1:',comb1)
    print('comb:',comb)
if __name__ == '__main__':
    #for i in range(100000):
        main()