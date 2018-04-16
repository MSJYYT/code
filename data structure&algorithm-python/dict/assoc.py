class Assoc:
    def __init__(self,key,value):
        self.key = key
        self.value = value
    def __lt__(self, other):        #有时需要考虑序
        return self.key < other.key
    def __le__(self, other):
        return self.key < other.key or self.key == other.key
    def __str__(self):              #定义字符串表示形式便于输出和交互
        return "Assoc{{0},{1}}".format(self.key,self.value)