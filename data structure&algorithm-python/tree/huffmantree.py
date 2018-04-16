from tree import prioqueue as pq
from tree import classtreeNode as classNode
#哈夫曼数
class HTNode(classNode.BinTNode):
    def __lt__(self, othernode):
        return self.data < othernode.data
class HuffmanPrioQ(pq.PrioQueue):
    def number(self):
        return len(self._elems)
    def HuffmanTree(weights):
        trees = HuffmanPrioQ()
        for w in weights:
            trees.enqueue(HTNode(w))
        while trees.number() > 1:
            t1 = trees.dequeue()
            t2 = trees.dequeue()
            x = t1.data + t2.data
            trees.enqueue(HTNode(x, t1, t2))
        return trees.dequeue()