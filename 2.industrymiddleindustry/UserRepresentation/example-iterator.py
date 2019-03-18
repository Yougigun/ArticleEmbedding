from collections import Iterator
class Fib(object):
    def __init__(self):
        self.a, self.b = 0, 1
        self.last_n=len(self)
        self.n=0
    
    # 返回迭代器对象本身
    def __len__(self):
        return 10

    def __iter__(self):
        return self
    
    # 返回容器下一个元素
    def __next__(self):
        if self.last_n==self.n : raise StopIteration
        self.a, self.b = self.b, self.a + self.b
        self.n+=1
        return self.a

def main():
    fib = Fib()    # fib 是一个迭代器
    print( 'isinstance(fib, Iterator): ', isinstance(fib, Iterator))

    for i in fib:
        print (i)