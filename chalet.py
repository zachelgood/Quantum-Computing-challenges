from itertools import groupby
l=[1,1,1,1,0,0,1]
list=([list(v) for k,v in groupby(l, key = lambda x: x != 0) if k != 0])
if len(list)%2 ==0:
    print(0)
else:
    print(1)
