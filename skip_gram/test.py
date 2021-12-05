import collections
from functools import cmp_to_key

res = collections.defaultdict(int)

dict1 = input().rstrip()
dict2 = input().rstrip()

dict1 = dict1[1:-1]
dict2 = dict2[1:-1]

dict1 = dict1.split(",")
dict2 = dict2.split(",")


def compare_(x1, x2):
    if type(x1[0]) == str:
        x1 = int(ord(x1[0]))
    else:
        x1 = x1[0]
    if type(x2[0]) == str:
        x2 = int(ord(x2[0]))
    else:
        x2 = x2[0]
    
    if x1 > x2:
        return 1
    elif x1 < x2:
        return -1
    else:
        return 0

for item in dict1:
    item = item.split(":")
    if '"' in item[0]:
        res[item[0][1:-1]] += int(item[1])
    else:
        res[int(item[0])] += int(item[1])

for item in dict2:
    item = item.split(":")
    if '"' in item[0]:
        res[item[0][1:-1]] += int(item[1])
    else:
        res[int(item[0])] += int(item[1])

res = sorted(res.items(), key=cmp_to_key(lambda x,y:compare_(x,y)))
res = dict(res)
res = str(res)
res = res.replace("'",'"')
res = res.replace(" ","")
for i in range(len(res)):
    print(res[i] == ' ')
print(len(res))
print(res)
