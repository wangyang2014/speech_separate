import math

inValue = input().split(' ')

N,M = int(inValue[0]),int(inValue[1])

value = min(int(math.log(N,2)) ,M)
data = N / 2**(value)

if data == int(data):
    p = data
else:
    p = int(data) + 1



count = value + p + 1
print(count)



    
        

    


