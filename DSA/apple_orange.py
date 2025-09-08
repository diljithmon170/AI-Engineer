s = 7      # start of the house
t = 11     # end of the house
a = 5      # apple tree position
b = 15     # orange tree position
apples = [-2, 2, 1]   # distances apples fall
oranges = [5, -6]     # distances oranges fall

count_a=0
count_o=0
for i in apples:
    asum = a+i
    if asum>=s and asum<=t:
        count_a+=1
for j in oranges:
    osum = b+j
    if osum>=s and osum<=t:
        count_o+=1
print(count_a,'\n',count_o)