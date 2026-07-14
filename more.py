# Question 1: Find the Largest Element in an Array ⭐

# Problem Statement:
# Given an integer array, find the largest element.

# Example 1

# Input:
# arr = [2, 5, 1, 9, 6]

# Output:
# 9

# Example 2

# Input:
# arr = [-5, -2, -10, -1]

# Output:
# -1
# Interview Difficulty



arr = [-5, -2, -10, -1]



# def maxinarr(arr):
#     maxx = float('-inf')
#     for i in arr:
#         if i > maxx:
#             maxx = i
#     return maxx


def smaxinarr(arr):
    maxx = float('-inf')
    second = float('-inf')
    for i in arr:
        if i > maxx:
            second = maxx
            maxx = i
        elif i > second and i != maxx:
            second = i
    return second if second != float('-inf') else -1






##check sorted array :


def check(arr):

    for i in range(len(arr)-1):
        if arr[i] > arr[i+1]:
            return False
    return True



# print(check(arr=[5,2,3,4,5]))
        
def reversearray(arr):
    low = 0
    high = len(arr)-1

    while low< high:
        temp = arr[low]
        arr[low] = arr[high]
        arr[high] = temp
        low +=1
        high -=1

    return arr

# print(reversearray(arr=[1,2,3,4,9,4,6]))



## count even and odd 


def counevenandodd(arr):
    even,odd = 0,0

    for i in arr:
        if i%2 ==0:
            even +=1
        else:
            odd +=1
    return f"even coun = {even}\nodd count = {odd}"


# print(counevenandodd(arr=[10,20,30,40,50,]))
        



#question no 6 average and sum

def avrgandsum(arr):
    sum = 0
    avera = 0
    for i in arr:
        sum += i
    return f"sum is {sum}\naverage is {sum//len(arr)}"


# print(avrgandsum(arr=[5,15,25]))

#day 4

def maxandmin(arr):
    maxx  = float('-inf')
    min = float('inf')
    for i in arr:
        if i > maxx :
            maxx = i
        elif i < min:
            min = i
    return min,maxx

# print(maxandmin(arr=[4,2,9,1,7,-2]))

##searching the elements using binary search


def binarysearch(arr,tobefoundelement):
    low = 0 
    high = len(arr)-1
    while low <= high:
        mid = low + (high-low)//2
        if arr[mid] == tobefoundelement:
            return f"the element is found at {mid}"
        elif tobefoundelement < arr[mid]:
            high = mid -1
        elif tobefoundelement > arr[mid]:
            low = mid +1
    return " not found"

arr=[1,2,3,4,5,6,7]
print(binarysearch(arr,5))

        
             
