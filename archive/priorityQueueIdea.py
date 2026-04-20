# https://www.geeksforgeeks.org/heap-queue-or-heapq-in-python/
# Python code to demonstrate working of
# nlargest() and nsmallest()

# importing "heapq" to implement heap queue
import heapq

# initializing list
values = [6, 7, 9, 4, 3, 5, 8, 10, 1]

# using heapify() to convert list into heap
heapq.heapify(values)

# using nlargest to print 3 largest numbers
# prints 10, 9 and 8
print("The 3 largest numbers in list are : ", end="")
print(heapq.nlargest(3, values))

# using nsmallest to print 3 smallest numbers
# prints 1, 3 and 4
print("The 3 smallest numbers in list are : ", end="")
print(heapq.nsmallest(3, values))


# Remove and return the smallest element from the heap
smallest = heapq.heappop(values)
 
# Print the smallest element and the updated heap
print("Smallest element:", smallest)
print("Heap after pop:", values)