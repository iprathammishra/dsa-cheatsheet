# Heap- Priority Queue

## Basic Heap Operations

(Min-Heap Implementation)

```python

import heapq

class MinHeap:
    def __init__(self):
        self.heap = []

    def insert(self, val):
        heapq.heappush(self.heap, val)

    def delete(self, val):
        self.heap.remove(val)
        heapq.heapify(self.heap)

    def extract_min(self):
        return heapq.heappop(self.heap)

    def get_min(self):
        return self.heap[0]

# Usage
min_heap = MinHeap()
min_heap.insert(3)
min_heap.insert(1)
min_heap.insert(6)
print(min_heap.extract_min())  # Output: 1

```

(Max-Heap Implementation)

```python

import heapq

class MaxHeap:
    def __init__(self):
        self.heap = []

    def insert(self, val):
        heapq.heappush(self.heap, -val)

    def delete(self, val):
        self.heap.remove(-val)
        heapq.heapify(self.heap)

    def extract_max(self):
        return -heapq.heappop(self.heap)

    def get_max(self):
        return -self.heap[0]

# Usage
max_heap = MaxHeap()
max_heap.insert(3)
max_heap.insert(1)
max_heap.insert(6)
print(max_heap.extract_max())  # Output: 6

```

## Kth Largest/Smallest Element

(Kth Largest Element)

```python

import heapq

def find_kth_largest(nums, k):
    return heapq.nlargest(k, nums)[-1]

# Usage
print(find_kth_largest([3,2,1,5,6,4], 2))  # Output: 5

```

(Kth Smallest Element)

```python

import heapq

def find_kth_smallest(nums, k):
    return heapq.nsmallest(k, nums)[-1]

# Usage
print(find_kth_smallest([3,2,1,5,6,4], 2))  # Output: 2

```

## Merge K Sorted Lists/Arrays

```python

import heapq

def merge_k_sorted_lists(lists):
    heap = []
    for i, lst in enumerate(lists):
        if lst:
            heapq.heappush(heap, (lst[0], i, 0))

    result = []
    while heap:
        val, list_idx, element_idx = heapq.heappop(heap)
        result.append(val)
        if element_idx + 1 < len(lists[list_idx]):
            heapq.heappush(heap, (lists[list_idx][element_idx + 1], list_idx, element_idx + 1))

    return result

# Usage
lists = [[1,4,5], [1,3,4], [2,6]]
print(merge_k_sorted_lists(lists))  # Output: [1, 1, 2, 3, 4, 4, 5, 6]

```

## Top K Frequent Elements

(Top K Frequent Elements)

```python

import heapq
from collections import Counter

def top_k_frequent(nums, k):
    count = Counter(nums)
    return heapq.nlargest(k, count.keys(), key=count.get)

# Usage
print(top_k_frequent([1,1,1,2,2,3], 2))  # Output: [1, 2]

```

(Top K Frequent Words)

```python

import heapq
from collections import Counter

def top_k_frequent_words(words, k):
    count = Counter(words)
    return heapq.nlargest(k, count.keys(), key=lambda word: (count[word], word))

# Usage
print(top_k_frequent_words(["i", "love", "leetcode", "i", "love", "coding"], 2))  # Output: ["i", "love"]
```

## Sliding Window Maximum/Minimum

(Sliding Window Maximum)

```python

import heapq
from collections import deque

def max_sliding_window(nums, k):
    result = []
    deq = deque()
    for i in range(len(nums)):
        if deq and deq[0] == i - k:
            deq.popleft()
        while deq and nums[deq[-1]] < nums[i]:
            deq.pop()
        deq.append(i)
        if i >= k - 1:
            result.append(nums[deq[0]])
    return result

# Usage
print(max_sliding_window([1,3,-1,-3,5,3,6,7], 3))  # Output: [3, 3, 5, 5, 6, 7]

```

(Sliding Window Minimum)

```python

import heapq
from collections import deque

def min_sliding_window(nums, k):
    result = []
    deq = deque()
    for i in range(len(nums)):
        if deq and deq[0] == i - k:
            deq.popleft()
        while deq and nums[deq[-1]] > nums[i]:
            deq.pop()
        deq.append(i)
        if i >= k - 1:
            result.append(nums[deq[0]])
    return result

# Usage
print(min_sliding_window([1,3,-1,-3,5,3,6,7], 3))  # Output: [-1, -3, -3, -3, 3, 3]

```

## Median in Data Stream

```python

import heapq

class MedianFinder:
    def __init__(self):
        self.low = []  # Max-heap
        self.high = []  # Min-heap

    def add_num(self, num):
        heapq.heappush(self.low, -num)
        heapq.heappush(self.high, -heapq.heappop(self.low))
        if len(self.low) < len(self.high):
            heapq.heappush(self.low, -heapq.heappop(self.high))

    def find_median(self):
        if len(self.low) > len(self.high):
            return -self.low[0]
        else:
            return (-self.low[0] + self.high[0]) / 2

# Usage
mf = MedianFinder()
mf.add_num(1)
mf.add_num(2)
print(mf.find_median())  # Output: 1.5
mf.add_num(3)
print(mf.find_median())  # Output: 2

```

## Reorganize String or Array

(Reorganize String)

```python

import heapq
from collections import Counter

def reorganize_string(s):
    count = Counter(s)
    max_heap = [(-freq, char) for char, freq in count.items()]
    heapq.heapify(max_heap)
    
    prev_freq, prev_char = 0, ''
    result = []
    
    while max_heap:
        freq, char = heapq.heappop(max_heap)
        result.append(char)
        if prev_freq < 0:
            heapq.heappush(max_heap, (prev_freq, prev_char))
        prev_freq, prev_char = freq + 1, char
    
    result = ''.join(result)
    return result if len(result) == len(s) else ""

# Usage
print(reorganize_string("aab"))  # Output: "aba"

```

## Find the Smallest Range Covering Elements from K Lists

```python

import heapq

def smallest_range(nums):
    min_heap = []
    max_val = float('-inf')
    
    for i, row in enumerate(nums):
        heapq.heappush(min_heap, (row[0], i, 0))
        max_val = max(max_val, row[0])
    
    best_range = float('-inf'), float('inf')
    
    while min_heap:
        min_val, i, j = heapq.heappop(min_heap)
        
        if max_val - min_val < best_range[1] - best_range[0]:
            best_range = min_val, max_val
        
        if j + 1 == len(nums[i]):
            return best_range
        
        next_val = nums[i][j + 1]
        heapq.heappush(min_heap, (next_val, i, j + 1))
        max_val = max(max_val, next_val)

# Usage
nums = [[4,10,15,24,26], [0,9,12,20], [5,18,22,30]]
print(smallest_range(nums))  # Output: (20, 24)

```

## Task Scheduling

```python

import heapq
from collections import Counter

def least_interval(tasks, n):
    count = Counter(tasks)
    max_heap = [-cnt for cnt in count.values()]
    heapq.heapify(max_heap)
    
    time = 0
    while max_heap:
        i, temp = 0, []
        while i <= n:
            if max_heap:
                cnt = heapq.heappop(max_heap)
                if cnt + 1 < 0:
                    temp.append(cnt + 1)
            time += 1
            if not max_heap and not temp:
                break
            i += 1
        for item in temp:
            heapq.heappush(max_heap, item)
    
    return time

# Usage
print(least_interval(["A","A","A","B","B","B"], 2))  # Output: 8

```

## Frequently Sorting

(Sort Characters by Frequency)

```python

import heapq
from collections import Counter

def frequency_sort(s):
    count = Counter(s)
    max_heap = [(-freq, char) for char, freq in count.items()]
    heapq.heapify(max_heap)
    
    result = []
    while max_heap:
        freq, char = heapq.heappop(max_heap)
        result.extend([char] * -freq)
    
    return ''.join(result)

# Usage
print(frequency_sort("tree"))  # Output: "eetr"

```

(Frequency Sort)

```python

import heapq
from collections import Counter

def frequency_sort_list(nums):
    count = Counter(nums)
    max_heap = [(-freq, num) for num, freq in count.items()]
    heapq.heapify(max_heap)
    
    result = []
    while max_heap:
        freq, num = heapq.heappop(max_heap)
        result.extend([num] * -freq)
    
    return result

# Usage
print(frequency_sort_list([1,1,1,2,2,3]))  # Output: [1, 1, 1, 2, 2, 3]

```

Here's an explanation for each of the heap problems with approaches and sample code.

**Kth Largest Element in an Array**

Problem: Find the Kth largest element in an unsorted array using a min-heap.

Approach:

- Use a min-heap of size K to maintain the K largest elements.
- Traverse through the array and push each element into the heap.
- If the heap size exceeds K, remove the smallest element.
- At the end, the root of the heap will be the Kth largest element.

```py
import heapq

def find_kth_largest(nums, k):
    min_heap = []
    for num in nums:
        heapq.heappush(min_heap, num)
        if len(min_heap) > k:
            heapq.heappop(min_heap)
    return min_heap[0]

nums = [3, 2, 1, 5, 6, 4]
k = 2
print(find_kth_largest(nums, k))  # Output: 5

```

**Merge K Sorted Lists**

Problem: Merge K sorted linked lists into one sorted list using a heap.

Approach:

- Use a min-heap to store the smallest element from each list.
- Pop the smallest element, add it to the result, and push the next element from the same list.
- Repeat until all lists are merged.

```py
import heapq

class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

def merge_k_sorted_lists(lists):
    min_heap = []
    for i, node in enumerate(lists):
        if node:
            heapq.heappush(min_heap, (node.val, i, node))

    dummy = ListNode()
    curr = dummy

    while min_heap:
        val, i, node = heapq.heappop(min_heap)
        curr.next = node
        curr = curr.next
        if node.next:
            heapq.heappush(min_heap, (node.next.val, i, node.next))

    return dummy.next

```

**Top K Frequent Elements**

Problem: Given an array, return the K most frequent elements.

Approach:

- Count the frequency of each element.
- Use a min-heap of size K to store the most frequent elements.
- At the end, the heap contains the K most frequent elements.

```py
import heapq
from collections import Counter

def top_k_frequent(nums, k):
    freq_map = Counter(nums)
    min_heap = []

    for num, freq in freq_map.items():
        heapq.heappush(min_heap, (freq, num))
        if len(min_heap) > k:
            heapq.heappop(min_heap)

    return [num for freq, num in min_heap]

nums = [1, 1, 1, 2, 2, 3]
k = 2
print(top_k_frequent(nums, k))  # Output: [1, 2]

```

**Reorganize String**

Problem: Rearrange characters in a string such that no two adjacent characters are the same.

Approach:

- Count the frequency of each character.
- Use a max-heap to store characters sorted by frequency.
- Build the result string by placing the most frequent character alternately.

```py
import heapq
from collections import Counter

def reorganize_string(s):
    freq_map = Counter(s)
    max_heap = [(-freq, char) for char, freq in freq_map.items()]
    heapq.heapify(max_heap)

    prev_freq, prev_char = 0, ''
    result = []

    while max_heap:
        freq, char = heapq.heappop(max_heap)
        result.append(char)

        if prev_freq < 0:
            heapq.heappush(max_heap, (prev_freq, prev_char))

        prev_freq, prev_char = freq + 1, char

    return "".join(result) if len(result) == len(s) else ""

s = "aab"
print(reorganize_string(s))  # Output: "aba"

```

**Find Median from Data Stream**

Problem: Efficiently find the median of a stream of integers.

Approach:

- Use two heaps: a max-heap for the left half and min-heap for the right half.
- Balance the heaps so their sizes differ by at most 1.
- The median is either the root of the max-heap or the average of the roots of both heaps.

```py
import heapq

class MedianFinder:
    def __init__(self):
        self.left = []  # Max-Heap
        self.right = []  # Min-Heap

    def add_num(self, num):
        heapq.heappush(self.left, -num)
        heapq.heappush(self.right, -heapq.heappop(self.left))
        if len(self.right) > len(self.left):
            heapq.heappush(self.left, -heapq.heappop(self.right))

    def find_median(self):
        if len(self.left) > len(self.right):
            return -self.left[0]
        return (-self.left[0] + self.right[0]) / 2

mf = MedianFinder()
mf.add_num(1)
mf.add_num(2)
print(mf.find_median())  # Output: 1.5
mf.add_num(3)
print(mf.find_median())  # Output: 2

```

** K Closest Points to Origin**

Problem: Find the K closest points to the origin (0,0) in a 2D plane.

Approach:

- Use a max-heap of size K to store the closest points.
- Push points into the heap with their distance from the origin.
- If the heap size exceeds K, remove the farthest point.

```py
import heapq

def k_closest(points, k):
    max_heap = []

    for x, y in points:
        dist = -(x**2 + y**2)  # Use negative distance for max-heap
        heapq.heappush(max_heap, (dist, (x, y)))
        if len(max_heap) > k:
            heapq.heappop(max_heap)

    return [point for _, point in max_heap]

points = [(1, 3), (-2, 2), (5, 8), (0, 1)]
k = 2
print(k_closest(points, k))  # Output: [(-2, 2), (0, 1)]

```

**Sliding Window Median**

Problem: Find the median of numbers in each sliding window of size K.

Approach:

- Use two heaps (max-heap and min-heap) to maintain the sliding window.
- Add and remove elements as the window slides, keeping the heaps balanced.
- Calculate the median from the heaps at each step.

```py
import heapq
from bisect import bisect_left

def sliding_window_median(nums, k):
    result = []
    min_heap, max_heap = [], []

    def balance_heaps():
        if len(max_heap) > len(min_heap) + 1:
            heapq.heappush(min_heap, -heapq.heappop(max_heap))
        if len(min_heap) > len(max_heap):
            heapq.heappush(max_heap, -heapq.heappop(min_heap))

    for i, num in enumerate(nums):
        heapq.heappush(max_heap, -num)
        heapq.heappush(min_heap, -heapq.heappop(max_heap))
        balance_heaps()

        if i >= k - 1:
            median = -max_heap[0] if k % 2 != 0 else (-max_heap[0] + min_heap[0]) / 2
            result.append(median)

            outgoing = nums[i - k + 1]
            if outgoing <= -max_heap[0]:
                max_heap.remove(-outgoing)
                heapq.heapify(max_heap)
            else:
                min_heap.remove(outgoing)
                heapq.heapify(min_heap)

            balance_heaps()

    return result

nums = [1, 3, -1, -3, 5, 3, 6, 7]
k = 3
print(sliding_window_median(nums, k))  # Output: [1, -1, -1, 3, 5, 6]

```

**Smallest Range Covering Elements from K Lists**

Problem: Find the smallest range that includes at least one number from each of K sorted lists.

Approach:

- Use a min-heap to track the smallest element from each list.
- Maintain the current maximum element across the lists.
- Track the range formed by the minimum and maximum elements.

```py
import heapq

def smallest_range(lists):
    min_heap = []
    max_val = float('-inf')
    for i, lst in enumerate(lists):
        heapq.heappush(min_heap, (lst[0], i, 0))
        max_val = max(max_val, lst[0])

    best_range = [float('-inf'), float('inf')]

    while min_heap:
        min_val, i, j = heapq.heappop(min_heap)
        if max_val - min_val < best_range[1] - best_range[0]:
            best_range = [min_val, max_val]

        if j + 1 < len(lists[i]):
            heapq.heappush(min_heap, (lists[i][j + 1], i, j + 1))
            max_val = max(max_val, lists[i][j + 1])
        else:
            break

    return best_range

lists = [[4, 10, 15, 24, 26], [0, 9, 12, 20], [5, 18, 22, 30]]
print(smallest_range(lists))  # Output: [20, 24]

```

**Maximum Sum Combination**

Problem: Find the K maximum sum combinations from two arrays.

Appraoch:

- Use a max-heap to track the largest sums.
- Start with the largest possible combination and push neighboring combinations into the heap.
- Extract K combinations.

```py
import heapq

def max_sum_combinations(A, B, k):
    A.sort(reverse=True)
    B.sort(reverse=True)

    max_heap = [(-(A[0] + B[0]), 0, 0)]
    visited = set((0, 0))
    result = []

    for _ in range(k):
        curr_sum, i, j = heapq.heappop(max_heap)
        result.append(-curr_sum)

        if i + 1 < len(A) and (i + 1, j) not in visited:
            heapq.heappush(max_heap, (-(A[i + 1] + B[j]), i + 1, j))
            visited.add((i + 1, j))

        if j + 1 < len(B) and (i, j + 1) not in visited:
            heapq.heappush(max_heap, (-(A[i] + B[j + 1]), i, j + 1))
            visited.add((i, j + 1))

    return result

A = [1, 4, 2, 3]
B = [2, 5, 1, 6]
k = 3
print(max_sum_combinations(A, B, k))  # Output: [10, 9, 9]

```
