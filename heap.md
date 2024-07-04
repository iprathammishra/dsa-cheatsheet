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
