# Binary Search

## Basic Binary Search

(Searching for a specific element in a sorted array)

```python

def binary_search(arr, target):
    left, right = 0, len(arr) - 1
    while left <= right:
        mid = left + (right - left) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    return -1

# Example usage:
arr = [1, 2, 3, 4, 5, 6, 7]
target = 4
print(binary_search(arr, target))  # Output: 3

```

(Find the first or last occurrence of a tarrget element in a sorted array)

```python

def find_first_occurrence(arr, target):
    left, right = 0, len(arr) - 1
    result = -1
    while left <= right:
        mid = left + (right - left) // 2
        if arr[mid] == target:
            result = mid
            right = mid - 1  # Move left to find the first occurrence
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    return result

def find_last_occurrence(arr, target):
    left, right = 0, len(arr) - 1
    result = -1
    while left <= right:
        mid = left + (right - left) // 2
        if arr[mid] == target:
            result = mid
            left = mid + 1  # Move right to find the last occurrence
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    return result

# Example usage:
arr = [1, 2, 2, 2, 3, 4, 5]
target = 2
print(find_first_occurrence(arr, target))  # Output: 1
print(find_last_occurrence(arr, target))   # Output: 3

```

(Determine the appropriate index to insert an element to mainitain sorted order)

```python

def lower_bound(arr, target):
    left, right = 0, len(arr)
    while left < right:
        mid = left + (right - left) // 2
        if arr[mid] < target:
            left = mid + 1
        else:
            right = mid
    return left

def upper_bound(arr, target):
    left, right = 0, len(arr)
    while left < right:
        mid = left + (right - left) // 2
        if arr[mid] <= target:
            left = mid + 1
        else:
            right = mid
    return left

# Example usage:
arr = [1, 2, 2, 2, 3, 4, 5]
target = 2
print(lower_bound(arr, target))  # Output: 1
print(upper_bound(arr, target))  # Output: 4

```

(Finding the minimum or maxium value that satisfies certain conditions in a function or sequence)

```python

def find_min_in_rotated_sorted_array(arr):
    left, right = 0, len(arr) - 1
    while left < right:
        mid = left + (right - left) // 2
        if arr[mid] > arr[right]:
            left = mid + 1
        else:
            right = mid
    return arr[left]

def find_peak_element(arr):
    left, right = 0, len(arr) - 1
    while left < right:
        mid = left + (right - left) // 2
        if arr[mid] < arr[mid + 1]:
            left = mid + 1
        else:
            right = mid
    return left

# Example usage:
rotated_arr = [4, 5, 6, 7, 0, 1, 2]
print(find_min_in_rotated_sorted_array(rotated_arr))  # Output: 0

arr = [1, 2, 3, 1]
print(find_peak_element(arr))  # Output: 2

```

(Search in a Rotated Sorted Array)

```python

def search_in_rotated_sorted_array(arr, target):
    left, right = 0, len(arr) - 1
    while left <= right:
        mid = left + (right - left) // 2
        if arr[mid] == target:
            return mid
        if arr[left] <= arr[mid]:
            if arr[left] <= target < arr[mid]:
                right = mid - 1
            else:
                left = mid + 1
        else:
            if arr[mid] < target <= arr[right]:
                left = mid + 1
            else:
                right = mid - 1
    return -1

# Example usage:
rotated_arr = [4, 5, 6, 7, 0, 1, 2]
target = 0
print(search_in_rotated_sorted_array(rotated_arr, target))  # Output: 4

```

(Search in a Matrix)

```python

def search_matrix(matrix, target):
    if not matrix or not matrix[0]:
        return False
    rows, cols = len(matrix), len(matrix[0])
    left, right = 0, rows * cols - 1
    while left <= right:
        mid = left + (right - left) // 2
        mid_value = matrix[mid // cols][mid % cols]
        if mid_value == target:
            return True
        elif mid_value < target:
            left = mid + 1
        else:
            right = mid - 1
    return False

# Example usage:
matrix = [
    [1, 3, 5, 7],
    [10, 11, 16, 20],
    [23, 30, 34, 60]
]
target = 3
print(search_matrix(matrix, target))  # Output: True

```

(Find the kth Smallest/Largest Element)

```python

def find_kth_smallest(matrix, k):
    import heapq
    min_heap = []
    for r in range(min(k, len(matrix))):
        heapq.heappush(min_heap, (matrix[r][0], r, 0))
    while k:
        element, r, c = heapq.heappop(min_heap)
        if c < len(matrix[0]) - 1:
            heapq.heappush(min_heap, (matrix[r][c + 1], r, c + 1))
        k -= 1
    return element

# Example usage:
matrix = [
    [1, 5, 9],
    [10, 11, 13],
    [12, 13, 15]
]
k = 8
print(find_kth_smallest(matrix, k))  # Output: 13

```

(Search for Range- Find the starting and ending position of a given target value.)

```python

def search_range(arr, target):
    def find_first():
        left, right = 0, len(arr) - 1
        while left <= right:
            mid = left + (right - left) // 2
            if arr[mid] < target:
                left = mid + 1
            else:
                right = mid - 1
        return left

    def find_last():
        left, right = 0, len(arr) - 1
        while left <= right:
            mid = left + (right - left) // 2
            if arr[mid] <= target:
                left = mid + 1
            else:
                right = mid - 1
        return right

    start, end = find_first(), find_last()
    if start <= end:
        return [start, end]
    return [-1, -1]

# Example usage:
arr = [5, 7, 7, 8, 8, 10]
target = 8
print(search_range(arr, target))  # Output: [3, 4]

```

(Finding the minimum or maximum value in functions that are strictly increasing or decreasing.)

```python

def minimum_speed_to_arrive_on_time(dist, hour):
    def can_arrive(speed):
        time = 0
        for d in dist[:-1]:
            time += (d + speed - 1) // speed
        time += dist[-1] / speed
        return time <= hour

    left, right = 1, 10**7
    while left < right:
        mid = left + (right - left) // 2
        if can_arrive(mid):
            right = mid
        else:
            left = mid + 1
    return left if can_arrive(left) else -1

# Example usage:
dist = [1, 3, 2]
hour = 2.7
print(minimum_speed_to_arrive_on_time(dist, hour))  # Output: 3

```

(Searching for an element in an array with unknown length, assuming it is infinite or very large.)

```python

def search_in_infinite_sorted_array(reader, target):
    # 'reader' is an interface with a method 'get(index)' that returns the element at index
    # If 'index' is out of bounds, 'get(index)' returns a large value (infinity)
    left, right = 0, 1
    while reader.get(right) < target:
        left = right
        right *= 2

    while left <= right:
        mid = left + (right - left) // 2
        mid_value = reader.get(mid)
        if mid_value == target:
            return mid
        elif mid_value < target:
            left = mid + 1
        else:
            right = mid - 1
    return -1

# Example usage:
class ArrayReader:
    def __init__(self, arr):
        self.arr = arr
    
    def get(self, index):
        if index >= len(self.arr):
            return float('inf')
        return self.arr[index]

arr = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
target = 7
reader = ArrayReader(arr)
print(search_in_infinite_sorted_array(reader, target))  # Output: 6

```

