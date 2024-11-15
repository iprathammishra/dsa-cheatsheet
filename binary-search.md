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

Here are 10 Binary Search questions, ranging from easy to hard, designed to cover a variety of binary search patterns.

# Easy Level

**Binary Search in Sorted Array**

Problem: Given a sorted array and a target, return the index of the target or -1 if not found.

Approach:

- Use classic binary search: repeatedly divide the search interval into halves.
- Check the middle element:
    - If it's the target, return the index.
    - If it's smaller, search the right half; otherwise search the left half.

Time Complexity: O(log n)

Example:

```py
nums = [1, 2, 3, 4, 5, 6]
target = 4
left, right = 0, len(nums) - 1
while left <= right:
    mid = (left + right) // 2
    if nums[mid] == target:
        print(mid)  # Output: 3
        break
    elif nums[mid] < target:
        left = mid + 1
    else:
        right = mid - 1
else:
    print(-1)

```

**Find First and Last Position of Element in Sorted Array**

Problem: Return the starting and ending indicies of a target in a sorted array.

Approach:

- Use binary search twice:
    -  First to find the leftmost position.
    - Then to find the rightmost position.

Time Complexity: O(log n)

Example:

```py
nums = [5, 7, 7, 8, 8, 10]
target = 8
def find_left(nums, target):
    left, right = 0, len(nums) - 1
    while left <= right:
        mid = (left + right) // 2
        if nums[mid] >= target:
            right = mid - 1
        else:
            left = mid + 1
    return left

def find_right(nums, target):
    left, right = 0, len(nums) - 1
    while left <= right:
        mid = (left + right) // 2
        if nums[mid] <= target:
            left = mid + 1
        else:
            right = mid - 1
    return right

left = find_left(nums, target)
right = find_right(nums, target)
print([left, right] if left <= right else [-1, -1])  # Output: [3, 4]

```

**Find Peak Element**

Problem: Find a peak element in an array (an element that is greater than its neighbors).

Approach:

- Use binary search to check midpoints:
    - If the middle element is greater than its right neighbor, move left.
    - Otherwise, move right.

Time complexity: O(log n)

Example:

```py
nums = [1, 2, 1, 3, 5, 6, 4]
left, right = 0, len(nums) - 1
while left < right:
    mid = (left + right) // 2
    if nums[mid] > nums[mid + 1]:
        right = mid
    else:
        left = mid + 1
print(left)  # Output: 5 (peak at index 5, value 6)

```

# Medium Level

**Search in Rotated Sorted Array**

Problem: Given a rotated sorted array, find the index of the target element.

Approach:
- Use binary search:
    - Determine which half of the array is sorted.
    - Adjust the search range accordingly.

Time complexity: O(log n)

Example:

```py
nums = [4, 5, 6, 7, 0, 1, 2]
target = 0
left, right = 0, len(nums) - 1
while left <= right:
    mid = (left + right) // 2
    if nums[mid] == target:
        print(mid)  # Output: 4
        break
    if nums[left] <= nums[mid]:
        if nums[left] <= target < nums[mid]:
            right = mid - 1
        else:
            left = mid + 1
    else:
        if nums[mid] < target <= nums[right]:
            left = mid + 1
        else:
            right = mid - 1
else:
    print(-1)

```

**Find Minimum in Rotated Sorted Array**

Problem: Find the minimum element in a rotated sorted array.

Approach:
- Use binary search:
    - Compare the middle element with the rightmost element to decide the direction.

Time complexity: O(log n)

Example:

```py
nums = [3, 4, 5, 1, 2]
left, right = 0, len(nums) - 1
while left < right:
    mid = (left + right) // 2
    if nums[mid] > nums[right]:
        left = mid + 1
    else:
        right = mid
print(nums[left])  # Output: 1

```

**Kth Smallest Element in a Sorted Matrix**

Problem: Find the k-th smallest element in an n x n sorted matrix.

Approach:

- Use binary search on the range of values.
- Count the number of elements less than or equal to the middle value.

Time complexity: O(n log(max-min))

Example:

```py
import bisect
matrix = [
    [1, 5, 9],
    [10, 11, 13],
    [12, 13, 15]
]
k = 8
n = len(matrix)
left, right = matrix[0][0], matrix[-1][-1]
while left < right:
    mid = (left + right) // 2
    count = 0
    for row in matrix:
        count += bisect.bisect_right(row, mid)
    if count < k:
        left = mid + 1
    else:
        right = mid
print(left)  # Output: 13

```
# Hard Level

**Median of Two Sorted Arrays**

Problem: Find the median of two sorted arrays.

Approach:

- Use binary search on the smaller array to partition the arrays into two halves.
- Ensure both halves have the same number of elements.

Time complexity: O(log(min(m,n))), where m and n are the lengths of the arrays.

Example:

```py
nums1 = [1, 3]
nums2 = [2]

def find_median(nums1, nums2):
    if len(nums1) > len(nums2):
        nums1, nums2 = nums2, nums1  # Ensure nums1 is the smaller array
    x, y = len(nums1), len(nums2)
    left, right = 0, x

    while left <= right:
        partition_x = (left + right) // 2
        partition_y = (x + y + 1) // 2 - partition_x

        max_left_x = nums1[partition_x - 1] if partition_x > 0 else float('-inf')
        min_right_x = nums1[partition_x] if partition_x < x else float('inf')

        max_left_y = nums2[partition_y - 1] if partition_y > 0 else float('-inf')
        min_right_y = nums2[partition_y] if partition_y < y else float('inf')

        if max_left_x <= min_right_y and max_left_y <= min_right_x:
            if (x + y) % 2 == 0:
                return (max(max_left_x, max_left_y) + min(min_right_x, min_right_y)) / 2
            else:
                return max(max_left_x, max_left_y)
        elif max_left_x > min_right_y:
            right = partition_x - 1
        else:
            left = partition_x + 1

print(find_median(nums1, nums2))  # Output: 2
```

**Split Array Largest Sum**

Problem: Divide an array into m subarrays such that the largest sum amoung them is minimized.

Approach:

- Use binary search on the range of possible sums.
- Check if a mid value can be a valid maximum sum using a greedy approach.

Time complexity: O(n* log(sum(nums)))

Example:

```py
nums = [7, 2, 5, 10, 8]
m = 2

def is_valid(mid):
    subarray_count = 1
    current_sum = 0
    for num in nums:
        if current_sum + num > mid:
            subarray_count += 1
            current_sum = num
            if subarray_count > m:
                return False
        else:
            current_sum += num
    return True

left, right = max(nums), sum(nums)
while left < right:
    mid = (left + right) // 2
    if is_valid(mid):
        right = mid
    else:
        left = mid + 1

print(left)  # Output: 18

```

**Aggresive Cows (Minimum Distance Between Cows)**

Problem: Place cows in stalls such that the minimum distance between any two cows is maximized.

Approach:

- Use binary search on the distance range.
- Check if a distance is feasible using a greedy approach.

Time complexity: O(n * log(max-min))

Example:

```py
stalls = [1, 2, 4, 8, 9]
k = 3

def can_place_cows(mid):
    count = 1  # Place the first cow
    last_position = stalls[0]
    for i in range(1, len(stalls)):
        if stalls[i] - last_position >= mid:
            count += 1
            last_position = stalls[i]
            if count == k:
                return True
    return False

stalls.sort()
left, right = 1, stalls[-1] - stalls[0]
while left <= right:
    mid = (left + right) // 2
    if can_place_cows(mid):
        left = mid + 1
    else:
        right = mid - 1

print(right)  # Output: 4

```

**Allocate Minimum Pages**

Problem: Allocate books to students such that the maximum pages assigned to a student are minimized.

Approach:

- Similar to the "Split Array Largest Sum" problem.

Time complexity: O(n *log(sum(pages)))

Example:

```py
books = [12, 34, 67, 90]
students = 2

def is_valid(mid):
    student_count = 1
    current_pages = 0
    for pages in books:
        if current_pages + pages > mid:
            student_count += 1
            current_pages = pages
            if student_count > students:
                return False
        else:
            current_pages += pages
    return True

left, right = max(books), sum(books)
while left < right:
    mid = (left + right) // 2
    if is_valid(mid):
        right = mid
    else:
        left = mid + 1

print(left)  # Output: 113

```

**Count of Smaller Numbers After Self**

Problem: Given an array, return an array counts where counts[i] is the number of smaller elements to the right of nums[i].

Approach:

- Traverse the array from right to left to process elements in reverse order.
- Use a sorted list s to keep track of elements to the right of the current number in sorted order.
- For each number:
    - Use binary search (bisect_left) to find the position where the current number should be inserted in s. This position gives the count of smaller elements.
    - Append this position to the result array.
    - Insert the current number into the sorted list s at the found position to maintain the sorted order.
- Finally, reverse the result array to match the original order of nums.


Time complexity: O(n^2)

Example:

```py
from bisect import bisect_left

class Solution:
    def countSmaller(self, nums: List[int]) -> List[int]:
        s = []  # A sorted list to maintain the right-side elements.
        r = []  # Result array to store counts of smaller elements.
        
        # Traverse the array from right to left.
        for num in nums[::-1]:
            # Find the position where the current number would fit.
            i = bisect_left(s, num)
            
            # Append the index (count of smaller numbers) to the result.
            r.append(i)
            
            # Insert the current number at the correct position in the sorted list.
            s.insert(i, num)
        
        # Since we traversed nums in reverse, reverse the result to match original order.
        return r[::-1]

```