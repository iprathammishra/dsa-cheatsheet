# Prefix Sum

## Range Sum Queries

(Range Sum Query- Immutable)

```python

class NumArray:
    def __init__(self, nums):
        self.prefix_sums = [0] * (len(nums) + 1)
        for i in range(len(nums)):
            self.prefix_sums[i + 1] = self.prefix_sums[i] + nums[i]
    
    def sum_range(self, i, j):
        return self.prefix_sums[j + 1] - self.prefix_sums[i]

# Example usage:
nums = [1, 2, 3, 4, 5]
obj = NumArray(nums)
print(obj.sum_range(1, 3))

```

(Range Sum Query 2D- Immutable)

```python

class NumMatrix:
    def __init__(self, matrix):
        if not matrix:
            return
        rows, cols = len(matrix), len(matrix[0])
        self.prefix_sums = [[0] * (cols + 1) for _ in range(rows + 1)]
        for r in range(rows):
            for c in range(cols):
                self.prefix_sums[r + 1][c + 1] = (matrix[r][c] + self.prefix_sums[r + 1][c] +
                                                  self.prefix_sums[r][c + 1] - self.prefix_sums[r][c])
    
    def sum_region(self, row1, col1, row2, col2):
        return (self.prefix_sums[row2 + 1][col2 + 1] - self.prefix_sums[row2 + 1][col1] -
                self.prefix_sums[row1][col2 + 1] + self.prefix_sums[row1][col1])

# Example usage:
matrix = [
    [3, 0, 1, 4, 2],
    [5, 6, 3, 2, 1],
    [1, 2, 0, 1, 5],
    [4, 1, 0, 1, 7],
    [1, 0, 3, 0, 5]
]
obj = NumMatrix(matrix)
print(obj.sum_region(2, 1, 4, 3))

```

## Subarray Sum Equals K

(Subarray Sum Equals K)

```python

def subarray_sum(nums, k):
    count = 0
    prefix_sum = 0
    prefix_sums = {0: 1}
    for num in nums:
        prefix_sum += num
        if prefix_sum - k in prefix_sums:
            count += prefix_sums[prefix_sum - k]
        prefix_sums[prefix_sum] = prefix_sums.get(prefix_sum, 0) + 1
    return count

# Example usage:
nums = [1, 1, 1]
k = 2
print(subarray_sum(nums, k))

```

(Continuous Subarray Sum)

```python

def check_subarray_sum(nums, k):
    prefix_sum = 0
    prefix_sums = {0: -1}
    for i, num in nums:
        prefix_sum += num
        if k != 0:
            prefix_sum %= k
        if prefix_sum in prefix_sums:
            if i - prefix_sums[prefix_sum] > 1:
                return True
        else:
            prefix_sums[prefix_sum] = i
    return False

# Example usage:
nums = [23, 2, 4, 6, 7]
k = 6
print(check_subarray_sum(nums, k))

```

## Maximum/Minimum Sum Subarrays

(Maximum Subarray (Kadane's Algorithm))

```python

def max_subarray(nums):
    max_so_far = nums[0]
    max_ending_here = nums[0]
    for num in nums[1:]:
        max_ending_here = max(num, max_ending_here + num)
        max_so_far = max(max_so_far, max_ending_here)
    return max_so_far

# Example usage:
nums = [-2,1,-3,4,-1,2,1,-5,4]
print(max_subarray(nums))

```

(Maximum Sum Circular Subarray)

```python

def max_subarray_sum_circular(nums):
    def kadane(arr):
        max_so_far = arr[0]
        max_ending_here = arr[0]
        for num in arr[1:]:
            max_ending_here = max(num, max_ending_here + num)
            max_so_far = max(max_so_far, max_ending_here)
        return max_so_far

    total_sum = sum(nums)
    max_kadane = kadane(nums)
    max_wrap = total_sum - kadane([-num for num in nums])

    if max_wrap == 0:
        return max_kadane
    return max(max_kadane, max_wrap)

# Example usage:
nums = [1,-2,3,-2]
print(max_subarray_sum_circular(nums))

```

## Minimum Size Subarray Sum

```python

def min_subarray_len(target, nums):
    n = len(nums)
    min_len = float('inf')
    current_sum = 0
    left = 0
    for right in range(n):
        current_sum += nums[right]
        while current_sum >= target:
            min_len = min(min_len, right - left + 1)
            current_sum -= nums[left]
            left += 1
    return min_len if min_len != float('inf') else 0

# Example usage:
target = 7
nums = [2,3,1,2,4,3]
print(min_subarray_len(target, nums))

```

## Prefix Sum with Hashing

(Subarray Sum Equals K)

```python

def subarray_sum(nums, k):
    count = 0
    prefix_sum = 0
    prefix_sums = {0: 1}
    for num in nums:
        prefix_sum += num
        if prefix_sum - k in prefix_sums:
            count += prefix_sums[prefix_sum - k]
        prefix_sums[prefix_sum] = prefix_sums.get(prefix_sum, 0) + 1
    return count

# Example usage:
nums = [1, 1, 1]
k = 2
print(subarray_sum(nums, k))

```

(Continuous Subarray Sum)

```python

def check_subarray_sum(nums, k):
    prefix_sum = 0
    prefix_sums = {0: -1}
    for i, num in enumerate(nums):
        prefix_sum += num
        if k != 0:
            prefix_sum %= k
        if prefix_sum in prefix_sums:
            if i - prefix_sums[prefix_sum] > 1:
                return True
        else:
            prefix_sums[prefix_sum] = i
    return False

# Example usage:
nums = [23, 2, 4, 6, 7]
k = 6
print(check_subarray_sum(nums, k))

```

(Subarrays with K different Integers)

```python

from collections import defaultdict

def subarrays_with_k_distinct(nums, k):
    def at_most_k(k):
        count = defaultdict(int)
        left = 0
        result = 0
        for right in range(len(nums)):
            if count[nums[right]] == 0:
                k -= 1
            count[nums[right]] += 1
            while k < 0:
                count[nums[left]] -= 1
                if count[nums[left]] == 0:
                    k += 1
                left += 1
            result += right - left + 1
        return result
    
    return at_most_k(k) - at_most_k(k - 1)

# Example usage:
nums = [1, 2, 1, 2, 3]
k = 2
print(subarrays_with_k_distinct(nums, k))

```
