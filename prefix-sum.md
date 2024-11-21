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

Here are some patterns to take a look for revision.

**Running Sum of 1D Array**

Problem: Given an array, return a new array where each element is the sum of all previous elements, including itself.

Approach:

- Use a single traversal and maintain a running sum. Update each element by adding it to the cumulative sum so far.

```py
def runningSum(nums):
    for i in range(1, len(nums)):
        nums[i] += nums[i - 1]
    return nums

```

**Find Pivot Index**

Problem: Find the index where the sum of elements on its left equals the sum on its right.

Approach:

- Use the total sum and calculate the left sum as you iterate. The right sum is `total - leftSum - nums[i]`.

```py
def pivotIndex(nums):
    total = sum(nums)
    leftSum = 0
    for i, num in enumerate(nums):
        if leftSum == total - leftSum - num:
            return i
        leftSum += num
    return -1

```

**Subarray Sum Equals K**

Problem: Find the total number of continuous subarrays that sum to k.

Approach:

- Use a dictionary to store the prefix sums. For each prefix sum, check if (prefixSum - k) exists, indicating a subarray that sums to k.

```py
def subarraySum(nums, k):
    count = 0
    prefixSum = 0
    prefixMap = {0: 1}
    for num in nums:
        prefixSum += num
        count += prefixMap.get(prefixSum - k, 0)
        prefixMap[prefixSum] = prefixMap.get(prefixSum, 0) + 1
    return count

```

**Find the Longest Subarray with Equal 0s and 1s**

Problem: Replace 0 with -1 and find the longest subarray with a sum of 0.

Approach:

- Use a prefix sum and a hash map to store the first occurrence of each prefix sum. If a prefix sum repeats, the subarray between these indices has a sum of 0.

```py
def findMaxLength(nums):
    prefixMap = {0: -1}
    maxLength = 0
    prefixSum = 0
    for i, num in enumerate(nums):
        prefixSum += 1 if num == 1 else -1
        if prefixSum in prefixMap:
            maxLength = max(maxLength, i - prefixMap[prefixSum])
        else:
            prefixMap[prefixSum] = i
    return maxLength

```

**Range Sum Query - Immutable**

Problem: Given an array, calculate the sum of elements in a given range multiple times.

Approach:

- Precompute the prefix sum array. The sum of any range `[i, j]` is `prefix[j + 1] - prefix[i]`.

```py
class NumArray:
    def __init__(self, nums):
        self.prefix = [0]
        for num in nums:
            self.prefix.append(self.prefix[-1] + num)

    def sumRange(self, i, j):
        return self.prefix[j + 1] - self.prefix[i]

```

**Minimum Size Subarray Sum**

Problem: Find the minimal length of a subarray whose sum is greater than or equal to s.

Approach:

- Use a sliding window approach combined with the prefix sum to calculate sums on the fly and minimize the window size.

```py
def minSubArrayLen(target, nums):
    left = 0
    currentSum = 0
    minLength = float('inf')
    for right in range(len(nums)):
        currentSum += nums[right]
        while currentSum >= target:
            minLength = min(minLength, right - left + 1)
            currentSum -= nums[left]
            left += 1
    return minLength if minLength != float('inf') else 0

```

**Maximum Sum Rectangle in a 2D Matrix**

Problem: Find the submatrix with the maximum sum in a 2D matrix.

Approach:

- Use a prefix sum for rows and reduce the problem to a 1D maximum subarray sum using Kadane's algorithm.

```py
def maxSumRectangle(matrix):
    rows, cols = len(matrix), len(matrix[0])
    maxSum = float('-inf')
    for top in range(rows):
        temp = [0] * cols
        for bottom in range(top, rows):
            for col in range(cols):
                temp[col] += matrix[bottom][col]
            maxSum = max(maxSum, kadane(temp))
    return maxSum

def kadane(arr):
    maxSum = float('-inf')
    currentSum = 0
    for num in arr:
        currentSum = max(num, currentSum + num)
        maxSum = max(maxSum, currentSum)
    return maxSum

```

**Count Range Sum**

Problem: Count the number of subarrays with sums in a given range [lower, upper].

Approach:

- Use a prefix sum and a sorted list to count valid subarrays using binary search.

```py
from sortedcontainers import SortedList
def countRangeSum(nums, lower, upper):
    prefixSum = 0
    sortedList = SortedList([0])
    count = 0
    for num in nums:
        prefixSum += num
        count += sortedList.bisect_right(prefixSum - lower) - sortedList.bisect_left(prefixSum - upper)
        sortedList.add(prefixSum)
    return count

```

**Maximum Sum of Two Non-Overlapping Subarrays**

Problem: Find two non-overlapping subarrays with the maximum sum.

Approach:

- Use prefix sums to precompute sums for subarrays and track the best left and right subarray sums.

```py
def maxSumTwoNoOverlap(nums, firstLen, secondLen):
    def maxSum(L, M):
        prefix = [0] * (len(nums) + 1)
        for i in range(len(nums)):
            prefix[i + 1] = prefix[i] + nums[i]
        maxL = maxM = res = 0
        for i in range(L + M, len(prefix)):
            maxL = max(maxL, prefix[i - M] - prefix[i - M - L])
            res = max(res, maxL + prefix[i] - prefix[i - M])
        return res
    return max(maxSum(firstLen, secondLen), maxSum(secondLen, firstLen))

```

**Split Array into Subarrays with Maximum Sum**

Problem: Split an array into m subarrays such that the largest sum among them is  minimized.

Approach:

- Use binary search on the result (minimum possible largest sum). For each mid, check if the split is feasible using prefix sums.

```py
def splitArray(nums, m):
    def canSplit(maxSum):
        count = 1
        currentSum = 0
        for num in nums:
            currentSum += num
            if currentSum > maxSum:
                count += 1
                currentSum = num
                if count > m:
                    return False
        return True
    left, right = max(nums), sum(nums)
    while left < right:
        mid = (left + right) // 2
        if canSplit(mid):
            right = mid
        else:
            left = mid + 1
    return left

```

