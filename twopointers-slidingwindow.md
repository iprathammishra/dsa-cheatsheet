# Two Pointers / Sliding Window
## Sliding Window Pattern

### Fixed-Size Window 

(Maximum Sum Subarray of Size K)
```python

def max_sum_subarray_of_size_k(arr, k):
    max_sum = 0
    window_sum = 0

    for i in range(k):
        window_sum += arr[i]

    max_sum = window_sum
    for i in range(k, len(arr)):
        window_sum += arr[i] - arr[i - k]
        max_sum = max(max_sum, window_sum)
    
    return max_sum

```

(Longest Substring with K Distinct Characters)

```python

def longest_substring_k_distinct(s, k):
    char_count = {}
    max_length = 0
    window_start = 0

    for window_end in range(len(s)):
        right_char = s[window_end]
        if right_char not in char_count:
            char_count[right_char] = 0
        char_count[right_char] += 1

        while len(char_count) > k:
            left_char = s[window_start]
            char_count[left_char] -= 1
            if char_count[left_char] == 0:
                del char_count[left_char]
            window_start += 1

        max_length = max(max_length, window_end - window_start + 1)

    return max_length

```

(Average of Subarrays of Size K)

```python

def average_of_subarrays(arr, k):
    result = []
    window_sum = 0
    window_start = 0

    for window_end in range(len(arr)):
        window_sum += arr[window_end]

        if window_end >= k - 1:
            result.append(window_sum / k)
            window_sum -= arr[window_start]
            window_start += 1

    return result

```

### Variable-size Window

(Longest Substring Without Repeating Characters)

```python

def longest_substring_without_repeating_characters(s):
    char_index = {}
    max_length = 0
    window_start = 0

    for window_end in range(len(s)):
        right_char = s[window_end]
        if right_char in char_index:
            window_start = max(window_start, char_index[right_char] + 1)

        char_index[right_char] = window_end
        max_length = max(max_length, window_end - window_start + 1)

    return max_length

```

(Minimum Window Substring)

```python

def min_window_substring(s, t):
    if not s or not t:
        return ""

    char_count_t = {}
    for char in t:
        if char not in char_count_t:
            char_count_t[char] = 0
        char_count_t[char] += 1

    required = len(char_count_t)
    window_count = {}
    formed = 0
    l, r = 0, 0
    min_length = float("inf")
    min_window = (0, 0)

    while r < len(s):
        char = s[r]
        if char not in window_count:
            window_count[char] = 0
        window_count[char] += 1

        if char in char_count_t and window_count[char] == char_count_t[char]:
            formed += 1

        while l <= r and formed == required:
            char = s[l]

            if r - l + 1 < min_length:
                min_length = r - l + 1
                min_window = (l, r)

            window_count[char] -= 1
            if char in char_count_t and window_count[char] < char_count_t[char]:
                formed -= 1

            l += 1

        r += 1

    l, r = min_window
    return s[l:r+1] if min_length != float("inf") else ""

```

### Dynamic Window (Expanding and Contracting)

(Smallest Subarray with a Given Sum)

```python

def smallest_subarray_with_given_sum(s, arr):
    min_length = float("inf")
    window_sum = 0
    window_start = 0

    for window_end in range(len(arr)):
        window_sum += arr[window_end]

        while window_sum >= s:
            min_length = min(min_length, window_end - window_start + 1)
            window_sum -= arr[window_start]
            window_start += 1

    return min_length if min_length != float("inf") else 0

```

## Two Pointers Pattern
### Opposite Direction Two Pointers

(Two Sum- Sorted Array)

```python

def two_sum_sorted_array(arr, target):
    left, right = 0, len(arr) - 1

    while left < right:
        current_sum = arr[left] + arr[right]

        if current_sum == target:
            return [left, right]
        elif current_sum < target:
            left += 1
        else:
            right -= 1

    return []

```

(Reverse a String)

```python

def reverse_string(s):
    left, right = 0, len(s) - 1
    s = list(s)

    while left < right:
        s[left], s[right] = s[right], s[left]
        left += 1
        right -= 1

    return ''.join(s)

```

(Valid Palindrome)

```python

def is_valid_palindrome(s):
    left, right = 0, len(s) - 1

    while left < right:
        if s[left] != s[right]:
            return False
        left += 1
        right -= 1

    return True

```

### Same Direction Two Pointers

(Remove Duplicates from Sorted Array)

```python

def remove_duplicates_from_sorted_array(arr):
    if not arr:
        return 0

    write_index = 1

    for i in range(1, len(arr)):
        if arr[i] != arr[i - 1]:
            arr[write_index] = arr[i]
            write_index += 1

    return write_index

```

(Move Zeros)

```python

def move_zeros(nums):
    write_index = 0

    for i in range(len(nums)):
        if nums[i] != 0:
            nums[write_index], nums[i] = nums[i], nums[write_index]
            write_index += 1

    return nums

```

### Fixed and Moving Pointer

(Remove Element)

```python

def remove_element(arr, val):
    write_index = 0

    for i in range(len(arr)):
        if arr[i] != val:
            arr[write_index] = arr[i]
            write_index += 1

    return write_index

```

(Find All Anagrams in a String)

```python

from collections import Counter

def find_all_anagrams(s, p):
    p_count = Counter(p)
    s_count = Counter()
    result = []
    p_len = len(p)

    for i in range(len(s)):
        s_count[s[i]] += 1

        if i >= p_len:
            if s_count[s[i - p_len]] == 1:
                del s_count[s[i - p_len]]
            else:
                s_count[s[i - p_len]] -= 1

        if s_count == p_count:
            result.append(i - p_len + 1)

    return result

```

## Combined Patterns
### Sliding Window With Two Pointers

(Longest Substring Without Repeating Characters)

```python

def longest_substring_without_repeating_characters(s):
    char_index = {}
    max_length = 0
    window_start = 0

    for window_end in range(len(s)):
        right_char = s[window_end]
        if right_char in char_index:
            window_start = max(window_start, char_index[right_char] + 1)

        char_index[right_char] = window_end
        max_length = max(max_length, window_end - window_start + 1)

    return max_length

```

### Two Pointers with Sorting

(3Sum)

```python

def three_sum(nums):
    nums.sort()
    result = []

    for i in range(len(nums)):
        if i > 0 and nums[i] == nums[i - 1]:
            continue
        left, right = i + 1, len(nums) - 1
        while left < right:
            total = nums[i] + nums[left] + nums[right]
            if total < 0:
                left += 1
            elif total > 0:
                right -= 1
            else:
                result.append([nums[i], nums[left], nums[right]])
                while left < right and nums[left] == nums[left + 1]:
                    left += 1
                while left < right and nums[right] == nums[right - 1]:
                    right -= 1
                left += 1
                right -= 1

    return result

```

(4Sum)

```python

def four_sum(nums, target):
    nums.sort()
    result = []

    for i in range(len(nums)):
        if i > 0 and nums[i] == nums[i - 1]:
            continue
        for j in range(i + 1, len(nums)):
            if j > i + 1 and nums[j] == nums[j - 1]:
                continue
            left, right = j + 1, len(nums) - 1
            while left < right:
                total = nums[i] + nums[j] + nums[left] + nums[right]
                if total < target:
                    left += 1
                elif total > target:
                    right -= 1
                else:
                    result.append([nums[i], nums[j], nums[left], nums[right]])
                    while left < right and nums[left] == nums[left + 1]:
                        left += 1
                    while left < right and nums[right] == nums[right - 1]:
                        right -= 1
                    left += 1
                    right -= 1

    return result

```

(Two Sum II- Input array is sorted)

```python

def two_sum_sorted(nums, target):
    left, right = 0, len(nums) - 1

    while left < right:
        current_sum = nums[left] + nums[right]
        if current_sum == target:
            return [left + 1, right + 1]  # assuming 1-indexed positions
        elif current_sum < target:
            left += 1
        else:
            right -= 1

    return []

```

Here are 10 questions, ranging from easy to hard. These will help to build a solid foundation and tackle problems from simple to complex.

# Easy Level

**Find All Pairs with a Given Sum in an Array**

Problem: Given an array of integers and a target sum, return all pairs of numbers from the array whose sum is equal to the target.

Approach:

-  Use a hashset to store the elements you've already seen while iterating through the array.
- For each number x, check if target - x is already in the set. If so, you've found a pair.
- This ensures each pair is counted only once.

Time Complexity: O(n), where n is the size of the array.

Example:

```py
nums = [1, 2, 3, 4, 3, 5]
target = 6
seen = set()
result = []
for num in nums:
    complement = target - num
    if complement in seen:
        result.append((complement, num))
    seen.add(num)
print(result)  # Output: [(1, 5), (2, 4), (3, 3)]
```

**Reverse String**

Problem: Reverse a string using two pointers.

Approach:
- Use two pointers: one starting at the beginning (left) and one at the end (right).
- Swap characters while moving left forward and right backward until they cross each other.

Time Complexity: O(n), where n is the length of the string.

Example:

```py
s = "hello"
s = list(s)  # Convert to list to mutate
left, right = 0, len(s) - 1
while left < right:
    s[left], s[right] = s[right], s[left]
    left += 1
    right -= 1
print("".join(s))  # Output: "olleh"
```

**Remove Duplicates from Sorted Array**

Problem: Remove duplicates in a sorted array and return the new length.

Approach:
- Use a slow pointer (i) to track the unique elements in the array and a fast pointer (j) to iterate over the array.
- When a new unique element is found, move i and replace nums[i] with the new unique element at nums[j].




