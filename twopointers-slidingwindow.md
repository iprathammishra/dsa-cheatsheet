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

Time Complexity: O(n), where n is the size of the array.

Example:

```py
nums = [1, 1, 2, 2, 3]
i = 0
for j in range(1, len(nums)):
    if nums[j] != nums[i]:
        i += 1
        nums[i] = nums[j]
print(i + 1)  # Output: 3 (unique elements: [1, 2, 3])

```

**Valid Palindrome**

Problem: Check is a string is a palindrome, considering only alphanumeric characters and ignoring case.

Approach: 

- Use two pointers: one at the start and one at the end.
- Skip non-alphanumeric characters and check if the characters at the two pointers and equal, moving the pointers toward the center.

Time Complexity: O(n), where n is the length of the string.

Example:

```py
s = "A man, a plan, a canal, Panama"
s = ''.join([c.lower() for c in s if c.isalnum()])
left, right = 0, len(s) - 1
while left < right:
    if s[left] != s[right]:
        print(False)
        return
    left += 1
    right -= 1
print(True)  # Output: True

```

# Medium Level

**Find Subarray with Sum Equals to Target**

Problem: Given an array and a target sum, find a contiguous subarray whose sum is equal to the target.

Approach:

- Use a sliding window (or prefix sum) approach to track the sum of elements in the current window.
- If the sum exceeds the target, shrink the window from the left.

Time Complexity: O(n), where n is the size of the array.

Example:

```py
nums = [1, 2, 3, 7, 5]
target = 12
current_sum = 0
left = 0
for right in range(len(nums)):
    current_sum += nums[right]
    while current_sum > target:
        current_sum -= nums[left]
        left += 1
    if current_sum == target:
        print(True)
        return
print(False)  # Output: True
```

**Longest Substring Without Repeating Characters**

Problem: Find the length of the longest substring with no repeating characters.

Approach:

- Use a sliding window with two pointers: left and right. Expand the window by moving right, and contract it by moving left when a duplicate character is found.
- Keep track of characters in a set to quickly check for duplicates.

Time Complexity: O(n), where n is the length of the string.

Example:

```py
s = "abcabcbb"
char_set = set()
left, result = 0, 0
for right in range(len(s)):
    while s[right] in char_set:
        char_set.remove(s[left])
        left += 1
    char_set.add(s[right])
    result = max(result, right - left + 1)
print(result)  # Output: 3 (longest substring: "abc")
```

**Minimum Window Substring**

Problem: Find the minimum window in string s that contains all characters from string t.

Approach:

- Use a sliding window approach with two pointers. Expand the window by moving the right pointer, and when all characters from t are included, try to shrink the window by moving the left pointer.
- Use a frequency map for string t and track the count of characters in the current window.

Time complexity: O(n), where n is the length of string s.

Example:

```py
s = "ADOBECODEBANC"
t = "ABC"
from collections import Counter
t_freq = Counter(t)
s_freq = Counter()
left, right = 0, 0
min_len = float('inf')
min_substr = ""
while right < len(s):
    s_freq[s[right]] += 1
    while all(s_freq[char] >= t_freq[char] for char in t_freq):
        if right - left + 1 < min_len:
            min_len = right - left + 1
            min_substr = s[left:right+1]
        s_freq[s[left]] -= 1
        left += 1
    right += 1
print(min_substr)  # Output: "BANC"

```

# Hard Level

**Longest Subarray with At Most K Distinct Characters**

Problem: Find the length of the longest subarray that contains at most k distinct characters.

Approach:

- Use a sliding window where you expand the window by moving the right pointer. If the number of distinct characters exceeds k, shrink the window by moving the left pointer.
- Use a dictionary to track the count of characters in the window.

Time Complexity: O(n), where n is the length of the string.

Example:

```py
s = "eceba"
k = 2
from collections import defaultdict
char_count = defaultdict(int)
left, max_len = 0, 0
for right in range(len(s)):
    char_count[s[right]] += 1
    while len(char_count) > k:
        char_count[s[left]] -= 1
        if char_count[s[left]] == 0:
            del char_count[s[left]]
        left += 1
    max_len = max(max_len, right - left + 1)
print(max_len)  # Output: 3 (longest substring: "ece")

```

**Substring with Concatenation of All Words**

Problem: Find all starting indices of substrings in s that are a concatenation of each word in *words* exactly once.

Approach:

- Use a sliding window to check for substrings of size len(words)*len(word) where all words from *words* are present exactly once. Use a hash map to track the frequency of words in the substring.

Time Complexity: O(n*m), where n is the length of string s and m is the length of each word.

Example:

```py
s = "barfoothefoobarman"
words = ["foo", "bar"]
word_len = len(words[0])
word_count = Counter(words)
result = []
for i in range(word_len):
    left, right = i, i
    current_count = defaultdict(int)
    while right + word_len <= len(s):
        word = s[right:right + word_len]
        right += word_len
        if word in word_count:
            current_count[word] += 1
            while current_count[word] > word_count[word]:
                current_count[s[left:left + word_len]] -= 1
                left += word_len
            if right - left == len(words) * word_len:
                result.append(left)
        else:
            current_count.clear()
            left = right
print(result)  # Output: [0, 9]

```

**Count of Substrings with Exactly K Distinct Characters**

Problem: Count the number of substrings in s with exactly k distinct characters.

Approach:

- Use a helper function to count substrings with at most k distinct characters. The result for exactly k distinct characters can be obtained by calculating:
    - count(at most k) - count(at most k-1)

Time Complexity: O(n), where n is the length of the string.

Example:

```py
s="abcba"
k = 2
def at_most_k(s, k):
    from collections import defaultdict
    char_count = defaultdict(int)
    left, result = 0, 0
    for right in range(len(s)):
        char_count[s[right]] += 1
        while len(char_count) > k:
            char_count[s[left]] -= 1
            if char_count[s[left]] == 0:
                del char_count[s[left]]
            left += 1
        result += right - left + 1
    return result
print(at_most_k(s, k) - at_most_k(s, k - 1))

```

