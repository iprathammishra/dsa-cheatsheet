# Bit Manipulation

(Checking if a Number is a Power of Two)

```python
# Check if a number is a power of two
def is_power_of_two(n):
    # A power of two has exactly one bit set to 1 (e.g., 1, 2, 4, 8, 16...)
    # n & (n - 1) clears the lowest set bit. If n is a power of two, the result will be 0.
    return n > 0 and (n & (n - 1)) == 0

# Example usage
print(is_power_of_two(16))  # Output: True
print(is_power_of_two(18))  # Output: False

```

(Counting the Number of 1 bits- Hamming Weight)

```python
# Count the number of 1 bits in an integer
def hamming_weight(n):
    count = 0
    while n:
        # n & (n - 1) flips the lowest set bit of n to 0
        n &= (n - 1)
        count += 1  # Increment count for each 1 bit
    return count

# Example usage
print(hamming_weight(11))  # Output: 3 (binary: 1011)


```

(Finding the Only Non-Repeating Element)

```python
# Find the single number that does not repeat
def single_number(nums):
    result = 0
    for num in nums:
        # XOR-ing all numbers together cancels out the numbers that appear twice,
        # leaving the single number that does not repeat
        result ^= num
    return result

# Example usage
print(single_number([4, 1, 2, 1, 2]))  # Output: 4


```

(Generating All Subsets of a Set)

```python
# Generate all subsets of a set
def subsets(nums):
    n = len(nums)
    result = []
    # Loop through all possible subsets (2^n possibilities)
    for i in range(1 << n):
        subset = []
        for j in range(n):
            # Check if the j-th bit is set in i
            if i & (1 << j):
                subset.append(nums[j])
        result.append(subset)
    return result

# Example usage
print(subsets([1, 2, 3]))  
# Output: [[], [1], [2], [1, 2], [3], [1, 3], [2, 3], [1, 2, 3]]


```

(Finding the Two Non-Repeating Elements)

```python
# Find the two numbers that do not repeat
def single_number_two(nums):
    xor = 0
    for num in nums:
        # XOR all numbers together to get xor = x ^ y,
        # where x and y are the two non-repeating numbers
        xor ^= num
    
    # Get the rightmost set bit (difference between x and y)
    diff = xor & -xor
    x, y = 0, 0
    for num in nums:
        # Divide the numbers into two groups based on the rightmost set bit
        if num & diff:
            x ^= num
        else:
            y ^= num
    return [x, y]

# Example usage
print(single_number_two([1, 2, 1, 3, 2, 5]))  # Output: [3, 5]


```

(Bitwise AND of Numbers Range)

```python
# Bitwise AND of all numbers in a range
def range_bitwise_and(m, n):
    shift = 0
    while m < n:
        # Right shift both m and n until they are equal
        # This removes the differing least significant bits
        m >>= 1
        n >>= 1
        shift += 1
    # Left shift to restore the common bits
    return m << shift

# Example usage
print(range_bitwise_and(5, 7))  # Output: 4


```

(Counting Bits- Number of 1 bits up to n)

```python
# Count the number of 1 bits for all numbers up to n
def count_bits(n):
    dp = [0] * (n + 1)
    for i in range(1, n + 1):
        # The number of 1 bits in i is the number of 1 bits in i // 2 (i >> 1)
        # plus 1 if the last bit of i is 1 (i & 1)
        dp[i] = dp[i >> 1] + (i & 1)
    return dp

# Example usage
print(count_bits(5))  # Output: [0, 1, 1, 2, 1, 2]


```

(Reverse Bits)

```python
# Reverse bits of a given 32 bits unsigned integer
def reverse_bits(n):
    result = 0
    for i in range(32):
        # Left shift result to make room for the next bit
        result <<= 1
        # Add the least significant bit of n to result
        result |= n & 1
        # Right shift n to process the next bit
        n >>= 1
    return result

# Example usage
print(reverse_bits(43261596))  
# Output: 964176192 (binary: 00000010100101000001111010011100 -> 00111001011110000010100101000000)


```

Here are some of the best pattern to take a look for revision.

**Check if a Number is Power of Two**

Problem: Determine if a given integer is a power of two.

Approach: 

- A number that is a power of two has exactly one bit set in its binary representation.
- `n & (n - 1)` removes the rightmost set bit. If the result is `0`, the number is a power of two.

```py
def isPowerOfTwo(n):
    return n > 0 and (n & (n - 1)) == 0

```

**Count Set Bits**

Problem: Count the numnber of 1 bit in the binary representation of a number.

Approach: 

- Use Brian Kernighan's algorithm: repeatedly perform `n = n & (n - 1)` until `n` becomes 0. Each operation removes one set bit.

```py
def countSetBits(n):
    count = 0
    while n:
        n &= (n - 1)
        count += 1
    return count

```

**Find the Single Number**

Problem: In an array where every number appears twize except one, find the single number.

Approach:

- Use XOR: `a ^ a = 0` and `a ^ 0 = a`. XOR all numbers; duplicates cancel out, leaving the unique number.

```py
def singleNumber(nums):
    result = 0
    for num in nums:
        result ^= num
    return result

```

**Reverse Bits**

Problem: Reverse the bits of a given 32-bit integer.

Approach:

- Iterate through all 32 bits, extract the least significant bit, and shift it into its reversed position.

```py
def reverseBits(n):
    result = 0
    for i in range(32):
        result = (result << 1) | (n & 1)
        n >>= 1
    return result

```

**Find Missing Number**

Problem: Find the missing number in an array containing numbers from 0 to n.

Approach:

- XOR all numbers in the array and from 0 to n. The result is the missing number.

```py
def missingNumber(nums):
    n = len(nums)
    result = n
    for i, num in enumerate(nums):
        result ^= i ^ num
    return result

```

**Maximum XOR of Two Numbers in an Array**

Problem: Find the maximum XOR of two numbers in an array.

Approach:

- Use a trie to store binary representations of numbers.
- For each number, try to maximize XOR by choosing the opposite bit at each level of the trie.

```py
class TrieNode:
    def __init__(self):
        self.children = {}

def findMaximumXOR(nums):
    root = TrieNode()
    for num in nums:
        node = root
        for i in range(31, -1, -1):
            bit = (num >> i) & 1
            if bit not in node.children:
                node.children[bit] = TrieNode()
            node = node.children[bit]

    max_xor = 0
    for num in nums:
        node = root
        curr_xor = 0
        for i in range(31, -1, -1):
            bit = (num >> i) & 1
            opposite_bit = 1 - bit
            if opposite_bit in node.children:
                curr_xor = (curr_xor << 1) | 1
                node = node.children[opposite_bit]
            else:
                curr_xor = curr_xor << 1
                node = node.children[bit]
        max_xor = max(max_xor, curr_xor)
    return max_xor

```

**Divide Two Integers Without Using Division**

Problem: Implement division of two integers without using `/`, `*`, or `%`.

Approach:

- Use bit manipulation to perform subtraction and shifting.
- Subtract the largest multiple of the divior from the divident by shifting the divisor left until it exceeds the dividend.

```py
def divide(dividend, divisor):
    if dividend == -2**31 and divisor == -1:
        return 2**31 - 1
    sign = (dividend > 0) == (divisor > 0)
    dividend, divisor = abs(dividend), abs(divisor)
    result = 0
    while dividend >= divisor:
        temp, count = divisor, 1
        while dividend >= (temp << 1):
            temp <<= 1
            count <<= 1
        dividend -= temp
        result += count
    return result if sign else -result

```

**Subset XOR Sum**

Problem: Find the sum of XOR values of all subsets of an array.

Approach:

- Each bit contributes to the XOR sum in `2^(n-1)` subsets. Multiply the count by the bit value for each bit position.

```py
def subsetXORSum(nums):
    return sum(nums) * (1 << (len(nums) - 1))

```

**Find Two Missing Numbers**

Problem: In an array of `n` integers from `1` to `n+2`, find the two missing numbers.

Approach:

- XOR all numbers from 1 to n+2 and all array elements to get `a^b` (the missing numbers).
- Use a set bit in `a ^ b` to divide numbers into two groups and find a and b.

```py
def findTwoMissingNumbers(nums):
    n = len(nums) + 2
    xor_all = 0
    for i in range(1, n + 1):
        xor_all ^= i
    for num in nums:
        xor_all ^= num

    diff_bit = xor_all & -xor_all
    a = b = 0
    for i in range(1, n + 1):
        if i & diff_bit:
            a ^= i
        else:
            b ^= i
    for num in nums:
        if num & diff_bit:
            a ^= num
        else:
            b ^= num
    return a, b

```

**Nim Game**

Problem: Determine if the first player will win a Nim game with n stones.

Approach:

- If `n % 4 == 0`, the first player always loses. Otherwise, the first player can always leave a multiple of 4 for the opponent.

```py
def canWinNim(n):
    return n % 4 != 0

```
