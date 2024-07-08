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
