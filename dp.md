# Dynamic Programming

Don't forget to solve these questions:
1. State Machine.
 - Climbing Stairs
 - Best Time to Buy and Sell Stock (series)
 - House Robber (series)
 - Target Sum
 - Count number of subsets with given difference
 - Delete and Earn
 - Jump Game (series)
 - Edit Distance
2. Bounded and Un-Bounded Knapsack.
 - Coin Change (series)
 - Rod Cutting
3. Partitions.
 - Partition Equal Subset Sum
4. Trivial/Classic.
 - Fibonacci Series
 - Best Time to Buy and Sell Stock
 - Longest Increasing Subsequence (tab)
5. Subsequence/Substring/Possibilities/Min/Max.
 - Longest Common Subsequence
 - Longest Palindrome (substring, subsequence)

## Fibonacci Numbers

```python

# Iterative approach
def fibonacci(n):
    if n <= 1:
        return n
    dp = [0] * (n + 1)
    dp[1] = 1
    for i in range(2, n + 1):
        dp[i] = dp[i - 1] + dp[i - 2]
    return dp[n]

# Example usage:
print(fibonacci(10))

```

## Knapsack Problems

(0/1 Knapsack)

```python

def knapsack(weights, values, capacity):
    n = len(weights)
    dp = [[0] * (capacity + 1) for _ in range(n + 1)]
    
    for i in range(1, n + 1):
        for w in range(1, capacity + 1):
            if weights[i - 1] <= w:
                dp[i][w] = max(dp[i - 1][w], dp[i - 1][w - weights[i - 1]] + values[i - 1])
            else:
                dp[i][w] = dp[i - 1][w]
    
    return dp[n][capacity]

# Example usage:
weights = [1, 2, 3]
values = [60, 100, 120]
capacity = 5
print(knapsack(weights, values, capacity))

```

(Unbounded Knapsack)

```python
def unbounded_knapsack(weights, values, capacity):
    n = len(weights)
    dp = [0] * (capacity + 1)
    
    for i in range(n):
        for w in range(weights[i], capacity + 1):
            dp[w] = max(dp[w], dp[w - weights[i]] + values[i])
    
    return dp[capacity]

# Example usage:
weights = [1, 3, 4, 5]
values = [10, 40, 50, 70]
capacity = 8
print(unbounded_knapsack(weights, values, capacity))

```

## Longest Common Subsequence- LCS

```python
def lcs(text1, text2):
    m, n = len(text1), len(text2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if text1[i - 1] == text2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    
    return dp[m][n]

# Example usage:
text1 = "abcde"
text2 = "ace"
print(lcs(text1, text2))

```

## Longest Increasing Subsequence- LIS

```python
def lis(nums):
    if not nums:
        return 0
    
    dp = [1] * len(nums)
    
    for i in range(1, len(nums)):
        for j in range(i):
            if nums[i] > nums[j]:
                dp[i] = max(dp[i], dp[j] + 1)
    
    return max(dp)

# Example usage:
nums = [10, 9, 2, 5, 3, 7, 101, 18]
print(lis(nums))

```

## Matrix Chain Multiplication

```python
def matrix_chain_order(p):
    n = len(p) - 1
    dp = [[0] * n for _ in range(n)]
    
    for length in range(2, n + 1):
        for i in range(n - length + 1):
            j = i + length - 1
            dp[i][j] = float('inf')
            for k in range(i, j):
                cost = dp[i][k] + dp[k + 1][j] + p[i] * p[k + 1] * p[j + 1]
                if cost < dp[i][j]:
                    dp[i][j] = cost
    
    return dp[0][n - 1]

# Example usage:
p = [1, 2, 3, 4]
print(matrix_chain_order(p))
```

## Edit Distance

```python

def edit_distance(word1, word2):
    m, n = len(word1), len(word2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    for i in range(m + 1):
        for j in range(n + 1):
            if i == 0:
                dp[i][j] = j
            elif j == 0:
                dp[i][j] = i
            elif word1[i - 1] == word2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])
    
    return dp[m][n]

# Example usage:
word1 = "horse"
word2 = "ros"
print(edit_distance(word1, word2))

```

## Partition Problems

(Partition Equal Subset Sum)

```python

def can_partition(nums):
    total_sum = sum(nums)
    if total_sum % 2 != 0:
        return False
    
    target = total_sum // 2
    dp = [False] * (target + 1)
    dp[0] = True
    
    for num in nums:
        for i in range(target, num - 1, -1):
            dp[i] = dp[i] or dp[i - num]
    
    return dp[target]

# Example usage:
nums = [1, 5, 11, 5]
print(can_partition(nums))

```

(Minimum Subset Sum Difference)

```python

def min_subset_sum_diff(nums):
    total_sum = sum(nums)
    n = len(nums)
    dp = [[False] * (total_sum // 2 + 1) for _ in range(n + 1)]
    dp[0][0] = True
    
    for i in range(1, n + 1):
        for j in range(total_sum // 2 + 1):
            dp[i][j] = dp[i - 1][j]
            if j >= nums[i - 1]:
                dp[i][j] = dp[i][j] or dp[i - 1][j - nums[i - 1]]
    
    for j in range(total_sum // 2, -1, -1):
        if dp[n][j]:
            return total_sum - 2 * j

# Example usage:
nums = [1, 2, 3, 9]
print(min_subset_sum_diff(nums))

```

## Subset Sum Problems

(Subset Sum Problem)

```python

def subset_sum(nums, target):
    dp = [False] * (target + 1)
    dp[0] = True
    
    for num in nums:
        for i in range(target, num - 1, -1):
            dp[i] = dp[i] or dp[i - num]
    
    return dp[target]

# Example usage:
nums = [1, 2, 3, 7]
target = 6
print(subset_sum(nums, target))

```

(Count of Subset Sum)

```python

def count_subsets(nums, target):
    dp = [0] * (target + 1)
    dp[0] = 1
    
    for num in nums:
        for i in range(target, num - 1, -1):
            dp[i] += dp[i - num]
    
    return dp[target]

# Example usage:
nums = [1, 1, 2, 3]
target = 4
print(count_subsets(nums, target))

```

## Rod Cutting

```python
def rod_cutting(prices, length):
    dp = [0] * (length + 1)
    
    for i in range(1, length + 1):
        max_val = 0
        for j in range(i):
            max_val = max(max_val, prices[j] + dp[i - j - 1])
        dp[i] = max_val
    
    return dp[length]

# Example usage:
prices = [2, 5, 7, 8]
length = 5
print(rod_cutting(prices, length))

```

## Coin Change Problems

(Coin Change)

```python

def coin_change(coins, amount):
    dp = [float('inf')] * (amount + 1)
    dp[0] = 0
    
    for coin in coins:
        for i in range(coin, amount + 1):
            dp[i] = min(dp[i], dp[i - coin] + 1)
    
    return dp[amount] if dp[amount] != float('inf') else -1

# Example usage:
coins = [1, 2, 5]
amount = 11
print(coin_change(coins, amount))

```

(Coin Change- II)

```python

def change(amount, coins):
    dp = [0] * (amount + 1)
    dp[0] = 1
    
    for coin in coins:
        for i in range(coin, amount + 1):
            dp[i] += dp[i - coin]
    
    return dp[amount]

# Example usage:
coins = [1, 2, 5]
amount = 5
print(change(amount, coins))
```

## Palindrome Problems

(Longest Palindrome Subsequence)

```python
def longest_palindromic_subsequence(s):
    n = len(s)
    dp = [[0] * n for _ in range(n)]
    
    for i in range(n):
        dp[i][i] = 1
    
    for length in range(2, n + 1):
        for i in range(n - length + 1):
            j = i + length - 1
            if s[i] == s[j]:
                dp[i][j] = dp[i + 1][j - 1] + 2
            else:
                dp[i][j] = max(dp[i + 1][j], dp[i][j - 1])
    
    return dp[0][n - 1]

# Example usage:
s = "bbbab"
print(longest_palindromic_subsequence(s))

```

(Longest Palindromic Substring)

```python
def longest_palindromic_substring(s):
    n = len(s)
    dp = [[False] * n for _ in range(n)]
    start, max_length = 0, 1
    
    for i in range(n):
        dp[i][i] = True
    
    for length in range(2, n + 1):
        for i in range(n - length + 1):
            j = i + length - 1
            if s[i] == s[j] and (length == 2 or dp[i + 1][j - 1]):
                dp[i][j] = True
                if length > max_length:
                    start = i
                    max_length = length
    
    return s[start:start + max_length]

# Example usage:
s = "babad"
print(longest_palindromic_substring(s))

```

(Palindrome Partitioning)

```python
def palindrome_partitioning(s):
    n = len(s)
    dp = [[False] * n for _ in range(n)]
    cuts = [0] * n
    
    for i in range(n):
        min_cut = i
        for j in range(i + 1):
            if s[j] == s[i] and (i - j < 2 or dp[j + 1][i - 1]):
                dp[j][i] = True
                min_cut = 0 if j == 0 else min(min_cut, cuts[j - 1] + 1)
        cuts[i] = min_cut
    
    return cuts[-1]

# Example usage:
s = "aab"
print(palindrome_partitioning(s))

```

## Path Finding in Grid

(Unique Paths)

```python

def unique_paths(m, n):
    dp = [[1] * n for _ in range(m)]
    
    for i in range(1, m):
        for j in range(1, n):
            dp[i][j] = dp[i - 1][j] + dp[i][j - 1]
    
    return dp[m - 1][n - 1]

# Example usage:
m, n = 3, 7
print(unique_paths(m, n))

```

(Minimum Path Sum)

```python
def min_path_sum(grid):
    m, n = len(grid), len(grid[0])
    dp = [[0] * n for _ in range(m)]
    dp[0][0] = grid[0][0]
    
    for i in range(1, m):
        dp[i][0] = dp[i - 1][0] + grid[i][0]
    
    for j in range(1, n):
        dp[0][j] = dp[0][j - 1] + grid[0][j]
    
    for i in range(1, m):
        for j in range(1, n):
            dp[i][j] = min(dp[i - 1][j], dp[i][j - 1]) + grid[i][j]
    
    return dp[m - 1][n - 1]

# Example usage:
grid = [
    [1, 3, 1],
    [1, 5, 1],
    [4, 2, 1]
]
print(min_path_sum(grid))

```

(Cherry Pickup)

```python
def cherry_pickup(grid):
    n = len(grid)
    dp = [[[float('-inf')] * n for _ in range(n)] for _ in range(n)]
    dp[0][0][0] = grid[0][0]
    
    for x1 in range(n):
        for y1 in range(n):
            for x2 in range(n):
                y2 = x1 + y1 - x2
                if 0 <= y2 < n and grid[x1][y1] != -1 and grid[x2][y2] != -1:
                    if x1 > 0:
                        dp[x1][y1][x2] = max(dp[x1][y1][x2], dp[x1 - 1][y1][x2])
                    if y1 > 0:
                        dp[x1][y1][x2] = max(dp[x1][y1][x2], dp[x1][y1 - 1][x2])
                    if x2 > 0:
                        dp[x1][y1][x2] = max(dp[x1][y1][x2], dp[x1][y1][x2 - 1])
                    if y2 > 0:
                        dp[x1][y1][x2] = max(dp[x1][y1][x2], dp[x1][y1][x2][y2 - 1])
                    
                    if x1 == x2 and y1 == y2:
                        dp[x1][y1][x2] += grid[x1][y1]
                    else:
                        dp[x1][y1][x2] += grid[x1][y1] + grid[x2][y2]
    
    return max(0, dp[n - 1][n - 1][n - 1])

# Example usage:
grid = [
    [0, 1, -1],
    [1, 0, -1],
    [1, 1, 1]
]
print(cherry_pickup(grid))

```

## DP on Trees

(House Robber III)

```python
class TreeNode:
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None

def rob(root):
    def helper(node):
        if not node:
            return (0, 0)
        
        left = helper(node.left)
        right = helper(node.right)
        
        rob_this = node.val + left[1] + right[1]
        not_rob_this = max(left) + max(right)
        
        return (rob_this, not_rob_this)
    
    return max(helper(root))

# Example usage:
root = TreeNode(3)
root.left = TreeNode(2)
root.right = TreeNode(3)
root.left.right = TreeNode(3)
root.right.right = TreeNode(1)
print(rob(root))

```

(Diameter of Binary Tree)

```python
class TreeNode:
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None

def diameter_of_binary_tree(root):
    def helper(node):
        nonlocal diameter
        if not node:
            return 0
        
        left = helper(node.left)
        right = helper(node.right)
        
        diameter = max(diameter, left + right)
        return max(left, right) + 1
    
    diameter = 0
    helper(root)
    return diameter

# Example usage:
root = TreeNode(1)
root.left = TreeNode(2)
root.right = TreeNode(3)
root.left.left = TreeNode(4)
root.left.right = TreeNode(5)
print(diameter_of_binary_tree(root))
```

(Maximum Path Sum in Binary Tree)

```python
class TreeNode:
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None

def max_path_sum(root):
    def helper(node):
        nonlocal max_sum
        if not node:
            return 0
        
        left = max(helper(node.left), 0)
        right = max(helper(node.right), 0)
        
        current_sum = node.val + left + right
        max_sum = max(max_sum, current_sum)
        
        return node.val + max(left, right)
    
    max_sum = float('-inf')
    helper(root)
    return max_sum

# Example usage:
root = TreeNode(-10)
root.left = TreeNode(9)
root.right = TreeNode(20)
root.right.left = TreeNode(15)
root.right.right = TreeNode(7)
print(max_path_sum(root))
```

## Bitmasking

(Travelling Salesaman Problem)

```python

def tsp(graph, start):
    n = len(graph)
    dp = [[float('inf')] * n for _ in range(1 << n)]
    dp[1 << start][start] = 0
    
    for mask in range(1 << n):
        for u in range(n):
            if mask & (1 << u):
                for v in range(n):
                    if mask & (1 << v) == 0:
                        dp[mask | (1 << v)][v] = min(dp[mask | (1 << v)][v], dp[mask][u] + graph[u][v])
    
    return min(dp[(1 << n) - 1][i] for i in range(n))

# Example usage:
graph = [
    [0, 10, 15, 20],
    [10, 0, 35, 25],
    [15, 35, 0, 30],
    [20, 25, 30, 0]
]
start = 0
print(tsp(graph, start))
```

(Counting Bits)

```python

def count_bits(num):
    dp = [0] * (num + 1)
    for i in range(1, num + 1):
        dp[i] = dp[i >> 1] + (i & 1)
    return dp

# Example usage:
num = 5
print(count_bits(num))

```

(Maximum AND Sum)

```python

def max_and_sum(nums, k):
    n = len(nums)
    dp = [-float('inf')] * (1 << n)
    dp[0] = 0
    
    for mask in range(1 << n):
        count = bin(mask).count('1')
        if count % k == 0:
            continue
        for i in range(n):
            if mask & (1 << i) == 0:
                next_mask = mask | (1 << i)
                dp[next_mask] = max(dp[next_mask], dp[mask] + (count // k + 1) & nums[i])
    
    return max(dp)

# Example usage:
nums = [1, 2, 3, 4]
k = 2
print(max_and_sum(nums, k))
```

## Stock Buy and Sell Problems

(Best Time to Buy and Sell Stock)

```python
def max_profit(prices):
    min_price = float('inf')
    max_profit = 0
    
    for price in prices:
        if price < min_price:
            min_price = price
        elif price - min_price > max_profit:
            max_profit = price - min_price
    
    return max_profit

# Example usage:
prices = [7, 1, 5, 3, 6, 4]
print(max_profit(prices))

```

(Best Time to Buy and Sell Stock with Cooldown)

```python

def max_profit_with_cooldown(prices):
    n = len(prices)
    if n <= 1:
        return 0
    
    dp = [0] * n
    max_diff = -prices[0]
    
    for i in range(1, n):
        if i > 1:
            dp[i] = max(dp[i - 1], prices[i] + max_diff)
            max_diff = max(max_diff, dp[i - 2] - prices[i])
        else:
            dp[i] = max(dp[i - 1], prices[i] + max_diff)
            max_diff = max(max_diff, -prices[i])
    
    return dp[-1]

# Example usage:
prices = [1, 2, 3, 0, 2]
print(max_profit_with_cooldown(prices))
```

(Best Time to Buy and Sell Stock with Transaction Fee)

```python
def max_profit_with_fee(prices, fee):
    n = len(prices)
    if n <= 1:
        return 0
    
    cash, hold = 0, -prices[0]
    for price in prices:
        cash = max(cash, hold + price - fee)
        hold = max(hold, cash - price)

    return cash

prices = [1, 3, 2, 8, 4, 9]
fee = 2
print(max_profit_with_fee(prices, fee))
```

Here are some patterns and their approaches to visit to revision.

**Fibonacci Number**

Problem: Compute the nth Fibonacci number.

Pattern: Classic recursion + memoization.

Approach:

- Use an array (or two variables) to store previously computed Fibonacci numbers.
- Start with base cases (`fib(0) = 0)`, `fib(1) = 1`).
- Build up to the nth Fibonacci using `fib(n) = fib(n-1) + fib(n-2)`.

```py
def fibonacci(n):
    if n <= 1:
        return n
    dp = [0] * (n + 1)
    dp[1] = 1
    for i in range(2, n + 1):
        dp[i] = dp[i - 1] + dp[i - 2]
    return dp[n]

```

**Climbing Stairs**

Problem: Find the number of distinct ways to climb n stairs, taking 1 or 2 steps at a time.

Pattern: Subproblems depend on 1 or 2 previous steps.

Approach:

- Define `dp[i]` as the number of ways to climb `i` stairs.
- Use `dp[i] = dp[i-1] + dp[i-2]`.

```py
def climbStairs(n):
    if n <= 2:
        return n
    dp = [0] * (n + 1)
    dp[1], dp[2] = 1, 2
    for i in range(3, n + 1):
        dp[i] = dp[i - 1] + dp[i - 2]
    return dp[n]

```

**Minimum Path Sum**

Problem: Find the minimum sum path from the top-left to the bottom-right in a grid.

Pattern: Grid traversal with overlapping subproblems.

Approach:

- Define `dp[i][j]` as the minimum path sum to cell (i, j).
- Use `dp[i][j] = grid[i][j] + min(dp[i-1][j], dp[i][j-1])`.

```py
def minPathSum(grid):
    rows, cols = len(grid), len(grid[0])
    for i in range(rows):
        for j in range(cols):
            if i == 0 and j == 0:
                continue
            elif i == 0:
                grid[i][j] += grid[i][j - 1]
            elif j == 0:
                grid[i][j] += grid[i - 1][j]
            else:
                grid[i][j] += min(grid[i - 1][j], grid[i][j - 1])
    return grid[-1][-1]

```

**House Robber**

Problem: Maximize the sum of values in a list while skipping adjacent elements.

Pattern: Non-adjacent sum optimization.

Approach: Use `dp[i] = max(dp[i-1], nums[i] + dp[i-2])`.

```py
def rob(nums):
    if not nums:
        return 0
    if len(nums) == 1:
        return nums[0]
    dp = [0] * len(nums)
    dp[0], dp[1] = nums[0], max(nums[0], nums[1])
    for i in range(2, len(nums)):
        dp[i] = max(dp[i-1], nums[i] + dp[i-2])
    return dp[-1]

```

**Coin Change (Combinations)**

Problem: Count the number of ways to make a target amount using given coin denominations.

Pattern: Unbounded knapsack.

Approach:

- Define `dp[i]` as the number of ways to make sum i.
- Use `dp[i] += dp[i-coin]` for each coin.

```py
def coinChangeCombinations(coins, amount):
    dp = [0] * (amount + 1)
    dp[0] = 1
    for coin in coins:
        for i in range(coin, amount + 1):
            dp[i] += dp[i - coin]
    return dp[amount]

```

**Longest Increasing Subsequence**

Problem: Find the length of the longest subsequence in increasing order.

Pattern: 1D DP with binary search.

Approach:

- Use an array dp where dp[i] stores the smallest ending value of a subsequence of length i+1.

```py
def lengthOfLIS(nums):
    dp = []
    for num in nums:
        i = bisect_left(dp, num)
        if i == len(dp):
            dp.append(num)
        else:
            dp[i] = num
    return len(dp)

```

**Partition Equal Subset Sum**

Problem: Determine if an array can be partitioned into two subsets with equal sums.

Pattern: Subset sum optimization.

Approach:

- Calculate the total sum of the array. If it's odd, return False.
- Use a DP array (dp[i]) to determine if a subset with sum j is possible.
- Transition: `dp[j] = dp[j] or dp[j - num]`.

```py
def canPartition(nums):
    total_sum = sum(nums)
    if total_sum % 2 != 0:
        return False
    target = total_sum // 2
    dp = [False] * (target + 1)
    dp[0] = True
    for num in nums:
        for j in range(target, num - 1, -1):
            dp[j] = dp[j] or dp[j - num]
    return dp[target]

```

**Unique Paths**

Problem: Count all unique paths in an m x n grid, moving only down or right.

Pattern: Grid traversal with combinatorics.

Approach:

- Define `dp[i][j]` as the number of unique paths to call (i, j).
- Transition: `dp[i][j] = dp[i-1][j] + dp[i][j-1]`.

```py
def uniquePaths(m, n):
    dp = [[1] * n for _ in range(m)]
    for i in range(1, m):
        for j in range(1, n):
            dp[i][j] = dp[i-1][j] + dp[i][j-1]
    return dp[-1][-1]

```

**Edit Distance**

Problem: Find the minimum number of operations (insert, delete, replace) to convert one string into another.

Pattern: String manipulation with 2D DP.

Approach:

- Define `dp[i][j]` as the minimum edits to convert the first i characters to word1 to the first j characters to word2.
- Transition:
    - If characters match: `dp[i][j] = dp[i-1][j-1]`.
    - Otherwise: `dp[i][j] = 1 + min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])`.

```py
def minDistance(word1, word2):
    m, n = len(word1), len(word2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        for j in range(n + 1):
            if i == 0:
                dp[i][j] = j
            elif j == 0:
                dp[i][j] = i
            elif word1[i - 1] == word2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])
    return dp[m][n]

```

**Longest Palindromic Subsequence**

Problem: Find the length of the longest subsequence in a string that is a palindrome.

Pattern: Reverse-and-compare strategy with 2D DP.

Approach:

- Define `dp[i][j]` as the length of the longest palindromic subsequence between indices i and j.
- Transition:
    - If `s[i] == s[j]`: `dp[i][j] = dp[i+1][j-1] + 2`.
    - Else: `dp[i][j] = max(dp[i+1][j], dp[i][j-1])`.

```py
def longestPalindromeSubseq(s):
    n = len(s)
    dp = [[0] * n for _ in range(n)]
    for i in range(n - 1, -1, -1):
        dp[i][i] = 1
        for j in range(i + 1, n):
            if s[i] == s[j]:
                dp[i][j] = dp[i + 1][j - 1] + 2
            else:
                dp[i][j] = max(dp[i + 1][j], dp[i][j - 1])
    return dp[0][-1]

```

**Burst Ballons**

Problem: Maximize coins by bursting balloons wisely.

Pattern: Interval DP.

Approach:

- Define `dp[i][j]` as the maximum coins for bursting balloons between indices i and j.
- For each subinterval [i, j], consider each balloons k as the last one to burst.
- Transition: `dp[i][j] = max(dp[i][k-1] + dp[k+1][j] + nums[i-1]*nums[k]*nums[j+1])`.

```py
def maxCoins(nums):
    nums = [1] + nums + [1]
    n = len(nums)
    dp = [[0] * n for _ in range(n)]
    for length in range(2, n):
        for i in range(0, n - length):
            j = i + length
            for k in range(i + 1, j):
                dp[i][j] = max(dp[i][j], nums[i] * nums[k] * nums[j] + dp[i][k] + dp[k][j])
    return dp[0][-1]

```

**Matrix Chain Multiplication**

Problem: Find the minimum number of multiplications needed to multiply a sequence of matrices.

Pattern: Parenthesis grouping optimization.

Approach:

- Define `dp[i][j]` as the minimum multiplication for multiplying matrices from i to j.
- Transition: `dp[i][j] = min(dp[i][k] + dp[k+1][j] + dimensions[i-1]*dimensions[k]*dimensions[j])`.

```py
def matrixChainOrder(dimensions):
    n = len(dimensions) - 1
    dp = [[0] * n for _ in range(n)]
    for length in range(2, n + 1):
        for i in range(n - length + 1):
            j = i + length - 1
            dp[i][j] = float('inf')
            for k in range(i, j):
                cost = dp[i][k] + dp[k+1][j] + dimensions[i]*dimensions[k+1]*dimensions[j+1]
                dp[i][j] = min(dp[i][j], cost)
    return dp[0][-1]

```

**Work Break**

Problem: Check if a string can be segmented into words from a given dictionary.

Pattern: String segmentation with overlapping subproblems.

Approach:

- Define dp[i] as True if the substring `s[0:i]` can be segmented into valid words.
- Transition: For every substring `s[j:i]`, check if it exists in the dictionary and dp[j] is True.

```py
def wordBreak(s, wordDict):
    word_set = set(wordDict)
    dp = [False] * (len(s) + 1)
    dp[0] = True
    for i in range(1, len(s) + 1):
        for j in range(i):
            if dp[j] and s[j:i] in word_set:
                dp[i] = True
                break
    return dp[-1]

```

**Wildcard Matching**

Problem: Match a string with a pattern containing wildcards * (matches any sequence) and ? (matches any single character).

Pattern: Advanced string DP.

Approach:

- Define `dp[i][j]` as True if the first i characters of the string match the first j characters of the pattern.
- Transition:
    - If `p[j-1] == s[i-1] or p[j-1] == '?'`: `dp[i][j] = dp[i-1][j-1]`.
    - If `p[j-1] == '*'`: `dp[i][j] = dp[i-1][j] or dp[i][j-1]`.

```py
def isMatch(s, p):
    m, n = len(s), len(p)
    dp = [[False] * (n + 1) for _ in range(m + 1)]
    dp[0][0] = True
    for j in range(1, n + 1):
        if p[j-1] == '*':
            dp[0][j] = dp[0][j-1]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if p[j-1] == s[i-1] or p[j-1] == '?':
                dp[i][j] = dp[i-1][j-1]
            elif p[j-1] == '*':
                dp[i][j] = dp[i-1][j] or dp[i][j-1]
    return dp[m][n]

```

**Maximum Profit in Job Scheduling**

Problem: Given jobs with start times, end times, and profits, find the maximum profit you can earn by scheduling non-overlapping jobs.

Pattern: Weighted interval scheduling with DP.

Approach:

- Sort jobs by their end times.
- Use binary search to find the last non-overlapping job for each job.
- Define `dp[i]` as the maximum profit using the first i jobs.
- Transition: `dp[i] = max(dp[i-1], profit[i-1] + dp[last_non_overlapping_job])`.

```py
def jobScheduling(startTime, endTime, profit):
    jobs = sorted(zip(startTime, endTime, profit), key=lambda x: x[1])
    n = len(jobs)
    dp = [0] * (n + 1)
    
    def binary_search(index):
        low, high = 0, index - 1
        while low <= high:
            mid = (low + high) // 2
            if jobs[mid][1] <= jobs[index][0]:
                if jobs[mid + 1][1] <= jobs[index][0]:
                    low = mid + 1
                else:
                    return mid
            else:
                high = mid - 1
        return -1

    for i in range(1, n + 1):
        include_profit = jobs[i - 1][2]
        last_index = binary_search(i - 1)
        if last_index != -1:
            include_profit += dp[last_index + 1]
        dp[i] = max(dp[i - 1], include_profit)

    return dp[-1]

```
