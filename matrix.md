# Matrix

(Traversal)

```python

# Traversal - Iterate through each cell of the matrix using nested loops.
def traverse_matrix(matrix):
    for i in range(len(matrix)):
        for j in range(len(matrix[0])):
            # Perform operations on matrix[i][j]
            print(matrix[i][j])

```

(Row-wise / Column-wise Traversal)

```python

# Row-wise Traversal
def row_wise_traversal(matrix):
    for row in matrix:
        for element in row:
            # Perform operations on element
            print(element)

# Column-wise Traversal
def column_wise_traversal(matrix):
    for col in range(len(matrix[0])):
        for row in range(len(matrix)):
            # Perform operations on matrix[row][col]
            print(matrix[row][col])

```

(Search)

```python

# Linear Search
def search_matrix(matrix, target):
    for row in matrix:
        for element in row:
            if element == target:
                return True
    return False

# Binary Search (for sorted matrix)
def binary_search_matrix(matrix, target):
    if not matrix:
        return False
    rows, cols = len(matrix), len(matrix[0])
    left, right = 0, rows * cols - 1

    while left <= right:
        mid = (left + right) // 2
        mid_value = matrix[mid // cols][mid % cols]
        if mid_value == target:
            return True
        elif mid_value < target:
            left = mid + 1
        else:
            right = mid - 1

    return False

```

(BFS/DFS)

```python

from collections import deque

# BFS
def bfs_matrix(matrix, start):
    rows, cols = len(matrix), len(matrix[0])
    queue = deque([start])
    visited = set(start)

    while queue:
        x, y = queue.popleft()
        # Process the node (x, y)
        print(matrix[x][y])
        
        # Explore neighbors (4-directional)
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < rows and 0 <= ny < cols and (nx, ny) not in visited:
                queue.append((nx, ny))
                visited.add((nx, ny))

# DFS (recursive)
def dfs_matrix(matrix, x, y, visited):
    rows, cols = len(matrix), len(matrix[0])
    # Process the node (x, y)
    print(matrix[x][y])
    visited.add((x, y))

    # Explore neighbors (4-directional)
    for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        nx, ny = x + dx, y + dy
        if 0 <= nx < rows and 0 <= ny < cols and (nx, ny) not in visited:
            dfs_matrix(matrix, nx, ny, visited)

# DFS (using stack)
def dfs_stack_matrix(matrix, start):
    rows, cols = len(matrix), len(matrix[0])
    stack = [start]
    visited = set(start)

    while stack:
        x, y = stack.pop()
        # Process the node (x, y)
        print(matrix[x][y])
        
        # Explore neighbors (4-directional)
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < rows and 0 <= ny < cols and (nx, ny) not in visited:
                stack.append((nx, ny))
                visited.add((nx, ny))

```

(Matrix Rotation- In-place rotation)

```python

# Rotate matrix 90 degrees in-place
def rotate_matrix_in_place(matrix):
    n = len(matrix)
    for i in range(n // 2):
        for j in range(i, n - i - 1):
            temp = matrix[i][j]
            matrix[i][j] = matrix[n - j - 1][i]
            matrix[n - j - 1][i] = matrix[n - i - 1][n - j - 1]
            matrix[n - i - 1][n - j - 1] = matrix[j][n - i - 1]
            matrix[j][n - i - 1] = temp

```

(Matrix Rotation- Creating a new rotated matrix)

```python

# Rotate matrix 90 degrees by creating a new matrix
def rotate_matrix_new(matrix):
    n = len(matrix)
    rotated_matrix = [[0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            rotated_matrix[j][n - 1 - i] = matrix[i][j]
    return rotated_matrix

```

(Matrix Transformation- Transposition)

```python
# Matrix Transposition
def transpose_matrix(matrix):
    rows, cols = len(matrix), len(matrix[0])
    transposed_matrix = [[0] * rows for _ in range(cols)]
    for i in range(rows):
        for j in range(cols):
            transposed_matrix[j][i] = matrix[i][j]
    return transposed_matrix

```

(Matrix Transformation- Flipping)

```python
# Flip along the horizontal axis
def flip_horizontal(matrix):
    return matrix[::-1]

# Flip along the vertical axis
def flip_vertical(matrix):
    return [row[::-1] for row in matrix]

```

Here are 10 matrix-based questions, ordered from easy to hard, designed to cover a wide range of patterns:

**Traverse a Matrix (Row-wise and Column-wise Traversal)**

Problem: Given a m x n matrix, print its elements row-wise and column-wise.

Pattern: Traversal

Approach: Constructive

```py
# Column-wise Traversal
rows, cols = len(matrix), len(matrix[0])
for col in range(cols):
    for row in range(rows):
        print(matrix[row][col], end=" ")
```

**Search in a Sorted Matrix**

Problem: Given a matrix where each row is sorted and the first element of each row is greater than the last element of the previous row, search for a target.

Pattern: Binary Search

Approach:

- Treat the sorted matrix as a 1D sorted array using the relationship between indices.
- Use binary search on the flattened structure.

Why this problem?: This introduces the binary search pattern applied to 2D grids. It's efficient for searching in sorted strucutures.

```py
def search_in_matrix(matrix, target):
    rows, cols = len(matrix), len(matrix[0])
    left, right = 0, rows * cols - 1

    while left <= right:
        mid = (left + right) // 2
        mid_value = matrix[mid // cols][mid % cols]

        if mid_value == target:
            return True
        elif mid_value < target:
            left = mid + 1
        else:
            right = mid - 1
    return False

matrix = [[1, 3, 5], [7, 9, 11], [13, 15, 17]]
print(search_in_matrix(matrix, 9))  # Output: True

```

**Find the Maximum Element in Each Row**

Problem: For a given m X n matrix, find the maximum element in each row.

Pattern: Row-wise Analysis

Approach: Constructive.

**Spiral Matrix Traversal**

For solution look here [here](patterns.md).


**Rotate a Matrix by 90 Degrees**

Problem: Rotate the matrix 90 degrees clockwise in place.

Pattern: Transformation

Approach:

- Transpose the matrix (swap rows with columns)
- Reverse each row.

```py
def rotate_matrix(matrix):
    matrix[:] = [list(row) for row in zip(*matrix[::-1])]

matrix = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
rotate_matrix(matrix)
print(matrix)
# Output: [[7, 4, 1], [8, 5, 2], [9, 6, 3]]

```

**Find the Largest Rectangle of 1s in a Binary Matrix**

Problem: Given a binary matrix, find the largest rectangle of 1s.

Pattern: Dynamic Programming + stack

Approach:

- Use a histogram-like approach:
    - For each row, treat it as the base of a histogram.
    - Use a stack to calculate the largest rectangle.

Why this problem?: It introduces dynamic programming combined with stack operations to solve matrix problems efficiently.

```py
def maximal_rectangle(matrix):
    def largest_histogram_area(heights):
        stack, max_area = [], 0
        for i, h in enumerate(heights + [0]):
            while stack and heights[stack[-1]] > h:
                height = heights[stack.pop()]
                width = i if not stack else i - stack[-1] - 1
                max_area = max(max_area, height * width)
            stack.append(i)
        return max_area

    if not matrix: return 0
    n = len(matrix[0])
    heights = [0] * n
    max_area = 0

    for row in matrix:
        for i in range(n):
            heights[i] = heights[i] + 1 if row[i] == 1 else 0
        max_area = max(max_area, largest_histogram_area(heights))

    return max_area

matrix = [[1, 0, 1, 0, 0], [1, 0, 1, 1, 1], [1, 1, 1, 1, 1], [1, 0, 0, 1, 0]]
print(maximal_rectangle(matrix))  # Output: 6

```

**Find the Kth Smallest Element in a Sorted Matrix**

Problem: Given an n x n matrix where rows and columns are sorted, find the k-th smallest element.

Pattern: Binary Search + Heap

Approach:

- Use a min-heap to store elements row by row, or apply binary search:
    - Treat the matrix as a sorted array using the indices.
    - Use binary search to narrow the range of possible values.

Why this problem?: It combines heap-based optimizations with matrix traversal. Understanding this problem is key for tackling other sorted matrix challenges.

```py
import heapq

def kth_smallest(matrix, k):
    n = len(matrix)
    min_heap = []
    
    # Push the first element of each row into the heap
    for row in range(min(k, n)):  # Only need to consider at most k rows
        heapq.heappush(min_heap, (matrix[row][0], row, 0))
    
    # Extract k-1 elements from the heap
    while k > 1:
        val, row, col = heapq.heappop(min_heap)
        if col + 1 < n:
            heapq.heappush(min_heap, (matrix[row][col + 1], row, col + 1))
        k -= 1

    return heapq.heappop(min_heap)[0]

matrix = [[1, 5, 9], [10, 11, 13], [12, 13, 15]]
k = 8
print(kth_smallest(matrix, k))  # Output: 13

```

**Word Search in a Matrix**

Problem: Find if a given word exists in a m x n matrix, where the word can be formed by sequentially adjacent characters (up, down, left, right).

Pattern: Backtracking

Approach:

- Use backtracking to explore all possible paths:
    - Chech each cell as the starting point.
    - Move to adjacent cells (up, down, left, right) to match the next character.
    - Mark cells as visited to avoid revisiting.

Why this problem?: This introduces the backtracking pattern, which is crucial for problems involving exploration or recursion.

```py
def exist(board, word):
    rows, cols = len(board), len(board[0])

    def backtrack(r, c, index):
        if index == len(word):
            return True
        if r < 0 or c < 0 or r >= rows or c >= cols or board[r][c] != word[index]:
            return False

        # Mark the cell as visited
        temp, board[r][c] = board[r][c], '#'
        found = (backtrack(r + 1, c, index + 1) or
                 backtrack(r - 1, c, index + 1) or
                 backtrack(r, c + 1, index + 1) or
                 backtrack(r, c - 1, index + 1))
        board[r][c] = temp  # Restore the cell

        return found

    for r in range(rows):
        for c in range(cols):
            if backtrack(r, c, 0):
                return True
    return False

board = [["A", "B", "C", "E"], ["S", "F", "C", "S"], ["A", "D", "E", "E"]]
word = "ABCCED"
print(exist(board, word))  # Output: True

```

**Count All Unique Paths in a Matrix**

Problem: Find the number of unique paths to move from the top-left to the bottom-right of a matrix, considering only down and right moves.

Pattern: Dynamic Programming

Approach:

- Use dynamic programming:
    - Create a DP table where dp[i][j] represents the number of ways to reach cell (i,j).
    - Base case: There's only one way to reach the first row or column.
    - Transition: dp[i][j] = dp[i-1][j] + dp[i][j-1]
- Optimize with a single row DP for space efficiency.

Why this problem?: It teaches DP for grid traversal and helps in visualizing overlapping subproblems in 2D structures.

```py
def unique_paths(m, n):
    dp = [1] * n  # Initialize the first row with 1s
    for i in range(1, m):
        for j in range(1, n):
            dp[j] += dp[j - 1]  # Update paths based on the previous row and column
    return dp[-1]

m, n = 3, 3
print(unique_paths(m, n))  # Output: 6

```

**Maximal Square**

Problem: Find the largest square containing only 1st in a binary matrix and return its area.

Pattern: Dynamic Programming

Approach:

- Use dynamic programming.
    - Let dp[i][j] represent the side length of the largest square ending at (i,j).
    - Transition: If matrix[i][j]=1, then dp[i][j]=min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1]) + 1.
    - Track the maximum value of dp[i][j] to compute the area.
- Optimize by using only two rows for DP.

Why this problem?: It combines DP and geometric intuition, crucial for finding patterns like areas and perimeters in matrices.

```py 
def maximal_square(matrix):
    if not matrix: return 0
    rows, cols = len(matrix), len(matrix[0])
    dp = [0] * (cols + 1)
    max_side = 0
    prev = 0  # Represents dp[i-1][j-1]

    for i in range(1, rows + 1):
        for j in range(1, cols + 1):
            temp = dp[j]
            if matrix[i - 1][j - 1] == 1:
                dp[j] = min(dp[j], dp[j - 1], prev) + 1
                max_side = max(max_side, dp[j])
            else:
                dp[j] = 0
            prev = temp

    return max_side * max_side

matrix = [[1, 0, 1, 0, 0], [1, 0, 1, 1, 1], [1, 1, 1, 1, 1], [1, 0, 0, 1, 0]]
print(maximal_square(matrix))  # Output: 4

```
