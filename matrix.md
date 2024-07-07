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
