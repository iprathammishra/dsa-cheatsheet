# Patterns

## KMP- Knutt-Morris-Pratt Algorithm

Purpose: Efficiently find a pattern in a text by skipping redundant comparisons.

How it Works:

1. Build LPS Array (Longest Prefix Suffix): 
    - Preprocess the pattern to create an array of longest prefixes that are also suffixes.
    - This helps skip parts of the text that don’t match, minimizing re-checks.
2. Pattern Matching:
    - Use two pointers, one for text (i) and one for pattern (j).
    - If characters match, move both pointers. If they mismatch, use LPS to reset j without moving i back.
    - Repeat until the pattern is found or text is fully checked.
    
Time Complexity: 
O(m+n) 
m is pattern length, 
n is text length.


```py
def compute_lps(pattern):
    m = len(pattern)
    lps = [0] * m
    length = 0
    i = 1
    while i < m:
        if pattern[i] == pattern[length]:
            length += 1
            lps[i] = length
            i += 1
        else:
            if length != 0:
                length = lps[length - 1]
            else:
                lps[i] = 0
                i += 1
    return lps

def kmp_search(text, pattern):
    n = len(text)
    m = len(pattern)
    lps = compute_lps(pattern)
    i = j = 0
    matches = []
    
    while i < n:
        if text[i] == pattern[j]:
            i += 1
            j += 1
        if j == m:
            matches.append(i - j)
            j = lps[j - 1]
        elif i < n and text[i] != pattern[j]:
            if j != 0:
                j = lps[j - 1]
            else:
                i += 1
    return matches

# Example usage
text = "ABABABC"
pattern = "ABABC"
print(kmp_search(text, pattern))  # Output: [2]

```

## Minimum Cost Path in a Grid - Dijkstra 

Purpose: Find the path with the minimum sum of weights from the top-left to the bottom-right corner of a grid.

How it Works:

1. Set Up Priority Queue:
- Use a min-heap (priority queue) to explore paths with the lowest current cost.
- Track minimum costs to reach each cell, initializing all cells as infinity except the starting cell (0, 0).
2. Explore Neighbors:
- For each cell (i, j), add its neighbors (i+1, j), (i, j+1), (i-1, j), (i, j-1) with updated costs if the path cost is lower than the current known cost.
- Continue until reaching the target cell (n-1, m-1).
- Return Minimum Cost to reach the target.

Time complexity is (Elogv) Where E is the number of edges and V is the number of cells.

```py
import heapq

def min_cost_path(grid):
    n, m = len(grid), len(grid[0])
    directions = [(1, 0), (0, 1), (-1, 0), (0, -1)]
    min_cost = [[float('inf')] * m for _ in range(n)]
    min_cost[0][0] = grid[0][0]
    min_heap = [(grid[0][0], 0, 0)]  # (cost, row, col)

    while min_heap:
        cost, i, j = heapq.heappop(min_heap)
        if i == n - 1 and j == m - 1:
            return cost  # reached bottom-right corner
        for di, dj in directions:
            ni, nj = i + di, j + dj
            if 0 <= ni < n and 0 <= nj < m:
                new_cost = cost + grid[ni][nj]
                if new_cost < min_cost[ni][nj]:
                    min_cost[ni][nj] = new_cost
                    heapq.heappush(min_heap, (new_cost, ni, nj))
    return -1  # if no path found

# Example usage
grid = [
    [1, 3, 1],
    [1, 5, 1],
    [4, 2, 1]
]
print(min_cost_path(grid))  # Output: 7

```

## 0/1 Knapsack or Bounded Knapsack

Purpose: Maximize the total value of items in a knapsack without exceeding a given weight limit, where each item can either be included (1) or excluded (0) from the knapsack.

How it Works:

1. Define DP Array:
- Use a 2D DP table, dp[i][w], where i is the index of the item and w is the weight capacity.
- dp[i][w] stores the maximum value achievable with items up to i and weight w.
2. Fill the DP Table (Bottom-Up):
- Initialize dp[0][*] and dp[*][0] to zero, as zero items or zero capacity yields zero value.
- For each item i, and for each capacity w:
- Exclude the item: Set dp[i][w] = dp[i-1][w].
- Include the item (if it fits): Set dp[i][w] = max(dp[i-1][w], dp[i-1][w - weight[i]] + value[i]).
- Result: dp[n][W] gives the maximum value achievable for capacity W using n items.

Note: This can also we done using 1D DP for more space optimized approach.

```py
def knapsack(values, weights, W):
    n = len(values)
    dp = [[0] * (W + 1) for _ in range(n + 1)]

    for i in range(1, n + 1):
        for w in range(W + 1):
            if weights[i - 1] <= w:
                dp[i][w] = max(dp[i - 1][w], dp[i - 1][w - weights[i - 1]] + values[i - 1])
            else:
                dp[i][w] = dp[i - 1][w]

    return dp[n][W]

# Example usage
values = [60, 100, 120]
weights = [10, 20, 30]
W = 50
print(knapsack(values, weights, W))  # Output: 220

```

## Vertical Traversal of a Tree

Purpose: Traverse and group nodes of a binary tree by vertical columns, ordered from leftmost to rightmost.

How it Works:

1. Use a Data Structure for Tracking Columns:
- Use a dictionary to map each column to its nodes.
- Traverse the tree with a queue or recursive DFS, tracking:
    - Column index: Shift left (-1) for left children and right (+1) for right children.
    - Row index: Track depth to sort nodes at the same vertical level.
2. Sort and Group Nodes:
- For each column, sort nodes by their row index, and if tied, by node value.
- Collect nodes column by column from leftmost to rightmost.
3. Result: Output a list of lists, where each sublist represents nodes in each vertical column from left to right.

Time complexity is NlogN where N is the number of nodes and because we sort them.

```py
from collections import defaultdict, deque

def vertical_traversal(root):
    if not root:
        return []

    # Dictionary to hold nodes per column: {column: [(row, value)]}
    col_dict = defaultdict(list)
    queue = deque([(root, 0, 0)])  # (node, column, row)

    # BFS traversal
    while queue:
        node, col, row = queue.popleft()
        col_dict[col].append((row, node.val))
        
        if node.left:
            queue.append((node.left, col - 1, row + 1))
        if node.right:
            queue.append((node.right, col + 1, row + 1))

    # Sort columns and each column's nodes by row, then by value
    result = []
    for col in sorted(col_dict.keys()):
        column_nodes = sorted(col_dict[col], key=lambda x: (x[0], x[1]))
        result.append([val for _, val in column_nodes])

    return result

# Example usage
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

# Constructing example tree
root = TreeNode(1)
root.left = TreeNode(2)
root.right = TreeNode(3)
root.left.left = TreeNode(4)
root.left.right = TreeNode(5)
root.right.left = TreeNode(6)
root.right.right = TreeNode(7)

print(vertical_traversal(root))  # Output: [[4], [2], [1, 5, 6], [3], [7]]

```

## Serialize and De-serialize a Tree

Purpose: Convert a binary tree into a string (serialization) for easy storage or transmission, and reconstruct the original tree from the string (deserialization).

How it Works:

1. Serialization:
- Use a traversal (preorder or level-order) to capture the tree structure and node values.
- Represent null nodes with a placeholder (e.g., #) to preserve structure.
- Output a string with node values and # separated by a delimiter (e.g., comma).
2. Deserialization:
- Split the serialized string into a list of values.
- Use recursive or iterative logic to rebuild the tree using these values, treating # as a null node.

Time complexity is O(N) where N is the number of nodes.

```py
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

class Codec:
    def serialize(self, root):
        """Encodes a tree to a single string."""
        def dfs(node):
            if not node:
                result.append("#")
                return
            result.append(str(node.val))
            dfs(node.left)
            dfs(node.right)
        
        result = []
        dfs(root)
        return ",".join(result)

    def deserialize(self, data):
        """Decodes your encoded data to tree."""
        def dfs():
            val = next(vals)
            if val == "#":
                return None
            node = TreeNode(int(val))
            node.left = dfs()
            node.right = dfs()
            return node

        vals = iter(data.split(","))
        return dfs()

# Example usage
# Constructing example tree
root = TreeNode(1)
root.left = TreeNode(2)
root.right = TreeNode(3)
root.right.left = TreeNode(4)
root.right.right = TreeNode(5)

codec = Codec()
data = codec.serialize(root)
print(data)  # Serialized string
new_root = codec.deserialize(data)

```

## Binary Tree to BST.

Purpose: Transform a binary tree into a binary search tree (BST) while preserving the structure of the original tree (i.e., nodes’ positions remain the same).

How it Works:

1. Inorder Traversal to Collect Values:
- Perform an inorder traversal of the binary tree to collect all node values in a list.
- Inorder traversal captures the nodes in their natural order (left-to-right).
2. Sort the Values:
- Sort the list of collected values, as an inorder traversal of a BST will yield sorted values.
3. Reassign Values Using Inorder Traversal:
- Perform a second inorder traversal of the binary tree and replace each node’s value with the next sorted value, thus converting the tree into a BST.

Time complexity is O(NlogN) where N is the number of nodes and because we sorted it.

```py 
from collections import defaultdict, deque

def vertical_traversal(root):
    if not root:
        return []

    # Dictionary to hold nodes per column: {column: [(row, value)]}
    col_dict = defaultdict(list)
    queue = deque([(root, 0, 0)])  # (node, column, row)

    # BFS traversal
    while queue:
        node, col, row = queue.popleft()
        col_dict[col].append((row, node.val))
        
        if node.left:
            queue.append((node.left, col - 1, row + 1))
        if node.right:
            queue.append((node.right, col + 1, row + 1))

    # Sort columns and each column's nodes by row, then by value
    result = []
    for col in sorted(col_dict.keys()):
        column_nodes = sorted(col_dict[col], key=lambda x: (x[0], x[1]))
        result.append([val for _, val in column_nodes])

    return result

# Example usage
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

# Constructing example tree
root = TreeNode(1)
root.left = TreeNode(2)
root.right = TreeNode(3)
root.left.left = TreeNode(4)
root.left.right = TreeNode(5)
root.right.left = TreeNode(6)
root.right.right = TreeNode(7)

print(vertical_traversal(root))  # Output: [[4], [2], [1, 5, 6], [3], [7]]

```

## Spiral Matrix 

Classic problem but the solution is advanced.

```py
class Solution:
    def spiralOrder(self, matrix: List[List[int]]) -> List[int]:
        rows, cols = len(matrix), len(matrix[0])
        x, y, dx, dy = 0, 0, 1, 0
        res = []

        for _ in range(rows * cols):
            res.append(matrix[y][x])
            matrix[y][x] = "."

            if not 0 <= x + dx < cols or not 0 <= y + dy < rows or matrix[y+dy][x+dx] == ".":
                dx, dy = -dy, dx
            
            x += dx
            y += dy
        
        return res
```