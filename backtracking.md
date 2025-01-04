# Backtracking

For Backtracking and Recursion don't forget to solve these questions:
1. Rat in a Maze
2. Combination Sum I and II
3. N-queens
4. Letter combination of a phone number
5. Subsets I and II
6. Word search
7. Validate BST
8. Binary Tree Maximum path sum
9. Find all paths from SRC to TAR in graph
10. Number of islands
11. Find if path exists in graph
12. is Graph Bipartite

## Permutations

(Permutations)

```python

def permute(nums):
    def backtrack(start, end):
        if start == end:
            result.append(nums[:])
        for i in range(start, end):
            nums[start], nums[i] = nums[i], nums[start]
            backtrack(start + 1, end)
            nums[start], nums[i] = nums[i], nums[start]
    
    result = []
    backtrack(0, len(nums))
    return result

# Example usage:
nums = [1, 2, 3]
print(permute(nums))

```

(Permutations II- with Duplicates)

```python

def permute_unique(nums):
    def backtrack(start, end):
        if start == end:
            result.append(nums[:])
        seen = set()
        for i in range(start, end):
            if nums[i] in seen:
                continue
            seen.add(nums[i])
            nums[start], nums[i] = nums[i], nums[start]
            backtrack(start + 1, end)
            nums[start], nums[i] = nums[i], nums[start]
    
    result = []
    nums.sort()
    backtrack(0, len(nums))
    return result

# Example usage:
nums = [1, 1, 2]
print(permute_unique(nums))

```

## Combinations

(Combinations)

```python

def combine(n, k):
    def backtrack(start, path):
        if len(path) == k:
            result.append(path[:])
            return
        for i in range(start, n + 1):
            path.append(i)
            backtrack(i + 1, path)
            path.pop()
    
    result = []
    backtrack(1, [])
    return result

# Example usage:
n = 4
k = 2
print(combine(n, k))

```

(Combination Sum)

```python

def combination_sum(candidates, target):
    def backtrack(start, path, target):
        if target < 0:
            return
        if target == 0:
            result.append(path[:])
            return
        for i in range(start, len(candidates)):
            path.append(candidates[i])
            backtrack(i, path, target - candidates[i])
            path.pop()
    
    result = []
    candidates.sort()
    backtrack(0, [], target)
    return result

# Example usage:
candidates = [2, 3, 6, 7]
target = 7
print(combination_sum(candidates, target))

```

(Combination Sum II)

```python

def combination_sum2(candidates, target):
    def backtrack(start, path, target):
        if target < 0:
            return
        if target == 0:
            result.append(path[:])
            return
        for i in range(start, len(candidates)):
            if i > start and candidates[i] == candidates[i - 1]:
                continue
            path.append(candidates[i])
            backtrack(i + 1, path, target - candidates[i])
            path.pop()
    
    result = []
    candidates.sort()
    backtrack(0, [], target)
    return result

# Example usage:
candidates = [10, 1, 2, 7, 6, 1, 5]
target = 8
print(combination_sum2(candidates, target))

```

(Letter Combinations of a Phone Number)

```python

def letter_combinations(digits):
    if not digits:
        return []
    
    phone_map = {
        "2": "abc", "3": "def", "4": "ghi", "5": "jkl",
        "6": "mno", "7": "pqrs", "8": "tuv", "9": "wxyz"
    }
    
    def backtrack(index, path):
        if index == len(digits):
            result.append("".join(path))
            return
        possible_letters = phone_map[digits[index]]
        for letter in possible_letters:
            path.append(letter)
            backtrack(index + 1, path)
            path.pop()
    
    result = []
    backtrack(0, [])
    return result

# Example usage:
digits = "23"
print(letter_combinations(digits))

```

## Subsets

(Subsets)

```python

def subsets(nums):
    def backtrack(start, path):
        result.append(path[:])
        for i in range(start, len(nums)):
            path.append(nums[i])
            backtrack(i + 1, path)
            path.pop()
    
    result = []
    backtrack(0, [])
    return result

# Example usage:
nums = [1, 2, 3]
print(subsets(nums))

```

(Subsets II- with Duplicates)

```python

def subsets_with_dup(nums):
    def backtrack(start, path):
        result.append(path[:])
        for i in range(start, len(nums)):
            if i > start and nums[i] == nums[i - 1]:
                continue
            path.append(nums[i])
            backtrack(i + 1, path)
            path.pop()
    
    result = []
    nums.sort()
    backtrack(0, [])
    return result

# Example usage:
nums = [1, 2, 2]
print(subsets_with_dup(nums))

```

## Palindromic Structures

(Palindrome Partitioning)

```python

def partition(s):
    def is_palindrome(sub):
        return sub == sub[::-1]
    
    def backtrack(start, path):
        if start == len(s):
            result.append(path[:])
            return
        for end in range(start + 1, len(s) + 1):
            if is_palindrome(s[start:end]):
                path.append(s[start:end])
                backtrack(end, path)
                path.pop()
    
    result = []
    backtrack(0, [])
    return result

# Example usage:
s = "aab"
print(partition(s))

```

## Path and Grid Problems

(Word Search)

```python

def exist(board, word):
    def backtrack(i, j, suffix):
        if not suffix:
            return True
        if i < 0 or i >= len(board) or j < 0 or j >= len(board[0]) or board[i][j] != suffix[0]:
            return False
        ret = False
        board[i][j], temp = '#', board[i][j]
        for di, dj in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
            ret = backtrack(i + di, j + dj, suffix[1:])
            if ret:
                break
        board[i][j] = temp
        return ret
    
    for i in range(len(board)):
        for j in range(len(board[0])):
            if backtrack(i, j, word):
                return True
    return False

# Example usage:
board = [
  ['A','B','C','E'],
  ['S','F','C','S'],
  ['A','D','E','E']
]
word = "ABCCED"
print(exist(board, word))

```

(N-Queens)

```python

def solve_n_queens(n):
    def is_not_under_attack(row, col):
        return not (cols[col] or hills[row - col] or dales[row + col])
    
    def place_queen(row, col):
        queens.add((row, col))
        cols[col] = True
        hills[row - col] = True
        dales[row + col] = True
    
    def remove_queen(row, col):
        queens.remove((row, col))
        cols[col] = False
        hills[row - col] = False
        dales[row + col] = False
    
    def add_solution():
        solution = []
        for _, col in sorted(queens):
            solution.append('.' * col + 'Q' + '.' * (n - col - 1))
        solutions.append(solution)
    
    def backtrack(row):
        for col in range(n):
            if is_not_under_attack(row, col):
                place_queen(row, col)
                if row + 1 == n:
                    add_solution()
                else:
                    backtrack(row + 1)
                remove_queen(row, col)
    
    solutions = []
    queens = set()
    cols = [False] * n
    hills = [False] * (2 * n - 1)
    dales = [False] * (2 * n - 1)
    backtrack(0)
    return solutions

# Example usage:
n = 4
print(solve_n_queens(n))

```

(Rat in a Maze)

```python

def solve_maze(maze):
    def is_safe(x, y):
        return 0 <= x < N and 0 <= y < N and maze[x][y] == 1
    
    def solve(x, y):
        if x == N - 1 and y == N - 1:
            sol[x][y] = 1
            return True
        if is_safe(x, y):
            sol[x][y] = 1
            if solve(x + 1, y):
                return True
            if solve(x, y + 1):
                return True
            sol[x][y] = 0
            return False
        return False
    
    N = len(maze)
    sol = [[0] * N for _ in range(N)]
    if solve(0, 0):
        return sol
    return []

# Example usage:
maze = [
    [1, 0, 0, 0],
    [1, 1, 0, 1],
    [0, 1, 0, 0],
    [1, 1, 1, 1]
]
print(solve_maze(maze))

```

(Sudoku Solver)

```python

def solve_sudoku(board):
    def is_valid(board, r, c, k):
        for i in range(9):
            if board[i][c] == k: return False
            if board[r][i] == k: return False
            if board[3 * (r // 3) + i // 3][3 * (c // 3) + i % 3] == k: return False
        return True
    
    def solve():
        for r in range(9):
            for c in range(9):
                if board[r][c] == '.':
                    for k in map(str, range(1, 10)):
                        if is_valid(board, r, c, k):
                            board[r][c] = k
                            if solve():
                                return True
                            board[r][c] = '.'
                    return False
        return True
    
    solve()

# Example usage:
board = [
    ["5","3",".",".","7",".",".",".","."],
    ["6",".",".","1","9","5",".",".","."],
    [".","9","8",".",".",".",".","6","."],
    ["8",".",".",".","6",".",".",".","3"],
    ["4",".",".","8",".","3",".",".","1"],
    ["7",".",".",".","2",".",".",".","6"],
    [".","6",".",".",".",".","2","8","."],
    [".",".",".","4","1","9",".",".","5"],
    [".",".",".",".","8",".",".","7","9"]
]
solve_sudoku(board)
for row in board:
    print(row)

```

## Combinatorial Generation

(Generate Parentheses)

```python

def generate_parenthesis(n):
    def backtrack(s='', left=0, right=0):
        if len(s) == 2 * n:
            result.append(s)
            return
        if left < n:
            backtrack(s + '(', left + 1, right)
        if right < left:
            backtrack(s + ')', left, right + 1)
    
    result = []
    backtrack()
    return result

# Example usage:
n = 3
print(generate_parenthesis(n))

```

(Gray Code)

```python

def gray_code(n):
    result = []
    for i in range(1 << n):
        result.append(i ^ (i >> 1))
    return result

# Example usage:
n = 2
print(gray_code(n))

```

(Beautiful Arrangement)

```python

def count_arrangement(n):
    def backtrack(pos, visited):
        if pos > n:
            return 1
        count = 0
        for i in range(1, n + 1):
            if not visited[i] and (pos % i == 0 or i % pos == 0):
                visited[i] = True
                count += backtrack(pos + 1, visited)
                visited[i] = False
        return count
    
    visited = [False] * (n + 1)
    return backtrack(1, visited)

# Example usage:
n = 2
print(count_arrangement(n))

```

Here are 10 backtracking problems ranging from easy to hard, with detailed explanations of the problem, pattern, approach, and code:

**Generate All Subsets**

Pattern: Subset Generation

Problem: Given a set of distinct integers, return all possible subsets.

Approach:

- Use backtracking to explore the inclusion or exclusion of each element.
- At each recursive step, add the current subset to the result.

```py
def subsets(nums):
    result = []

    def backtrack(start, current):
        result.append(current[:])  # Add the current subset
        for i in range(start, len(nums)):
            current.append(nums[i])  # Include nums[i]
            backtrack(i + 1, current)  # Explore further
            current.pop()  # Exclude nums[i]

    backtrack(0, [])
    return result

print(subsets([1, 2, 3]))  # Output: [[], [1], [1, 2], [1, 2, 3], [1, 3], [2], [2, 3], [3]]
```

**Permutations**

Pattern: Permutation Generation

Problem: Given a list of numbers, return all possible permutations.

Approach:

- Swap elements to fix the first position, then recursively permute the remaining elements.

```py
def permute(nums):
    result = []

    def backtrack(start):
        if start == len(nums):
            result.append(nums[:])  # Add the current permutation
        for i in range(start, len(nums)):
            nums[start], nums[i] = nums[i], nums[start]  # Swap
            backtrack(start + 1)  # Recurse
            nums[start], nums[i] = nums[i], nums[start]  # Backtrack

    backtrack(0)
    return result

print(permute([1, 2, 3]))  # Output: [[1, 2, 3], [1, 3, 2], ..., [3, 2, 1]]

```

**Combination Sum**

Pattern: Combination Generation with Constraints.

Problem: Find all unique combinations of numbers that sum up to a target.

Approach:

- Use backtracking to explore combinations.
- Skip the current number after picking it to avoid duplicates.

```py
def combination_sum(candidates, target):
    result = []

    def backtrack(start, remaining, current):
        if remaining == 0:
            result.append(current[:])  # Add valid combination
            return
        if remaining < 0:
            return

        for i in range(start, len(candidates)):
            current.append(candidates[i])  # Include candidates[i]
            backtrack(i, remaining - candidates[i], current)  # Allow reuse
            current.pop()  # Backtrack

    backtrack(0, target, [])
    return result

print(combination_sum([2, 3, 6, 7], 7))  # Output: [[2, 2, 3], [7]]

```

**N-Queens Problem**

Pattern: Constraint Satisfaction Problem

Problem: Place n queens on an n x x chessboard so no two queens threaten each other.

Approach:

- Use backtracking to place queens row by row.
- Check constraints for column, diagonal, and anti-diagonal conflicts.

```py
def solve_n_queens(n):
    result = []
    board = [["."] * n for _ in range(n)]

    def is_safe(row, col):
        for i in range(row):
            if board[i][col] == "Q" or \
               (col - (row - i) >= 0 and board[i][col - (row - i)] == "Q") or \
               (col + (row - i) < n and board[i][col + (row - i)] == "Q"):
                return False
        return True

    def backtrack(row):
        if row == n:
            result.append(["".join(r) for r in board])  # Add board configuration
            return
        for col in range(n):
            if is_safe(row, col):
                board[row][col] = "Q"  # Place queen
                backtrack(row + 1)  # Recurse to the next row
                board[row][col] = "."  # Backtrack

    backtrack(0)
    return result

print(solve_n_queens(4))
# Output: [['.Q..', '...Q', 'Q...', '..Q.'], ...]

```

**Sudoku Solver**

Pattern: Constraint Satisfaction with Pruning.

Problem: Solve a 9 x 9 Sudoku puzzle.

Approach:

- Use backtracking to fill empty cells.
- Check row, column, and 3x3 subgrid constraints.

```py
def solve_sudoku(board):
    def is_valid(num, row, col):
        for i in range(9):
            if board[row][i] == num or board[i][col] == num or \
               board[row - row % 3 + i // 3][col - col % 3 + i % 3] == num:
                return False
        return True

    def backtrack():
        for i in range(9):
            for j in range(9):
                if board[i][j] == ".":
                    for num in map(str, range(1, 10)):
                        if is_valid(num, i, j):
                            board[i][j] = num
                            if backtrack():
                                return True
                            board[i][j] = "."  # Backtrack
                    return False
        return True

    backtrack()

board = [["5", "3", ".", ".", "7", ".", ".", ".", "."], ...]
solve_sudoku(board)
print(board)  # Output: Solved Sudoku

```

**Word Search**

Pattern: Pathfinding with Backtracking.

Problem: Determine if a word exists in a grid by moving in adjacent cells.

Approach:

- Use backtracking to explore all possible paths for forming the word.
- Mark cells as visited during the recursion and restore them during backtracking.

```py
def exist(board, word):
    rows, cols = len(board), len(board[0])

    def backtrack(r, c, index):
        if index == len(word):  # Word is completely matched
            return True
        if r < 0 or r >= rows or c < 0 or c >= cols or board[r][c] != word[index]:
            return False

        temp, board[r][c] = board[r][c], "#"  # Mark the cell as visited
        found = (backtrack(r + 1, c, index + 1) or
                 backtrack(r - 1, c, index + 1) or
                 backtrack(r, c + 1, index + 1) or
                 backtrack(r, c - 1, index + 1))
        board[r][c] = temp  # Restore the cell
        return found

    for r in range(rows):
        for c in range(cols):
            if board[r][c] == word[0] and backtrack(r, c, 0):
                return True
    return False

board = [["A", "B", "C", "E"], ["S", "F", "C", "S"], ["A", "D", "E", "E"]]
word = "ABCCED"
print(exist(board, word))  # Output: True

```

**Rat in a Maze**

Problem: Given a maze represented by a binary matrix, find all paths from the top-left corner to the bottom-right corner. You can move in all four directions.

Approach:

- Use backtracking to explore all possible directions.
- Maintain a visited matrix to prevent revisiting cells.

```py
def rat_in_maze(maze):
    n = len(maze)
    result = []
    path = []
    directions = [(1, 0, 'D'), (0, -1, 'L'), (0, 1, 'R'), (-1, 0, 'U')]

    def backtrack(r, c):
        if r == n - 1 and c == n - 1:  # Reached the destination
            result.append("".join(path))
            return

        for dr, dc, move in directions:
            nr, nc = r + dr, c + dc
            if 0 <= nr < n and 0 <= nc < n and maze[nr][nc] == 1:
                maze[r][c] = -1  # Mark as visited
                path.append(move)
                backtrack(nr, nc)
                path.pop()
                maze[r][c] = 1  # Backtrack

    if maze[0][0] == 1:
        backtrack(0, 0)
    return result

maze = [[1, 0, 0, 0], [1, 1, 0, 1], [0, 1, 0, 0], [1, 1, 1, 1]]
print(rat_in_maze(maze))  # Output: ['DDRDRR', 'DRDDRR']

```

**Generate Parentheses**

Problem: Generate all combinations of well-formed parentheses for n pairs.

Approach:

- Use backtracking with constraints:
    - Add an open parenthesis if it's allowed.
    - Add a close parenthesis only if it doesn't exceed the count of open parentheses.

```py
def generate_parentheses(n):
    result = []

    def backtrack(open_count, close_count, current):
        if open_count == close_count == n:
            result.append(current)
            return

        if open_count < n:
            backtrack(open_count + 1, close_count, current + "(")
        if close_count < open_count:
            backtrack(open_count, close_count + 1, current + ")")

    backtrack(0, 0, "")
    return result

print(generate_parentheses(3))  # Output: ['((()))', '(()())', '(())()', '()(())', '()()()']

```

**Palindrome Partitioning**

Problem: Partition a string such that every substring is a palindrome.

Approach:

- Use backtracking to generate all possible partitions.
- Check if each substring is a palindrome before proceeding.

```py
def partition(s):
    result = []

    def is_palindrome(sub):
        return sub == sub[::-1]

    def backtrack(start, current):
        if start == len(s):
            result.append(current[:])
            return

        for end in range(start + 1, len(s) + 1):
            if is_palindrome(s[start:end]):
                current.append(s[start:end])
                backtrack(end, current)
                current.pop()

    backtrack(0, [])
    return result

print(partition("aab"))  # Output: [['a', 'a', 'b'], ['aa', 'b']]

```

**Hamiltonian Path**

Problem: Determine if a Hamiltonian path (visiting each vertex exactly once) exists in a graph.

Approach:

- Use backtracking to explore all paths, marking vertices as visited and restoring them during backtracking.

```py
def hamiltonian_path(graph, start):
    n = len(graph)
    visited = [False] * n

    def backtrack(path):
        if len(path) == n:  # All vertices visited
            return True

        for neighbor in range(n):
            if graph[path[-1]][neighbor] == 1 and not visited[neighbor]:
                visited[neighbor] = True
                path.append(neighbor)
                if backtrack(path):
                    return True
                path.pop()
                visited[neighbor] = False

        return False

    visited[start] = True
    return backtrack([start])

graph = [[0, 1, 1, 0], [1, 0, 1, 1], [1, 1, 0, 1], [0, 1, 1, 0]]
print(hamiltonian_path(graph, 0))  # Output: True

```
