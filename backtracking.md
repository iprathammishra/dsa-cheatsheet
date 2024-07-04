# Backtracking

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
