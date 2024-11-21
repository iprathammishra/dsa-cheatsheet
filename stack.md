# Stack

## Monotonic Stack

A monotonic stack is a stack where the elements are either entirely in increasing order (monotonic increasing stack) or in decreasing order (monotonic decreasing stack). Monotonic stacks are useful for problems where you need to maintain a sequence of elements with a specific order property, such as finding the next greaater or smaller elenent for each element in an array.

(Finding the next greater element for each element in an array)

```python

def next_greater_element(nums):
    result = [-1] * len(nums)
    stack = []
    for i in range(len(nums)):
        while stack and nums[stack[-1]] < nums[i]:
            result[stack.pop()] = nums[i]
        stack.append(i)
    return result

# Example usage:
nums = [2, 1, 2, 4, 3]
print(next_greater_element(nums))  # Output: [4, 2, 4, -1, -1]

```

## Stack Patterns

### Basic Stack Operations

(Implement fundamental operations of a stack)

```python

class Stack:
    def __init__(self):
        self.stack = []

    def push(self, x):
        self.stack.append(x)

    def pop(self):
        if not self.is_empty():
            return self.stack.pop()
        return None

    def top(self):
        if not self.is_empty():
            return self.stack[-1]
        return None

    def is_empty(self):
        return len(self.stack) == 0

# Example usage:
s = Stack()
s.push(1)
s.push(2)
print(s.top())  # Output: 2
print(s.pop())  # Output: 2
print(s.is_empty())  # Output: False

```

(Parenthesis Matching- Use a stack to check whether parentheses in a string are properly matched)

```python

def is_valid_parentheses(s):
    stack = []
    mapping = {')': '(', '}': '{', ']': '['}
    for char in s:
        if char in mapping:
            top_element = stack.pop() if stack else '#'
            if mapping[char] != top_element:
                return False
        else:
            stack.append(char)
    return not stack

# Example usage:
s = "({[]})"
print(is_valid_parentheses(s))  # Output: True

```

(Expression Evaluation- Evaluate arithmetic expressions using a stack)

```python

def eval_rpn(tokens):
    stack = []
    for token in tokens:
        if token not in "+-*/":
            stack.append(int(token))
        else:
            b = stack.pop()
            a = stack.pop()
            if token == '+':
                stack.append(a + b)
            elif token == '-':
                stack.append(a - b)
            elif token == '*':
                stack.append(a * b)
            elif token == '/':
                stack.append(int(a / b))  # Truncate towards zero
    return stack[0]

# Example usage:
tokens = ["2", "1", "+", "3", "*"]
print(eval_rpn(tokens))  # Output: 9

```

(Find the next greater element for each element in the array using a stack)

```python

def next_greater_element(nums):
    result = [-1] * len(nums)
    stack = []
    for i in range(len(nums)):
        while stack and nums[stack[-1]] < nums[i]:
            result[stack.pop()] = nums[i]
        stack.append(i)
    return result

# Example usage:
nums = [2, 1, 2, 4, 3]
print(next_greater_element(nums))  # Output: [4, 2, 4, -1, -1]

```

(Find the largest rectangular area in a histogram using a stack)

```python

def largest_rectangle_area(heights):
    stack = []
    max_area = 0
    heights.append(0)
    for i in range(len(heights)):
        while stack and heights[i] < heights[stack[-1]]:
            h = heights[stack.pop()]
            w = i if not stack else i - stack[-1] - 1
            max_area = max(max_area, h * w)
        stack.append(i)
    heights.pop()
    return max_area

# Example usage:
heights = [2, 1, 5, 6, 2, 3]
print(largest_rectangle_area(heights))  # Output: 10

```

(Evaluate an RPN expression using a stack)

```python

def eval_rpn(tokens):
    stack = []
    for token in tokens:
        if token not in "+-*/":
            stack.append(int(token))
        else:
            b = stack.pop()
            a = stack.pop()
            if token == '+':
                stack.append(a + b)
            elif token == '-':
                stack.append(a - b)
            elif token == '*':
                stack.append(a * b)
            elif token == '/':
                stack.append(int(a / b))  # Truncate towards zero
    return stack[0]

# Example usage:
tokens = ["2", "1", "+", "3", "*"]
print(eval_rpn(tokens))  # Output: 9

```

(Implement a stack that supports retrieving the minimum element in constant time)

```python

class MinStack:
    def __init__(self):
        self.stack = []
        self.min_stack = []

    def push(self, x):
        self.stack.append(x)
        if not self.min_stack or x <= self.min_stack[-1]:
            self.min_stack.append(x)

    def pop(self):
        if self.stack:
            if self.stack[-1] == self.min_stack[-1]:
                self.min_stack.pop()
            return self.stack.pop()

    def top(self):
        return self.stack[-1] if self.stack else None

    def get_min(self):
        return self.min_stack[-1] if self.min_stack else None

# Example usage:
min_stack = MinStack()
min_stack.push(-2)
min_stack.push(0)
min_stack.push(-3)
print(min_stack.get_min())  # Output: -3
min_stack.pop()
print(min_stack.top())  # Output: 0
print(min_stack.get_min())  # Output: -2

```

(Simplify an absolute path for a file- Unix style)

```python

def simplify_path(path):
    stack = []
    parts = path.split("/")
    for part in parts:
        if part == "..":
            if stack:
                stack.pop()
        elif part and part != ".":
            stack.append(part)
    return "/" + "/".join(stack)

# Example usage:
path = "/home//foo/"
print(simplify_path(path))  # Output: "/home/foo"

```

(Decode an encoded string based on certain rules using a stack)

```python

def decode_string(s):
    stack = []
    current_num = 0
    current_string = ""
    for char in s:
        if char.isdigit():
            current_num = current_num * 10 + int(char)
        elif char == '[':
            stack.append((current_string, current_num))
            current_string, current_num = "", 0
        elif char == ']':
            last_string, num = stack.pop()
            current_string = last_string + num * current_string
        else:
            current_string += char
    return current_string

# Example usage:
s = "3[a]2[bc]"
print(decode_string(s))  # Output: "aaabcbc"

```

(Find the maximum nesting depth of parentheses)

```python

def max_depth(s):
    stack = []
    max_depth = 0
    current_depth = 0
    for char in s:
        if char == '(':
            stack.append(char)
            current_depth += 1
            max_depth = max(max_depth, current_depth)
        elif char == ')':
            if stack:
                stack.pop()
                current_depth -= 1
    return max_depth

# Example usage:
s = "(1+(2*3)+((8)/4))+1"
print(max_depth(s))  # Output: 3

```

(Find the span of stock's price for all the days using a stack)

```python

def calculate_span(prices):
    stack = []
    spans = [0] * len(prices)
    for i, price in enumerate(prices):
        while stack and prices[stack[-1]] <= price:
            stack.pop()
        spans[i] = i + 1 if not stack else i - stack[-1]
        stack.append(i)
    return spans

# Example usage:
prices = [100, 80, 60, 70, 60, 75, 85]
print(calculate_span(prices))  # Output: [1, 1, 1, 2, 1, 4, 6]

```

(Remove adjacent characters that are the same using a stack)

```python

def remove_adjacent_duplicates(s):
    stack = []
    for char in s:
        if stack and stack[-1] == char:
            stack.pop()
        else:
            stack.append(char)
    return ''.join(stack)

# Example usage:
s = "abbaca"
print(remove_adjacent_duplicates(s))  # Output: "ca"

```

Here are some patterns to look after later.




