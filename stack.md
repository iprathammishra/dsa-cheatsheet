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

**Valid Parentheses**

Problem: Check if a string of parentheses is valid (e.g., `{[()]}`).

Approach:

- Use a stack to track opening brackets. For each closing bracket, check if it matches the top of the stack.
- If unmatched or the stack is not empty at the end, return false.

```py
def isValid(s):
    stack = []
    mapping = {')': '(', '}': '{', ']': '['}
    for char in s:
        if char in mapping:
            top = stack.pop() if stack else '#'
            if mapping[char] != top:
                return False
        else:
            stack.append(char)
    return not stack

```

**Implement Stack using Queues**

Problem: Implement a stack using two queues.

Approach: 

- Use one queue (queue1) for push operations. For pop operations, transfer all but one element to a second queue, then remove the last element.

```py
from collections import deque

class MyStack:
    def __init__(self):
        self.queue = deque()

    def push(self, x):
        self.queue.append(x)
        for _ in range(len(self.queue) - 1):
            self.queue.append(self.queue.popleft())

    def pop(self):
        return self.queue.popleft()

    def top(self):
        return self.queue[0]

    def empty(self):
        return not self.queue

```

**Min Stack**

Problem: Design a stack that supports retrieving the minimum element in constant time.

Approach:

- Use an auxiliary stack to keep track of the minimum element. Push the current minimum whenever pushing onto the main stack.

```py
class MinStack:
    def __init__(self):
        self.stack = []
        self.min_stack = []

    def push(self, x):
        self.stack.append(x)
        if not self.min_stack or x <= self.min_stack[-1]:
            self.min_stack.append(x)

    def pop(self):
        if self.stack.pop() == self.min_stack[-1]:
            self.min_stack.pop()

    def top(self):
        return self.stack[-1]

    def getMin(self):
        return self.min_stack[-1]

```

**Next Greater Element**

Problem: For each element in an array, find the next greater element on the right. If none, return -1.

Approach:

- Traverse the array in reverse. Use a stack to track candidates for the next greater element.

```py
def nextGreaterElement(nums):
    stack = []
    result = [-1] * len(nums)
    for i in range(len(nums) - 1, -1, -1):
        while stack and stack[-1] <= nums[i]:
            stack.pop()
        if stack:
            result[i] = stack[-1]
        stack.append(nums[i])
    return result

```

**Daily Temperatures**

Problem: For each day's temperature, find how many days you have to wait for a warmer temperature.

Approach:

- Use a stack to store indices of temperatures. For each day, compare the current temperature with the stack's top.

```py
def dailyTemperatures(temperatures):
    stack = []
    result = [0] * len(temperatures)
    for i, temp in enumerate(temperatures):
        while stack and temperatures[stack[-1]] < temp:
            prev_day = stack.pop()
            result[prev_day] = i - prev_day
        stack.append(i)
    return result

```

**Largest Rectangle in Histogram**

Problem: Given an array of bar heights, find the largest rectangle that can be formed.

Approach:

- Use a stack to store indices of bars in increasing order. Compute areas when popping from the stack.

```py
def largestRectangleArea(heights):
    stack = []
    max_area = 0
    heights.append(0)  # Add a sentinel value
    for i, h in enumerate(heights):
        while stack and heights[stack[-1]] > h:
            height = heights[stack.pop()]
            width = i if not stack else i - stack[-1] - 1
            max_area = max(max_area, height * width)
        stack.append(i)
    return max_area

```

**Trapping Rain Water**

Problem: Calculate how much rainwater can be trapped between the bars of varying heights.

Approach:

- Use a stack to track the indices of bars. Calculate trapped water when popping from the stack.

```py
def trap(height):
    stack = []
    water = 0
    for i, h in enumerate(height):
        while stack and height[stack[-1]] < h:
            top = stack.pop()
            if not stack:
                break
            distance = i - stack[-1] - 1
            bounded_height = min(h, height[stack[-1]]) - height[top]
            water += distance * bounded_height
        stack.append(i)
    return water

```

**Basic Calculator**

Problem: Evaluate a mathematical expression string containing +, -, (, and ).

Approach: Use a stack to handle parentheses and intermediate results. Parses the string to evaluate.

```py
def calculate(s):
    stack = []
    num, sign = 0, 1
    result = 0
    for char in s:
        if char.isdigit():
            num = num * 10 + int(char)
        elif char in "+-":
            result += sign * num
            num = 0
            sign = 1 if char == '+' else -1
        elif char == '(':
            stack.append(result)
            stack.append(sign)
            result, sign = 0, 1
        elif char == ')':
            result += sign * num
            num = 0
            result *= stack.pop()
            result += stack.pop()
    return result + sign * num

```

**Decode String**

Problem: Decode a string patterns like `3[a2[bc]]` into `abcbcabcbcabcbc`.

Approach: Use two stacks: one for numbers and another for strings. Decode when encountering `]`.

```py
def decodeString(s):
    num_stack = []
    str_stack = []
    current_str = ""
    current_num = 0
    for char in s:
        if char.isdigit():
            current_num = current_num * 10 + int(char)
        elif char == '[':
            num_stack.append(current_num)
            str_stack.append(current_str)
            current_num, current_str = 0, ""
        elif char == ']':
            num = num_stack.pop()
            prev_str = str_stack.pop()
            current_str = prev_str + num * current_str
        else:
            current_str += char
    return current_str

```

**Longest Valid Parentheses**

Problem: Find the length of the longest valid parentheses substring.

Approach: Use a stack to track indices. Push -1 initially to handle valid substring lengths.

```py
def longestValidParentheses(s):
    stack = [-1]
    max_length = 0
    for i, char in enumerate(s):
        if char == '(':
            stack.append(i)
        else:
            stack.pop()
            if not stack:
                stack.append(i)
            else:
                max_length = max(max_length, i - stack[-1])
    return max_length

```
