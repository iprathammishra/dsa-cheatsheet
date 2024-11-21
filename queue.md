# Queue

## Monotonic Queue

A monotonic queue is a deque (double-ended queue) where elements are either entirely in increasing order (monotonic increasing deque) or in decreasing order (monotonic decreasing deque). Monotonic queues are useful for problems where you need to maintain a sequence of elements with a specific order property, such as finding the maximum or minimum element in a sliding window.

(Finding the maximum element in all sliding windows of size k in an array)

```python

from collections import deque

def max_sliding_window(nums, k):
    if not nums:
        return []
    
    # Monotonic deque to store indices of elements in decreasing order
    deque_index = deque()
    result = []

    for i in range(len(nums)):
        # Remove elements not within the window
        if deque_index and deque_index[0] < i - k + 1:
            deque_index.popleft()
        
        # Maintain monotonic decreasing order
        while deque_index and nums[deque_index[-1]] < nums[i]:
            deque_index.pop()

        deque_index.append(i)

        # Append maximum of current window
        if i >= k - 1:
            result.append(nums[deque_index[0]])

    return result

# Example usage:
nums = [1, 3, -1, -3, 5, 3, 6, 7]
k = 3
print(max_sliding_window(nums, k))  # Output: [3, 3, 5, 5, 6, 7]

```

## Queue Patterns

(Reverse the order of elements in a queue using auxiliary data structures)

```python

from collections import deque

def reverse_queue(queue):
    stack = []
    while queue:
        stack.append(queue.popleft())
    while stack:
        queue.append(stack.pop())
    return queue

# Example usage:
queue = deque([1, 2, 3, 4, 5])
print(reverse_queue(queue))  # Output: deque([5, 4, 3, 2, 1])

```

(Maintain a sub-list of elements from a larger list and move it one element at a time)

```python

from collections import deque

def sliding_window_max(nums, k):
    if not nums:
        return []
    
    deque_index = deque()
    result = []

    for i in range(len(nums)):
        # Remove elements not within the window
        if deque_index and deque_index[0] < i - k + 1:
            deque_index.popleft()
        
        # Maintain monotonic decreasing order
        while deque_index and nums[deque_index[-1]] < nums[i]:
            deque_index.pop()

        deque_index.append(i)

        # Append maximum of current window
        if i >= k - 1:
            result.append(nums[deque_index[0]])

    return result

# Example usage:
nums = [1, 3, -1, -3, 5, 3, 6, 7]
k = 3
print(sliding_window_max(nums, k))  # Output: [3, 3, 5, 5, 6, 7]

```

(Rearrange a queue of people based on their height and the number of people in front of them)

```python

def reconstruct_queue(people):
    people.sort(key=lambda x: (-x[0], x[1]))  # Sort by height descending, and by k ascending
    result = []
    for person in people:
        result.insert(person[1], person)
    return result

# Example usage:
people = [[7,0], [4,4], [7,1], [5,0], [6,1], [5,2]]
print(reconstruct_queue(people))  # Output: [[5,0], [7,0], [5,2], [6,1], [4,4], [7,1]]

```

(Implement a queue with a maximum capacity, discarding oldest elements when exceeding capacity)

```python

from collections import deque

class BoundedQueue:
    def __init__(self, capacity):
        self.capacity = capacity
        self.queue = deque()
    
    def enqueue(self, value):
        if len(self.queue) >= self.capacity:
            self.queue.popleft()
        self.queue.append(value)
    
    def dequeue(self):
        if self.queue:
            return self.queue.popleft()
        return None
    
    def size(self):
        return len(self.queue)
    
    def is_empty(self):
        return len(self.queue) == 0
    
    def is_full(self):
        return len(self.queue) == self.capacity

# Example usage:
queue = BoundedQueue(5)
queue.enqueue(1)
queue.enqueue(2)
queue.enqueue(3)
queue.enqueue(4)
queue.enqueue(5)
queue.enqueue(6)  # 1 will be discarded
print(queue.dequeue())  # Output: 2

```

Here are some patters to after for.

**Implement Queue using Stacks**

Problem: Implement a queue using two stacks.

Approach:

- Use two stacks: one for enqueue (input_stack) and one for dequeue (output_stack).
- Push new elements into input_stack. For dequeuing, transfer elements from input_stack to output_stack if output_stack is empty, and then pop from output_stack.

```py
class MyQueue:
    def __init__(self):
        self.input_stack = []
        self.output_stack = []

    def push(self, x):
        self.input_stack.append(x)

    def pop(self):
        if not self.output_stack:
            while self.input_stack:
                self.output_stack.append(self.input_stack.pop())
        return self.output_stack.pop()

    def peek(self):
        if not self.output_stack:
            while self.input_stack:
                self.output_stack.append(self.input_stack.pop())
        return self.output_stack[-1]

    def empty(self):
        return not self.input_stack and not self.output_stack

```

**Implement Circular Queue**

Problem: Implement a fixed-size circular queue.

Approach:

- Use a list of fixed size with pointers for `front` and `rear`. Handle wraparound using modulo arithmetic.

```py
class CircularQueue:
    def __init__(self, k):
        self.queue = [None] * k
        self.size = k
        self.front = self.rear = -1

    def enQueue(self, value):
        if (self.rear + 1) % self.size == self.front:
            return False  # Queue is full
        if self.front == -1:
            self.front = 0
        self.rear = (self.rear + 1) % self.size
        self.queue[self.rear] = value
        return True

    def deQueue(self):
        if self.front == -1:
            return False  # Queue is empty
        if self.front == self.rear:
            self.front = self.rear = -1
        else:
            self.front = (self.front + 1) % self.size
        return True

    def Front(self):
        return -1 if self.front == -1 else self.queue[self.front]

    def Rear(self):
        return -1 if self.rear == -1 else self.queue[self.rear]

    def isEmpty(self):
        return self.front == -1

    def isFull(self):
        return (self.rear + 1) % self.size == self.front

```

**Sliding Window Maximum**

Problem: Given an array and a window size k, find the maximum value in each sliding window.

Approach:

- Use a deque to store indices of elements in the window. Maintain decreasing order of values in the deque. The front of the deque is the maximum for the current window.

```py
from collections import deque

def maxSlidingWindow(nums, k):
    dq = deque()
    result = []
    for i in range(len(nums)):
        if dq and dq[0] < i - k + 1:
            dq.popleft()
        while dq and nums[dq[-1]] < nums[i]:
            dq.pop()
        dq.append(i)
        if i >= k - 1:
            result.append(nums[dq[0]])
    return result

```

**Design Hit Counter**

Problem: Count the number of hits received in the past 5 minutes.

Approach:

- Use a queue to store timestamps of hits. Remove timestamps olde than 300 seconds during each operation.

```py
from collections import deque

class HitCounter:
    def __init__(self):
        self.hits = deque()

    def hit(self, timestamp):
        self.hits.append(timestamp)

    def getHits(self, timestamp):
        while self.hits and self.hits[0] <= timestamp - 300:
            self.hits.popleft()
        return len(self.hits)

```

**Rotting Oranges**

Problem: Given a grid of oranges, calculate the minimum time to rot all oranges or return -1 if impossible.

Approach:

- Use BFS starting from all initially rotten oranges. Track the time taken to rot fresh oranges.

```py
from collections import deque

def orangesRotting(grid):
    rows, cols = len(grid), len(grid[0])
    queue = deque()
    fresh = 0
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == 2:
                queue.append((r, c, 0))
            elif grid[r][c] == 1:
                fresh += 1

    time = 0
    directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
    while queue:
        r, c, time = queue.popleft()
        for dr, dc in directions:
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols and grid[nr][nc] == 1:
                grid[nr][nc] = 2
                fresh -= 1
                queue.append((nr, nc, time + 1))

    return time if fresh == 0 else -1

```

**Shortest Path in Binary Matrix**

Problem: Find the shortest path from top-left to the bottom-right in a binary grid.

Approach:

- Use BFS to explore all 8 directions from each cell. Track the path length as levels in BFS.

```py
from collections import deque

def shortestPathBinaryMatrix(grid):
    if grid[0][0] or grid[-1][-1]:
        return -1
    n = len(grid)
    directions = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
    queue = deque([(0, 0, 1)])
    visited = set()
    visited.add((0, 0))

    while queue:
        r, c, path_len = queue.popleft()
        if (r, c) == (n - 1, n - 1):
            return path_len
        for dr, dc in directions:
            nr, nc = r + dr, c + dc
            if 0 <= nr < n and 0 <= nc < n and grid[nr][nc] == 0 and (nr, nc) not in visited:
                visited.add((nr, nc))
                queue.append((nr, nc, path_len + 1))

    return -1

```

**Course Schedule**

Problem: Determine if all courses can be finished given prerequisites.

Approach:

- Use Kahn's Algorithm for topological sorting with an in-degree array and a queue.

```py
from collections import deque, defaultdict

def canFinish(numCourses, prerequisites):
    graph = defaultdict(list)
    in_degree = [0] * numCourses

    for course, prereq in prerequisites:
        graph[prereq].append(course)
        in_degree[course] += 1

    queue = deque([i for i in range(numCourses) if in_degree[i] == 0])
    taken = 0

    while queue:
        course = queue.popleft()
        taken += 1
        for neighbor in graph[course]:
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                queue.append(neighbor)

    return taken == numCourses

```

**Shortest Path in a Weighted Graph (Dijkstra using Priority Queue)**

Problem: Find the shortest path from a source to all nodes in a weighted graph.

Approach:

- Use a priority queue (min-heap) to implement Dijkstra's algorithm.

```py
import heapq

def dijkstra(graph, start):
    pq = [(0, start)]
    distances = {node: float('inf') for node in graph}
    distances[start] = 0

    while pq:
        current_dist, current_node = heapq.heappop(pq)
        if current_dist > distances[current_node]:
            continue
        for neighbor, weight in graph[current_node]:
            distance = current_dist + weight
            if distance < distances[neighbor]:
                distances[neighbor] = distance
                heapq.heappush(pq, (distance, neighbor))
    return distances

```
