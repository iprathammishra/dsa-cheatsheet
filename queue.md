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
