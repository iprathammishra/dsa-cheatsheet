# Kadane's Algorithm

```python

def kadane(arr):
    max_current = max_global = arr[0]
    
    for num in arr[1:]:
        max_current = max(num, max_current + num)
        if max_current > max_global:
            max_global = max_current
    
    return max_global

```

# Fast and Slow Pointer Algorithm- (Tortoise and Hare's Algorithm)

```python

class ListNode:
    def __init__(self, value=0, next=None):
        self.value = value
        self.next = next
    
def detect_cycle(head):
    slow = fast = head

    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next
        if slow == fast:
            break
        
    if not fast or not fast.next:
        return None
    
    slow = head
    while slow != fast:
        slow = slow.next
        fast = fast.next
    
    return slow

```

# Trie- (Prefix Tree)

```python

class TrieNode:
    def __init__(self) -> None:
        self.children = {}
        self.is_endof_word = False

class Trie:
    def __init__(self) -> None:
        self.root = TrieNode()

    def insert(self, word) -> None:
        node = self.root
        for char in word:
            node = node.children.setdefault(char, TrieNode())
        node.is_endof_word = True
    
    def search(self, word) -> bool:
        node = self.root
        for char in word:
            if char not in node.children:
                return False
            node = node.children[char]
        return node.is_endof_word

    def starts_with(self, prefix) -> bool:
        node = self.root
        for char in prefix:
            if char not in node.children:
                return False
            node = node.children[char]
        return True

```

# Union Find

```python

class UnionFind:
    def __init__(self, nodes) -> None:
        self.parent = {node:node for node in nodes}
        self.rank = {node:0 for node in nodes}

    def find(self, node):
        if self.parent[node] != node:
            self.parent[node] = self.find(self.parent[node])
        return self.parent[node]
    
    def union(self, node1, node2):
        root1 = self.find(node1)
        root2 = self.find(node2)

        if self.rank[root1] < self.rank[root2]:
            self.parent[root1] = root2  
        elif self.rank[root1] > self.rank[root2]:
            self.parent[roo2] = root1
        else:
            self.parent[root1] = root2
            self.rank[root2] += 1

```

# Prims

# Kruskal

# Dijkstra's Algorithm

# Bellman Ford Algorithm

# Floyd Warshall Algorithm

# Kosaraju's Algorithm

# Topological Sort

# Sieve of Eratosthenes

```python

def sieve(n):
    is_prime = [True]*(n+1)
    is_prime[0] = is_prime[1] = False

    for i in range(2, int(n**0.5)+1):
        if is_prime(i):
            for j in range(i*i, n+1, i):
                is_prime[j] = False
    
    primes = [i for i in range(n+1) if is_prime[i]]
    return primes

```

# Longest Increasing Subsequence

# Longest Common Subsequence

# Ancestor in Graph

# Ancestor in Tree

# Lowest Common Ancestor

