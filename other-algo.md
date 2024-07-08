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

```python

import heapq

def prim(graph, start):
    mst = []
    visited = set([start])
    edges = [(cost, start, to) for to, cost in graph[start].items()]

    heapq.heapify(edges)

    while edges:
        cost, frm, to = heapq.heappop(edges)
        if to not in visited:
            visited.add(to)
            mst.append((frm, to, cost))

            for next_to not in visited:
                if next_to not in visited:
                    heapq.heappush(edges, (next_cost, to, next_to))

    return mst

graph = {
    'A': {'B': 1, 'C': 4},
    'B': {'A': 1, 'C': 2, 'D': 5},
    'C': {'A': 4, 'B': 2, 'D': 1},
    'D': {'B': 5, 'C': 1}
}

print(prim(graph, 'A'))
```

# Kruskal

```python

class UnionFind:

    def __init__(self, nodes) -> None:
        self.parent = {node: node for node in nodes}
        self.rank = {node: 0 for node in nodes}
    
    def find(self, node):
        if self.parent[node] != node:
            self.parent[node] = self.find(self.parent[node])
        return self.parent[node]

    def union(self, node1, node2):
        root1 = self.find(node1)
        root2 = self.find(node2)

        if root1 != root2:
            if self.rank[root1] < self.rank[root2]:
                self.parent[root1] = root2
            elif self.rank[root1] > self.rank[root2]:
                self.parent[root2] = root1
            else:
                self.parent[root2] = root1
                self.rank[root1] += 1
    
def kruskal(graph):
    edges = [(cost, frm, to)
             for frm in graph for to, cost in graph[frm].items()]
            
    edges.sort()
    mst = []

    uf = UnionFind(graph)
    for cost, frm, to in edges:
        if uf.find(frm) != uf.find(to):
            uf.union(frm, to)
            mst.append((frm, to, cost))
    
    return mst

graph = {
    'A': {'B': 1, 'C': 4},
    'B': {'A': 1, 'C': 2, 'D': 5},
    'C': {'A': 4, 'B': 2, 'D': 1},
    'D': {'B': 5, 'C': 1}
}

print(kruskal(graph))

```

# Dijkstra's Algorithm

```python

import heapq

def dijkstra(graph, start):
    distances = {vertex: float('inf') for vertex in graph}
    distances[start] = 0

    pq = [(0, start)]
    while pq:
        current_distance, current_vertex = heapq.heappop(pq)

        if current_distance > distances[current_vertex]:
            continue
        
        for nei, wei in graph[current_vertex].items():
            distance = current_distance + wei
            if distance < distances[nei]:
                distances[nei] = distance
                heapq.heappush(pq, (distance, nei))
    
    return distance

graph = {
    'A': {'B': 1, 'C': 4},
    'B': {'A': 1, 'C': 2, 'D': 5},
    'C': {'A': 4, 'B': 2, 'D': 1},
    'D': {'B': 5, 'C': 1}
}

print(dijkstra(graph, 'A'))
```

# Bellman Ford Algorithm

```python
import heapq

def bellman(graph, start):
    distances = {vertex: float('inf') for vertex in graph}
    distances[start] = 0

    for _ in range(len(graph) - 1):
        for vertex in graph:
            for nei, wei in graph[vertex].items():
                if distances[vertex] + wei < distances[nei]:
                    distances[nei] = distances[vertex]+wei
    
    # Check for negative cycles
    for vertex in graph:
        for nei, wei in graph[vertex].items():
            if distances[vertex] + wei < distances[nei]:
                raise ValueError("Negative Cycle")
    
    return distances

graph = {
    'A': {'B': 1, 'C': 4},
    'B': {'A': 1, 'C': 2, 'D': 5},
    'C': {'A': 4, 'B': 2, 'D': 1},
    'D': {'B': 5, 'C': 1}
}

print(bellman(graph, 'A'))

```

# Floyd Warshall Algorithm

```python

import heapq

def floyd(graph):
    distances = {u: {v: float("inf") for v in graph} for u in graph}
    
    for vertex in graph:
        distances[vertex][vertex] = 0
        for nei, wei in graph[vertex].items():
            distances[vertex][nei] = wei
    
    for k in graph:
        for i in graph:
            for j in graph:
                distances[i][j] = min(
                    distances[i][j], distances[i][k] + distances[k][j]
                )
    
    return distances

graph = {
    'A': {'B': 1, 'C': 4},
    'B': {'A': 1, 'C': 2, 'D': 5},
    'C': {'A': 4, 'B': 2, 'D': 1},
    'D': {'B': 5, 'C': 1}
}

print(floyd(graph))
}

```

# Kosaraju's Algorithm

```python
from collections import defaultdict

def dfs(graph, node, visited, stack):
    visited[node] = True
    for neighbor in graph[node]:
        if not visited[neighbor]:
            dfs(graph, neighbor, visited, stack)
    stack.append(node)

def dfs_transposed(transposed_graph, node, visited, scc):
    visited[node] = True
    scc.append(node)
    for neighbor in transposed_graph[node]:
        if not visited[neighbor]:
            dfs_transposed(transposed_graph, neighbor, visited, scc)

def kosaraju(graph):
    stack = []
    visited = {node: False for node in graph}

    for node in graph:
        if not visited[node]:
            dfs(graph, node, visited, stack)
    
    transposed_graph = defaultdict(list)
    for node in graph:
        for neighbor in graph[node]:
            transposed_graph[neighbor].append(node)

    visited = {node: False for node in graph}
    sccs = []

    while stack:
        node = stack.pop()
        if not visited[node]:
            scc = []
            dfs_transposed(transposed_graph, node, visited, scc)
            sccs.append(scc)

    return sccs

graph = {
    0: [1],
    1: [2],
    2: [0, 3],
    3: [4],
    4: [5],
    5: [3, 6],
    6: []
}

print(kosaraju(graph))

```

# Topological Sort

```python
def topoSort(self, V, adj):
    
    degree = [0]*V
    for i in range(V):
        for u in adj[i]:
            degree[u] += 1
    from collections import deque
    q = deque()
    
    for i in range(V):
        if degree[i] == 0:
            q.append(i)
    
    res = []
    while q:
        for _ in range(len(q)):
            node = q.popleft()
            res.append(node)
            
            for u in adj[node]:
                degree[u] -= 1
                if degree[u] == 0:
                    q.append(u)
                    
    return res
```

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

```python

def LIS(arr):
    N = len(arr)
    dp = [1]*N

    for i in range(1, N):
        for j in range(i):
            if arr[i] > arr[j]:
                dp[i] = max(dp[i], 1 + dp[j])
    
    return max(dp)

```

# Longest Common Subsequence

```python

#    "" B D C A B
# ""  0 0 0 0 0 0
#  A  0 0 0 0 1 1
#  B  0 1 1 1 1 2
#  C  0 1 1 2 2 2 
#  B  0 1 2 2 2 3
#  D  0 1 2 2 2 3
#  A  0 1 2 2 3 3 
#  B  0 1 2 2 3 4


def LCS(A, B):
    m = len(A)
    n = len(B)

    dp = [[0]*(n+1) for _ in range(m+1)]
    for i in range(1, m+1):
        for j in range(1, n+1):
            if A[i-1] = B[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])
    
    return dp[m][n]

```

# Ancestor in Graph

```python
def getAncestors(self, n: int, edges: List[List[int]]) -> List[List[int]]:
    graph = defaultdict(list)
    
    for u, v in edges:
        graph[v].append(u)
    
    def dfs(node, ancestors):
        for parent in graph[node]:
            if parent not in ancestors:
                ancestors.add(parent)
                dfs(parent, ancestors)
    
    result = []
    for i in range(n):
        ancestors = set()
        dfs(i, ancestors)
        result.append(sorted(ancestors))
    
    return result
```

# Ancestor in Tree

```python
def Ancestors(self, root, target):
    def dfs(node, path):
        if not node:
            return None
        if node.data == target:
            return path
        path.append(node.data)
        if dfs(node.left, path) or dfs(node.right, path):
            return path
        path.pop()
        return None
    
    if root.data == target:
        return []
        
    output = dfs(root, [])
    if output:
        return output[::-1]
    else:
        return []
```

# Lowest Common Ancestor

```python
def lowest_common_ancestor(root, p, q):

    def LCA(node):
        if not node:
            return Node
        if node in [p, q]:
            return node
        
        left = LCA(node.left)
        right = LCA(node.right)

        if left and right:
            return node
        else:
            return left or right

return LCA(root)

```
