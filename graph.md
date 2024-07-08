# Graph

## Graph Traversal Patterns

(Depth-First Search- DFS)

```python
def dfs(graph, start):
    visited = set()
    stack = [start]
    
    while stack:
        node = stack.pop()
        if node not in visited:
            visited.add(node)
            for neighbor in graph[node]:
                stack.append(neighbor)
    return visited

# Example usage:
graph = {
    'A': ['B', 'C'],
    'B': ['D', 'E'],
    'C': ['F'],
    'D': [],
    'E': ['F'],
    'F': []
}
print(dfs(graph, 'A'))

```

(Breadth-First Search- BFS)

```python
from collections import deque

def bfs(graph, start):
    visited = set()
    queue = deque([start])
    
    while queue:
        node = queue.popleft()
        if node not in visited:
            visited.add(node)
            for neighbor in graph[node]:
                queue.append(neighbor)
    return visited

# Example usage:
graph = {
    'A': ['B', 'C'],
    'B': ['D', 'E'],
    'C': ['F'],
    'D': [],
    'E': ['F'],
    'F': []
}
print(bfs(graph, 'A'))

```

## Topological Sort

```python
from collections import defaultdict

def topological_sort(graph):
    def dfs(node):
        if node in visited:
            return
        visited.add(node)
        for neighbor in graph[node]:
            dfs(neighbor)
        stack.append(node)
    
    visited = set()
    stack = []
    
    for node in graph:
        if node not in visited:
            dfs(node)
    
    return stack[::-1]

# Example usage:
graph = {
    'A': ['B', 'C'],
    'B': ['D', 'E'],
    'C': ['F'],
    'D': [],
    'E': ['F'],
    'F': []
}
print(topological_sort(graph))

```

## Cycle Detection

(Undirected Graph)

```python
def has_cycle_undirected(graph):
    def dfs(node, parent):
        visited.add(node)
        for neighbor in graph[node]:
            if neighbor not in visited:
                if dfs(neighbor, node):
                    return True
            elif neighbor != parent:
                return True
        return False
    
    visited = set()
    for node in graph:
        if node not in visited:
            if dfs(node, None):
                return True
    return False

# Example usage:
graph = {
    0: [1, 2],
    1: [0, 2],
    2: [0, 1, 3],
    3: [2]
}
print(has_cycle_undirected(graph))

```

(Directed Graph)

```python

def has_cycle_directed(graph):
    def dfs(node):
        if node in visited:
            return False
        if node in rec_stack:
            return True
        rec_stack.add(node)
        for neighbor in graph[node]:
            if dfs(neighbor):
                return True
        rec_stack.remove(node)
        visited.add(node)
        return False
    
    visited = set()
    rec_stack = set()
    for node in graph:
        if dfs(node):
            return True
    return False

# Example usage:
graph = {
    'A': ['B', 'C'],
    'B': ['D', 'E'],
    'C': ['F'],
    'D': [],
    'E': ['F'],
    'F': []
}
print(has_cycle_directed(graph))

```

## Dijkstra's Algorithm Patterns

(Single Source Shortest Path)

```python
import heapq

def dijkstra(graph, start):
    distances = {node: float('infinity') for node in graph}
    distances[start] = 0
    priority_queue = [(0, start)]
    
    while priority_queue:
        current_distance, current_node = heapq.heappop(priority_queue)
        
        if current_distance > distances[current_node]:
            continue
        
        for neighbor, weight in graph[current_node]:
            distance = current_distance + weight
            
            if distance < distances[neighbor]:
                distances[neighbor] = distance
                heapq.heappush(priority_queue, (distance, neighbor))
    
    return distances

# Example usage:
graph = {
    'A': [('B', 1), ('C', 4)],
    'B': [('C', 2), ('D', 5)],
    'C': [('D', 1)],
    'D': []
}
print(dijkstra(graph, 'A'))
```

(Shortest Path in Grid Based Graphs)

```python
import heapq

def dijkstra_grid(grid):
    rows, cols = len(grid), len(grid[0])
    directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
    distances = [[float('inf')] * cols for _ in range(rows)]
    distances[0][0] = grid[0][0]
    priority_queue = [(grid[0][0], 0, 0)]
    
    while priority_queue:
        current_distance, row, col = heapq.heappop(priority_queue)
        
        if (row, col) == (rows - 1, cols - 1):
            return current_distance
        
        for dr, dc in directions:
            r, c = row + dr, col + dc
            if 0 <= r < rows and 0 <= c < cols:
                distance = current_distance + grid[r][c]
                if distance < distances[r][c]:
                    distances[r][c] = distance
                    heapq.heappush(priority_queue, (distance, r, c))
    
    return distances[rows - 1][cols - 1]

# Example usage:
grid = [
    [1, 3, 1],
    [1, 5, 1],
    [4, 2, 1]
]
print(dijkstra_grid(grid))
```

(Weighted Graph with Constraints)

```python

import heapq

def dijkstra_with_constraints(graph, start, k):
    distances = {node: float('infinity') for node in graph}
    distances[start] = 0
    priority_queue = [(0, start, 0)]
    
    while priority_queue:
        current_distance, current_node, stops = heapq.heappop(priority_queue)
        
        if stops > k:
            continue
        
        for neighbor, weight in graph[current_node]:
            distance = current_distance + weight
            
            if distance < distances[neighbor]:
                distances[neighbor] = distance
                heapq.heappush(priority_queue, (distance, neighbor, stops + 1))
    
    return distances

# Example usage:
graph = {
    'A': [('B', 1), ('C', 4)],
    'B': [('C', 2), ('D', 5)],
    'C': [('D', 1)],
    'D': []
}
print(dijkstra_with_constraints(graph, 'A', 2))

```

(Dijkstra's Algorithm with Multiple Sources)

```python

import heapq

def multi_source_dijkstra(graph, sources):
    distances = {node: float('infinity') for node in graph}
    priority_queue = []
    
    for source in sources:
        distances[source] = 0
        heapq.heappush(priority_queue, (0, source))
    
    while priority_queue:
        current_distance, current_node = heapq.heappop(priority_queue)
        
        if current_distance > distances[current_node]:
            continue
        
        for neighbor, weight in graph[current_node]:
            distance = current_distance + weight
            
            if distance < distances[neighbor]:
                distances[neighbor] = distance
                heapq.heappush(priority_queue, (distance, neighbor))
    
    return distances

# Example usage:
graph = {
    'A': [('B', 1), ('C', 4)],
    'B': [('C', 2), ('D', 5)],
    'C': [('D', 1)],
    'D': []
}
sources = ['A', 'B']
print(multi_source_dijkstra(graph, sources))

```

## Additional Patterns

(Minimum Spanning Tree- Kruskal's Algorithm)

```python

class UnionFind:
    def __init__(self, size):
        self.root = list(range(size))
        self.rank = [1] * size

    def find(self, x):
        if self.root[x] != x:
            self.root[x] = self.find(self.root[x])
        return self.root[x]

    def union(self, x, y):
        rootX = self.find(x)
        rootY = self.find(y)
        if rootX != rootY:
            if self.rank[rootX] > self.rank[rootY]:
                self.root[rootY] = rootX
            elif self.rank[rootX] < self.rank[rootY]:
                self.root[rootX] = rootY
            else:
                self.root[rootY] = rootX
                self.rank[rootX] += 1

def kruskal(n, edges):
    uf = UnionFind(n)
    mst = []
    edges.sort(key=lambda x: x[2])
    
    for u, v, weight in edges:
        if uf.find(u) != uf.find(v):
            uf.union(u, v)
            mst.append((u, v, weight))
    
    return mst

# Example usage:
edges = [
    (0, 1, 10),
    (0, 2, 6),
    (0, 3, 5),
    (1, 3, 15),
    (2, 3, 4)
]
n = 4  # Number of nodes
print(kruskal(n, edges))

```

(Minimum Spanning Tree- Prim's Algorithm)

```python
import heapq

def prim(graph, start):
    mst = []
    visited = set()
    min_heap = [(0, start, None)]  # (cost, node, parent)
    
    while min_heap:
        cost, node, parent = heapq.heappop(min_heap)
        if node not in visited:
            visited.add(node)
            if parent is not None:
                mst.append((parent, node, cost))
            for neighbor, weight in graph[node]:
                if neighbor not in visited:
                    heapq.heappush(min_heap, (weight, neighbor, node))
    
    return mst

# Example usage:
graph = {
    'A': [('B', 1), ('C', 4)],
    'B': [('A', 1), ('C', 2), ('D', 5)],
    'C': [('A', 4), ('B', 2), ('D', 1)],
    'D': [('B', 5), ('C', 1)]
}
print(prim(graph, 'A'))

```

(Bellman-Ford Algorithm- Single Source Shortest Path with Negative Weights)

```python

def bellman_ford(graph, start):
    distances = {node: float('inf') for node in graph}
    distances[start] = 0
    
    for _ in range(len(graph) - 1):
        for node in graph:
            for neighbor, weight in graph[node]:
                if distances[node] + weight < distances[neighbor]:
                    distances[neighbor] = distances[node] + weight
    
    for node in graph:
        for neighbor, weight in graph[node]:
            if distances[node] + weight < distances[neighbor]:
                return "Negative weight cycle detected"
    
    return distances

# Example usage:
graph = {
    'A': [('B', -1), ('C', 4)],
    'B': [('C', 3), ('D', 2), ('E', 2)],
    'C': [],
    'D': [('B', 1), ('C', 5)],
    'E': [('D', -3)]
}
print(bellman_ford(graph, 'A'))

```

(Floyd-Warshall Algorithm- All Pairs Shortest Path)

```python
def floyd_warshall(graph):
    nodes = list(graph.keys())
    distances = {node: {neighbor: float('inf') for neighbor in nodes} for node in nodes}
    
    for node in nodes:
        distances[node][node] = 0
        for neighbor, weight in graph[node]:
            distances[node][neighbor] = weight
    
    for k in nodes:
        for i in nodes:
            for j in nodes:
                if distances[i][j] > distances[i][k] + distances[k][j]:
                    distances[i][j] = distances[i][k] + distances[k][j]
    
    return distances

# Example usage:
graph = {
    'A': [('B', 3), ('C', 8), ('E', -4)],
    'B': [('D', 1), ('E', 7)],
    'C': [('B', 4)],
    'D': [('A', 2), ('C', -5)],
    'E': [('D', 6)]
}
print(floyd_warshall(graph))

```
