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

Here are 10 graph-related question ranging from easy to hard.

**Depth First Search (DFS)**

Problem: Problem a depth-first traversal of a graph starting from a given node.

Approach:

- Use a stack or recursion to traverse the graph.
- Mark nodes as visited to avoid cycles.

```py
def dfs(graph, start):
    visited = set()
    result = []

    def traverse(node):
        if node in visited:
            return
        visited.add(node)
        result.append(node)
        for neighbor in graph[node]:
            traverse(neighbor)

    traverse(start)
    return result

graph = {
    0: [1, 2],
    1: [0, 3, 4],
    2: [0],
    3: [1],
    4: [1, 5],
    5: [4]
}
print(dfs(graph, 0))  # Output: [0, 1, 3, 4, 5, 2]

```

**Breadth First Search (BFS)**

Problem: Perform a breadth-first traversal of a graph starting from a given node.

Approach:

- Use a queue to explore nodes level by level.
- Mark nodes as visited to avoid cycles.

```py
from collections import deque

def bfs(graph, start):
    visited = set()
    queue = deque([start])
    result = []

    while queue:
        node = queue.popleft()
        if node not in visited:
            visited.add(node)
            result.append(node)
            queue.extend(graph[node])

    return result

graph = {
    0: [1, 2],
    1: [0, 3, 4],
    2: [0],
    3: [1],
    4: [1, 5],
    5: [4]
}
print(bfs(graph, 0))  # Output: [0, 1, 2, 3, 4, 5]

```

**Detect Cycle in an Undirected Graph**

Problem: Detech if there is a cycle in an undirected graph.

Approach:

- Use DFS to traverse the graph.
- Track visited nodes and ensure that no back edge leads to a previously visited node unless it's the parent.

```py
def has_cycle(graph):
    visited = set()

    def dfs(node, parent):
        visited.add(node)
        for neighbor in graph[node]:
            if neighbor not in visited:
                if dfs(neighbor, node):
                    return True
            elif neighbor != parent:
                return True
        return False

    for node in graph:
        if node not in visited:
            if dfs(node, -1):
                return True
    return False

graph = {0: [1, 2], 1: [0, 2], 2: [0, 1], 3: [4], 4: [3]}
print(has_cycle(graph))  # Output: True

```

**Dijkstra's Algorithm**

Problem: Find the shortest path from a source node to all other nodes in a weighted graph.

Approach:

- Use a priority queue to explore nodes with the smallest distance.
- Update the shortest distance to each node as you traverse.

```py
import heapq

def dijkstra(graph, start):
    min_heap = [(0, start)]
    distances = {node: float('inf') for node in graph}
    distances[start] = 0

    while min_heap:
        current_dist, node = heapq.heappop(min_heap)
        if current_dist > distances[node]:
            continue
        for neighbor, weight in graph[node]:
            distance = current_dist + weight
            if distance < distances[neighbor]:
                distances[neighbor] = distance
                heapq.heappush(min_heap, (distance, neighbor))

    return distances

graph = {
    0: [(1, 4), (2, 1)],
    1: [(3, 1)],
    2: [(1, 2), (3, 5)],
    3: []
}
print(dijkstra(graph, 0))  # Output: {0: 0, 1: 3, 2: 1, 3: 4}

```

**Detect Cycle in a Directed Graph**

Problem: Detect if there is a cycle in a directed graph.

Approach:

- Use DFS with a recursion stack to detect back edges.
- Back edges indicate cycles.

```py
def has_cycle_directed(graph):
    visited = set()
    recursion_stack = set()

    def dfs(node):
        visited.add(node)
        recursion_stack.add(node)
        for neighbor in graph[node]:
            if neighbor not in visited:
                if dfs(neighbor):
                    return True
            elif neighbor in recursion_stack:
                return True
        recursion_stack.remove(node)
        return False

    for node in graph:
        if node not in visited:
            if dfs(node):
                return True
    return False

graph = {0: [1], 1: [2], 2: [0], 3: [4], 4: []}
print(has_cycle_directed(graph))  # Output: True

```

**Topological Sort**

Problem: Find a valid topological order for a directed acyclic graph (DAG).

Approach:

- Use DFS to traverse the graph and add nodes to the result in reverse post-order.
- Ensure the graph is acyclic.

```py
def topological_sort(graph):
    visited = set()
    result = []

    def dfs(node):
        visited.add(node)
        for neighbor in graph[node]:
            if neighbor not in visited:
                dfs(neighbor)
        result.append(node)

    for node in graph:
        if node not in visited:
            dfs(node)

    return result[::-1]

graph = {0: [1, 2], 1: [3], 2: [3], 3: []}
print(topological_sort(graph))  # Output: [0, 2, 1, 3]

```

**Connected Components**

Problem: Find all connected components in an undirected graph.

Approach:

- Use DFS or BFS to explore all nodes in each component.
- Track visited nodes.

```py
def connected_components(graph):
    visited = set()
    components = []

    def dfs(node, component):
        visited.add(node)
        component.append(node)
        for neighbor in graph[node]:
            if neighbor not in visited:
                dfs(neighbor, component)

    for node in graph:
        if node not in visited:
            component = []
            dfs(node, component)
            components.append(component)

    return components

graph = {0: [1], 1: [0], 2: [3], 3: [2], 4: []}
print(connected_components(graph))  # Output: [[0, 1], [2, 3], [4]]

```

**Minimum Spanning Tree (Prim's Algorithm)

Problem: Find the minimum spanning tree of a weighted graph.

Approach:

- Use a priority queue to select the minimum weight edge at each step.
- Add the selected edge to the MST.

```py
import heapq

def prims_mst(graph):
    start = list(graph.keys())[0]
    min_heap = [(0, start)]
    visited = set()
    mst_cost = 0

    while min_heap:
        cost, node = heapq.heappop(min_heap)
        if node in visited:
            continue
        visited.add(node)
        mst_cost += cost
        for neighbor, weight in graph[node]:
            if neighbor not in visited:
                heapq.heappush(min_heap, (weight, neighbor))

    return mst_cost

graph = {
    0: [(1, 4), (2, 1)],
    1: [(0, 4), (2, 2), (3, 5)],
    2: [(0, 1), (1, 2), (3, 8)],
    3: [(1, 5), (2, 8)]
}
print(prims_mst(graph))  # Output: 7

```

**Shortest Path in a Weighted Grid (Dijkstra Variation)**

Problem: Find the shortest path in a weighted grid from top-left to bottom-right.

Approach:

- Use a priority queue to explore cells with the smallest distance.
- Update the shortest path to each cell.

```py
import heapq

def shortest_path_grid(grid):
    rows, cols = len(grid), len(grid[0])
    directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
    min_heap = [(grid[0][0], 0, 0)]
    distances = {(0, 0): grid[0][0]}

    while min_heap:
        cost, r, c = heapq.heappop(min_heap)
        if (r, c) == (rows - 1, cols - 1):
            return cost

        for dr, dc in directions:
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols:
                new_cost = cost + grid[nr][nc]
                if (nr, nc) not in distances or new_cost < distances[(nr, nc)]:
                    distances[(nr, nc)] = new_cost
                    heapq.heappush(min_heap, (new_cost, nr, nc))

grid = [
    [1, 3, 1],
    [1, 5, 1],
    [4, 2, 1]
]
print(shortest_path_grid(grid))  # Output: 7

```

**Strongly Connected Components (Tarjan's Algorithm)**

Problem: Find all strongly components (SCCs) in a directed graph.

Approach:

- Use Tarjan's Algorithm, which employs DFS to track discovery times and low-link values of nodes.
- Nodes belonging to an SCC will share the same low-link value.
- Use a stack to manage nodes during DFS and extract SCCs when an SCC root is identified.

```py
def tarjans_scc(graph):
    def dfs(node):
        nonlocal time
        discovery[node] = low[node] = time
        time += 1
        stack.append(node)
        in_stack.add(node)

        for neighbor in graph[node]:
            if discovery[neighbor] == -1:
                dfs(neighbor)
                low[node] = min(low[node], low[neighbor])
            elif neighbor in in_stack:
                low[node] = min(low[node], discovery[neighbor])

        if low[node] == discovery[node]:  # Found an SCC root
            scc = []
            while stack:
                top = stack.pop()
                in_stack.remove(top)
                scc.append(top)
                if top == node:
                    break
            sccs.append(scc)

    n = len(graph)
    discovery = [-1] * n
    low = [-1] * n
    time = 0
    stack = []
    in_stack = set()
    sccs = []

    for node in range(n):
        if discovery[node] == -1:
            dfs(node)

    return sccs

graph = {
    0: [1],
    1: [2],
    2: [0, 3],
    3: [4],
    4: [5],
    5: [3]
}
print(tarjans_scc(graph))  # Output: [[3, 5, 4], [0, 2, 1]]

```

**Shortest Path with Negative Weights (Bellman-Ford Algorithm)**

Problem: Find the shortest path from a source to all nodes in a graph with negative weight edges.

Approach:

- Use the Bellman-Ford Algorithm, which iterates V-1 times (where V is the number of vertices) to relax all edges.
- Check for negative weight cycles in the graph by performing one additional iteration.

```py
def bellman_ford(graph, vertices, start):
    distances = [float('inf')] * vertices
    distances[start] = 0

    for _ in range(vertices - 1):
        for u, v, weight in graph:
            if distances[u] != float('inf') and distances[u] + weight < distances[v]:
                distances[v] = distances[u] + weight

    # Check for negative weight cycles
    for u, v, weight in graph:
        if distances[u] != float('inf') and distances[u] + weight < distances[v]:
            return "Graph contains a negative weight cycle"

    return distances

graph = [
    (0, 1, 1),
    (1, 2, 3),
    (2, 3, -1),
    (3, 1, -2),
    (0, 2, 4)
]
vertices = 4
print(bellman_ford(graph, vertices, 0))  # Output: [0, 1, 4, 3]

```

**Union-Find for Cycle Detection**

Problem: Detect if an undirected graph contains a cycle using the union-find (disjoint set union) algorithm.

Approach:

- Use union-find to group connected components.
- If two nodes belong to the same components before union, a cycle exists.

```py
def find(parent, node):
    if parent[node] != node:
        parent[node] = find(parent, parent[node])
    return parent[node]

def union(parent, rank, u, v):
    root_u = find(parent, u)
    root_v = find(parent, v)

    if root_u != root_v:
        if rank[root_u] > rank[root_v]:
            parent[root_v] = root_u
        elif rank[root_u] < rank[root_v]:
            parent[root_u] = root_v
        else:
            parent[root_v] = root_u
            rank[root_u] += 1

def has_cycle(edges, vertices):
    parent = [i for i in range(vertices)]
    rank = [0] * vertices

    for u, v in edges:
        if find(parent, u) == find(parent, v):
            return True
        union(parent, rank, u, v)

    return False

edges = [(0, 1), (1, 2), (2, 0)]
vertices = 3
print(has_cycle(edges, vertices))  # Output: True

```

**Ancestor Matrix Problem**

Problem: Given a binary tree, create an ancestor matrix where M[i][j] = 1 if node `i` is an ancestor of node `j`, otherwise 0.

Approach:

- Perform a DFS to fill the ancestor matrix.
- Traverse all paths from each node to its descendants.

```py
def ancestor_matrix(root, n):
    matrix = [[0] * n for _ in range(n)]

    def dfs(curr, ancestors):
        if not curr:
            return
        for ancestor in ancestors:
            matrix[ancestor][curr.val] = 1
        dfs(curr.left, ancestors + [curr.val])
        dfs(curr.right, ancestors + [curr.val])

    dfs(root, [])
    return matrix

# Example tree
root = TreeNode(0)
root.left = TreeNode(1)
root.right = TreeNode(2)
root.left.left = TreeNode(3)
root.left.right = TreeNode(4)

matrix = ancestor_matrix(root, 5)
for row in matrix:
    print(row)
# Output: Ancestor matrix for nodes

```

**Strongly Connected Components (Kosaraju's Algorithm)**

Problem: Find all Strongly Connected Components (SCCs) in a directed graph using Kosaraju's Algorithm.

Approach:

- Perform a DFS and record the finish time of each node in a stack.
- Reverse the graph, and perform DFS in the order of decreasing finish times from the stack. Each DFS call on the reversed graph identifies an SCC.

This algorithm takes O(V + E) time.

```py
from collections import defaultdict

def kosaraju_scc(graph, vertices):
    def dfs1(node, visited, stack):
        visited[node] = True
        for neighbor in graph[node]:
            if not visited[neighbor]:
                dfs1(neighbor, visited, stack)
        stack.append(node)

    def dfs2(node, visited, component, reverse_graph):
        visited[node] = True
        component.append(node)
        for neighbor in reverse_graph[node]:
            if not visited[neighbor]:
                dfs2(neighbor, visited, component, reverse_graph)

    def reverse_graph(graph):
        reversed_g = defaultdict(list)
        for u in graph:
            for v in graph[u]:
                reversed_g[v].append(u)
        return reversed_g

    # Step 1: Perform DFS and store nodes in a stack by finish time
    visited = [False] * vertices
    stack = []
    for v in range(vertices):
        if not visited[v]:
            dfs1(v, visited, stack)

    # Step 2: Reverse the graph
    reversed_g = reverse_graph(graph)

    # Step 3: Perform DFS on reversed graph in the order of decreasing finish time
    visited = [False] * vertices
    sccs = []
    while stack:
        node = stack.pop()
        if not visited[node]:
            component = []
            dfs2(node, visited, component, reversed_g)
            sccs.append(component)

    return sccs

graph = {
    0: [1],
    1: [2],
    2: [0],
    1: [3],
    3: [4],
    4: [5],
    5: [3]
}
vertices = 6
print(kosaraju_scc(graph, vertices))
# Output: [[3, 5, 4], [0, 2, 1]]

```
