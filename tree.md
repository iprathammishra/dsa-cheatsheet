# Tree

## Depth-First Traversals

(Preorder Traversal)

```python

class TreeNode:
    def __init__(self, value=0, left=None, right=None):
        self.value = value
        self.left = left
        self.right = right

def preorder_traversal(root):
    if root is None:
        return []
    return [root.value] + preorder_traversal(root.left) + preorder_traversal(root.right)

# Example usage
# root = TreeNode(1, TreeNode(2), TreeNode(3))
# print(preorder_traversal(root))  # Output: [1, 2, 3]

```

(Inorder Traversal)

```python

def inorder_traversal(root):
    if root is None:
        return []
    return inorder_traversal(root.left) + [root.value] + inorder_traversal(root.right)

# Example usage
# root = TreeNode(1, TreeNode(2), TreeNode(3))
# print(inorder_traversal(root))  # Output: [2, 1, 3]

```

(Postorder Traversal)

```python

def postorder_traversal(root):
    if root is None:
        return []
    return postorder_traversal(root.left) + postorder_traversal(root.right) + [root.value]

# Example usage
# root = TreeNode(1, TreeNode(2), TreeNode(3))
# print(postorder_traversal(root))  # Output: [2, 3, 1]

```

## Breadth-First Traversal

(Level-order Traversal)

```python

from collections import deque

def level_order_traversal(root):
    if root is None:
        return []
    
    result = []
    queue = deque([root])
    
    while queue:
        level_size = len(queue)
        level_nodes = []
        
        for _ in range(level_size):
            node = queue.popleft()
            level_nodes.append(node.value)
            
            if node.left:
                queue.append(node.left)
            if node.right:
                queue.append(node.right)
        
        result.append(level_nodes)
    
    return result

# Example usage
# root = TreeNode(1, TreeNode(2), TreeNode(3))
# print(level_order_traversal(root))  # Output: [[1], [2, 3]]

```

## Special Traversals and Algorithms

(Reverse Level-order Traversal)

```python

def reverse_level_order_traversal(root):
    if root is None:
        return []
    
    result = []
    queue = deque([root])
    
    while queue:
        level_size = len(queue)
        level_nodes = []
        
        for _ in range(level_size):
            node = queue.popleft()
            level_nodes.append(node.value)
            
            if node.right:
                queue.append(node.right)
            if node.left:
                queue.append(node.left)
        
        result.insert(0, level_nodes)
    
    return result

# Example usage
# root = TreeNode(1, TreeNode(2), TreeNode(3))
# print(reverse_level_order_traversal(root))  # Output: [[2, 3], [1]]

```

(Ancestor Traversal)

```python

def find_ancestors(root, target):
    def helper(node, target, path):
        if not node:
            return False
        path.append(node.value)
        if node.value == target:
            return True
        if helper(node.left, target, path) or helper(node.right, target, path):
            return True
        path.pop()
        return False
    
    ancestors = []
    helper(root, target, ancestors)
    return ancestors

# Example usage
# root = TreeNode(1, TreeNode(2), TreeNode(3, TreeNode(4), TreeNode(5)))
# print(find_ancestors(root, 5))  # Output: [1, 3, 5]

```

(Max Heapify)

```python

def max_heapify(arr, n, i):
    largest = i
    left = 2 * i + 1
    right = 2 * i + 2
    
    if left < n and arr[largest] < arr[left]:
        largest = left
    
    if right < n and arr[largest] < arr[right]:
        largest = right
    
    if largest != i:
        arr[i], arr[largest] = arr[largest], arr[i]
        max_heapify(arr, n, largest)

# Example usage
# arr = [4, 10, 3, 5, 1]
# max_heapify(arr, len(arr), 0)
# print(arr)  # Output: [10, 5, 3, 4, 1]

```

(Min Heapify)

```python

def min_heapify(arr, n, i):
    smallest = i
    left = 2 * i + 1
    right = 2 * i + 2
    
    if left < n and arr[smallest] > arr[left]:
        smallest = left
    
    if right < n and arr[smallest] > arr[right]:
        smallest = right
    
    if smallest != i:
        arr[i], arr[smallest] = arr[smallest], arr[i]
        min_heapify(arr, n, smallest)

# Example usage
# arr = [4, 10, 3, 5, 1]
# min_heapify(arr, len(arr), 0)
# print(arr)  # Output: [1, 4, 3, 5, 10]

```

(Morris Traversal)

```python

def morris_inorder_traversal(root):
    result = []
    current = root
    
    while current:
        if current.left is None:
            result.append(current.value)
            current = current.right
        else:
            predecessor = current.left
            while predecessor.right and predecessor.right != current:
                predecessor = predecessor.right
            if predecessor.right is None:
                predecessor.right = current
                current = current.left
            else:
                predecessor.right = None
                result.append(current.value)
                current = current.right
    
    return result

# Example usage
# root = TreeNode(1, TreeNode(2), TreeNode(3))
# print(morris_inorder_traversal(root))  # Output: [2, 1, 3]

```

(Binary Tree Zigzag Traversal)

```python

def zigzag_level_order_traversal(root):
    if root is None:
        return []
    
    result = []
    queue = deque([root])
    left_to_right = True
    
    while queue:
        level_size = len(queue)
        level_nodes = deque()
        
        for _ in range(level_size):
            node = queue.popleft()
            if left_to_right:
                level_nodes.append(node.value)
            else:
                level_nodes.appendleft(node.value)
            
            if node.left:
                queue.append(node.left)
            if node.right:
                queue.append(node.right)
        
        result.append(list(level_nodes))
        left_to_right = not left_to_right
    
    return result

# Example usage
# root = TreeNode(1, TreeNode(2), TreeNode(3))
# print(zigzag_level_order_traversal(root))  # Output: [[1], [3, 2]]

```

(Serialize and Deserialize Tree)

```python

class Codec:
    def serialize(self, root):
        def helper(node):
            if node is None:
                result.append("null")
            else:
                result.append(str(node.value))
                helper(node.left)
                helper(node.right)
        
        result = []
        helper(root)
        return ",".join(result)
    
    def deserialize(self, data):
        def helper():
            value = next(values)
            if value == "null":
                return None
            node = TreeNode(int(value))
            node.left = helper()
            node.right = helper()
            return node
        
        values = iter(data.split(","))
        return helper()

# Example usage
# codec = Codec()
# tree_string = codec.serialize(root)
# new_root = codec.deserialize(tree_string)

```

(Lowest Common Ancestor- LCA)

```python

def lowest_common_cestor(root, p, q):
    if root is None or root == p or root == q:
        return root
    
    left = lowest_common_cestor(root.left, p, q)
    right = lowest_common_cestor(root.right, p, q)
    
    if left and right:
        return root
    return left if left else right

# Example usage
# root = TreeNode(1, TreeNode(2), TreeNode(3))
# lca = lowest_common_cestor(root, root.left, root.right)
# print(lca.value)  # Output: 1

```

(Binary Tree Diameter)

```python

def diameter_of_binary_tree(root):
    diameter = 0
    
    def depth(node):
        nonlocal diameter
        if not node:
            return 0
        left = depth(node.left)
        right = depth(node.right)
        diameter = max(diameter, left + right)
        return max(left, right) + 1
    
    depth(root)
    return diameter

# Example usage
# root = TreeNode(1, TreeNode(2), TreeNode(3, TreeNode(4), TreeNode(5)))
# print(diameter_of_binary_tree(root))  # Output: 3

```

# Binary Search Tree Patterns

## Insert a Node into BST

```python
class TreeNode:
    def __init__(self, value=0, left=None, right=None):
        self.value = value
        self.left = left
        self.right = right

def insert_into_bst(root, value):
    if root is None:
        return TreeNode(value)
    if value < root.value:
        root.left = insert_into_bst(root.left, value)
    else:
        root.right = insert_into_bst(root.right, value)
    return root

# Example usage
# root = TreeNode(4, TreeNode(2), TreeNode(7))
# root = insert_into_bst(root, 5)
# print(inorder_traversal(root))  # Output: [2, 4, 5, 7]

```

## Delete a Node from BST

```python

def delete_node(root, key):
    if root is None:
        return root
    
    if key < root.value:
        root.left = delete_node(root.left, key)
    elif key > root.value:
        root.right = delete_node(root.right, key)
    else:
        if root.left is None:
            return root.right
        elif root.right is None:
            return root.left
        
        temp = min_value_node(root.right)
        root.value = temp.value
        root.right = delete_node(root.right, temp.value)
    
    return root

def min_value_node(node):
    current = node
    while current.left:
        current = current.left
    return current

# Example usage
# root = TreeNode(5, TreeNode(3), TreeNode(6, None, TreeNode(7)))
# root = delete_node(root, 3)
# print(inorder_traversal(root))  # Output: [5, 6, 7]

```

## Search a Node in BST

```python

def search_bst(root, value):
    if root is None or root.value == value:
        return root
    if value < root.value:
        return search_bst(root.left, value)
    return search_bst(root.right, value)

# Example usage
# root = TreeNode(4, TreeNode(2), TreeNode(7))
# result = search_bst(root, 2)
# print(result.value if result else "Not found")  # Output: 2

```

## Validate BST

```python

def is_valid_bst(root):
    def validate(node, low=float('-inf'), high=float('inf')):
        if not node:
            return True
        if not (low < node.value < high):
            return False
        return validate(node.left, low, node.value) and validate(node.right, node.value, high)
    
    return validate(root)

# Example usage
# root = TreeNode(2, TreeNode(1), TreeNode(3))
# print(is_valid_bst(root))  # Output: True

```

## Find Kth Smallest element in BST

```python

def kth_smallest(root, k):
    def inorder(node):
        return inorder(node.left) + [node.value] + inorder(node.right) if node else []
    
    return inorder(root)[k-1]

# Example usage
# root = TreeNode(3, TreeNode(1, None, TreeNode(2)), TreeNode(4))
# print(kth_smallest(root, 1))  # Output: 1

```

## Lowest Common Ancestor in BST

```python

def lowest_common_ancestor_bst(root, p, q):
    while root:
        if p.value < root.value and q.value < root.value:
            root = root.left
        elif p.value > root.value and q.value > root.value:
            root = root.right
        else:
            return root

# Example usage
# root = TreeNode(6, TreeNode(2, TreeNode(0), TreeNode(4, TreeNode(3), TreeNode(5))), TreeNode(8, TreeNode(7), TreeNode(9)))
# p, q = root.left, root.left.right
# lca = lowest_common_ancestor_bst(root, p, q)
# print(lca.value)  # Output: 2

```

## Convert Sorted Array to BST

```python

def sorted_array_to_bst(nums):
    if not nums:
        return None
    
    mid = len(nums) // 2
    root = TreeNode(nums[mid])
    root.left = sorted_array_to_bst(nums[:mid])
    root.right = sorted_array_to_bst(nums[mid+1:])
    
    return root

# Example usage
# nums = [-10, -3, 0, 5, 9]
# root = sorted_array_to_bst(nums)
# print(inorder_traversal(root))  # Output: [-10, -3, 0, 5, 9]

```

## Convert BST to Greater Tree

```python

def convert_bst_to_greater_tree(root):
    total = 0
    
    def reverse_inorder(node):
        nonlocal total
        if node:
            reverse_inorder(node.right)
            total += node.value
            node.value = total
            reverse_inorder(node.left)
    
    reverse_inorder(root)
    return root

# Example usage
# root = TreeNode(4, TreeNode(1), TreeNode(6))
# root = convert_bst_to_greater_tree(root)
# print(inorder_traversal(root))  # Output: [7, 6, 6]

```

## Merge Two BSTs

```python

def merge_two_bsts(root1, root2):
    def inorder(node):
        return inorder(node.left) + [node.value] + inorder(node.right) if node else []
    
    def sorted_array_to_bst(nums):
        if not nums:
            return None
        mid = len(nums) // 2
        root = TreeNode(nums[mid])
        root.left = sorted_array_to_bst(nums[:mid])
        root.right = sorted_array_to_bst(nums[mid+1:])
        return root
    
    nums1 = inorder(root1)
    nums2 = inorder(root2)
    merged_nums = sorted(nums1 + nums2)
    return sorted_array_to_bst(merged_nums)

# Example usage
# root1 = TreeNode(1, None, TreeNode(3))
# root2 = TreeNode(2, TreeNode(1), TreeNode(4))
# merged_root = merge_two_bsts(root1, root2)
# print(inorder_traversal(merged_root))  # Output: [1, 1, 2, 3, 4]

```

## Range Sum of BST

```python

def range_sum_bst(root, low, high):
    if not root:
        return 0
    if root.value < low:
        return range_sum_bst(root.right, low, high)
    if root.value > high:
        return range_sum_bst(root.left, low, high)
    return root.value + range_sum_bst(root.left, low, high) + range_sum_bst(root.right, low, high)

# Example usage
# root = TreeNode(10, TreeNode(5, TreeNode(3), TreeNode(7)), TreeNode(15, None, TreeNode(18)))
# print(range_sum_bst(root, 7, 15))  # Output: 32

```
