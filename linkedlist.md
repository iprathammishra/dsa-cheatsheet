# LinkedList

## Reversal of LinkedList

```python

class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

def reverseLinkedList(head: ListNode) -> ListNode:
    prev, curr = None, head
    while curr:
        next_node = curr.next
        curr.next = prev
        prev = curr
        curr = next_node
    return prev

```

## Cycle Detection in LinkedList

```python

def hasCycle(head: ListNode) -> bool:
    slow, fast = head, head
    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next
        if slow == fast:
            return True
    return False

```

## Merge LinkedLists

```python

def mergeTwoLists(l1: ListNode, l2: ListNode) -> ListNode:
    dummy = ListNode()
    tail = dummy
    while l1 and l2:
        if l1.val < l2.val:
            tail.next = l1
            l1 = l1.next
        else:
            tail.next = l2
            l2 = l2.next
        tail = tail.next
    tail.next = l1 or l2
    return dummy.next

```

## Intersection of LinkedLists

```python

def getIntersectionNode(headA: ListNode, headB: ListNode) -> ListNode:
    if not headA or not headB:
        return None
    a, b = headA, headB
    while a != b:
        a = a.next if a else headB
        b = b.next if b else headA
    return a

```

## Palindromic LinkedList

```python

def isPalindrome(head: ListNode) -> bool:
    def reverse(node):
        prev = None
        while node:
            next_node = node.next
            node.next = prev
            prev = node
            node = next_node
        return prev
    
    slow, fast = head, head
    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next
    
    second_half = reverse(slow)
    first_half = head
    
    while second_half:
        if first_half.val != second_half.val:
            return False
        first_half = first_half.next
        second_half = second_half.next
    return True

```

## Reordering LinkedList

```python

def reorderList(head: ListNode) -> None:
    if not head:
        return
    slow, fast = head, head
    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next
    
    prev, curr = None, slow
    while curr:
        next_temp = curr.next
        curr.next = prev
        prev = curr
        curr = next_temp
    
    first, second = head, prev
    while second.next:
        tmp1, tmp2 = first.next, second.next
        first.next = second
        second.next = tmp1
        first = tmp1
        second = tmp2

```

## LinkedList as Number

```python

def addTwoNumbers(l1: ListNode, l2: ListNode) -> ListNode:
    dummy = ListNode()
    curr = dummy
    carry = 0
    while l1 or l2 or carry:
        v1 = l1.val if l1 else 0
        v2 = l2.val if l2 else 0
        val = v1 + v2 + carry
        carry = val // 10
        val = val % 10
        curr.next = ListNode(val)
        curr = curr.next
        l1 = l1.next if l1 else None
        l2 = l2.next if l2 else None
    return dummy.next

```

## Flattening LinkedList

```python

class Node:
    def __init__(self, val=0, next=None, child=None):
        self.val = val
        self.next = next
        self.child = child

def flatten(head: 'Node') -> 'Node':
    if not head:
        return head
    
    pseudo_head = Node(None, head, None)
    prev = pseudo_head
    
    stack = [head]
    while stack:
        curr = stack.pop()
        prev.next = curr
        curr.prev = prev
        
        if curr.next:
            stack.append(curr.next)
            curr.next = None
        
        if curr.child:
            stack.append(curr.child)
            curr.child = None
        
        prev = curr
    
    pseudo_head.next.prev = None
    return pseudo_head.next

```

## Clone LinkedList

```python

class Node:
    def __init__(self, val=0, next=None, random=None):
        self.val = val
        self.next = next
        self.random = random

def copyRandomList(head: 'Node') -> 'Node':
    if not head:
        return None

    old_to_new = {}
    curr = head
    while curr:
        old_to_new[curr] = Node(curr.val)
        curr = curr.next
    
    curr = head
    while curr:
        old_to_new[curr].next = old_to_new.get(curr.next)
        old_to_new[curr].random = old_to_new.get(curr.random)
        curr = curr.next
    
    return old_to_new[head]

```

## LinkedList Partition

```python

def partition(head: ListNode, x: int) -> ListNode:
    before_head = ListNode(0)
    before = before_head
    after_head = ListNode(0)
    after = after_head
    
    while head:
        if head.val < x:
            before.next = head
            before = before.next
        else:
            after.next = head
            after = after.next
        head = head.next
    
    after.next = None
    before.next = after_head.next
    return before_head.next

```

## Remove LinkedList Elements

```python

def removeElements(head: ListNode, val: int) -> ListNode:
    dummy = ListNode(next=head)
    prev, curr = dummy, head
    
    while curr:
        if curr.val == val:
            prev.next = curr.next
        else:
            prev = curr
        curr = curr.next
    
    return dummy.next

def removeNthFromEnd(head: ListNode, n: int) -> ListNode:
    dummy = ListNode(next=head)
    fast = slow = dummy
    
    for _ in range(n + 1):
        fast = fast.next
    
    while fast:
        fast = fast.next
        slow = slow.next
    
    slow.next = slow.next.next
    return dummy.next

```

## Swap Nodes in LinkedList

```python

def swapPairs(head: ListNode) -> ListNode:
    dummy = ListNode(0)
    dummy.next = head
    prev = dummy
    
    while head and head.next:
        first_node = head
        second_node = head.next
        
        prev.next = second_node
        first_node.next = second_node.next
        second_node.next = first_node
        
        prev = first_node
        head = first_node.next
    
    return dummy.next

def swapNodes(head: ListNode, k: int) -> ListNode:
    first = second = head
    for _ in range(k - 1):
        first = first.next
    
    kth = first
    while first.next:
        first = first.next
        second = second.next
    
    kth.val, second.val = second.val, kth.val
    return head

```

## Finding Middle of LinkedList

```python

def middleNode(head: ListNode) -> ListNode:
    slow = fast = head
    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next
    return slow

```

## LinkedList Intersection and Union

```python

def getIntersectionNode(headA: ListNode, headB: ListNode) -> ListNode:
    if not headA or not headB:
        return None
    a, b = headA, headB
    while a != b:
        a = a.next if a else headB
        b = b.next if b else headA
    return a

def union(headA: ListNode, headB: ListNode) -> ListNode:
    elements = set()
    dummy = ListNode()
    tail = dummy
    
    current = headA
    while current:
        if current.val not in elements:
            elements.add(current.val)
            tail.next = ListNode(current.val)
            tail = tail.next
        current = current.next
    
    current = headB
    while current:
        if current.val not in elements:
            elements.add(current.val)
            tail.next = ListNode(current.val)
            tail = tail.next
        current = current.next
    
    return dummy.next

```

## Segragrate LinkedList

```python

def sortList(head: ListNode) -> ListNode:
    if not head or not head.next:
        return head
    
    def merge(l1, l2):
        dummy = ListNode()
        tail = dummy
        while l1 and l2:
            if l1.val < l2.val:
                tail.next = l1
                l1 = l1.next
            else:
                tail.next = l2
                l2 = l2.next
            tail = tail.next
        tail.next = l1 or l2
        return dummy.next
    
    slow, fast = head, head.next
    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next
    
    mid = slow.next
    slow.next = None
    left = sortList(head)
    right = sortList(mid)
    return merge(left, right)

def segregateEvenOdd(head: ListNode) -> ListNode:
    if not head:
        return None
    
    even_head = even = ListNode(0)
    odd_head = odd = ListNode(0)
    
    while head:
        if head.val % 2 == 0:
            even.next = head
            even = even.next
        else:
            odd.next = head
            odd = odd.next
        head = head.next
    
    even.next = odd_head.next
    odd.next = None
    return even_head.next

```

Here's a curated list of 10 problems on LinkedLists covering various patterns.

**Reverse a LinkedList**

Problem: Reverse a singly linked list.

Appraoch:

- Use three pointers: `prev`, `curr`, and `next`.
- Traverse the list and reverse the direction of each pointer.

```py
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

def reverse_linked_list(head):
    prev = None
    curr = head

    while curr:
        next_node = curr.next
        curr.next = prev
        prev = curr
        curr = next_node

    return prev

# Example Usage
head = ListNode(1, ListNode(2, ListNode(3, ListNode(4))))
reversed_head = reverse_linked_list(head)
while reversed_head:
    print(reversed_head.val, end=" -> ")  # Output: 4 -> 3 -> 2 -> 1
    reversed_head = reversed_head.next

```

**Detect a Cycle in a LinkedList**

Problem: Determine if a linked list had a cycle.

Approach:

- Use the Floyd's Cycle Detection Algorithm (Slow and Fast Pointers).
- If slow and fast pointers meet, a cycle exits.

```py
def has_cycle(head):
    slow, fast = head, head

    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next
        if slow == fast:
            return True

    return False

# Example Usage
head = ListNode(1, ListNode(2, ListNode(3)))
head.next.next.next = head.next  # Creates a cycle
print(has_cycle(head))  # Output: True
```

**Merge Two Sorted LinkedLists**

Problem: Merge two sorted linked lists into a single sorted linked list.

Approach:

- Use two pointers to compare values from both lists.
- Build the result list by choosing the smaller value at each step.

```py
def merge_two_sorted_lists(l1, l2):
    dummy = ListNode()
    current = dummy

    while l1 and l2:
        if l1.val < l2.val:
            current.next = l1
            l1 = l1.next
        else:
            current.next = l2
            l2 = l2.next
        current = current.next

    current.next = l1 if l1 else l2
    return dummy.next

# Example Usage
l1 = ListNode(1, ListNode(2, ListNode(4)))
l2 = ListNode(1, ListNode(3, ListNode(5)))
merged = merge_two_sorted_lists(l1, l2)
while merged:
    print(merged.val, end=" -> ")  # Output: 1 -> 1 -> 2 -> 3 -> 4 -> 5
    merged = merged.next
```

**Remove N-th Node From End**

Problem: Remove the n-th node from the end of the list in a single pass.

Approach:

- Use two pointers with a gap of n+1 between them.
- Move both pointers until the fast pointer reaches the end.
- The slow pointer will now point to the node before the one to be removed.

```py
def remove_nth_from_end(head, n):
    dummy = ListNode(0, head)
    slow, fast = dummy, dummy

    for _ in range(n + 1):
        fast = fast.next

    while fast:
        slow = slow.next
        fast = fast.next

    slow.next = slow.next.next
    return dummy.next

# Example Usage
head = ListNode(1, ListNode(2, ListNode(3, ListNode(4, ListNode(5)))))
new_head = remove_nth_from_end(head, 2)
while new_head:
    print(new_head.val, end=" -> ")  # Output: 1 -> 2 -> 3 -> 5
    new_head = new_head.next

```

**Middle of a LinkedList**

Problem: Find the middle node of a linked list.

Approach:

- Use two pointers: slow and fast.
- Move slow one step and fast two steps.
- When fast reaches the end, slow is at the middle.

```py
def find_middle(head):
    slow, fast = head, head

    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next

    return slow

# Example Usage
head = ListNode(1, ListNode(2, ListNode(3, ListNode(4, ListNode(5)))))
middle = find_middle(head)
print(middle.val)  # Output: 3

```

**Reverse Nodes in K-groups**

Problem: Reverse nodes of a linked list in groups of k.

Approach:

- Count nodes to ensure enough for reversal.
- Reverse k nodes at a time using the standard reverse approach.
- Recursively call for the remaining list.

```py
def reverse_k_group(head, k):
    def reverse(head, k):
        prev, curr = None, head
        while k:
            next_node = curr.next
            curr.next = prev
            prev = curr
            curr = next_node
            k -= 1
        return prev

    count, temp = 0, head
    while temp and count < k:
        temp = temp.next
        count += 1

    if count == k:
        reversed_head = reverse(head, k)
        head.next = reverse_k_group(temp, k)
        return reversed_head

    return head

# Example Usage
head = ListNode(1, ListNode(2, ListNode(3, ListNode(4, ListNode(5)))))
reversed_head = reverse_k_group(head, 3)
while reversed_head:
    print(reversed_head.val, end=" -> ")  # Output: 3 -> 2 -> 1 -> 4 -> 5
    reversed_head = reversed_head.next

```

**Check Palindrome LinkedList**

Problem: Check if a linked list is a palindrome.

Approach:

- Find the middle of the list.
- Reverse the second half.
- Compare the two halves.

```py
def is_palindrome(head):
    def reverse(head):
        prev, curr = None, head
        while curr:
            next_node = curr.next
            curr.next = prev
            prev = curr
            curr = next_node
        return prev

    slow, fast = head, head
    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next

    second_half = reverse(slow)
    first_half = head

    while second_half:
        if first_half.val != second_half.val:
            return False
        first_half = first_half.next
        second_half = second_half.next

    return True

# Example Usage
head = ListNode(1, ListNode(2, ListNode(2, ListNode(1))))
print(is_palindrome(head))  # Output: True

```

**Flatten a Multilevel Doubly LinkedList**

Problem: Flatten a multilevel doubly linked list into a single-level list.

Approach:

- Traverse the list and recursively flatten child nodes.
- Attach the flattened child to current node.

```py
class Node:
    def __init__(self, val=0, next=None, child=None):
        self.val = val
        self.next = next
        self.child = child

def flatten_multilevel_list(head):
    if not head:
        return None

    stack = []
    current = head

    while current:
        if current.child:
            if current.next:
                stack.append(current.next)
            current.next = current.child
            current.child.prev = current
            current.child = None

        if not current.next and stack:
            next_node = stack.pop()
            current.next = next_node
            next_node.prev = current

        current = current.next

    return head

# Example Usage
# Create a multilevel list: 1 -> 2 -> 3 -> NULL, where 2 has a child 4 -> 5
head = Node(1)
head.next = Node(2)
head.next.prev = head
head.next.next = Node(3)
head.next.next.prev = head.next
head.next.child = Node(4, Node(5))

flattened = flatten_multilevel_list(head)
while flattened:
    print(flattened.val, end=" -> ")  # Output: 1 -> 2 -> 4 -> 5 -> 3
    flattened = flattened.next

```

**Copy List with Random Pointer**

Problem: Create a deep copy of a linked with random poiners.

Approach:

- Create a mapping between original nodes and their copies using a hashmap.
- Iterate through the list twice.
    - First to create the copies of the nodes.
    - Second to link next and random pointers.

```py
class RandomNode:
    def __init__(self, val=0, next=None, random=None):
        self.val = val
        self.next = next
        self.random = random

def copy_random_list(head):
    if not head:
        return None

    mapping = {}
    current = head

    # First pass: Create all nodes
    while current:
        mapping[current] = RandomNode(current.val)
        current = current.next

    # Second pass: Link next and random pointers
    current = head
    while current:
        if current.next:
            mapping[current].next = mapping[current.next]
        if current.random:
            mapping[current].random = mapping[current.random]
        current = current.next

    return mapping[head]

# Example Usage
head = RandomNode(1, RandomNode(2, RandomNode(3)))
head.random = head.next.next  # 1 -> random -> 3
head.next.random = head       # 2 -> random -> 1
copied_head = copy_random_list(head)

# Output copied list values and random pointers
current = copied_head
while current:
    print(f"Node: {current.val}, Random: {current.random.val if current.random else None}")
    current = current.next

```

**Merge K Sorted Linked Lists**

Problem: Merge K sorted linked lists into a single sorted linked list.

Approach:

- Use a priority queue (min-heap) to merge the lists.
- Add the head of each list to the heap.
- Extract the smallest element, move its pointer to the next node, and add it back to the heap.

```py
from heapq import heappush, heappop

class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

    def __lt__(self, other):
        return self.val < other.val  # For heap comparison

def merge_k_sorted_lists(lists):
    heap = []
    for l in lists:
        if l:
            heappush(heap, l)

    dummy = ListNode()
    current = dummy

    while heap:
        smallest = heappop(heap)
        current.next = smallest
        current = current.next

        if smallest.next:
            heappush(heap, smallest.next)

    return dummy.next

# Example Usage
l1 = ListNode(1, ListNode(4, ListNode(5)))
l2 = ListNode(1, ListNode(3, ListNode(4)))
l3 = ListNode(2, ListNode(6))
lists = [l1, l2, l3]
merged = merge_k_sorted_lists(lists)

while merged:
    print(merged.val, end=" -> ")  # Output: 1 -> 1 -> 2 -> 3 -> 4 -> 4 -> 5 -> 6
    merged = merged.next

```
