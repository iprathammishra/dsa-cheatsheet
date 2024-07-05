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
