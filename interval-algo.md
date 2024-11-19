# Interval Algorithms

(Merging Intervals)

```python

# Merging Intervals
def merge_intervals(intervals):
    if not intervals:
        return []

    intervals.sort(key=lambda x: x[0])
    merged = [intervals[0]]

    for current in intervals[1:]:
        last = merged[-1]
        if current[0] <= last[1]:
            last[1] = max(last[1], current[1])
        else:
            merged.append(current)

    return merged

# Example usage
intervals = [[1, 3], [2, 6], [8, 10], [15, 18]]
print(merge_intervals(intervals))  # Output: [[1, 6], [8, 10], [15, 18]]

```

(Inserting Intervals)

```python

# Inserting Intervals
def insert_interval(intervals, new_interval):
    result = []
    i, n = 0, len(intervals)

    # Add all intervals before the new interval
    while i < n and intervals[i][1] < new_interval[0]:
        result.append(intervals[i])
        i += 1

    # Merge all overlapping intervals with the new interval
    while i < n and intervals[i][0] <= new_interval[1]:
        new_interval[0] = min(new_interval[0], intervals[i][0])
        new_interval[1] = max(new_interval[1], intervals[i][1])
        i += 1
    result.append(new_interval)

    # Add all intervals after the new interval
    while i < n:
        result.append(intervals[i])
        i += 1

    return result

# Example usage
intervals = [[1, 3], [6, 9]]
new_interval = [2, 5]
print(insert_interval(intervals, new_interval))  # Output: [[1, 5], [6, 9]]

```

(Interval Intersection)

```python

# Interval Intersection
def interval_intersection(A, B):
    i, j = 0, 0
    result = []

    while i < len(A) and j < len(B):
        start = max(A[i][0], B[j][0])
        end = min(A[i][1], B[j][1])

        if start <= end:
            result.append([start, end])

        if A[i][1] < B[j][1]:
            i += 1
        else:
            j += 1

    return result

# Example usage
A = [[0, 2], [5, 10], [13, 23], [24, 25]]
B = [[1, 5], [8, 12], [15, 24], [25, 26]]
print(interval_intersection(A, B))  # Output: [[1, 2], [5, 5], [8, 10], [15, 23], [24, 24], [25, 25]]

```

(Meeting Rooms- Checking if intervals overlap)

```python

# Meeting Rooms - Check if a person can attend all meetings (no overlaps)
def can_attend_meetings(intervals):
    intervals.sort(key=lambda x: x[0])

    for i in range(1, len(intervals)):
        if intervals[i][0] < intervals[i - 1][1]:
            return False

    return True

# Example usage
intervals = [[0, 30], [5, 10], [15, 20]]
print(can_attend_meetings(intervals))  # Output: False

```

(Minimum Meeting Rooms- Finding the minimum number of meeting rooms required)

```python
# Minimum Meeting Rooms
import heapq

def min_meeting_rooms(intervals):
    if not intervals:
        return 0

    intervals.sort(key=lambda x: x[0])
    min_heap = [intervals[0][1]]

    for interval in intervals[1:]:
        if interval[0] >= min_heap[0]:
            heapq.heappop(min_heap)
        heapq.heappush(min_heap, interval[1])

    return len(min_heap)

# Example usage
intervals = [[0, 30], [5, 10], [15, 20]]
print(min_meeting_rooms(intervals))  # Output: 2

```

(Employee Free Time- Finding free time slots for employees)

```python

# Employee Free Time
from heapq import heappop, heappush

def employee_free_time(schedule):
    result = []
    min_heap = []

    for i, employee in enumerate(schedule):
        heappush(min_heap, (employee[0][0], i, 0))

    prev_end = min_heap[0][0]

    while min_heap:
        time, emp, interval_idx = heappop(min_heap)

        if prev_end < time:
            result.append([prev_end, time])
        
        prev_end = max(prev_end, schedule[emp][interval_idx][1])
        
        if interval_idx + 1 < len(schedule[emp]):
            heappush(min_heap, (schedule[emp][interval_idx + 1][0], emp, interval_idx + 1))

    return result

# Example usage
schedule = [[[1, 2], [5, 6]], [[1, 3]], [[4, 10]]]
print(employee_free_time(schedule))  # Output: [[3, 4]]

```

Here's a curated list of 10 problems on Interval Algorithms covering key concepts and patterns.

**Merge Intervals**

Problem: Given a list of intervals, merge overlapping intervals and return the resulting list.

Approach:

- Sort intervals by their start time.
- Traverse the sorted intervals and merge overlapping ones by comparing the current interval's start with the previous interval's end.

```py
def merge_intervals(intervals):
    intervals.sort(key=lambda x: x[0])
    merged = []

    for interval in intervals:
        if not merged or merged[-1][1] < interval[0]:
            merged.append(interval)
        else:
            merged[-1][1] = max(merged[-1][1], interval[1])

    return merged

intervals = [[1, 3], [2, 6], [8, 10], [15, 18]]
print(merge_intervals(intervals))  # Output: [[1, 6], [8, 10], [15, 18]]

```

**Insert Interval**

Problem: Insert a new interval into a list of sorted non-overlapping intervals and merge if necessary.

Approach:

- Traverse the intervals and odd all intervals that end before the new interval starts.
- Merge overlapping intervals with the new interval.
- Append all intervals starting after the new interval ends.

```py
def insert_interval(intervals, new_interval):
    result = []
    for interval in intervals:
        if interval[1] < new_interval[0]:
            result.append(interval)
        elif interval[0] > new_interval[1]:
            result.append(new_interval)
            new_interval = interval
        else:
            new_interval = [min(interval[0], new_interval[0]), max(interval[1], new_interval[1])]
    result.append(new_interval)
    return result

intervals = [[1, 3], [6, 9]]
new_interval = [2, 5]
print(insert_interval(intervals, new_interval))  # Output: [[1, 5], [6, 9]]

```

**Meeting Rooms**

Problem: Given meeting time intervals, determine if a person can attend all meetings.

Approach:

- Sort intervals by start time.
- Check if any two consecutive intervals overlap.

```py
def can_attend_meetings(intervals):
    intervals.sort(key=lambda x: x[0])
    for i in range(1, len(intervals)):
        if intervals[i][0] < intervals[i-1][1]:
            return False
    return True

intervals = [[0, 30], [5, 10], [15, 20]]
print(can_attend_meetings(intervals))  # Output: False

```

**Meeting Rooms II (Min Meeting Rooms)**

Problem: Find the minimum number of meeting rooms required to host all meetings.

Approach:

- Use a priority queue (min-heap) to keep track of ongoing meeting's end times.
- For each interval, remove meetings that have ended and add the current meeting to the heap.

```py
import heapq

def min_meeting_rooms(intervals):
    intervals.sort(key=lambda x: x[0])
    heap = []

    for interval in intervals:
        if heap and heap[0] <= interval[0]:
            heapq.heappop(heap)
        heapq.heappush(heap, interval[1])

    return len(heap)

intervals = [[0, 30], [5, 10], [15, 20]]
print(min_meeting_rooms(intervals))  # Output: 2

```

**Non-overlapping Intervals**

Problem: Find the minimum number of intervals to remove to make the rest non-overlapping.

Approach:

- Sort intervals by their end times.
- Count overlapping intervals by comparing the current interval's start with the previous interval's end.

```py
def erase_overlap_intervals(intervals):
    intervals.sort(key=lambda x: x[1])
    count, end = 0, float('-inf')

    for interval in intervals:
        if interval[0] < end:
            count += 1
        else:
            end = interval[1]

    return count

intervals = [[1, 2], [2, 3], [3, 4], [1, 3]]
print(erase_overlap_intervals(intervals))  # Output: 1

```

**Interval Intersection**

Problem: Find the intersection of two lists of intervals.

Approach:

- Use two pointers to traverse both lists and compare interval ranges.
- If intervals overlap, compute the intersection.
- Move the pointer for the interval with the smaller endpoint.

```py
def interval_intersection(A, B):
    i, j, result = 0, 0, []

    while i < len(A) and j < len(B):
        start = max(A[i][0], B[j][0])
        end = min(A[i][1], B[j][1])

        if start <= end:
            result.append([start, end])

        if A[i][1] < B[j][1]:
            i += 1
        else:
            j += 1

    return result

A = [[0, 2], [5, 10], [13, 23], [24, 25]]
B = [[1, 5], [8, 12], [15, 24], [25, 26]]
print(interval_intersection(A, B))  # Output: [[1, 2], [5, 5], [8, 10], [15, 23], [24, 24], [25, 25]]

```

**Employee Free Time**

Problem: Find the common free time intervals for all employees given their schedules.

Approach:

- Flatten all intervals into a single lists and sort by start times.
- Merge overlapping intervals.
- Identify gaps between merged intervals as free time.

```py
def employee_free_time(schedule):
    intervals = sorted([interval for emp in schedule for interval in emp], key=lambda x: x[0])
    merged = []
    for interval in intervals:
        if not merged or merged[-1][1] < interval[0]:
            merged.append(interval)
        else:
            merged[-1][1] = max(merged[-1][1], interval[1])

    free_time = []
    for i in range(1, len(merged)):
        free_time.append([merged[i-1][1], merged[i][0]])

    return free_time

schedule = [[[1, 2], [5, 6]], [[1, 3]], [[4, 10]]]
print(employee_free_time(schedule))  # Output: [[3, 4]]

```

**Max Number of Events That Can Be Attended**

Problem: Attend the maximum numbef of non-overlapping events given their start and end days.

Approach:

- Sort events by their end time.
- Use a set to track attended days.

```py
def max_events(events):
    events.sort(key=lambda x: x[1])
    attended = set()

    for start, end in events:
        for day in range(start, end + 1):
            if day not in attended:
                attended.add(day)
                break

    return len(attended)

events = [[1, 2], [2, 3], [3, 4]]
print(max_events(events))  # Output: 3

```

**Minimum Interval to Cover a Point**

Problem: Given a list of intervals, find the smallest interval that covers a given point.

Approach:

- Sort intervals by their length.
- Iterate to find the first interval covering the point.

```py
def minimum_interval_covering_point(intervals, point):
    intervals.sort(key=lambda x: x[1] - x[0])  # Sort by interval length
    for start, end in intervals:
        if start <= point <= end:
            return [start, end]
    return []

intervals = [[1, 4], [2, 3], [3, 6], [4, 8]]
point = 3
print(minimum_interval_covering_point(intervals, point))  # Output: [2, 3]

```

**Split Intervals Into K Group**

Problem: Split intervals into exactly k group such that the maxuimum group size is minimized.

Approach:

- Use binary search to minimize the maximum group size, validating a solution with a greedy algorithm.

```
def can_split_intervals(intervals, k, max_size):
    groups = [0] * k  # Groups' sizes
    for start, end in intervals:
        # Check if the interval can fit in an existing group
        placed = False
        for i in range(k):
            if groups[i] < max_size:
                groups[i] += 1
                placed = True
                break
        if not placed:  # Could not place interval
            return False
    return True

def split_intervals_k_groups(intervals, k):
    def is_valid(mid):
        return can_split_intervals(intervals, k, mid)

    intervals.sort()  # Sort intervals (can be based on start time or any criterion)
    low, high = 1, len(intervals)
    answer = high

    while low <= high:
        mid = (low + high) // 2
        if is_valid(mid):
            answer = mid
            high = mid - 1
        else:
            low = mid + 1

    return answer

intervals = [[1, 3], [2, 6], [8, 10], [15, 18]]
k = 2
print(split_intervals_k_groups(intervals, k))  # Output: 2

```