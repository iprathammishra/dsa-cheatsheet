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
