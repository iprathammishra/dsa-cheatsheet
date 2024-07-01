# Two Pointers / Sliding Window
## Sliding Window Pattern

### Fixed-Size Window 

(Maximum Sum Subarray of Size K)
```python

def max_sum_subarray_of_size_k(arr, k):
    max_sum = 0
    window_sum = 0

    for i in range(k):
        window_sum += arr[i]

    max_sum = window_sum
    for i in range(k, len(arr)):
        window_sum += arr[i] - arr[i - k]
        max_sum = max(max_sum, window_sum)
    
    return max_sum

```

(Longest Substring with K Distinct Characters)

