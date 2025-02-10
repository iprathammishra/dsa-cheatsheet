/*
The MergeSort function repeatedly divides the array into two halves until we reach a stage where we try to perform MergeSort on a subarray of size 1.

After that, the merge function comes into play and combines the sorted arrays into larger arrays until the whole array is merged.

Here is the dry run for:
6
5 2 9 1 5 6

We call merge_sort(0, 5) to sort the whole array.

First call: merge_sort(0, 5)
Midpoint calculation: (0 + 5) / 2 = 2
We split into two subarrays:
Left: merge_sort(0, 2) (subarray: {5, 2, 9})
Right: merge_sort(3, 5) (subarray: {1, 5, 6})

Second call: merge_sort(0, 2)
Midpoint calculation: (0 + 2) / 2 = 1
We split into two subarrays:
Left: merge_sort(0, 1) (subarray: {5, 2})
Right: merge_sort(2, 2) (subarray: {9})

Third call: merge_sort(0, 1)
Midpoint calculation: (0 + 1) / 2 = 0
We split into two subarrays:
Left: merge_sort(0, 0) (subarray: {5})
Right: merge_sort(1, 1) (subarray: {2})

Fourth call: merge_sort(0, 0)
Base case: Only one element (5), so return {5}.
Fifth call: merge_sort(1, 1)
Base case: Only one element (2), so return {2}.
Now, we merge the two sorted subarrays {5} and {2}.
The merge() function would combine them to return {2, 5}.

Sixth call: merge_sort(2, 2)
Base case: Only one element (9), so return {9}.
Now, we merge the two sorted subarrays {2, 5} and {9}.
The merge() function would combine them to return {2, 5, 9}.

Seventh call: merge_sort(3, 5)
Midpoint calculation: (3 + 5) / 2 = 4
We split into two subarrays:
Left: merge_sort(3, 4) (subarray: {1, 5})
Right: merge_sort(5, 5) (subarray: {6})

Eighth call: merge_sort(3, 4)
Midpoint calculation: (3 + 4) / 2 = 3
We split into two subarrays:
Left: merge_sort(3, 3) (subarray: {1})
Right: merge_sort(4, 4) (subarray: {5})

Ninth call: merge_sort(3, 3)
Base case: Only one element (1), so return {1}.
Tenth call: merge_sort(4, 4)
Base case: Only one element (5), so return {5}.
Now, we merge the two sorted subarrays {1} and {5}.
The merge() function would combine them to return {1, 5}.

Eleventh call: merge_sort(5, 5)
Base case: Only one element (6), so return {6}.
Now, we merge the two sorted subarrays {1, 5} and {6}.
The merge() function would combine them to return {1, 5, 6}.

Final merge: Merging {2, 5, 9} and {1, 5, 6}
Now, we merge the two final sorted subarrays {2, 5, 9} and {1, 5, 6}.
The merge() function would combine them to return the fully sorted array {1, 2, 5, 5, 6, 9}.

Final Result:
The sorted array is {1, 2, 5, 5, 6, 9}.
*/

#include<bits/stdc++.h>
using namespace std;
#define int long long
#define PRATHAM ios_base::sync_with_stdio(false); cin.tie(nullptr);

vector<int> merge(vector<int>& l, vector<int>& r) {
  int n = (int)l.size(), m = (int)r.size();
  vector<int> ans;
  int i = 0, j = 0;
  while (i < n && j < m) {
    if (l[i] < r[j])
      ans.push_back(l[i++]);
    else
      ans.push_back(r[j++]);
  }
  while (i < n)
    ans.push_back(l[i++]);
  while (j < m)
    ans.push_back(r[j++]);
  return ans;
}
vector<int> merge_sort(int l, int r, vector<int>& a) {
  if (l == r) 
    return {a[l]};
  int mid = l + (r-l)/2;
  vector<int> left = merge_sort(l, mid, a);
  vector<int> right = merge_sort(mid+1, r, a);
  return merge(left, right);
}
int32_t main() {
  int n; cin >> n;
  vector<int> a(n);
  for (auto& i : a) 
    {cin >> i;}
  vector<int> ans = merge_sort(0, n-1, a);
  for (int i = 0; i < n; i++)
    cout << ans[i] << " ";
  return 0;
}















