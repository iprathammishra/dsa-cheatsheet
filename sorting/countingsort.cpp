/*
Counting sort, effective but will only work on numbers in ORDERED range [1, n] or [0, n].

We count the occurrences of each element and we store them in a separate array of length of (max_element + 1) of the given array. 

We can then easily build the array from it.
*/

#include<bits/stdc++.h>
using namespace std;
#define int long long
#define PRATHAM ios_base::sync_with_stdio(false); cin.tie(nullptr);
int32_t main() {
  int n; cin >> n;
  vector<int> a(n);
  int MAX = -1;
  for (auto& i : a) 
    {cin >> i; MAX = max(MAX, i);}
  vector<int> counter(MAX+1, 0);
  for (auto& i : a)
    counter[i]++;
  for (int i = 1; i < MAX+1; i++) {
    while (counter[i] > 0) {
      cout << i << " ";
      counter[i]--;
    }
  }
  return 0;
}















