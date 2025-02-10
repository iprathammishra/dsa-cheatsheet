/*
Selection sort is a sorting algorithm that selects the smallest element from an unsorted list in each iteration and places that element at the beginning of the unsorted list.
*/

#include<bits/stdc++.h>
using namespace std;
#define int long long
#define PRATHAM ios_base::sync_with_stdio(false); cin.tie(nullptr);
int32_t main() {
  int n; cin >> n;
  vector<int> a(n);
  int swaps = 0; // if question asked about in how many adjacent swap can you make the given array non-decreasing
  for (auto& i : a) 
    cin >> i;
  for (int i = 0; i < n; i++) {
    int min_index = i;
    for (int j = i+1; j < n; j++)
      if (a[min_index] > a[j])
        min_index = j;
    if (i != min_index) {
      swaps++;
      swap(a[min_index], a[i]);
    }
  } 
  cout << swaps << endl;
  for (auto& i : a)
    cout << i << " ";
  cout << endl;
  return 0;
}















