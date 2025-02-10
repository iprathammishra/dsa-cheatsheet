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
  for (int _i = 0; _i < n; _i++) {
    for (int i = 0; i+1 < n; i++) {
      if (a[i] > a[i+1]) {
        swaps++;
        swap(a[i], a[i+1]);
      }
    }
  }
  cout << swaps << endl;
  for (auto& i : a)
    cout << i << " ";
  cout << endl;
  return 0;
}