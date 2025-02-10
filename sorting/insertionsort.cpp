/*
Insertion sort is a sorting algorithm that places an unsortd element at its suitable place in each iteration.

Insertion sort works similarly as we sort cards in our hand in a card game.

We assume that the first card is alreadt sorted then, we select an unsorted card. If the unsorted card is greater than the card in hand, it is placed on the right otherwise, to the left. In the same way, other unsorted cards are taken and put in their right place.
*/

#include<bits/stdc++.h>
using namespace std;
#define int long long
#define PRATHAM ios_base::sync_with_stdio(false); cin.tie(nullptr);
int32_t main() {
  int n; cin >> n;
  vector<int> a(n);
  for (auto& i : a) 
    cin >> i;
  for (int i = 1; i < n; i++) {
    int j = i; int key = a[i];
    while (j - 1 >= 0 && a[j-1] > key) {
      a[j] = a[j-1];
      j--;
    }
    a[j] = key;
  } 
  for (auto& i : a)
    cout << i << " ";
  cout << endl;
  return 0;
}















