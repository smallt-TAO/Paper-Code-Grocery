#include <algorithm>
#include <vector>
#include <iostream>
// #include "twoSum.h"
// #include "threeSum.h"
#include "quickSort.h"

using namespace std;

int main() {
	int a[] = {-1, 0, 1, 2, -2, 15};
	vector<int> test_nums(a, a + 6);
	vector<vector<int>>::iterator iter;
	vector<int>::iterator iter0;
	/*
	Solution threeSum;
	vector<int> test_result = threeSum.threeSum(test_nums);

	for (iter = test_result.begin(); iter != test_result.end(); iter++) {
		for (iter0 = (*iter).begin(); iter0 != (*iter).end(); iter0++) {
			cout << *iter0 << ' ';
		}
		cout << endl;
	}
	*/
	Solution quickSort;
	quickSort.quickSort(test_nums, 0, test_nums.size() - 1);
	for (iter0 = test_nums.begin(); iter0 != test_nums.end(); iter0++) {
		cout << *iter0 << ' ';
	}
	cout << endl;
	return 0;
}