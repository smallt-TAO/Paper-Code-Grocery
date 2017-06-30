#include <algorithm>
#include <vector>
#include <unordered_map>
#include <iostream>

using namespace std;

class Solution {
public:
	vector<int> twoSum(vector<int>& nums, int target) {
		unordered_map<int, int> hash;
		vector<int> result;
		for (int i = 0; i < nums.size(); ++i) {
			int numberToFind = target - nums[i];
			if (hash.find(numberToFind) != hash.end()) {
				result.push_back(hash[numberToFind]);
				result.push_back(i);
				return result;
			}
			hash[nums[i]] = i;
		}
		return result;
	}
};

int main() {
	int a[] = {2, 7, 11, 15};
	vector<int> test_nums(a, a + 4);
	vector<int>::iterator iter;
	Solution solution;
	vector<int> test_result = solution.twoSum(test_nums, 9);

	for (iter = test_result.begin(); iter != test_result.end(); iter++) {
		cout << *iter << " ";
	}
	cout << endl;
	return 0;
}
