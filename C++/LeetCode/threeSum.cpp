#include <algorithm>
#include <vector>
#include <iostream>

using namespace std;

class Solution {
public:
	vector<vector<int>> threeSum(vector<int>& nums) {
		sort(nums.begin(), nums.end());
		vector<vector<int>> result;

		for (int i = 0; i < nums.size(); ++i) {
			int target = -nums[i];
			int font = i + 1;
			int back = nums.size() - 1;
			while (font < back) {
				int sum = nums[font] + nums[back];
				if (sum < target) font++;
				else if (sum > target) back--;
				else {
					vector<int> temp(3, 0);
					temp[0] = nums[font];
					temp[1] = nums[back];
					temp[2] = nums[i];
					result.push_back(temp);

					while (font < back && nums[font] == temp[0]) font++;
					while (font < back && nums[back] == temp[1]) back--;
				}
			}
			while (i + 1 < nums.size() && nums[i] == nums[i + 1]) i++;
		}

		return result;
	}
};

int main() {
	int a[] = { -1, 0, 1, 15 };
	vector<int> test_nums(a, a + 5);
	vector<vector<int>>::iterator iter;
	Solution threeSum;
	vector<vector<int>> test_result = threeSum.threeSum(test_nums);

	for (iter = test_result.begin(); iter != test_result.end(); iter++) {
		cout << (*iter)[0] << " " << (*iter)[1] << " " << (*iter)[2] << " ";
	}
	cout << endl;
	return 0;
}
