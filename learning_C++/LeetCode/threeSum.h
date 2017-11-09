#include <algorithm>
#include <vector>

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