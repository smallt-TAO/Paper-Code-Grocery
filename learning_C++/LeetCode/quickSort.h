#include <vector>

using namespace std;
 
class Solution {
public:
	void quickSort(vector<int>& nums, int i, int j) {
		if (i < j) {
			int front = i;
			int back = j;
			int temp = nums[i];
			
			if (front < back) {
			    // swap the big var.
				while (front < back && nums[back] > temp) back--;
				nums[front++] = nums[back];

				// swap the small var.
				while (front < back && nums[front] < temp) front++;
				nums[back--] = nums[front];

			}
			nums[front] = temp;
			quickSort(nums, i, front - 1);
			quickSort(nums, front + 1, j);
		}
	}
};