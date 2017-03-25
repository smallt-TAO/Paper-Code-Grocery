#include <iostream>
#include <vector>

using std::cout;
using std::endl;
// using std::cin;
using std::vector;

int main() {
	vector<int> dou{1, 3, 4, 5};


	for (auto it = dou.begin(); it != dou.end(); ++it) {
		cout << *it << " ";
		*it *= 2;
	}

	cout << endl;
		
	for (auto it = dou.begin(); it != dou.end(); ++it)
		cout << *it << " ";

	return 0;
}