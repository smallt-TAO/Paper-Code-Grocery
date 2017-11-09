#include <iostream>
#include <vector>

using std::endl;
using std::cin;
using std::cout;
using std::vector;

int main() {
	int tmp;
	int sum = 0;
	vector<int> v1;

	while (cin >> tmp) {
		v1.push_back(tmp);
	}

	if (v1.size() % 2 == 0) {
		for (auto it = v1.begin(); it != v1.end(); it += 2)
			cout << *it + *(it + 1) << endl;
	}
	else {
		cout << v1[v1.size() - 1] << "can't be sumed. " << endl;
		for (vector<int>::iterator it = v1.begin(); it != v1.end() - 1; it += 2)
			cout << *it + *(it + 1) << endl;
	}


	cout << "Other project : " << endl;
	for (vector<int>::iterator it = v1.begin(), en = v1.end(); it != en; it++)
		cout << *it + *(--en) << endl;

	return 0;
}