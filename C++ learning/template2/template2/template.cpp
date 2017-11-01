#include <iostream>
#include "template.h"

using namespace std;

int main() {
	CompareDemo<double> cd;
	cout << cd.compare(3.2, 4.5) << endl;

	return 1;
}