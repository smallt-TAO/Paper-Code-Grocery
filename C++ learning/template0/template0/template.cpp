#include <iostream>
#include "template.h"

using namespace std;

template<class T> A<T>::A() {
	cout << "Show the code is right" << endl;
};
template<class T> T A<T>::g(T a, T b) {
	return a + b;
}

void main() {
	A<double> a;
	cout << a.g(3.2, 3) << endl;
	A<int> b;
	cout << b.g(4, 8) << endl;
}