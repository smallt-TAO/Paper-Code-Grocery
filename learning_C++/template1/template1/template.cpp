#include <iostream>
#include <string>
#include "template.h"

using namespace std;

int main() {
	try {
		Stack<int, 20> int20Stack;
		int20Stack.push(8);
		cout << int20Stack.top() << endl;
		int20Stack.pop();

		Stack<string, 20> stringStack;
		stringStack.push("hello");
		cout << stringStack.top() << endl;
		stringStack.pop();
		stringStack.pop();
	}
	catch (std::exception const& ex) {
		std::cerr << "Exception : " << ex.what() << endl;
		return EXIT_FAILURE;
	}
}
