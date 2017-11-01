#include <iostream>
#include <string>

using std::cout;
using std::endl;
using std::cos;
using std::string;

int main() {
	string text("This is an example,"
		"in order to write this program, "
		"an example of it, in order to achieve "
		"the first section are rewritten into "
		"uppercase letters, and then output it.");

	for (auto it = text.begin(); it != text.end(); ++it)
		*it = tolower(*it);

	cout << text << endl;

	return 0;
}
