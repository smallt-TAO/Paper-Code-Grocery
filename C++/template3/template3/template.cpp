#include <iostream>
#include "template.h"

using namespace std;

void main() {
	SumDemo<double, double> sd;
	cout << sd.Cell(3, 4.2, 5.4) << endl;
}