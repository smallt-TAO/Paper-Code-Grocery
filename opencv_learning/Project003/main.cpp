# include <opencv2/opencv.hpp>
# include <opencv2\highgui\highgui.hpp>
# include <iostream>

using namespace cv;
using namespace std;

int main() {
	Mat I(4, 2, CV_8UC3, Scalar(25, 100, 125));
	cout << "I = " << endl << I << endl;

	Mat J(3, 3, CV_8UC2, Scalar::all(6));
	cout << "J = " << endl << J << endl;

	// using the create()
	I.create(3, 1, CV_8UC(4));
	cout << "I1 = " << endl << I << endl;

	// simlier to matlab.
	Mat A = Mat::eye(4, 4, CV_64F);
	cout << "A = " << endl << A << endl;

	Mat B = Mat::ones(3, 7, CV_16UC1);
	cout << "B = " << endl << B << endl;

	Mat C = Mat::zeros(3, 5, CV_8UC1);
	cout << "C = " << endl << C << endl;

	Mat M = Mat(4, 3, CV_8UC3);
	randu(M, Scalar::all(0), Scalar::all(36));
	cout << "M = " << endl << M << endl;

	Mat D = (Mat_<double>(3, 3) << 0, -1, 0, -1, 0, -1, 0, -1, 0);
	cout << "D = " << endl << D << endl;

	Mat D1 = D.row(2).clone();
	cout << "D1 = " << endl << D1 << endl;
	Mat D2;
	D.copyTo(D2);
	cout << "D2 = " << endl << D2 << endl;

	return 0;
}