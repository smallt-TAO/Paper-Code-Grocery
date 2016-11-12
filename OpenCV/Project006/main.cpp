# include <opencv2/core/core.hpp>
# include <opencv2/highgui/highgui.hpp>
# include <iostream>

using namespace std;
using namespace cv;

int main() {
	// load the gray image.
	Mat srcImage = imread("Bookmark.jpg", 0);
	if (!srcImage.data) {
		cout << endl << "Load Image Error" << endl;
		return false;
	}
	imshow("Origin Gray Image", srcImage);

	// Extend the Image to the Perfect Scale.
	int m = getOptimalDFTSize(srcImage.rows);
	int n = getOptimalDFTSize(srcImage.cols);
	cout << "The Scale of the origin Image " << srcImage.cols <<
		"X" << srcImage.rows << endl;
	cout << "The Scale of the Opti Image " << m << "X" << n << endl;
	int delta1 = (m - srcImage.rows) / 2;
	int delta2 = (n - srcImage.cols) / 2;

	Mat padded;  // To save the extended Image.
	copyMakeBorder(srcImage, padded, delta1, delta1, delta2, delta2, BORDER_CONSTANT, Scalar::all(0));
	imshow("The Perfect Scale Image", padded);

	// Arrangement Space for FFT Image
	Mat planes[] = { Mat_<float>(padded), Mat::zeros(padded.size(), CV_32F) };
	Mat complexI;
	merge(planes, 2, complexI);
	dft(complexI, complexI);
	split(complexI, planes);
	magnitude(planes[0], planes[1], planes[0]);
	Mat magnitudeImage = planes[0];

	// Scale Extend.
	magnitudeImage = magnitudeImage + Scalar::all(1);
	log(magnitudeImage, magnitudeImage);
	magnitudeImage = magnitudeImage(Rect(0, 0, magnitudeImage.cols & -2, magnitudeImage.rows & -2));
	int cx = magnitudeImage.cols / 2;
	int cy = magnitudeImage.rows / 2;
	Mat q0(magnitudeImage, Rect(0, 0, cx, cy));
	Mat q1(magnitudeImage, Rect(cx, 0, cx, cy));
	Mat q2(magnitudeImage, Rect(0, cy, cx, cy));
	Mat q3(magnitudeImage, Rect(cx, cy, cx, cy));

	Mat temp;
	q0.copyTo(temp);
	q3.copyTo(q0);
	temp.copyTo(q3);
	q1.copyTo(temp);
	q2.copyTo(q1);
	temp.copyTo(q2);

	normalize(magnitudeImage, magnitudeImage, 0, 1, NORM_MINMAX);
	normalize(magnitudeImage, magnitudeImage, 0, 1, NORM_MINMAX);

	imshow("FFT", magnitudeImage);
	waitKey(0);

	return 0;
}