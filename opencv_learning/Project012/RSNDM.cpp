#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace cv;
using namespace std;

int main() {
	// Mat srcImage = imread("1.jpg");
	Mat srcImage = imread("rsndm.jpg");
	Mat midImage;
	Mat dstImage;  // destation Image

	Canny(srcImage, midImage, 50, 200, 3);
	cvtColor(midImage, dstImage, COLOR_GRAY2BGR);

	vector<Vec2f> lines;
	HoughLines(midImage, lines, 1, CV_PI / 180, 150, 0, 0);

	// Point the ever lines in picture
	for (size_t i = 0; i < lines.size(); i++) {
		float rho = lines[i][0];
		float theta = lines[i][1];
		Point pt1;
		Point pt2;
		double a = cos(theta);
		double b = sin(theta);
		double x0 = a * rho;
		double y0 = b * rho;
		pt1.x = cvRound(x0 + 1000 * (-b));
		pt1.y = cvRound(y0 + 1000 * (a));
		pt2.x = cvRound(x0 - 1000 * (-b));
		pt2.y = cvRound(y0 - 1000 * (a));
		line(dstImage, pt1, pt2, Scalar(55, 100, 195), 1, LINE_AA);
	}

	imshow("Origin Image", srcImage);
	imshow("Canny Image", midImage);
	imshow("Result Image", dstImage);

	waitKey(0);

	return 0;
}