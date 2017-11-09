#include "opencv2/core/core.hpp" 
#include "opencv2/highgui/highgui.hpp" 
#include "opencv2/imgproc/imgproc.hpp" 

using namespace cv;

int main() {
	Mat image = imread("1.jpg");

	namedWindow("Box Filter Origin Image");
	namedWindow("Box Filter Result Image");

	imshow("Box Filter Origin Image", image);

	Mat out;
	boxFilter(image, out, -1, Size(5, 5));

	imshow("Box Filter Result Image", out);

	waitKey(0);
	return 0;
}