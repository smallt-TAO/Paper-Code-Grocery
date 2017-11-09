#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>

using namespace cv;
using namespace std;

int main() {
	Mat srcImage = imread("example001.jpg");
	Mat img = srcImage.clone();
	imshow("Origin Image", img);
	if (!srcImage.data) {
		cout << "Load image Error" << endl;
		return false;
	}
	Mat grayImage;
	Mat addingImage;
	Mat blurImage;
	Mat canny_detection;

	addingImage.create(img.size(), img.type());
	cvtColor(img, grayImage, COLOR_BGR2GRAY);
	imshow("Gray Image", grayImage);

	blur(grayImage, blurImage, Size(7, 7));
	imshow("Gray + Blur", blurImage);

	Canny(blurImage, canny_detection, 3, 9, 3);
	imshow("Gray + Blur + Detection", canny_detection);

	addingImage = Scalar::all(0);
	img.copyTo(addingImage, canny_detection);
	imshow("Origin + Dectecion", addingImage);

	waitKey(0);
	return 0;
}