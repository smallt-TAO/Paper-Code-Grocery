//
// This code for Dilate and Erode.
//
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>

using namespace std;
using namespace cv;

Mat g_srcImage, g_dstImage;
int g_nTrackbarNumber = 0;  // 0 stand for erode, and 1 shand for dilateint.
int g_nStructElementSize = 3;


void Process();  // Code for erode and dilateint.
void on_TrackbarNumChange(int, void *);
void on_ElementSizeChange(int, void *);
void ShowHelpText();

int main() {
	system("color 3F");

	g_srcImage = imread("1.jpg");
	if (!g_srcImage.data) { 
		printf("Load srcImage ERROR! \n"); 
		return false; 
	}

	ShowHelpText();

	namedWindow("Origin Image");
	imshow("Origin Image", g_srcImage);

	namedWindow("Result Image");
	Mat element = getStructuringElement(
		MORPH_RECT, 
		Size(2 * g_nStructElementSize + 1, 2 * g_nStructElementSize + 1), 
		Point(g_nStructElementSize, 
		g_nStructElementSize));
	erode(g_srcImage, g_dstImage, element);
	imshow("Result Image", g_dstImage);

	createTrackbar(
		"ero/dil", 
		"Result Image", 
		&g_nTrackbarNumber, 
		1, 
		on_TrackbarNumChange);
	createTrackbar(
		"Kernel", 
		"Result Image",
		&g_nStructElementSize, 
		21, 
		on_ElementSizeChange);

	waitKey(0);

	return 0;
}

void Process() {
	Mat element = getStructuringElement(
		MORPH_RECT, 
		Size(2 * g_nStructElementSize + 1, 2 * g_nStructElementSize + 1), 
		Point(g_nStructElementSize, g_nStructElementSize));

	if (g_nTrackbarNumber == 0) {
		erode(g_srcImage, g_dstImage, element);
	}
	else {
		dilate(g_srcImage, g_dstImage, element);
	}

	imshow("Result Image", g_dstImage);
}

void on_TrackbarNumChange(int, void *) {
	Process();
}

void on_ElementSizeChange(int, void *) {
	Process();
}

void ShowHelpText() {
	cout << "OpenCV version of the This Platform" << CV_VERSION << endl;
}
