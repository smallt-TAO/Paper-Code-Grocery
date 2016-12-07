#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/photo/photo.hpp>
#include <iostream>

using namespace cv;
using namespace std;

#define WINDOW_NAME0 "Origin Image"
#define WINDOW_NAME1 "Origin img"
#define WINDOW_NAME2 "Reuslt Image"

Mat srcImage0;
Mat srcImage1;
Mat inpaintMask;
Point previousPoint(-1, -1);

static void ShowHelpText() {
	cout << "The version of the platform" << CV_VERSION << endl;
	cout << "\n\n"
		<< "mouse can paint the white line"
		<< "\n"
		<< "Key of 1 can repaint the image"
		<< endl;
}

static void On_Mouse(int event, int x, int y, int flags, void*) {
	if (event == EVENT_LBUTTONUP || !(flags & EVENT_FLAG_LBUTTON))
		previousPoint = Point(-1, -1);

	else if (event == EVENT_LBUTTONDOWN)
		previousPoint = Point(x, y);

	else if (event == EVENT_MOUSEMOVE && (flags & EVENT_FLAG_LBUTTON)) {
		Point pt(x, y);
		if (previousPoint.x < 0)
			previousPoint = pt;
		line(inpaintMask, previousPoint, pt, Scalar::all(255), 6, 8, 0);
		line(srcImage1, previousPoint, pt, Scalar::all(255), 6, 8, 0);
		previousPoint = pt;
		imshow(WINDOW_NAME1, srcImage1);
	}
}

int main(int argc, char** argv) {
	// change the words color of console
	system("color 3F");

	ShowHelpText();
	Mat srcImage = imread("1.jpg", -1);
	if (!srcImage.data) {
		cout << "Load Image Error" << endl;
		return false;
	}
	srcImage0 = srcImage.clone();
	srcImage1 = srcImage.clone();
	inpaintMask = Mat::zeros(srcImage1.size(), CV_8U);
	imshow(WINDOW_NAME0, srcImage0);
	imshow(WINDOW_NAME1, srcImage1);
	setMouseCallback(WINDOW_NAME1, On_Mouse, 0);

	while (1) {
		char c = (char)waitKey();
		if (c == 27)
			break;
		if (c == '2') {
			inpaintMask = Scalar::all(0);
			srcImage.copyTo(srcImage1);
			imshow(WINDOW_NAME1, srcImage);
		}
		if (c == '1') {
			Mat inpaintedImage;
			inpaint(
				srcImage1, 
				inpaintMask, 
				inpaintedImage, 
				3, 
				INPAINT_TELEA);
			imshow(WINDOW_NAME2, inpaintedImage);
		}
	}

	return 0;

}