# include <opencv2/opencv.hpp>
# include "opencv2/highgui/highgui.hpp"

using namespace cv;

# define WINDOW_NAME "线性混合实例" // define the hong.
const int g_nMaxAlphaValue = 100;
int g_nAlphaValueSlider;
double g_dAlphaValue;
double g_dBetaValue;

Mat g_srcImage1; // source image
Mat g_srcImage2;
Mat g_dstImage; // mix image

void on_Trackbar(int, void*) {
	g_dAlphaValue = double(g_nAlphaValueSlider) / g_nMaxAlphaValue;
	g_dBetaValue = (1.0 - g_dAlphaValue);
	addWeighted(g_srcImage1, g_dAlphaValue, g_srcImage2, g_dBetaValue, 0, g_dstImage);
	imshow(WINDOW_NAME, g_dstImage);
}

int main(int argc, char** argv[]) {
	g_srcImage1 = imread("333.jpg");
	g_srcImage2 = imread("111.jpg");

	if (!g_srcImage1.data) {
		std::cout << "Can't find the image.";
		return -1;
	}
	if (!g_srcImage2.data) {
		std::cout << "Can't find the image.";
		return -1;
	}

	g_nAlphaValueSlider = 50;
	namedWindow(WINDOW_NAME);
	char TrackbarName[50];
	sprintf(TrackbarName, "透明度%d", g_nMaxAlphaValue);

	createTrackbar(TrackbarName, WINDOW_NAME, &g_nAlphaValueSlider,
		g_nMaxAlphaValue, on_Trackbar);
	on_Trackbar(g_nAlphaValueSlider, 0);

	waitKey(0);
	return 0;
}