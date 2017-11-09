#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace std;
using namespace cv;

Mat g_srcImage, g_dstImage;
int g_nElementShape = MORPH_RECT;

int g_nMaxIterationNum = 10;
int g_nOpenCloseNum = 0;
int g_nErodeDilateNum = 0;
int g_nTopBlackHatNum = 0;

static void on_OpenClose(int, void*);
static void on_ErodeDilate(int, void*);
static void on_TopBlackHat(int, void*);
static void ShowHelpText();

int main() {
	system("color 2F");

	ShowHelpText();

	g_srcImage = imread("1.jpg");
	if (!g_srcImage.data) { 
		printf("Oh，no, Load srcImage ERROR！ \n"); 
		return false; 
	}

	namedWindow("Origin Image");
	imshow("Origin Image", g_srcImage);

	namedWindow("Open/Shut", 1);
	namedWindow("erode/dilateint", 1);
	namedWindow("Top/Black", 1);

	g_nOpenCloseNum = 9;
	g_nErodeDilateNum = 9;
	g_nTopBlackHatNum = 2;

	createTrackbar("Iter", "Open/Shut", &g_nOpenCloseNum, g_nMaxIterationNum * 2 + 1, on_OpenClose);
	createTrackbar("Iter", "erode/dilateint", &g_nErodeDilateNum, g_nMaxIterationNum * 2 + 1, on_ErodeDilate);
	createTrackbar("Iter", "Top/Black", &g_nTopBlackHatNum, g_nMaxIterationNum * 2 + 1, on_TopBlackHat);

	while (1) {
		int c;

		on_OpenClose(g_nOpenCloseNum, 0);
		on_ErodeDilate(g_nErodeDilateNum, 0);
		on_TopBlackHat(g_nTopBlackHatNum, 0);

		c = waitKey(0);

		if ((char)c == 'q' || (char)c == 27)
			break;
		if ((char)c == '1')
			g_nElementShape = MORPH_ELLIPSE;
		else if ((char)c == '2')
			g_nElementShape = MORPH_RECT;
		else if ((char)c == '3')
			g_nElementShape = MORPH_CROSS;
		else if ((char)c == ' ')
			g_nElementShape = (g_nElementShape + 1) % 3;
	}

	return 0;
}

static void on_OpenClose(int, void*) {
	int offset = g_nOpenCloseNum - g_nMaxIterationNum;
	int Absolute_offset = offset > 0 ? offset : -offset;

	Mat element = getStructuringElement(
		g_nElementShape, 
		Size(Absolute_offset * 2 + 1, Absolute_offset * 2 + 1), 
		Point(Absolute_offset, Absolute_offset));

	if (offset < 0)
		morphologyEx(g_srcImage, g_dstImage, MORPH_OPEN, element);
	else
		morphologyEx(g_srcImage, g_dstImage, MORPH_CLOSE, element);

	imshow("Open/Shut", g_dstImage);
}

static void on_ErodeDilate(int, void*) {
	int offset = g_nErodeDilateNum - g_nMaxIterationNum;
	int Absolute_offset = offset > 0 ? offset : -offset;

	Mat element = getStructuringElement(
		g_nElementShape, 
		Size(Absolute_offset * 2 + 1, Absolute_offset * 2 + 1), 
		Point(Absolute_offset, Absolute_offset));

	if (offset < 0)
		erode(g_srcImage, g_dstImage, element);
	else
		dilate(g_srcImage, g_dstImage, element);

	imshow("erode/dilateint", g_dstImage);
}

static void on_TopBlackHat(int, void*) {
	int offset = g_nTopBlackHatNum - g_nMaxIterationNum;
	int Absolute_offset = offset > 0 ? offset : -offset;

	Mat element = getStructuringElement(
		g_nElementShape, 
		Size(Absolute_offset * 2 + 1, Absolute_offset * 2 + 1), 
		Point(Absolute_offset, Absolute_offset));

	if (offset < 0)
		morphologyEx(g_srcImage, g_dstImage, MORPH_TOPHAT, element);
	else
		morphologyEx(g_srcImage, g_dstImage, MORPH_BLACKHAT, element);

	imshow("Top/Black", g_dstImage);
}

static void ShowHelpText() {
	printf("\n\n\t\t\t   当前使用的OpenCV版本为：" CV_VERSION);
	printf("\n\n  ----------------------------------------------------------------------------\n");

	printf("\n\t请调整滚动条观察图像效果\n\n");
	printf("\n\t按键操作说明: \n\n"
		"\t\t键盘按键【ESC】或者【Q】- 退出程序\n"
		"\t\t键盘按键【1】- 使用椭圆(Elliptic)结构元素\n"
		"\t\t键盘按键【2】- 使用矩形(Rectangle )结构元素\n"
		"\t\t键盘按键【3】- 使用十字型(Cross-shaped)结构元素\n"
		"\t\t键盘按键【空格SPACE】- 在矩形、椭圆、十字形结构元素中循环\n");
}