#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
using namespace cv;
 
#define WINDOW_NAME1 "Origin Window"        
#define WINDOW_NAME2 "Match Window"        

Mat g_srcImage; 
Mat g_templateImage; 
Mat g_resultImage;
int g_nMatchMethod;
int g_nMaxTrackbarNum = 5;

void on_Matching(int, void*);
static void ShowHelpText();

int main() {
	system("color 2F");

	ShowHelpText();

	g_srcImage = imread("1.jpg", 1);
	g_templateImage = imread("2.jpg", 1);

	namedWindow(WINDOW_NAME1, WINDOW_AUTOSIZE);
	namedWindow(WINDOW_NAME2, WINDOW_AUTOSIZE);

	createTrackbar(
		"Method", WINDOW_NAME1, 
		&g_nMatchMethod, 
		g_nMaxTrackbarNum, 
		on_Matching);
	on_Matching(0, 0);

	waitKey(0);
	return 0;
}

void on_Matching(int, void*) {
	Mat srcImage;
	g_srcImage.copyTo(srcImage);

	int resultImage_cols = g_srcImage.cols - g_templateImage.cols + 1;
	int resultImage_rows = g_srcImage.rows - g_templateImage.rows + 1;
	g_resultImage.create(resultImage_cols, resultImage_rows, CV_32FC1);

	matchTemplate(
		g_srcImage, 
		g_templateImage, 
		g_resultImage, 
		g_nMatchMethod);
	normalize(
		g_resultImage, 
		g_resultImage, 
		0, 
		1, 
		NORM_MINMAX, 
		-1, 
		Mat());

	//【4】通过函数 minMaxLoc 定位最匹配的位置
	double minValue; double maxValue; Point minLocation; Point maxLocation;
	Point matchLocation;
	minMaxLoc(
		g_resultImage, 
		&minValue, 
		&maxValue, 
		&minLocation, 
		&maxLocation, 
		Mat());

	//【5】对于方法 SQDIFF 和 SQDIFF_NORMED, 越小的数值有着更高的匹配结果. 而其余的方法, 数值越大匹配效果越好
	if (g_nMatchMethod == TM_SQDIFF || g_nMatchMethod == TM_SQDIFF_NORMED)
	{
		matchLocation = minLocation;
	}
	else
	{
		matchLocation = maxLocation;
	}

	//【6】绘制出矩形，并显示最终结果
	rectangle(
		srcImage, 
		matchLocation, 
		Point(matchLocation.x + g_templateImage.cols, matchLocation.y + g_templateImage.rows), 
		Scalar(0, 0, 255), 
		2, 
		8, 
		0);
	rectangle(
		g_resultImage, 
		matchLocation, 
		Point(matchLocation.x + g_templateImage.cols, matchLocation.y + g_templateImage.rows), 
		Scalar(0, 0, 255), 
		2, 
		8, 
		0);

	imshow(WINDOW_NAME1, srcImage);
	imshow(WINDOW_NAME2, g_resultImage);

}

static void ShowHelpText() {
	printf("\n\n\t\t\t   当前使用的OpenCV版本为：" CV_VERSION);

	printf("\t欢迎来到【模板匹配】示例程序~\n");
	printf("\n\t滑动条对应的方法数值说明: \n\n"
		"\t\t方法【0】- 平方差匹配法(SQDIFF)\n"
		"\t\t方法【1】- 归一化平方差匹配法(SQDIFF NORMED)\n"
		"\t\t方法【2】- 相关匹配法(TM CCORR)\n"
		"\t\t方法【3】- 归一化相关匹配法(TM CCORR NORMED)\n"
		"\t\t方法【4】- 相关系数匹配法(TM COEFF)\n"
		"\t\t方法【5】- 归一化相关系数匹配法(TM COEFF NORMED)\n");
}
