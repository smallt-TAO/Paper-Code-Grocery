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

	//��4��ͨ������ minMaxLoc ��λ��ƥ���λ��
	double minValue; double maxValue; Point minLocation; Point maxLocation;
	Point matchLocation;
	minMaxLoc(
		g_resultImage, 
		&minValue, 
		&maxValue, 
		&minLocation, 
		&maxLocation, 
		Mat());

	//��5�����ڷ��� SQDIFF �� SQDIFF_NORMED, ԽС����ֵ���Ÿ��ߵ�ƥ����. ������ķ���, ��ֵԽ��ƥ��Ч��Խ��
	if (g_nMatchMethod == TM_SQDIFF || g_nMatchMethod == TM_SQDIFF_NORMED)
	{
		matchLocation = minLocation;
	}
	else
	{
		matchLocation = maxLocation;
	}

	//��6�����Ƴ����Σ�����ʾ���ս��
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
	printf("\n\n\t\t\t   ��ǰʹ�õ�OpenCV�汾Ϊ��" CV_VERSION);

	printf("\t��ӭ������ģ��ƥ�䡿ʾ������~\n");
	printf("\n\t��������Ӧ�ķ�����ֵ˵��: \n\n"
		"\t\t������0��- ƽ����ƥ�䷨(SQDIFF)\n"
		"\t\t������1��- ��һ��ƽ����ƥ�䷨(SQDIFF NORMED)\n"
		"\t\t������2��- ���ƥ�䷨(TM CCORR)\n"
		"\t\t������3��- ��һ�����ƥ�䷨(TM CCORR NORMED)\n"
		"\t\t������4��- ���ϵ��ƥ�䷨(TM COEFF)\n"
		"\t\t������5��- ��һ�����ϵ��ƥ�䷨(TM COEFF NORMED)\n");
}
