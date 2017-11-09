#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>

using namespace cv;
using namespace std;

#define WINDOW_NAME1 "Code One"        
#define WINDOW_NAME2 "Water Shed Image Result"        

Mat g_maskImage, g_srcImage;
Point prevPt(-1, -1);

static void ShowHelpText();
static void on_Mouse(int event, int x, int y, int flags, void*);

int main(int argc, char** argv) {
	system("color 6F");
	ShowHelpText();

	g_srcImage = imread("1.jpg", 1);
	imshow(WINDOW_NAME1, g_srcImage);
	Mat srcImage, grayImage;
	g_srcImage.copyTo(srcImage);
	cvtColor(g_srcImage, g_maskImage, COLOR_BGR2GRAY);
	cvtColor(g_maskImage, grayImage, COLOR_GRAY2BGR);
	g_maskImage = Scalar::all(0);

	setMouseCallback(WINDOW_NAME1, on_Mouse, 0);

	while (1) {
		int c = waitKey(0);

		if ((char)c == 27)
			break;

		if ((char)c == '2') {
			g_maskImage = Scalar::all(0);
			srcImage.copyTo(g_srcImage);
			imshow("image", g_srcImage);
		}

		if ((char)c == '1' || (char)c == ' ') {
			int i, j, compCount = 0;
			vector<vector<Point> > contours;
			vector<Vec4i> hierarchy;

			findContours(
				g_maskImage, 
				contours, 
				hierarchy, 
				RETR_CCOMP, 
				CHAIN_APPROX_SIMPLE);

			if (contours.empty())
				continue;

			Mat maskImage(g_maskImage.size(), CV_32S);
			maskImage = Scalar::all(0);

			for (int index = 0; index >= 0; index = hierarchy[index][0], compCount++)
				drawContours(
				maskImage, 
				contours, 
				index, 
				Scalar::all(compCount + 1), 
				-1, 
				8, 
				hierarchy, 
				INT_MAX);

			if (compCount == 0)
				continue;

			vector<Vec3b> colorTab;
			for (i = 0; i < compCount; i++) {
				int b = theRNG().uniform(0, 255);
				int g = theRNG().uniform(0, 255);
				int r = theRNG().uniform(0, 255);

				colorTab.push_back(Vec3b((uchar)b, (uchar)g, (uchar)r));
			}

			double dTime = (double)getTickCount();
			watershed(srcImage, maskImage);
			dTime = (double)getTickCount() - dTime;
			printf("\thandle Time = %gms\n", dTime*1000. / getTickFrequency());

			Mat watershedImage(maskImage.size(), CV_8UC3);
			for (i = 0; i < maskImage.rows; i++)
			for (j = 0; j < maskImage.cols; j++) {
				int index = maskImage.at<int>(i, j);
				if (index == -1)
					watershedImage.at<Vec3b>(i, j) = Vec3b(255, 255, 255);
				else if (index <= 0 || index > compCount)
					watershedImage.at<Vec3b>(i, j) = Vec3b(0, 0, 0);
				else
					watershedImage.at<Vec3b>(i, j) = colorTab[index - 1];
			}

			watershedImage = watershedImage*0.5 + grayImage*0.5;
			imshow(WINDOW_NAME2, watershedImage);
		}
	}

	return 0;
}


static void on_Mouse(int event, int x, int y, int flags, void*) {

	if (x < 0 || x >= g_srcImage.cols || y < 0 || y >= g_srcImage.rows)
		return;

	if (event == EVENT_LBUTTONUP || !(flags & EVENT_FLAG_LBUTTON))
		prevPt = Point(-1, -1);
	else if (event == EVENT_LBUTTONDOWN)
		prevPt = Point(x, y);

	else if (event == EVENT_MOUSEMOVE && (flags & EVENT_FLAG_LBUTTON)) {
		Point pt(x, y);
		if (prevPt.x < 0)
			prevPt = pt;
		line(g_maskImage, prevPt, pt, Scalar::all(255), 5, 8, 0);
		line(g_srcImage, prevPt, pt, Scalar::all(255), 5, 8, 0);
		prevPt = pt;
		imshow(WINDOW_NAME1, g_srcImage);
	}
}


static void ShowHelpText() {
	printf("当前使用的OpenCV版本为：" CV_VERSION);
	printf("\n");
  
	printf("\n\n\n\t欢迎来到【分水岭算法】示例程序~\n\n");
	printf("\t请先用鼠标在图片窗口中标记出大致的区域，\n\n\t然后再按键【1】或者【SPACE】启动算法。"
		"\n\n\t按键操作说明: \n\n"
		"\t\t键盘按键【1】或者【SPACE】- 运行的分水岭分割算法\n"
		"\t\t键盘按键【2】- 恢复原始图片\n"
		"\t\t键盘按键【ESC】- 退出程序\n\n\n");
}
