# include <opencv2/core/core.hpp>
# include <opencv2/highgui/highgui.hpp>
# include <opencv2/imgproc/imgproc.hpp>
# include <iostream>

using namespace std;
using namespace cv;

Mat g_srcImage;
Mat g_dstImage1;
Mat g_dstImage2;
Mat g_dstImage3;
int g_nBoxFilterValue = 3;
int g_nBlurValue = 3;
int g_nGaussianFilterValue = 3;

static void on_BoxFilter(int, void*);
static void on_Blur(int, void*);
static void on_Gaussian(int, void*);

int main() {
	system("color 7F");
	g_srcImage = imread("example001.jpg", 1);
	if (!g_srcImage.data) {
		cout << "Load srcImage is Error" << endl;
		return false;
	}
	imshow("Origin Image", g_srcImage);
	g_dstImage1 = g_srcImage.clone();
	g_dstImage2 = g_srcImage.clone();
	g_dstImage3 = g_srcImage.clone();

	namedWindow("BlurFilter", 1);
	createTrackbar("Kernal Num", "BlurFilter", &g_nBlurValue, 20, on_Blur);
	on_Blur(g_nBlurValue, 0);

	namedWindow("BoxFilter", 1);
	createTrackbar("Kernal Num", "BoxFilter", &g_nBoxFilterValue, 20, on_BoxFilter);
	on_BoxFilter(g_nBoxFilterValue, 0);

	namedWindow("GaussianFilter", 1);
	createTrackbar("kernal Num", "GaussianFilter", &g_nGaussianFilterValue, 20, on_Gaussian);
	on_Gaussian(g_nGaussianFilterValue, 0);

	cout << "All is Well" << endl;
	cout << "Come from the Smalltao" << endl;
	waitKey(0);
	return 0;
}

static void on_Blur(int, void*) {
	blur(g_srcImage, g_dstImage1, Size(g_nBlurValue + 1, g_nBlurValue + 1), Point(-1, -1));
	imshow("BlurFilter", g_dstImage1);
}

static void on_BoxFilter(int, void*) {
	boxFilter(g_srcImage, g_dstImage2, -1,
		Size(g_nBoxFilterValue + 1, g_nBoxFilterValue + 1), Point(-1, -1),
		true, 4);
	imshow("BoxFilter", g_dstImage2);
}

static void on_Gaussian(int, void*) {
	GaussianBlur(g_srcImage, g_dstImage3,
		Size(2 * g_nGaussianFilterValue + 1, 2 * g_nGaussianFilterValue + 1),
		0, 0, 4);
	imshow("GaussianFilter", g_dstImage3);
}