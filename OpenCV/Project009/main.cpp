#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace cv;
using namespace std;

# define NameWindow "Hanled Image"

void imshowMany(const std::string& _winName, const vector<Mat>& ployImages);

int main() {
	Mat srcImage = imread("20.jpg");
	imshow("Origin Image", srcImage);
	Mat temImage, grayImage, midImage, edgeImage, dstImage;
	cvtColor(srcImage, grayImage, COLOR_RGB2GRAY);
	// blur(grayImage, grayImage, Size(5, 5), Point(-1, -1));
	GaussianBlur(grayImage, grayImage, Size(5, 5), 0, 0);
	imshow("Gray image + blurFilter", grayImage);
	cvtColor(grayImage, midImage, COLOR_GRAY2RGB);
	Canny(grayImage, edgeImage, 50, 100, 3);
	imshow("Gray Image + Canny", edgeImage);
	cvtColor(edgeImage, temImage, COLOR_GRAY2RGB);
	cvtColor(edgeImage, dstImage, COLOR_GRAY2RGB);
	imshow("Gray Image + Dst", dstImage);
	
	vector<Vec4f> lines;  // Vector Structure for lines vector.
	// HoughLines(edgeImage, lines, 1, CV_PI / 180, 200, 0, 0);
	cv::HoughLinesP(edgeImage, lines, 1, CV_PI / 180, 50, 60, 5);

	for (size_t i = 0; i < lines.size(); i++) {
		Vec4f l = lines[i];
		line(edgeImage, Point(l[0], l[1]), Point(l[2], l[3]), Scalar(0, 0, 255), 3, 8);
	}
	imshow("Canny + Hoough", dstImage);
	
	vector<Mat> Images(4);
	Images[0] = srcImage;
	Images[1] = midImage;
	Images[2] = temImage;
	Images[3] = dstImage;

	// namedWindow(NameWindow);
	// imshowMany(NameWindow, Images);

	waitKey();
	return 0;
}

void imshowMany(const std::string& _winName, const vector<Mat>& ployImages) {
	int nImg = (int)ployImages.size();  

	Mat dispImg;
	int size;
	int x, y; 
	int w, h;

	float scale; 
	int max;

	if (nImg <= 0) {
		printf("Number of arguments too small....\n");
		return;
	}
	else if (nImg > 12) {
		printf("Number of arguments too large....\n");
		return;
	}

	else if (nImg == 1) {
		w = h = 1;
		size = 400;
	}
	else if (nImg == 2) {
		w = 2; h = 1; 
		size = 400;
	}
	else if (nImg == 3 || nImg == 4) {
		w = 2; h = 2; 
		size = 400;
	}
	else if (nImg == 5 || nImg == 6) {
		w = 3; h = 2;//3x2  
		size = 300;
	}
	else if (nImg == 7 || nImg == 8) {
		w = 4; h = 2;//4x2  
		size = 300;
	}
	else {
		w = 4; h = 3;//4x3  
		size = 200;
	}

	dispImg.create(Size(80 + size*w, 30 + size*h), CV_8UC3);

	for (int i = 0, m = 20, n = 20; i<nImg; i++, m += (20 + size)) {
		x = ployImages[i].cols;     
		y = ployImages[i].rows; 

		max = (x > y) ? x : y;  
		scale = (float)((float)max / size);  

		if (i%w == 0 && m != 20) {
			m = 20;
			n += 20 + size;
		}

		Mat imgROI = dispImg(Rect(m, n, (int)(x / scale), (int)(y / scale)));    
		resize(ployImages[i], imgROI, Size((int)(x / scale), (int)(y / scale))); 
	}
	namedWindow(_winName);
	imshow(_winName, dispImg);
}