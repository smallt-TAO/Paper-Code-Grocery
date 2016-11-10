# include <opencv2/core/core.hpp>
# include <opencv2/highgui/highgui.hpp>
# include "opencv2/imgproc/imgproc.hpp"
# include <iostream> 

using namespace cv;

# define WINDOW_NAME "AUDI"
# define WINDOW_NAME1 "TOYOTA"
# define WINDOW_NAME2 "RED"
# define WINDOW_NAME3 "Wuziqi"

int main(int argc, char ** argv[]) {
	// AUDI logo
	Mat image = Mat::zeros(600, 850, CV_8UC3);
	circle(image, Point(200, 300), 100, Scalar(225, 0, 255), 7, 8);
	circle(image, Point(350, 300), 100, Scalar(255, 0, 255), 7, 8);
	circle(image, Point(500, 300), 100, Scalar(255, 0, 255), 7, 8);
	circle(image, Point(650, 300), 100, Scalar(255, 0, 255), 7, 8);

	imshow(WINDOW_NAME, image);

	// TOYOTA logo
	Mat image1 = Mat::zeros(900, 900, CV_8UC3);
	ellipse(image1, Point(450, 450), Size(400, 250), 0, 0, 360, Scalar(0, 0, 225), 5, 8);
	ellipse(image1, Point(450, 450), Size(250, 110), 90, 0, 360, Scalar(0, 0, 225), 5, 8);
	ellipse(image1, Point(450, 320), Size(280, 120), 0, 0, 360, Scalar(0, 0, 255), 5, 8);

	imshow(WINDOW_NAME1, image1);

	Mat image3 = Mat::zeros(800, 800, CV_8UC3); 
	Rect rec1 = Rect(100, 300, 600, 200);
	Rect rec2 = Rect(300, 100, 200, 600);
	rectangle(image3, rec1, Scalar(0, 0, 255), -1, 8, 0); 
	rectangle(image3, rec2, Scalar(0, 0, 255), -1, 8, 0); 
	rectangle(image3, Point(100, 300), Point(700, 500), Scalar(0, 255, 255), 2, 8, 0);  
	rectangle(image3, Point(300, 100), Point(500, 700), Scalar(0, 255, 255), 2, 8, 0);
	rectangle(image3, Point(300, 300), Point(500, 500), Scalar(0, 0, 255), 3, 8); 
	
	imshow(WINDOW_NAME2, image3);

	Mat image4(600, 600, CV_8UC3, Scalar(255, 70, 0));
	for (int i = 0; i < 20; i++) {
		line(image4, Point(i * 30, 0), Point(i * 30, 600), Scalar(0, 0, 0), 1.5);
		line(image4, Point(0, i * 30), Point(600, i * 30), Scalar(0, 0, 0), 1.5);
	}
	putText(image4, "VolksWagen", Point(260, 450), FONT_HERSHEY_COMPLEX_SMALL, 2, Scalar(0, 0, 0));
	imshow(WINDOW_NAME3, image4);

	waitKey(0);
	return 0;
}