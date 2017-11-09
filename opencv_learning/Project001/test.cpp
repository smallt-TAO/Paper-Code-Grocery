#include <opencv2/core/core.hpp>  
#include <opencv2/highgui/highgui.hpp>  
#include <iostream>  

using namespace cv;
using namespace std;

int main() {
	Mat image;
	Mat image1;
	// image = imread("example001.jpg", IMREAD_COLOR);
	// image = imread("example001.jpg", 2 | 4);
	// image = imread("example001.jpg", 0);
	image1 = imread("example001.jpg", 0);
	image = imread("example002.jpg", 199);

	if (!image.data) {                                 // Check for invalid input  
		cout << "Could not open or find the image" << std::endl;
		return -1;
	}

	namedWindow("Display window", WINDOW_AUTOSIZE);    // Create a window for display.  
	imshow("Display window", image);                   // Show our image inside it. 

	namedWindow("Show time", WINDOW_AUTOSIZE);
	imshow("Show time", image1);

	waitKey(0);                                        // Wait for a keystroke in the window  
	return 0;
}