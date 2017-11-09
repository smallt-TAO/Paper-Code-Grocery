# include <opencv2/core/core.hpp>
# include <opencv2/highgui/highgui.hpp>
# include <iostream>

using namespace std;
using namespace cv;

bool MultiChannelBlending();
void ShowHelpText();

int main() {
	system("color 8F");
	if (MultiChannelBlending()) {
		cout << endl << "Image blending Success" << endl;
	}
	while (char(waitKey(1)) != 'q') {}

	return 0;
}

bool MultiChannelBlending() {
	Mat logoWH = imread("whlogo.jpg", 0);
	Mat logoHZ = imread("hzlogo.jpg", 0);
	Mat srcImage = imread("scenery.jpg");
	vector<Mat> channels;  // vector in std
	Mat imageBlueChannel;
	Mat imageGreenChannel;
	Mat imageRedChannel;
	if (!logoWH.data)
		cout << "Read the logoWH is wrong." << endl;
	else if (!logoHZ.data)
		cout << "Read the logoHZ is wrong." << endl;
	else if (!srcImage.data)
		cout << "Read the srcImage is wrong." << endl;
	else
		cout << "Load Image Success." << endl;

	split(srcImage, channels);
	imageBlueChannel = channels.at(0);
	imageGreenChannel = channels.at(1);
	imageRedChannel = channels.at(2);
	
	namedWindow("[1]BlueChannel", 1);
	imshow("[1]BlueChannel", srcImage);
	//imshow("[2]GreenChannel", imageGreenChannel);
	//imshow("[3]RedChannel", imageRedChannel);

	/*
	addWeighted(
	imageRedChannel(Rect(0, 0, logoWH.cols, logoWH.rows)),
	1.0,
	logoWH,
	0.3,
	0,
	imageRedChannel(Rect(0, 0, logoWH.cols, logoWH.rows)));
	imshow("RedChannel + logoWH", imageRedChannel);


	addWeighted(
	imageGreenChannel(Rect(2, 10, logoHZ.cols, logoHZ.rows)),
	1.0,
	logoHZ,
	0.3,
	0,
	imageGreenChannel(Rect(2, 10, logoHZ.cols, logoHZ.rows)));
	imshow("GreenChannel + logoHZ", imageGreenChannel);

	Mat mergedImage(srcImage.rows, srcImage.cols, CV_8UC3, Scalar(0));
	merge(channels, mergedImage);
	imshow("Channels Merged", mergedImage);
	*/

	return 0;
}