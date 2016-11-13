#include<opencv2/opencv.hpp>  
#include<opencv2/imgproc/imgproc.hpp>  

using namespace cv;
using namespace std;

#define NameWindow "��0��ͼƬ����Ч������"  
void imshowMany(const std::string& _winName, const vector<Mat>& ployImages);//  

int main()
{

	Mat srcImage = imread("building.jpg");
	imshow("��1��ԭͼ", srcImage);
	Mat  temImage, grayImage, midImage, edgeImage, dstImage;

	cvtColor(srcImage, grayImage, COLOR_BGR2GRAY);//ת��Ϊ�Ҷ�ͼ  
	blur(grayImage, grayImage, Size(5, 5), Point(-1, -1));//��ֵģ��  
	//GaussianBlur(grayImage, grayImage, Size(5, 5), 0, 0);  
	imshow("��2��ԭͼ�Ҷ�ͼ+��ֵģ��", grayImage);
	cvtColor(grayImage, midImage, COLOR_GRAY2BGR);//תΪ��ͨ��ͼ,���ڶ�ͼ��ʾ��λ�ã�1��2��  
	Canny(grayImage, edgeImage, 50, 100, 3);//tempImageΪ��ͨ����ֵͼ���ڶ�ͼ�����в�����ʾ  
	imshow("��3���Ҷ�ͼCanny��Ե���", edgeImage);
	cvtColor(edgeImage, temImage, COLOR_GRAY2BGR);//תΪ��ͨ��ͼ�����ڶ�ͼ��ʾ��λ�ã�2��1��  
	cvtColor(edgeImage, dstImage, COLOR_GRAY2BGR);//תΪ��ͨ��ͼ�����ڶ�ͼ��ʾ��λ�ã�2��2��  

	vector<Vec2f>lines;//ʸ���ṹlines���ڴ�ŵõ����߶�ʸ������  
	HoughLines(edgeImage, lines, 1, CV_PI / 180, 123, 0, 0);//��ֵ��ѡ��Խ��Ӱ��ܴ�  

	for (size_t i = 0; i < lines.size(); i++)//��ͼ�л��Ƴ�ÿ���߶�  
	{
		float rho = lines[i][0], theta = lines[i][1];
		Point pt1, pt2;
		double a = cos(theta), b = sin(theta);
		double x0 = a*rho, y0 = b*rho;
		pt1.x = cvRound(x0 + 1000 * (-b));
		pt1.y = cvRound(y0 + 1000 * (a));
		pt2.x = cvRound(x0 - 1000 * (-b));
		pt2.y = cvRound(y0 - 1000 * (a));
		line(dstImage, pt1, pt2, Scalar(255, 0, 255), 1, 8);//������ɫΪ��ɫ  
	}
	imshow("��4������canny��Houghֱ�߱任", dstImage);


	vector<Mat>Images(4);//ģ����vector�����ڷ���4������ΪMat��Ԫ�أ�������ͼƬ  
	Images[0] = srcImage;
	Images[1] = midImage;
	Images[2] = temImage;
	Images[3] = dstImage;


	namedWindow(NameWindow);
	imshowMany(NameWindow, Images);//����һ��������ʾ��ͼ����  

	waitKey(0);
	return 0;

}

//�Զ���һ��������ʾ��ͼ����  
void imshowMany(const std::string& _winName, const vector<Mat>& ployImages)
{
	int nImg = (int)ployImages.size();//��ȡ��ͬһ��������ʾ��ͼ����Ŀ  

	Mat dispImg;

	int size;
	int x, y;
	//��Ҫ��OpenCVʵ��ͬһ������ʾ���ͼƬ��ͼƬҪ������ʽ���У�������Matlab��subplot();     
	//w����ͼ���������е�����  ��h: ��ͼ���������еĵ���    
	int w, h;

	float scale;//���ű���  
	int max;

	if (nImg <= 0)
	{
		printf("Number of arguments too small....\n");
		return;
	}
	else if (nImg > 12)
	{
		printf("Number of arguments too large....\n");
		return;
	}

	else if (nImg == 1)
	{
		w = h = 1;
		size = 400;
	}
	else if (nImg == 2)
	{
		w = 2; h = 1;//2x1  
		size = 400;
	}
	else if (nImg == 3 || nImg == 4)
	{
		w = 2; h = 2;//2x2  
		size = 400;
	}
	else if (nImg == 5 || nImg == 6)
	{
		w = 3; h = 2;//3x2  
		size = 300;
	}
	else if (nImg == 7 || nImg == 8)
	{
		w = 4; h = 2;//4x2  
		size = 300;
	}
	else
	{
		w = 4; h = 3;//4x3  
		size = 200;
	}

	dispImg.create(Size(100 + size*w, 30 + size*h), CV_8UC3);//����ͼƬ����w*h���������������ߵ�ͼƬ����Ϊw*h  

	for (int i = 0, m = 20, n = 20; i<nImg; i++, m += (20 + size))
	{
		x = ployImages[i].cols;   //��(i+1)����ͼ��Ŀ��(����)  
		y = ployImages[i].rows;//��(i+1)����ͼ��ĸ߶ȣ�������  

		max = (x > y) ? x : y;//�Ƚ�ÿ��ͼƬ��������������ȡ��ֵ  
		scale = (float)((float)max / size);//�������ű���  

		if (i%w == 0 && m != 20)
		{
			m = 20;
			n += 20 + size;
		}

		Mat imgROI = dispImg(Rect(m, n, (int)(x / scale), (int)(y / scale)));   //�ڻ���dispImage�л���ROI����  
		resize(ployImages[i], imgROI, Size((int)(x / scale), (int)(y / scale))); //��Ҫ��ʾ��ͼ������ΪROI�����С  
	}
	namedWindow(_winName);
	imshow(_winName, dispImg);
}