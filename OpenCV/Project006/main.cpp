#include<opencv2/core/core.hpp>  
#include<opencv2/highgui/highgui.hpp>  
#include<iostream>  

using namespace std;
using namespace cv;

int main()
{
	//1.载入灰度图并显示  
	Mat srcImage = imread("bookmark.jpg", 0);
	if (!srcImage.data)
	{
		printf("读取图片错误，请确定目录下是否有imread函数指定图片存在!\n");
		return false;
	}
	imshow("原图灰度图   ", srcImage);

	//2.将图像延展到最佳尺寸，边界用0补充  
	int m = getOptimalDFTSize(srcImage.rows);//得到最佳列数  
	int n = getOptimalDFTSize(srcImage.cols);//得到最佳行数  
	cout << "原灰度图尺寸:" << srcImage.cols << " X " << srcImage.rows << endl;
	cout << "优化后图片尺寸:" << n << " X " << m << endl;
	int delta1 = (m - srcImage.rows) / 2;
	int delta2 = (n - srcImage.cols) / 2;


	Mat padded;//用于存储优化填补后的图像  
	copyMakeBorder(srcImage, padded, delta1, delta1, delta2, delta2, BORDER_CONSTANT, Scalar::all(0));//开始环图填补  
	imshow("最佳尺寸图", padded);//显示优化填补图片  

	//3.为傅里叶变换后的实部和虚部分配存储空间  
	//将planes数组组合合并成一个多通道的数组complextI  
	Mat planes[] = { Mat_<float>(padded), Mat::zeros(padded.size(), CV_32F) };//将输入图像转换成浮点类型  
	Mat complexI;//存储复数部分  
	merge(planes, 2, complexI);

	//4.进行就地离散傅里叶变换  
	dft(complexI, complexI);

	//5.将复数转换为幅值  
	split(complexI, planes);
	magnitude(planes[0], planes[1], planes[0]);
	Mat magnitudeImage = planes[0];

	//6.进行地鼠尺度缩放  
	magnitudeImage = magnitudeImage + Scalar::all(1);
	log(magnitudeImage, magnitudeImage);

	//7.剪切和重分布幅度图像限，若有奇数列，行，进行频谱剪裁  
	magnitudeImage = magnitudeImage(Rect(0, 0, magnitudeImage.cols & -2, magnitudeImage.rows & -2));
	int cx = magnitudeImage.cols / 2;
	int cy = magnitudeImage.rows / 2;
	//重新排列象限，使得远点位于中心  
	Mat q0(magnitudeImage, Rect(0, 0, cx, cy));
	Mat q1(magnitudeImage, Rect(cx, 0, cx, cy));
	Mat q2(magnitudeImage, Rect(0, cy, cx, cy));
	Mat q3(magnitudeImage, Rect(cx, cy, cx, cy));

	Mat temp;
	//左上和右下调换  
	q0.copyTo(temp);
	q3.copyTo(q0);
	temp.copyTo(q3);
	//左下和右上调换  
	q1.copyTo(temp);
	q2.copyTo(q1);
	temp.copyTo(q2);

	//归一化  
	normalize(magnitudeImage, magnitudeImage, 0, 1, NORM_MINMAX);
	normalize(magnitudeImage, magnitudeImage, 0, 1, NORM_MINMAX);

	imshow("频谱幅值", magnitudeImage);
	//while (char(waitKey(1))!='q') { }//按下q键退出  
	waitKey(0);
	return 0;
}