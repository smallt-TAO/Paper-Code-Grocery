#include<opencv2/core/core.hpp>  
#include<opencv2/highgui/highgui.hpp>  
#include<iostream>  

using namespace std;
using namespace cv;

int main()
{
	//1.����Ҷ�ͼ����ʾ  
	Mat srcImage = imread("bookmark.jpg", 0);
	if (!srcImage.data)
	{
		printf("��ȡͼƬ������ȷ��Ŀ¼���Ƿ���imread����ָ��ͼƬ����!\n");
		return false;
	}
	imshow("ԭͼ�Ҷ�ͼ   ", srcImage);

	//2.��ͼ����չ����ѳߴ磬�߽���0����  
	int m = getOptimalDFTSize(srcImage.rows);//�õ��������  
	int n = getOptimalDFTSize(srcImage.cols);//�õ��������  
	cout << "ԭ�Ҷ�ͼ�ߴ�:" << srcImage.cols << " X " << srcImage.rows << endl;
	cout << "�Ż���ͼƬ�ߴ�:" << n << " X " << m << endl;
	int delta1 = (m - srcImage.rows) / 2;
	int delta2 = (n - srcImage.cols) / 2;


	Mat padded;//���ڴ洢�Ż�����ͼ��  
	copyMakeBorder(srcImage, padded, delta1, delta1, delta2, delta2, BORDER_CONSTANT, Scalar::all(0));//��ʼ��ͼ�  
	imshow("��ѳߴ�ͼ", padded);//��ʾ�Ż��ͼƬ  

	//3.Ϊ����Ҷ�任���ʵ�����鲿����洢�ռ�  
	//��planes������Ϻϲ���һ����ͨ��������complextI  
	Mat planes[] = { Mat_<float>(padded), Mat::zeros(padded.size(), CV_32F) };//������ͼ��ת���ɸ�������  
	Mat complexI;//�洢��������  
	merge(planes, 2, complexI);

	//4.���о͵���ɢ����Ҷ�任  
	dft(complexI, complexI);

	//5.������ת��Ϊ��ֵ  
	split(complexI, planes);
	magnitude(planes[0], planes[1], planes[0]);
	Mat magnitudeImage = planes[0];

	//6.���е���߶�����  
	magnitudeImage = magnitudeImage + Scalar::all(1);
	log(magnitudeImage, magnitudeImage);

	//7.���к��طֲ�����ͼ���ޣ����������У��У�����Ƶ�׼���  
	magnitudeImage = magnitudeImage(Rect(0, 0, magnitudeImage.cols & -2, magnitudeImage.rows & -2));
	int cx = magnitudeImage.cols / 2;
	int cy = magnitudeImage.rows / 2;
	//�����������ޣ�ʹ��Զ��λ������  
	Mat q0(magnitudeImage, Rect(0, 0, cx, cy));
	Mat q1(magnitudeImage, Rect(cx, 0, cx, cy));
	Mat q2(magnitudeImage, Rect(0, cy, cx, cy));
	Mat q3(magnitudeImage, Rect(cx, cy, cx, cy));

	Mat temp;
	//���Ϻ����µ���  
	q0.copyTo(temp);
	q3.copyTo(q0);
	temp.copyTo(q3);
	//���º����ϵ���  
	q1.copyTo(temp);
	q2.copyTo(q1);
	temp.copyTo(q2);

	//��һ��  
	normalize(magnitudeImage, magnitudeImage, 0, 1, NORM_MINMAX);
	normalize(magnitudeImage, magnitudeImage, 0, 1, NORM_MINMAX);

	imshow("Ƶ�׷�ֵ", magnitudeImage);
	//while (char(waitKey(1))!='q') { }//����q���˳�  
	waitKey(0);
	return 0;
}