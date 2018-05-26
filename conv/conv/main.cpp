#include <opencv2/opencv.hpp>
#include <iostream>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

using namespace std;
using namespace cv;


// görüntü dýþýnda kalan piksel aþaðýdaki algoritmayla yansýtýlýyor
int reflect(int M, int x)
{
	if (x < 0)
	{
		return -x - 1;
	}
	if (x >= M)
	{
		return 2 * M - x - 1;
	}
	return x;
}
// görüntü sýnýrý aþýldýðýnda karþý tarafa geç
int circular(int M, int x)
{
	if (x<0)
		return x + M;
	if (x >= M)
		return x - M;
	return x;
}

// sýnýrdaki piksellerin ihmal edikdiði fonksiyon
/*void noBorderProcessing(Mat src, Mat dst, float Kernel[][3])
{

	float sum;
	for (int y = 1; y < src.rows - 1; y++) {
		for (int x = 1; x < src.cols - 1; x++) {
			sum = 0.0;
			for (int k = 0; k <= 2; k++) {
				for (int j = 0; j <= 2; j++) {
					sum = sum + Kernel[j][k] * src.at<uchar>(y + j, x + k);
				}
			}
			if (sum>255) sum = 255;
			if (sum<0)sum = 0;
			dst.at<uchar>(y, x) = sum;
		}
	}
}*/

// yansýtarak indeksleme
void refletedIndexing(Mat src, Mat dst, float Kernel[][3])
{
	float sum, x1, y1;
	for (int y = 0; y < src.rows; y++) {
		for (int x = 0; x < src.cols; x++) {
			sum = 0.0;
			for (int k = -1; k <= 1; k++) {
				for (int j = -1; j <= 1; j++) {
					x1 = reflect(src.cols, x - j);
					y1 = reflect(src.rows, y - k);
					sum = sum + Kernel[j + 1][k + 1] * src.at<uchar>(y1, x1);
				}
			}
			if (sum>255) sum = 255;
			if (sum<0)sum = 0;
			dst.at<uchar>(y, x) = sum;
		}
	}
}

//dairesel indeksleme
void circularIndexing(Mat src, Mat dst, float Kernel[][3])
{
	float sum, x1, y1;
	for (int y = 0; y < src.rows; y++) {
		for (int x = 0; x < src.cols; x++) {
			sum = 0.0;
			for (int k = -1; k <= 1; k++) {
				for (int j = -1; j <= 1; j++) {
					x1 = circular(src.cols, x - j);
					y1 = circular(src.rows, y - k);
					sum = sum + Kernel[j + 1][k + 1] * src.at<uchar>(y1, x1);
				}
			}
			if (sum>255) sum = 255;
			if (sum<0)sum = 0;
			dst.at<uchar>(y, x) = sum;
		}
	}
}



int main()
{

	Mat src, dst;


	/// yüklediðimiz resmi gri resme çeviriyoruz
	src = imread("h.jpg", CV_LOAD_IMAGE_GRAYSCALE);

	if (!src.data)//resim yükleme hatasý
	{
		return -1;
	}

	namedWindow("orjinal resim");
	imshow("orjinal resim", src);

	dst = src.clone(); //ilk resim boyutu kadar boþ bi resim oluþturuyoruz
	for (int y = 0; y < src.rows; y++)
		for (int x = 0; x < src.cols; x++)
			dst.at<uchar>(y, x) = 0.0;

	//kerneller

	//yumuþaklaþtýrma
	float yumusak[3][3] = {
		{ 1 / 9.0, 1 / 9.0, 1 / 9.0 },
		{ 1 / 9.0, 1 / 9.0, 1 / 9.0 },
		{ 1 / 9.0, 1 / 9.0, 1 / 9.0 }
	};


	circularIndexing(src, dst, yumusak);


	namedWindow("yumusak");
	imshow("yumusak", dst);

	//sol sobel

	float sobelSol[3][3] = {
		{ -1, -2, -1 },
		{ 0, 0, 0 },
		{ 1, 2, 1 }
	};


	circularIndexing(src, dst, sobelSol);


	namedWindow("Sol Sobel");
	imshow("Sol Sobel", dst);

	//sag sobel
	float sobelSag[3][3] = {
		{ -1, 0, 1 },
		{ -2, 0, 2 },
		{ -1, 0, 1 }
	};


	circularIndexing(src, dst, sobelSag);


	namedWindow("Sag Sobel");
	imshow("Sag Sobel", dst);

	// keskinleþtirme
	float keskin[3][3] = {
		{ 0, -1.0, 0 },
		{ -1.0, 5.0, -1.0 },
		{ 0, -1.0, 0 }
	};


	// cvFilter2D(src, dst2, deneme);

	circularIndexing(src, dst, keskin);


	namedWindow("keskinlestirme");
	imshow("keskinlestirme", dst);

	// kenar
	float kenar[3][3] = {
		{ -1.0, -1.0, -1.0 },
		{ -1.0, 8.0, -1.0 },
		{ -1.0, -1.0, -1.0 }
	};


	circularIndexing(src, dst, kenar);

	namedWindow("kenar buldurma");
	imshow("kenar buldurma", dst);
	
	// gauss
	float gauss[3][3] = {
		{ 1 / 16.0, 1 / 8.0, 1 / 16.0 },
		{ 1 / 8.0, 1 / 4.0, 1 / 8.0 },
		{ 1 / 16.0, 1 / 8.0, 1 / 16.0 }
	};

	circularIndexing(src, dst, gauss);

	namedWindow("gauss");
	imshow("gauss", dst);
	
	// prewitt
	float prewitt[3][3] = {
		{ -1, 0, 1 },
		{ -1, 0, 1 },
		{ -1, 0, 1 }
	};

	circularIndexing(src, dst, prewitt);

	namedWindow("prewitt");
	imshow("prewitt", dst);
	
	// laplace
	float laplace[3][3] = {
		{ 0, -1, 0 },
		{ -1, 4, -1 },
		{ 0, -1, 0 }
	};

	circularIndexing(src, dst, laplace);

	namedWindow("laplace");
	imshow("laplace", dst);
	

	//bekle
	waitKey();

	return 0;
}
