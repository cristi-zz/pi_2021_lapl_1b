// OpenCVApplication.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include "common.h"
#include <random>
#include <vector>
#include <iostream>
#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

void testOpenImage()
{
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat src;
		src = imread(fname);
		imshow("opened image", src);
		waitKey();
	}
}

void testOpenImagesFld()
{
	char folderName[MAX_PATH];
	if (openFolderDlg(folderName) == 0)
		return;
	char fname[MAX_PATH];
	FileGetter fg(folderName, "bmp");
	while (fg.getNextAbsFile(fname))
	{
		Mat src;
		src = imread(fname);
		imshow(fg.getFoundFileName(), src);
		if (waitKey() == 27) //ESC pressed
			break;
	}
}

void testColor2Gray()
{
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat_<Vec3b> src = imread(fname, IMREAD_COLOR);

		int height = src.rows;
		int width = src.cols;

		Mat_<uchar> dst(height, width);

		for (int i = 0; i < height; i++)
		{
			for (int j = 0; j < width; j++)
			{
				Vec3b v3 = src(i, j);
				uchar b = v3[0];
				uchar g = v3[1];
				uchar r = v3[2];
				dst(i, j) = (r + g + b) / 3;
			}
		}

		imshow("original image", src);
		imshow("gray image", dst);
		waitKey();
	}
}

Mat_<Vec3b> convertIntToVec3b(Mat_<Vec3i> matrix) {
	Mat_<Vec3b> result(matrix.rows, matrix.cols);
	for (int i = 0; i < result.rows; i++)
		for (int j = 0; j < result.cols; j++) {
			for (int k = 0; k < 3; k++) {
				int x = ((int)matrix[i][j][k]);
				if (x < 0) {
					x = 0;
				}
				else if (x > 255) {
					x = 255;
				}
				result[i][j][k] = x;
			}
		}
	return result;
}

Mat_<Vec3i> convertVec3bToInt(Mat_<Vec3b> matrix) {
	Mat_<Vec3i> result(matrix.rows, matrix.cols);
	for (int i = 0; i < result.rows; ++i) {
		for (int j = 0; j < result.cols; ++j) {
			for (int k = 0; k < 3; ++k) {
				result[i][j][k] = (int)matrix[i][j][k];
			}
		}
	}
	return result;
}

vector<Mat_<Vec3b>> genGauss(Mat_<Vec3b> img, int levels) {

	vector<Mat_<Vec3b>> gaussPyramid;
	Mat_<Vec3b> level;

	gaussPyramid.push_back(img);

	for (int i = 0; i < levels; i++) {

		pyrDown(gaussPyramid[i], level);
		gaussPyramid.push_back(level);
	}
	return gaussPyramid;
}
void testGauss(int noOfLayers) {
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat_<Vec3b> src = imread(fname, IMREAD_COLOR);
		std::vector<Mat_<Vec3b>> gaussianPyr = genGauss(src, noOfLayers);
		for (int i = 0; i < gaussianPyr.size(); i++) {
			std::string x = "gaussian pyr #";
			x += std::to_string(i);
			cv::resize(gaussianPyr[i], gaussianPyr[i], cv::Size(256, 256));
			imshow(x, gaussianPyr[i]);

		}

		waitKey();
	}
}

vector<Mat_<Vec3i>> genLaplace(Mat_<Vec3b> src, int levels) {

	vector<Mat_<Vec3i>> aux;
	vector<Mat_<Vec3b>> gaussPyramid = genGauss(src, levels);

	aux.push_back(convertVec3bToInt(gaussPyramid.back()));
	for (int i = gaussPyramid.size() - 1; i >= 1; --i) {



		Mat_<Vec3b> upperLevel;
		pyrUp(gaussPyramid[i], upperLevel, Size(gaussPyramid[i - 1].cols, gaussPyramid[i - 1].rows));
		aux.push_back(convertVec3bToInt(gaussPyramid[i - 1]) - convertVec3bToInt(upperLevel));
	}
	return aux;
}
void printLaplace(Mat_<Vec3i> laplaceImg, std::string text) {
	Mat_<Vec3b> ret(laplaceImg.rows, laplaceImg.cols);
	for (int i = 0; i < laplaceImg.rows; ++i) {
		for (int j = 0; j < laplaceImg.cols; ++j) {
			for (int k = 0; k < 3; ++k) {
				int val = ((int)laplaceImg[i][j][k] + 128);
				if (val > 255) {
					val = 255;
				}
				else if (val < 0) {
					val = 0;
				}
				ret[i][j][k] = (uchar)val;
			}
		}
	}
	cv::resize(ret, ret, cv::Size(256, 256));
	imshow(text, ret);
}
void testLaplace(int layers) {
	char fname[MAX_PATH];
	openFileDlg(fname);
	Mat src;
	src = imread(fname, IMREAD_COLOR);
	std::vector<Mat_<Vec3i>> laplacianPyr = genLaplace(src, layers);
	Mat laplacian0 = convertIntToVec3b(laplacianPyr[0]);

	cv::resize(laplacian0, laplacian0, cv::Size(256, 256));

	imshow("lapace pyr #0", laplacian0);

	for (int i = 1; i < laplacianPyr.size(); ++i) {
		std::string x = "laplacian pyr #";
		x += std::to_string(i);
		printLaplace(laplacianPyr[i], x);
		cv::resize(src, src, cv::Size(256, 256));
	}

	imshow("image", src);
	waitKey();
}

Mat_<Vec3b> reconstructImgFromLapPyr(vector<Mat_<Vec3i>> lapPyr) {

	Mat_<Vec3b> image = convertIntToVec3b(lapPyr[0]);
	Mat_<Vec3b> layer;

	for (int i = 1; i < lapPyr.size(); i++) {
		pyrUp(image, layer, Size(lapPyr[i].cols, lapPyr[i].rows));
		image = convertIntToVec3b(lapPyr[i] + convertVec3bToInt(layer));
	}
	return image;
}

Mat_<Vec3i> filter(Mat_<Vec3i> img, float threshold) {
	Mat_<Vec3i> dst(img.rows, img.cols);
	for (int i = 0; i < img.rows; i++)
		for (int j = 0; j < img.cols; j++) {
			float lenght = abs(sqrt(img(i, j)[0] * img(i, j)[0] + img(i, j)[1] * img(i, j)[1] + img(i, j)[2] * img(i, j)[2]));
			if (lenght < threshold) {
				dst(i, j) = 0;
			}
			else
				dst(i, j) = img(i, j);
		}

	return dst;
}


void testFiltrare(int layers, int threshold) {
	char fname[MAX_PATH];
	openFileDlg(fname);
	Mat_<Vec3b> src = imread(fname, IMREAD_COLOR);
	std::vector<Mat_<Vec3i>> laplacianPyr = genLaplace(src, layers);

	//filter pyramid levels
	std::vector<Mat_<Vec3i>> filteredPyr;

	for (int i = 0; i < laplacianPyr.size(); i++) {
		filteredPyr.push_back(filter(laplacianPyr[i], threshold));
	}


	//reconstruct image with filtered pyramid
	Mat_<Vec3b> rec = reconstructImgFromLapPyr(filteredPyr);
	float mae = compute_mae(src, rec);
	printf("%f", mae);



	imshow("Diference", (rec - src) * 10 + 128);

	imshow("reconstructed", rec);

	imshow("image", src);
	waitKey();


}


void merge() {

	char fname1[MAX_PATH];
	openFileDlg(fname1);
	Mat_<Vec3b> src1 = imread(fname1, IMREAD_COLOR);

	char fname2[MAX_PATH];
	openFileDlg(fname2);
	Mat_<Vec3b> src2 = imread(fname2, IMREAD_COLOR);


	Mat_<Vec3b> g1 = src1.clone();
	Mat_<Vec3b> g2 = src2.clone();


	std::vector<Mat_<Vec3b>> gpA;
	gpA.push_back(g1);
	for (int i = 0; i < 6; i++)
	{
		pyrDown(gpA[i], g1);
		gpA.push_back(g1);
	}

	std::vector<Mat_<Vec3b>> gpB;
	gpB.push_back(g2);
	for (int i = 0; i < 6; i++)
	{
		pyrDown(gpB[i], g2);
		gpB.push_back(g2);
	}


	std::vector<Mat_<Vec3b>> lpA;
	lpA.push_back(gpA[5]);
	for (int i = 5; i > 0; i--)
	{
		Mat_<Vec3b> ge;
		Mat_<Vec3b> l;
		Size size = Size(gpA[i - 1].cols, gpA[i - 1].rows);
		pyrUp(gpA[i], ge, size);
		subtract(gpA[i - 1], ge, l);
		lpA.push_back(l);
	}


	std::vector<Mat_<Vec3b>> lpB;
	lpB.push_back(gpB[5]);
	for (int i = 5; i > 0; i--)
	{
		Mat_<Vec3b> ge;
		Mat_<Vec3b> l;
		Size size = Size(gpB[i - 1].cols, gpB[i - 1].rows);
		pyrUp(gpB[i], ge, size);
		subtract(gpB[i - 1], ge, l);
		lpB.push_back(l);
	}

	std::vector<Mat_<Vec3b>> LS;

	for (int i = 0; i < 6; i++)
	{
		Mat_<Vec3b> ls(lpA[i].rows, lpA[i].cols);
		for (int j = 0; j < lpA[i].rows; j++)
		{
			for (int k = 0; k < lpA[i].cols; k++)
			{
				if (k < (lpA[i].cols / 2))
				{
					ls(j, k) = lpA[i](j, k);
				}
				else
				{
					ls(j, k) = lpB[i](j, k);
				}
			}
		}
		LS.push_back(ls);
	}

	Mat_<Vec3b> ls_ = LS[0];
	for (int i = 1; i < 6; i++)
	{
		Size size = Size(LS[i].cols, LS[i].rows);
		pyrUp(ls_, ls_, size);
		add(ls_, LS[i], ls_);
	}

	imshow("Pyramid blending", ls_);

	Mat_<Vec3b> real(src1.rows, src1.cols);
	for (int j = 0; j < src1.rows; j++)
	{
		for (int k = 0; k < src1.cols; k++)
		{
			if (k < (src1.cols / 2))
			{
				real(j, k) = src1(j, k);
			}
			else
			{
				real(j, k) = src2(j, k);
			}
		}
	}

	imshow("Direct connection", real);


	waitKey();
}


int main()
{
	int op;
	do
	{
		int n;
		system("cls");
		destroyAllWindows();
		printf("Menu:\n");
		printf(" 1 - Calculate n levels for Gaussian Pyramid .\n");
		printf(" 2 - Calculate n levels for Laplace Pyramid .\n");
		printf(" 3 - Reconstruct the image .\n");
		printf(" 4 - Blending two images.\n");
		printf(" 5 - Filtered pyramid\n");
		printf("Option: ");
		scanf("%d", &op);
		switch (op)
		{
		case 1:
			printf(" levels = ");
			scanf("%d", &n);
			testGauss(n);
			break;
		case 2:
			printf(" levels = ");
			scanf("%d", &n);
			testLaplace(n);
			break;
		case 3:
			printf(" levels = ");
			scanf("%d", &n);
			testReconstruction(n);
			break;
		case 4:
			merge();
			break;
		case 5:
			printf("levels = ");
			scanf("%d", &n);
			int threshold;
			printf("threshold = ");
			scanf("%d", &threshold);
			testFiltrare(n, threshold);
			break;
		}

	} while (op != 0);
	return 0;
}

