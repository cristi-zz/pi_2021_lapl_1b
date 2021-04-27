// OpenCVApplication.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include "common.h"
#include <stdio.h>
#include <stdlib.h>

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
	while (openFileDlg(fname)) {
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
	}
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
void testReconstruction(int layers) {
	char fname[MAX_PATH];
	while (openFileDlg(fname)) {
		Mat src;
		src = imread(fname, IMREAD_COLOR);

		std::vector<Mat_<Vec3i>> laplacianPyr = genLaplace(src, layers);
		Mat_<Vec3b> rec = reconstructImgFromLapPyr(laplacianPyr);

		imshow("Diference", (rec - src) * 10 + 128);

		imshow("reconstructed", rec);
		imshow("image", src);
		waitKey(0);
	}
}

float* compute_MAE(Mat_<Vec3b> firstImage, Mat_<Vec3b> secondImage) {

	float* MAE = (float*)calloc(3, sizeof(float));
	for (int i = 0; i < firstImage.rows; i++)
		for (int j = 0; j < firstImage.cols; j++) {
			MAE[0] = MAE[0] + abs(firstImage(i, j)[0] - secondImage(i, j)[0]);
			MAE[1] = MAE[1] + abs(firstImage(i, j)[1] - secondImage(i, j)[1]);
			MAE[2] = MAE[2] + abs(firstImage(i, j)[2] - secondImage(i, j)[2]);
		}
	MAE[0] = MAE[0] / (firstImage.rows * firstImage.cols);
	MAE[1] = MAE[1] / (firstImage.rows * firstImage.cols);
	MAE[2] = MAE[2] / (firstImage.rows * firstImage.cols);

	return MAE;
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
		printf(" 4 - Calculate the mean absolute error .\n");
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
		}
	} while (op != 0);
	return 0;
}
