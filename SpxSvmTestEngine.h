#pragma once
/*
Derue François-Xavier

Objectif : Test SVM classification on Superpixel
input : Training image, ROI, TestImage
output : Classification of Superpixel on TestImage

ex : 
SpxSvmTestEngine::Settings settings;
SpxSvmTestEngine testEngine;
testEngine.initialize(imTrain, ROImouse, settings);
testEngine.run(imTest);
testEngine.showResults(imTrain,imTest);
*/


#include <opencv2/opencv.hpp>
#include <opencv2/ml.hpp>
#include "Slic.h"
#include "funUtils.h"
#include "Superpixel.h"


using namespace std;
using namespace cv;
using namespace cv::ml;

class SpxSvmTestEngine{
private:
	Rect fgndROI;
	Mat grabSegMask;
	Slic slicTrain, slicTest;
	Ptr<SVM> svm;
	vector<Superpixel> v_spxTrain, v_spxTest;


public:
	SpxSvmTestEngine(){}
	~SpxSvmTestEngine(){}


	static struct Settings
	{
		int sizeSpx = 16;
		int compactSpx = 35;
		Slic::InitType initTypeSpx = Slic::SLIC_SIZE;
		Superpixel::ColorSpace spaceColorSpx = Superpixel::BGR;
		Superpixel::FeatType featTypeSpx = Superpixel::MEAN_COLOR;
		int histNbin1d = 6;
		int scaleBROI = 2;
		bool fullFrame = false;

		SVM::KernelTypes kernelSVM= SVM::RBF;
		SVM::Types typeSVM = SVM::C_SVC;
	};
	Settings m_settings;

	void initialize(Mat& imTrain, Rect ROI, Settings& settings);
	void run(Mat& imTest);
	void showResults(Mat& imTrain, Mat& imTest);



};