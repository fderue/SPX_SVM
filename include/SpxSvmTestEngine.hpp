/*
Objectif : Test SVM binary classification of foreground/background based on Superpixel
input : 
- Training image : image used to train the svm classifier
- ROI : bounding box splitting the training image. Inside of ROI is foreground (positive sample), outside is background (negative sample)
- TestImage : apply the classifier on the superpixel found in TestImage
output : Classification of the Superpixels extracted from TestImage

ex : 
SpxSvmTestEngine::Settings settings;
SpxSvmTestEngine testEngine;
testEngine.initialize(imTrain, ROImouse, settings);
testEngine.run(imTest);
testEngine.showResults(imTrain,imTest);

author : Derue François-Xavier
francois.xavier.derue<at>gmail.com
*/
#pragma once
#include <opencv2/opencv.hpp>
#include <opencv2/ml.hpp>
#include "Slic.hpp"
#include "funUtils.hpp"
#include "Superpixel.hpp"
#include <memory>


using namespace std;
using namespace cv;
using namespace cv::ml;

class SpxSvmTestEngine{
public:
	static struct Settings
	{
		int sizeSpxOrNbSpx0 = 16;
		Slic::InitType initTypeSpx = Slic::SLIC_SIZE;
		int compactSpx = 35;
		Superpixel::ColorSpace spaceColorSpx = Superpixel::BGR;
		Superpixel::FeatType featTypeSpx = Superpixel::MEAN_COLOR;
		int histNbin1d = 6;
		int scaleBROI = 2;
		bool fullFrame = false;

		SVM::KernelTypes kernelSVM = SVM::RBF;
		SVM::Types typeSVM = SVM::C_SVC;
	};
	SpxSvmTestEngine();
	~SpxSvmTestEngine(){}

	void initialize(const Mat& imTrain, const Rect& ROI, const Settings& settings);
	void run(const Mat& imTest);
	void showResults(const Mat& imTrain, const Mat& imTest);

	

private:
	Rect m_FgndROI;
	unique_ptr<Slic> m_pSlicTrain;
	unique_ptr<Slic> m_pSlicTest;
	Ptr<SVM> m_SvmClassifier;
	vector<Superpixel> vSpxTrain;
	vector<Superpixel> vSpxTest;
	Settings m_Settings;
};