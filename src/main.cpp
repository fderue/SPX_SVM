#include <opencv2/opencv.hpp>
#include "SpxSvmTestEngine.hpp"

using namespace std;
using namespace cv;
using namespace cv::ml;


bool noROI = true; 
bool clicked = false;
Rect ROImouse;
void CallBackFunc(int event, int x, int y, int flags, void* userdata)
{
	if (event == EVENT_LBUTTONDOWN)
	{
		//cout << "Left button of the mouse is clicked - position (" << x << ", " << y << ")" << endl;
		ROImouse.x = x;
		ROImouse.y = y;
		clicked = true;
	}
	else if (event == EVENT_LBUTTONUP)
	{
		//cout << "Left button of the mouse is released - position (" << x << ", " << y << ")" << endl;
		noROI = false;
	}
	else if (event == EVENT_MBUTTONDOWN)
	{
		//cout << "Middle button of the mouse is clicked - position (" << x << ", " << y << ")" << endl;
	}
	else if (event == EVENT_MOUSEMOVE)
	{
		if (clicked)
		{
			ROImouse.width = abs(x - ROImouse.x);
			ROImouse.height = abs(y - ROImouse.y);
		}
	}
}

int main(int argc, char** argv)
{
	namedWindow("My Window", 1);
	Mat imTrain = imread("E:/Videos/CVPR_benchmark/doll/img/0001.jpg");
	Mat imTest = imread("E:/Videos/CVPR_benchmark/doll/img/0001.jpg");

	//ROI selection
	CV_Assert(imTrain.data != nullptr);
	setMouseCallback("My Window", CallBackFunc, NULL);
	while (noROI){
		Mat im_clone = imTrain.clone();
		rectangle(im_clone, ROImouse, Scalar(255, 0, 0));
		imshow("My Window", im_clone);
		waitKey(30);
	}

	//test
	SpxSvmTestEngine::Settings settings;
	settings.sizeSpxOrNbSpx0 = 15;
	settings.initTypeSpx = Slic::SLIC_SIZE;
	settings.compactSpx = 35;
	settings.spaceColorSpx = Superpixel::Lab;
	settings.featTypeSpx = Superpixel::HISTO3D;
	settings.histNbin1d = 6;
	settings.scaleBROI = 2;
	settings.fullFrame = false;
	settings.kernelSVM = SVM::RBF;
	settings.typeSVM = SVM::C_SVC;

	SpxSvmTestEngine testEngine;
	testEngine.initialize(imTrain, ROImouse, settings);
	testEngine.run(imTest);
	testEngine.showResults(imTrain,imTest);

	return 0;

}