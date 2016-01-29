#pragma once
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

namespace funUtils{

	enum HistColor{
		BGR,
		HSV,
		Lab
	};

	void getGrabCutSeg(Mat& inIm, Mat& mask_fgnd, Rect ROI);
	Mat makeMask(Rect ROIin, int wFrame, int hFrame, float scale = 2, bool fullFrame = false);
	void adaptROI(Rect& ROI, int wFrame, int hFrame);
	void hist3D(Mat& image, Mat& hist, int Nbin,HistColor histColorSpace);

}
