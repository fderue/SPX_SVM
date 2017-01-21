#include "funUtils.hpp"

using namespace std;
using namespace cv;


namespace funUtils{
	void getGrabCutSeg(const Mat& inIm, Mat& mask_fgnd, Rect ROI)
	{
		Mat mask_out;
		Rect rect = ROI;
		Mat fgnd, bgnd;
		grabCut(inIm, mask_out, rect, bgnd, fgnd, 5, GC_INIT_WITH_RECT);
		bitwise_and(mask_out, GC_FGD, mask_fgnd);
	}

	Mat makeMask(Rect ROIin, int wFrame, int hFrame, float scale, bool fullFrame)
	{
		if (fullFrame){ Mat mask(hFrame, wFrame, CV_8U, Scalar(1)); mask(ROIin) = 0; return mask; }
		Mat mask(hFrame, wFrame, CV_8U, Scalar(0));
		float facx = ROIin.width / 2 * (1 - scale);
		float facy = ROIin.height / 2 * (1 - scale);
		Rect ROIex(ROIin.x + facx, ROIin.y + facy, ROIin.width*scale, ROIin.height*scale);
		adaptROI(ROIex, wFrame, hFrame);
		mask(ROIex) = 1;
		mask(ROIin) = 0;

		return mask;
	}


	void hist3D(Mat& image, Mat& hist, int Nbin, funUtils::HistColor histColor)
	{
		int h_bins = Nbin; int s_bins = Nbin; int v_bins = Nbin;
		int histSize[] = { h_bins, s_bins, v_bins };

		// hue varies from 0 to 179, saturation from 0 to 255
		float x_ranges[2];
		float y_ranges[2];
		float z_ranges[2];
		switch (histColor){
		case funUtils::HSV:
			x_ranges[0] = 0; x_ranges[1] = 180;
			y_ranges[0] = 0; y_ranges[1] = 256;
			z_ranges[0] = 0; z_ranges[1] = 256;
			break;
		case funUtils::BGR:
			x_ranges[0] = 0; x_ranges[1] = 256;
			y_ranges[0] = 0; y_ranges[1] = 256;
			z_ranges[0] = 0; z_ranges[1] = 256;
			break;
		case funUtils::Lab:
			x_ranges[0] = 0; x_ranges[1] = 256;
			y_ranges[0] = 0; y_ranges[1] = 256;
			z_ranges[0] = 0; z_ranges[1] = 256;
			break;
		default:
			x_ranges[0] = 0; x_ranges[1] = 256;
			y_ranges[0] = 0; y_ranges[1] = 256;
			z_ranges[0] = 0; z_ranges[1] = 256;
			break;
		}

		const float* ranges[] = { x_ranges, y_ranges, z_ranges };

		int channels[] = { 0,1,2};

		calcHist(&image,1, channels, Mat(), hist, 3, histSize, ranges, true,false);

		//normalization 3D
		float* hist_ptr = hist.ptr<float>();
		int a, b;
		int col_row = hist.size[0] * hist.size[1];
		int facNorm = h_bins*s_bins*v_bins;
		for (int k = 0; k < hist.size[2]; k++)
		{
			a = k*col_row;
			for (int i = 0; i < hist.size[0]; i++)
			{
				b = i*hist.size[1];
				for (int j = 0; j < hist.size[1]; j++)
				{
					hist_ptr[a + b + j] /= facNorm;
				}
			}
		}
	}

	void adaptROI(Rect& ROI, int wFrame, int hFrame)
	{
		if (ROI.x >= wFrame || ROI.y >= hFrame) { cerr << "error adaptROI out of frame" << endl; return; }
		if (ROI.x < 0){ ROI.width = ROI.width + ROI.x; ROI.x = 0; }
		if (ROI.y < 0){ ROI.height = ROI.height + ROI.y; ROI.y = 0; }
		int dX, dY;
		if ((dX = ROI.x + ROI.width - wFrame) > 0){ ROI.width -= dX; }
		if ((dY = ROI.y + ROI.height - hFrame)>0){ ROI.height -= dY; }
	}
}