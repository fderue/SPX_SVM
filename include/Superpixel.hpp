#pragma once
/*
Derue François-Xavier
*/

#include <opencv2\opencv.hpp>
#include "funUtils.hpp"

using namespace std;
using namespace cv;


class Pixel
{
public:
	static enum State
	{
		FGND,
		BGND,
		NEUT
	};

	static enum ColorSpace
	{
		BGR,
		HSV,
		Lab
	};

	Point xy;
	Vec3f color;
	State state;
	ColorSpace colorSpace;

	Pixel():xy(Point(-1, -1)), color(Vec3f(0, 0, 0)), state(NEUT),colorSpace(BGR){}
	Pixel(Point xy, Vec3f color, ColorSpace colorSpace = BGR, State state = NEUT) :xy(xy), color(color), state(NEUT), colorSpace(colorSpace){}
	friend ostream& operator<<(ostream& os, const Pixel& px){os << "xy : " << px.xy << "| color : " << px.color << endl;return os;}

};
class Superpixel : public Pixel
{
public:
	static enum FeatType
	{
		MEAN_COLOR,
		HISTO3D
	};

	vector<Pixel> v_pixels;
	Mat histo;
	FeatType featType;

	Superpixel() :Pixel(){}
	Superpixel(Point xy, Vec3f color, ColorSpace colorSpace = BGR, FeatType featType = MEAN_COLOR, State neut = NEUT) :Pixel(xy, color, colorSpace, neut){ this->featType = featType; }
	
	void computeMean();
	void computeHisto(int nBin1d=6);
	void alight(Mat& out, Vec3b color = Vec3b(255, 0, 0));
	Mat getFeatMat();

};

