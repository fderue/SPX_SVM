#include "SpxSvmTestEngine.h"



void makeSpxVec(Slic& slic, vector<Superpixel>& v_spx, Mat& imageBGR8, Superpixel::ColorSpace cs = Superpixel::BGR, Superpixel::FeatType ft = Superpixel::MEAN_COLOR, int nBin1d = 6)
{
	Mat imCvt;
	switch (cs)
	{
	case Superpixel::Lab:
		cvtColor(imageBGR8, imCvt, CV_BGR2Lab);
		break;
	case Superpixel::HSV:
		cvtColor(imageBGR8, imCvt, CV_BGR2HSV);
		break;
	default:
		break;
	}
	int Nspx = slic.getNspx();
	v_spx.resize(Nspx);
	vec2di& labels = slic.getLabels();
	for (int i = 0; i < labels.size(); i++){
		Vec3b* image_ptr = imageBGR8.ptr<Vec3b>(i);
		for (int j = 0; j < labels[0].size(); j++){
			v_spx[labels[i][j]].v_pixels.push_back(Pixel(Point(j, i), Vec3f(image_ptr[j]), cs));
		}
	}
	for (int i = 0; i < Nspx; i++){
		v_spx[i].computeMean();
		if (ft == Superpixel::HISTO3D) v_spx[i].computeHisto(nBin1d);
		v_spx[i].colorSpace = cs;
		v_spx[i].featType = ft;
	}
}

void setFgndSpx(Slic& slic, vector<Superpixel>& v_spx_in, Mat& grabSeg, Rect& ROI)
{
	vec2di& labels = slic.getLabels();
	vector<int> accum(v_spx_in.size(), 0);
	float spxFgndCount = 0;
	for (int i = ROI.y; i < ROI.y + ROI.height; i++){
		uchar* grabSeg_ptr = grabSeg.ptr<uchar>(i);
		for (int j = ROI.x; j < ROI.x + ROI.width; j++){
			if (grabSeg_ptr[j] != 0){
				if (accum[labels[i][j]] == 0) spxFgndCount++;
				accum[labels[i][j]] += 1;
			}
		}
	}
	float ratio = spxFgndCount / (float)v_spx_in.size();
	float thr_ratio = 0.75;

	if (ratio < thr_ratio) // all spx in ROI are fgnd
	{
		for (int i = 0; i < v_spx_in.size(); i++){
			if (v_spx_in[i].xy.x >= ROI.x && v_spx_in[i].xy.x < ROI.x + ROI.width
				&& v_spx_in[i].xy.y >= ROI.y && v_spx_in[i].xy.y < ROI.y + ROI.height){
				v_spx_in[i].state = Superpixel::FGND;
			}
		}
	}
	else // only spx with some amount of fgnd px are fgnd
	{
		int threshold = slic.getSspx()*slic.getSspx() / 4;
		for (int i = 0; i < accum.size(); i++){
			if (accum[i] > threshold){
				v_spx_in[i].state = Superpixel::FGND;
			}
		}
	}
}
void setBgndSpx(vector<Superpixel>& v_spx, Mat& maskBgnd8)
{
	for (int i = 0; i < v_spx.size(); i++){
		if (maskBgnd8.at<uchar>(v_spx[i].xy.y, v_spx[i].xy.x)){
			v_spx[i].state = Superpixel::BGND;
		}
	}
}

Mat computeLabelsMat(const vector<Superpixel*>& v_spx)
{
	Mat labelsMat(v_spx.size(), 1, CV_32SC1, Scalar(-1));
	float* labelsMat_ptr = (float*)labelsMat.data;
	for (int i = 0; i < v_spx.size(); i++){
		if (v_spx[i]->state == Pixel::FGND) labelsMat_ptr[i] = 1;
		else if (v_spx[i]->state == Pixel::BGND) labelsMat_ptr[i] = -1;
		else cerr << "error : no NEUT is allowed when computing labels Mat " << endl;
	}
	return labelsMat;
}

Mat createFeatMat(vector<Superpixel*>& v_spx)
{
	CV_Assert(!v_spx.empty());
	Superpixel::FeatType ft = v_spx[0]->featType;
	int D; //feature dimension
	Mat m_spx;
	switch (ft)
	{
	case Superpixel::MEAN_COLOR:
		D = 3;
		m_spx = Mat(v_spx.size(), D, CV_32F);
		for (int i = 0; i < m_spx.rows; i++){
			float* m_spx_ptr = m_spx.ptr<float>(i);
			for (int j = 0; j < D; j++){
				m_spx_ptr[j] = v_spx[i]->color[j];
			}
		}
		return m_spx;
		break;
	case Superpixel::HISTO3D:
		D = v_spx[0]->histo.size[0] * v_spx[0]->histo.size[1] * v_spx[0]->histo.size[2];
		m_spx = Mat(v_spx.size(), D, CV_32F);
		for (int i = 0; i < m_spx.rows; i++){
			float* m_spx_ptr = m_spx.ptr<float>(i);
			float* hist_ptr = (float*)v_spx[i]->histo.data;
			for (int j = 0; j < D; j++){
				m_spx_ptr[j] = hist_ptr[j];
			}
		}
		return m_spx;
		break;
	default:
		break;
	}

}

void trainSVM(Ptr<SVM>& svm, vector<Superpixel>& v_spx, SVM::Types svmType, SVM::KernelTypes kernelType)
{
	CV_Assert(!v_spx.empty());
	svm->setType(svmType);
	svm->setKernel(kernelType);
	svm->setTermCriteria(TermCriteria(TermCriteria::MAX_ITER, 100, 1e-6));
	vector<Superpixel*> v_spx_ptr;
	Mat labelsMat;
	if (svmType == SVM::C_SVC){ // TWO-CLASS SVM
		for (int i = 0; i < v_spx.size(); i++)if (v_spx[i].state != Superpixel::NEUT) v_spx_ptr.push_back(&v_spx[i]);
		Mat fbSpxFeatMat = createFeatMat(v_spx_ptr);
		labelsMat = computeLabelsMat(v_spx_ptr);
		Ptr<TrainData> tdata = TrainData::create(fbSpxFeatMat, ROW_SAMPLE, labelsMat);
		svm->trainAuto(tdata); // optimize parameter with k-fold
	}
	else{ //ONE-CLASS SVM
		svm->setNu(0.01); // only needed for one_class SVM (does not work well)
		for (int i = 0; i < v_spx.size(); i++) if (v_spx[i].state == Superpixel::FGND)v_spx_ptr.push_back(&v_spx[i]);

		//Mat fgndSpxMat = cvtPxVec2Mat(v_spx_ptr);
		Mat fbSpxFeatMat = createFeatMat(v_spx_ptr);

		labelsMat = computeLabelsMat(v_spx_ptr);
		svm->train(fbSpxFeatMat, ROW_SAMPLE, labelsMat);
	}
}
float predictSVM(Ptr<SVM>& svm, Superpixel& spx)
{
	Mat spxTest = spx.getFeatMat();

	Mat out;
	return svm->predict(spxTest);//, out,StatModel::RAW_OUTPUT);
	//return out.at<float>(0,0);
}

//=============================== Main functions =================================
void SpxSvmTestEngine::initialize(Mat& imTrain, Rect ROI, Settings& settings)
{
	m_settings = settings;
	fgndROI = ROI;
	//Grabcut (coarse fgnd/bgnd subtraction)
	funUtils::getGrabCutSeg(imTrain, grabSegMask, ROI);

	//Superpixel segmentation
	slicTrain.initialize(imTrain, settings.sizeSpx, settings.compactSpx, settings.initTypeSpx);
	slicTrain.generateSpx(imTrain);

	//create Superpixel vector from slic
	makeSpxVec(slicTrain, v_spxTrain, imTrain, settings.spaceColorSpx, settings.featTypeSpx);

	//set Fgnd and Bgnd Superpixel according to Grabcut
	setFgndSpx(slicTrain, v_spxTrain, grabSegMask, fgndROI);
	Mat maskBgnd = funUtils::makeMask(fgndROI, imTrain.cols, imTrain.rows, settings.scaleBROI, settings.fullFrame);
	setBgndSpx(v_spxTrain, maskBgnd);

	//train a classifier
	svm = SVM::create();
	trainSVM(svm, v_spxTrain, settings.typeSVM, settings.kernelSVM);

}
void SpxSvmTestEngine::run(Mat& imTest)
{
	CV_Assert(imTest.data != nullptr);
	slicTest.initialize(imTest, m_settings.sizeSpx, m_settings.compactSpx, m_settings.initTypeSpx);
	slicTest.generateSpx(imTest);

	makeSpxVec(slicTest, v_spxTest, imTest, m_settings.spaceColorSpx,m_settings.featTypeSpx);

	//classify all spx (or just in a search area where the samples will be)
	Mat svmTest = imTest.clone();

	auto start = getTickCount();
	for (int i = 0; i < v_spxTest.size(); i++){
		float response = predictSVM(svm, v_spxTest[i]);
		if (response > 0){
			v_spxTest[i].state = Superpixel::FGND;
		}
	}
	auto end = getTickCount();
	cout << "total runTime prediction : " << (end - start) / getTickFrequency() << endl;
}

void SpxSvmTestEngine::showResults(Mat& imTrain, Mat& imTest)
{
	//training samples
	Mat sampleSelection = imTrain.clone();
	for (int i = 0; i < v_spxTrain.size(); i++){
		if (v_spxTrain[i].state == Superpixel::FGND)v_spxTrain[i].alight(sampleSelection, Vec3b(0, 255, 0));
		else if (v_spxTrain[i].state == Superpixel::BGND)v_spxTrain[i].alight(sampleSelection, Vec3b(0, 0, 255));
	}
	slicTrain.display_contours(sampleSelection);

	imshow("Sample selection", sampleSelection);

	//classification result
	Mat classResult = imTest.clone();
	for (int i = 0; i < v_spxTest.size(); i++){
		if (v_spxTest[i].state == Superpixel::FGND)v_spxTest[i].alight(classResult, Vec3b(0, 255, 0));
	}
	slicTest.display_contours(classResult);
	imshow("imTest class", classResult);
}