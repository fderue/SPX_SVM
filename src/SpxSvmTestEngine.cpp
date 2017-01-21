#include "SpxSvmTestEngine.hpp"

static void makeSpxVec(Slic& slic, vector<Superpixel>& v_spx, const Mat& imageBGR8, Superpixel::ColorSpace cs = Superpixel::BGR, Superpixel::FeatType ft = Superpixel::MEAN_COLOR, int nBin1d = 6)
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
	int Nspx = slic.getNbSpx();
	v_spx.resize(Nspx);
	const Mat labels = slic.getLabels();
	for (int i = 0; i < labels.rows; i++){
		const Vec3b* image_ptr = imageBGR8.ptr<Vec3b>(i);
		const int* label_ptr = labels.ptr<int>(i);
		for (int j = 0; j < labels.cols; j++){
			v_spx[label_ptr[j]].v_pixels.push_back(Pixel(Point(j, i), Vec3f(image_ptr[j]), cs));
		}
	}
	for (int i = 0; i < Nspx; i++){
		v_spx[i].computeMean();
		if (ft == Superpixel::HISTO3D) v_spx[i].computeHisto(nBin1d);
		v_spx[i].colorSpace = cs;
		v_spx[i].featType = ft;
	}
}

static void setFgndSpx(Slic& slic, vector<Superpixel>& v_spx_in, Mat& grabSeg, Rect& ROI)
{
	const Mat labels = slic.getLabels();
	vector<int> accum(v_spx_in.size(), 0);
	float spxFgndCount = 0;
	for (int i = ROI.y; i < ROI.y + ROI.height; i++){
		uchar* grabSeg_ptr = grabSeg.ptr<uchar>(i);
		for (int j = ROI.x; j < ROI.x + ROI.width; j++){
			if (grabSeg_ptr[j] != 0){
				if (accum[labels.at<int>(i, j)] == 0) spxFgndCount++;
				accum[labels.at<int>(i, j)] += 1;
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
		int threshold = slic.getSpxDiam()*slic.getSpxDiam() / 4;
		for (int i = 0; i < accum.size(); i++){
			if (accum[i] > threshold){
				v_spx_in[i].state = Superpixel::FGND;
			}
		}
	}
}
static void setBgndSpx(vector<Superpixel>& v_spx, Mat& maskBgnd8)
{
	for (int i = 0; i < v_spx.size(); i++){
		if (maskBgnd8.at<uchar>(v_spx[i].xy.y, v_spx[i].xy.x)){
			v_spx[i].state = Superpixel::BGND;
		}
	}
}

static Mat computeLabelsMat(const vector<Superpixel*>& v_spx)
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

static Mat createFeatMat(vector<Superpixel*>& v_spx)
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

static void trainSVM(Ptr<SVM>& svm, vector<Superpixel>& v_spx, SVM::Types svmType, SVM::KernelTypes kernelType)
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
static float predictSVM(Ptr<SVM>& svm, Superpixel& spx)
{
	Mat spxTest = spx.getFeatMat();

	Mat out;
	return svm->predict(spxTest);//, out,StatModel::RAW_OUTPUT);
	//return out.at<float>(0,0);
}


SpxSvmTestEngine::SpxSvmTestEngine()
{
	m_pSlicTrain = make_unique<Slic>();
	m_pSlicTest = make_unique<Slic>();
	m_SvmClassifier = SVM::create();

}
void SpxSvmTestEngine::initialize(const Mat& imTrain, const Rect& ROI, const Settings& settings)
{
	m_Settings = settings;
	m_FgndROI = ROI;
	//Grabcut (coarse fgnd/bgnd subtraction)
	Mat grabCutSegMask;
	funUtils::getGrabCutSeg(imTrain, grabCutSegMask, ROI);

	//Superpixel segmentation
	m_pSlicTrain->initialize(imTrain, settings.sizeSpxOrNbSpx0, settings.compactSpx, 5, settings.initTypeSpx);
	m_pSlicTrain->generateSpx(imTrain);

	//create Superpixel vector from slic
	makeSpxVec(*m_pSlicTrain, vSpxTrain, imTrain, settings.spaceColorSpx, settings.featTypeSpx);

	//set Fgnd and Bgnd Superpixel according to Grabcut
	setFgndSpx(*m_pSlicTrain, vSpxTrain, grabCutSegMask, m_FgndROI);
	Mat maskBgnd = funUtils::makeMask(m_FgndROI, imTrain.cols, imTrain.rows, settings.scaleBROI, settings.fullFrame);
	setBgndSpx(vSpxTrain, maskBgnd);

	//train a classifier
	trainSVM(m_SvmClassifier, vSpxTrain, settings.typeSVM, settings.kernelSVM);

}
void SpxSvmTestEngine::run(const Mat& imTest)
{
	CV_Assert(imTest.data != nullptr);
	
	m_pSlicTest->initialize(imTest, m_Settings.sizeSpxOrNbSpx0, m_Settings.compactSpx, 5, m_Settings.initTypeSpx);
	m_pSlicTest->generateSpx(imTest);

	makeSpxVec(*m_pSlicTest, vSpxTest, imTest, m_Settings.spaceColorSpx, m_Settings.featTypeSpx);

	//classify all spx (or just in a search area where the samples will be)
	Mat svmTest = imTest.clone();

	auto start = getTickCount();
	for (int i = 0; i < vSpxTest.size(); i++){
		float response = predictSVM(m_SvmClassifier, vSpxTest[i]);
		if (response > 0){
			vSpxTest[i].state = Superpixel::FGND;
		}
	}
	auto end = getTickCount();
	cout << "total runTime prediction : " << (end - start) / getTickFrequency() << endl;
}

void SpxSvmTestEngine::showResults(const Mat& imTrain, const Mat& imTest)
{
	//training samples
	Mat sampleSelection = imTrain.clone();
	for (int i = 0; i < vSpxTrain.size(); i++){
		if (vSpxTrain[i].state == Superpixel::FGND)vSpxTrain[i].alight(sampleSelection, Vec3b(0, 255, 0));
		else if (vSpxTrain[i].state == Superpixel::BGND)vSpxTrain[i].alight(sampleSelection, Vec3b(0, 0, 255));
	}
	m_pSlicTrain->display_contours(sampleSelection);

	imshow("Sample selection", sampleSelection);

	//classification result
	Mat classResult = imTest.clone();
	for (int i = 0; i < vSpxTest.size(); i++){
		if (vSpxTest[i].state == Superpixel::FGND)vSpxTest[i].alight(classResult, Vec3b(0, 255, 0));
	}
	m_pSlicTest->display_contours(classResult);
	imshow("imTest class", classResult);
	waitKey();
}

