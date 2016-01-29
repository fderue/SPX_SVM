# SPX_SVM
SVM classification on Superpixel
Input : TrainImage, ROI and TestImage
Output : Superpixel of TestImage classified by SVM 

Detail : 
1) Training
- Draw ROI on TrainImage to select Foreground region
- Apply grabCut (from openCV) to have coarse fgnd/bgnd subtraction
- Oversegment TrainImage with SLIC to get the superpixels
- Select foreground superpixel samples if #pixel_fgnd>threshold inside that superpixel
- Select background superpixel samples in area around ROI
- Do SVM training with those samples

2) Test :
- Segment TestImage in Superpixel
- Extract feature from each Superpixel
- Classify each Superpixel with SVM


Note : Different settings are available 
- Color Space : BGR, HSV, Lab
- Feature : MEAN_COLOR, HISTOGRAM3D
- SVM Types : ONE CLASS , TWO CLASS
- SVM Kernel : LINEAR, RBF
- SLIC parameter : size/number of Superpixel, compactness
- Background ROI size : fullFrame -> all the superpixel out of ROI, or scaleROI -> BROI =  X*ROI-ROI
