/////////////////////////////////////////
////Author: Akash Jadhav ///////////////
////////////////////////////////////////
#include<opencv2/core/core.hpp>
#include<opencv2/highgui/highgui.hpp>
#include<opencv2/imgproc/imgproc.hpp>
#include<opencv2/ml/ml.hpp>

#include<iostream>
#include<sstream>

using namespace cv;
using namespace std;

const int MIN_CONTOUR_AREA = 100;
const int RESIZED_IMAGE_WIDTH = 20;
const int RESIZED_IMAGE_HEIGHT = 30;

class ContourWithData {
public:
	
	vector<cv::Point> ptContour; 
	Rect boundingRect;            
	float fltArea;                    
												
	bool checkIfContourIsValid() {              
		if (fltArea < MIN_CONTOUR_AREA) return false;        
		return true;                                         
	}

	
	static bool sortByBoundingRectXPosition(const ContourWithData& cwdLeft, const ContourWithData& cwdRight) {   
		return(cwdLeft.boundingRect.x < cwdRight.boundingRect.x);                                                
	}

};

int main() {
	vector<ContourWithData> allContoursWithData;         
	vector<ContourWithData> validContoursWithData;       
																
	Mat matClassificationInts;     
	FileStorage fsClassifications("classifications.xml", cv::FileStorage::READ);  
	
	if (fsClassifications.isOpened() == false) {               
		cout << "error, unable to open training classifications file, exiting program\n\n";  
		return(0);                                                                                
	}

	fsClassifications["classifications"] >> matClassificationInts;  
	fsClassifications.release();                                    
	Mat matTrainingImagesAsFlattenedFloats;     
	FileStorage fsTrainingImages("images.xml", cv::FileStorage::READ);     
	
	if (fsTrainingImages.isOpened() == false) {                                          
		cout << "error, unable to open training images file, exiting program\n\n";  
		return(0);                                                                       
	}

	fsTrainingImages["images"] >> matTrainingImagesAsFlattenedFloats;       
	fsTrainingImages.release();                                             
	Ptr<cv::ml::KNearest>  kNearest(cv::ml::KNearest::create());   
	kNearest->train(matTrainingImagesAsFlattenedFloats, cv::ml::ROW_SAMPLE, matClassificationInts);

	// test ///////////////////////////////////////////////////////////////////////////////

	Mat matTestingNumbers = cv::imread("drishti.png");      //read     

	if (matTestingNumbers.empty()) {                               
		cout << "error: image not read from file\n\n";        
		return(0);                                                 
	}

	Mat matGrayscale;        
	Mat matBlurred;          
	Mat matThresh;           
	Mat matThreshCopy;       

	cvtColor(matTestingNumbers, matGrayscale, CV_BGR2GRAY);     
	
	GaussianBlur(matGrayscale, matBlurred, cv::Size(5, 5), 0);
									   
	adaptiveThreshold(matBlurred, matThresh, 255, cv::ADAPTIVE_THRESH_GAUSSIAN_C, cv::THRESH_BINARY_INV, 11, 2);

	matThreshCopy = matThresh.clone();

	vector<std::vector<cv::Point> > ptContours;
	vector<cv::Vec4i> v4iHierarchy;            
	findContours(matThreshCopy,	ptContours, v4iHierarchy, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

	for (int i = 0; i < ptContours.size(); i++) {        
		ContourWithData contourWithData;                 
		contourWithData.ptContour = ptContours[i];       
		contourWithData.boundingRect = cv::boundingRect(contourWithData.ptContour);
		contourWithData.fltArea = cv::contourArea(contourWithData.ptContour);      
		allContoursWithData.push_back(contourWithData);                            
	}

	for (int i = 0; i < allContoursWithData.size(); i++) {         
		if (allContoursWithData[i].checkIfContourIsValid()) {      
			validContoursWithData.push_back(allContoursWithData[i]);
		}
	}
	
	sort(validContoursWithData.begin(), validContoursWithData.end(), ContourWithData::sortByBoundingRectXPosition);

	string strFinalString;      

	for (int i = 0; i < validContoursWithData.size(); i++) {       
																	
		rectangle(matTestingNumbers, validContoursWithData[i].boundingRect, cv::Scalar(0, 255, 0), 2);
		Mat matROI = matThresh(validContoursWithData[i].boundingRect); 

		Mat matROIResized;
		resize(matROI, matROIResized, cv::Size(RESIZED_IMAGE_WIDTH, RESIZED_IMAGE_HEIGHT));   

		Mat matROIFloat;
		matROIResized.convertTo(matROIFloat, CV_32FC1);         

		Mat matROIFlattenedFloat = matROIFloat.reshape(1, 1);

		Mat matCurrentChar(0, 0, CV_32F);

		kNearest->findNearest(matROIFlattenedFloat, 1, matCurrentChar); 

		float fltCurrentChar = (float)matCurrentChar.at<float>(0, 0);

		strFinalString = strFinalString + char(int(fltCurrentChar));        
	}

	cout << "\n\n" << "numbers read = " << strFinalString << "\n\n";     

	imshow("matTestingNumbers", matTestingNumbers);   
	waitKey(0);                                       
	return(0);
}


