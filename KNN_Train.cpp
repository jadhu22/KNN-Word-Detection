///////////////////////////////////////////
///////Author: Akash Jadhav //////////////
//////////////////////////////////////////
#include<opencv2/core/core.hpp>
#include<opencv2/highgui/highgui.hpp>
#include<opencv2/imgproc/imgproc.hpp>
#include<opencv2/ml/ml.hpp>

#include<iostream>
#include<vector>

using namespace cv;
using namespace std;
const int MIN_CONTOUR_AREA = 100;
const int RESIZED_IMAGE_WIDTH = 20;
const int RESIZED_IMAGE_HEIGHT = 30;

int main() {

	Mat imgTrainingNumbers;        
	Mat imgGrayscale;               
	Mat imgBlurred;                
	Mat imgThresh;                 
	Mat imgThreshCopy;            

	vector<std::vector<cv::Point> > ptContours;        
	vector<cv::Vec4i> v4iHierarchy;                    

	Mat matClassificationInts;      
	Mat matTrainingImagesAsFlattenedFloats;

	vector<int> intValidChars = { '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
		'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J',
		'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T',
		'U', 'V', 'W', 'X', 'Y', 'Z' };

	imgTrainingNumbers = cv::imread("train_data.png");          //read

	if (imgTrainingNumbers.empty()) {                               
		cout << "error: image not read from file\n\n";         
		return(0);                                                  
	}

	cvtColor(imgTrainingNumbers, imgGrayscale, CV_BGR2GRAY);        
	GaussianBlur(imgGrayscale, imgBlurred, cv::Size(5, 5), 0);
													
	adaptiveThreshold(imgBlurred, imgThresh, 255, cv::ADAPTIVE_THRESH_GAUSSIAN_C, cv::THRESH_BINARY_INV, 11, 2);

	imshow("imgThresh", imgThresh);         

	imgThreshCopy = imgThresh.clone();          

	findContours(imgThreshCopy,	ptContours,	v4iHierarchy, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
	for (int i = 0; i < ptContours.size(); i++) {                          
		if (contourArea(ptContours[i]) > MIN_CONTOUR_AREA) {              
			Rect boundingRect = cv::boundingRect(ptContours[i]);             

			rectangle(imgTrainingNumbers, boundingRect, cv::Scalar(0, 0, 255), 2);     
			Mat matROI = imgThresh(boundingRect);           
			Mat matROIResized;
			resize(matROI, matROIResized, cv::Size(RESIZED_IMAGE_WIDTH, RESIZED_IMAGE_HEIGHT)); 
			imshow("matROI", matROI);                               
			imshow("matROIResized", matROIResized);                 
			imshow("imgTrainingNumbers", imgTrainingNumbers);       

			int intChar = cv::waitKey(0);           

			if (intChar == 27) {        
				return(0);              
			}
			else if (find(intValidChars.begin(), intValidChars.end(), intChar) != intValidChars.end()) {

				matClassificationInts.push_back(intChar);       

				Mat matImageFloat;                          
				matROIResized.convertTo(matImageFloat, CV_32FC1);     

				Mat matImageFlattenedFloat = matImageFloat.reshape(1, 1);    

				matTrainingImagesAsFlattenedFloats.push_back(matImageFlattenedFloat);      
																							
			}   
		}   
	}  

	cout << "training complete\n\n";
		
	FileStorage fsClassifications("classifications.xml", FileStorage::WRITE);          
	if (fsClassifications.isOpened() == false) {                                               
		cout << "error, unable to open training classifications file, exiting program\n\n";       
		return(0);                                                                                    
	}

	fsClassifications << "classifications" << matClassificationInts;       
	fsClassifications.release();                                           
																			
	FileStorage fsTrainingImages("images.xml", cv::FileStorage::WRITE); 
	
	if (fsTrainingImages.isOpened() == false) {
		cout << "error, unable to open training images file, exiting program\n\n";  
		return(0);                                                                       
	}

	fsTrainingImages << "images" << matTrainingImagesAsFlattenedFloats;     
	fsTrainingImages.release();                                             
	return(0);
}




