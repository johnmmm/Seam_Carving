//
//  Enlarge.hpp
//  myfirstopencvpro
//
//  Created by mac on 17/6/22.
//  Copyright © 2017年 mac. All rights reserved.
//

#ifndef Enlarge_hpp
#define Enlarge_hpp

#include <stdio.h>
#include "Seam_carving.hpp"
#include "Carving.hpp"

void enlarge_cols(cv::Mat& image,cv::Mat& outImage, cv::Mat& outMinTrace,cv::Mat& outLargeLine);
void enlarge_rows(cv::Mat& image,cv::Mat& outImage, cv::Mat& outMinTrace,cv::Mat& outLargeLine);
void enlargeLines_cols(cv::Mat& inImage,cv::Mat&inTrace,cv::Mat& outImage, cv::Mat& matrecord);
void enlargeLines_rows(cv::Mat& inImage,cv::Mat&inTrace,cv::Mat& outImage, cv::Mat& matrecord);


#endif /* Enlarge_hpp */
