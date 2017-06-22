//
//  Carving.hpp
//  myfirstopencvpro
//
//  Created by mac on 17/6/22.
//  Copyright © 2017年 mac. All rights reserved.
//

#ifndef Carving_hpp
#define Carving_hpp

#include <stdio.h>
#include "Seam_carving.hpp"

void calculateColEnergy(cv::Mat& srcMat,cv::Mat& dstMat,cv::Mat& traceMat);
void getMinColEnergyTrace(const cv::Mat& energyMat,const cv::Mat& traceMat,cv::Mat& minTrace);
void delOneCol(cv::Mat& srcMat,cv::Mat& dstMat,cv::Mat& minTrace,cv::Mat& beDeletedLine);
int run_a_col(cv::Mat& image,cv::Mat& outImage);
int run_a_row(cv::Mat& image,cv::Mat& outImage);



#endif /* Carving_hpp */
