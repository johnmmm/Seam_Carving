//
//  Remove.hpp
//  myfirstopencvpro
//
//  Created by mac on 17/6/22.
//  Copyright © 2017年 mac. All rights reserved.
//

#ifndef Remove_hpp
#define Remove_hpp

#include <stdio.h>
#include "Carving.hpp"


class Remove
{
public:
    Remove(cv::Mat& img, cv::Mat& canvas, uchar remove = 1, uchar protect = 2): img(img), canvas(canvas)
    {
        this->img.copyTo(this->imgCopy);
        this->canvas = cv::Mat::zeros(this->img.size(), CV_8UC1);
        this->canvasColor[0] = remove;
        this->canvasColor[1] = protect;
    }
    ~Remove(){}
    
    void mark();
    
private:
    cv::Mat& img;
    cv::Mat& canvas;
    cv::Mat imgCopy;
    cv::Mat imgBlend;
    char windowName[10] = "Forgive";
    cv::Scalar painterColor[2] = {cv::Scalar(0, 255, 0), cv::Scalar(0, 0, 255)};
    uchar canvasColor[2] = {1, 2};
    int mode = 0;
    static void onMouse(int event, int x, int y, int flags, void* obj);
};

void paintremoving(cv::Mat& result, int type);
void calculateSeams(cv::Mat& canvas, cv::Mat& img, cv::Mat& result, int type);
int search_col(cv::Mat& image,cv::Mat& outImage, cv::Mat& canvas);

#endif /* Remove_hpp */
