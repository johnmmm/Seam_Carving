//
//  Remove.cpp
//  myfirstopencvpro
//
//  Created by mac on 17/6/22.
//  Copyright © 2017年 mac. All rights reserved.
//

#include "Remove.hpp"

void Remove::mark()
{
    printf("Paint the area you want to reomve <RIGHT click to switch>\n");
    cv::namedWindow(this->windowName);
    cv::setMouseCallback(this->windowName, this->onMouse, this);
    cv::imshow(this->windowName, this->imgCopy);
    cv::waitKey(0);
}

void Remove::onMouse(int event, int x, int y, int flags, void* obj)
{
    Remove* marker = reinterpret_cast<Remove*>(obj);
    if(event == CV_EVENT_LBUTTONDOWN || (event == CV_EVENT_MOUSEMOVE && (flags & CV_EVENT_FLAG_LBUTTON)))
    {
        cv::circle(marker->imgCopy, cv::Point(x, y), 20, marker->painterColor[marker->mode], -1);
        cv::circle(marker->canvas, cv::Point(x, y), 20, marker->canvasColor[marker->mode], -1);
        cv::addWeighted(marker->img, 0.6, marker->imgCopy, 0.4, 0, marker->imgBlend);
        cv::imshow(marker->windowName, marker->imgBlend);
    }
    if(event == CV_EVENT_RBUTTONDOWN)
    {
        if(marker->mode == 1)
        {
            marker->mode = 0;
            printf("Paint the area you want to remove <RIGHT click to switch>\n");
        }
        else
        {
            marker->mode = 1;
            printf("Paint the area you want to protect <RIGHT click to switch>\n");
        }
    }
}



void paintremoving(cv::Mat& result, int type)//0代表纵切，1代表横切
{
    cv::Mat image = cv::imread("/Users/mac/Desktop/programme/program/4st_term/seamcarving/7.jpg");
    
    cv::Mat canvas = cv::Mat::zeros(image.rows, image.cols, CV_8UC1);
    Remove marker(image, canvas);
    marker.mark();
    calculateSeams(canvas, image, result, type);
}

void calculateSeams(cv::Mat& canvas, cv::Mat& img, cv::Mat& result, int type)
{
    if(type == 0)
    {
        cv::Mat tmpMat;
        img.copyTo(tmpMat);
        for (int i = 0;i < LNUM;i++)
        {
            search_col(img, tmpMat, canvas);
            tmpMat.copyTo(img);
            //cout << i << endl;
        }
        img.copyTo(result);
    }
    else
    {
        
    }
}

int search_col(cv::Mat& image,cv::Mat& outImage, cv::Mat& canvas)
{
    cv::Mat image_gray(image.rows,image.cols,CV_8U,cv::Scalar(0));  //scalar 初始化为0 cv_8u 单通道阵列
    cv::cvtColor(image,image_gray,CV_BGR2GRAY); //彩色图像转换为灰度图像
    
    cv::Mat gradiant_H(image.rows,image.cols,CV_32F,cv::Scalar(0));//水平梯度矩阵  32bit浮点数
    cv::Mat gradiant_V(image.rows,image.cols,CV_32F,cv::Scalar(0));//垂直梯度矩阵
    
    cv::Mat kernel_H = (cv::Mat_<float>(3,3) << 0, 0, 0, 0, 1, -1, 0, 0, 0); //求水平梯度所使用的卷积核（赋初始值）
    cv::Mat kernel_V = (cv::Mat_<float>(3,3) << 0, 0, 0, 0, 1, 0, 0, -1, 0); //求垂直梯度所使用的卷积核（赋初始值） x的转置
    
    cv::filter2D(image_gray,gradiant_H,gradiant_H.depth(),kernel_H);  //实现函数的卷积运算  depth()为精度
    cv::filter2D(image_gray,gradiant_V,gradiant_V.depth(),kernel_V);
    
    //    int scale = 1; int delta = 0;
    //    Sobel(image_gray, gradiant_H, gradiant_H.depth(), 1, 0, 3, scale, delta);  //dx dy ksize
    //    Sobel(image_gray, gradiant_V, gradiant_V.depth(), 0, 1, 3, scale, delta);
    
    cv::Mat gradMag_mat(image.rows,image.rows,CV_32F,cv::Scalar(0));
    cv::add(cv::abs(gradiant_H),cv::abs(gradiant_V),gradMag_mat);//水平与垂直滤波结果的绝对值相加，可以得到近似梯度大小
    
    for(int i = 0; i < image.rows; i++)
        for(int j = 0; j < image.cols; j++)
        {
            if(canvas.at<uchar>(i,j) == 1)
                gradMag_mat.at<float>(i,j) = -10000000000;
            else if(canvas.at<uchar>(i,j) == 2)
                gradMag_mat.at<float>(i,j) += 100000000;
        }
    
    //计算能量线
    cv::Mat energyMat(image.rows,image.cols,CV_32F,cv::Scalar(0));//累计能量矩阵
    cv::Mat traceMat(image.rows,image.cols,CV_32F,cv::Scalar(0));//能量最小轨迹矩阵
    calculateColEnergy(gradMag_mat,energyMat,traceMat);
    
    //找出最小能量线
    cv::Mat minTrace(image.rows,1,CV_32F,cv::Scalar(0));//能量最小轨迹矩阵中的最小的一条的轨迹
    getMinColEnergyTrace(energyMat,traceMat,minTrace);
    
    //显示最小能量线
    int totalEnergy = 0;
    cv::Mat tmpImage(image.rows,image.cols,image.type());
    image.copyTo(tmpImage);  //复制
    for (int i = 0;i < image.rows;i++)
    {
        int k = minTrace.at<float>(i,0);
        tmpImage.at<cv::Vec3b>(i,k)[0] = 0;
        tmpImage.at<cv::Vec3b>(i,k)[1] = 0;
        tmpImage.at<cv::Vec3b>(i,k)[2] = 255;
        totalEnergy += energyMat.at<float>(i,k);//计算总的值，用来计算第二层矩阵
    }
    cv::imshow("Image Show Window (A)",tmpImage);  //显示最小能量线，过程
    
    //删除一列
    cv::Mat image2(image.rows,image.cols-1,image.type());
    cv::Mat beDeletedLine(image.rows,1,CV_8UC3);//记录被删掉的那一列的值
    delOneCol(image,image2,minTrace,beDeletedLine);
    
    cv::Mat tmping;
    canvas.copyTo(tmping);
    
    for(int i = 0; i < tmping.rows; i++)
    {
        int k = minTrace.at<float>(i,0);
        
        for(int j = 0; j < k; j++)
        {
            tmping.at<uchar>(i,j) = canvas.at<uchar>(i,j);
        }
        for (int j = k; j < tmping.cols-1; j++)
        {
            if(j == tmping.cols - 1)
                continue;
            tmping.at<uchar>(i,j) = canvas.at<uchar>(i,j+1);
        }
    }
    tmping.copyTo(canvas);
    
    image2.copyTo(outImage);

    return totalEnergy;
}