//
//  Enlarge.cpp
//  myfirstopencvpro
//
//  Created by mac on 17/6/22.
//  Copyright © 2017年 mac. All rights reserved.
//

#include "Enlarge.hpp"


//计算放大的时候所需要的曲线
void enlarge_cols(cv::Mat& image,cv::Mat& outImage, cv::Mat& outMinTrace,cv::Mat& outLargeLine)
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
    
    //计算能量线
    cv::Mat energyMat(image.rows,image.cols,CV_32F,cv::Scalar(0));//累计能量矩阵
    cv::Mat traceMat(image.rows,image.cols,CV_32F,cv::Scalar(0));//能量最小轨迹矩阵
    calculateColEnergy(gradMag_mat,energyMat,traceMat);
    
    //找出最小能量线
    cv::Mat minTrace(image.rows,1,CV_32F,cv::Scalar(0));//能量最小轨迹矩阵中的最小的一条的轨迹
    getMinColEnergyTrace(energyMat,traceMat,minTrace);
    
    //显示最小能量线
    cv::Mat tmpImage(image.rows,image.cols,image.type());
    image.copyTo(tmpImage);  //复制
    for (int i = 0;i < image.rows;i++)
    {
        int k = minTrace.at<float>(i,0);
        tmpImage.at<cv::Vec3b>(i,k)[0] = 0;
        tmpImage.at<cv::Vec3b>(i,k)[1] = 0;
        tmpImage.at<cv::Vec3b>(i,k)[2] = 255;
    }
    cv::imshow("Image Show Window (A)",tmpImage);  //显示最小能量线，过程
    
    minTrace.copyTo(outMinTrace);
    tmpImage.copyTo(outImage);
}

void enlarge_rows(cv::Mat& image,cv::Mat& outImage, cv::Mat& outMinTrace,cv::Mat& outLargeLine)
{
    cv::Mat reverseY, reverseM(image.cols,image.rows,image.type());
    image.copyTo(reverseY);
    for(int i = 0; i < image.rows; i++)
        for(int j = 0; j < image.cols; j++)
        {
            reverseM.at<cv::Vec3b>(j,i)[0] = reverseY.at<cv::Vec3b>(i,j)[0];
            reverseM.at<cv::Vec3b>(j,i)[1] = reverseY.at<cv::Vec3b>(i,j)[1];
            reverseM.at<cv::Vec3b>(j,i)[2] = reverseY.at<cv::Vec3b>(i,j)[2];
        }
    reverseM.copyTo(image);
    
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
    
    //计算能量线
    cv::Mat energyMat(image.rows,image.cols,CV_32F,cv::Scalar(0));//累计能量矩阵
    cv::Mat traceMat(image.rows,image.cols,CV_32F,cv::Scalar(0));//能量最小轨迹矩阵
    calculateColEnergy(gradMag_mat,energyMat,traceMat);
    
    //找出最小能量线
    cv::Mat minTrace(image.rows,1,CV_32F,cv::Scalar(0));//能量最小轨迹矩阵中的最小的一条的轨迹
    getMinColEnergyTrace(energyMat,traceMat,minTrace);
    
    //显示最小能量线
    cv::Mat tmpImage(image.rows,image.cols,image.type());
    image.copyTo(tmpImage);  //复制
    for (int i = 0;i < image.rows;i++)
    {
        int k = minTrace.at<float>(i,0);
        tmpImage.at<cv::Vec3b>(i,k)[0] = 0;
        tmpImage.at<cv::Vec3b>(i,k)[1] = 0;
        tmpImage.at<cv::Vec3b>(i,k)[2] = 255;
    }
    cv::imshow("Image Show Window (A)",tmpImage);  //显示最小能量线，过程
    
    cv::Mat reverseY1, reverseM1(tmpImage.cols,tmpImage.rows,tmpImage.type());
    tmpImage.copyTo(reverseY1);
    for(int i = 0; i < tmpImage.rows; i++)
        for(int j = 0; j < tmpImage.cols; j++)
        {
            reverseM1.at<cv::Vec3b>(j,i)[0] = reverseY1.at<cv::Vec3b>(i,j)[0];
            reverseM1.at<cv::Vec3b>(j,i)[1] = reverseY1.at<cv::Vec3b>(i,j)[1];
            reverseM1.at<cv::Vec3b>(j,i)[2] = reverseY1.at<cv::Vec3b>(i,j)[2];
        }
    reverseM1.copyTo(tmpImage);
    
    minTrace.copyTo(outMinTrace);
    tmpImage.copyTo(outImage);
}

//进行放大操作
void enlargeLines_cols(cv::Mat& inImage,cv::Mat&inTrace,cv::Mat& outImage, cv::Mat& matrecord)
{
    cv::Mat enlargedImage(inImage.rows,inImage.cols+1,CV_8UC3);  //三通道，8bit
    for (int i = 0; i < inImage.rows; i++)
    {
        int k = inTrace.at<float>(i);
        k = k + matrecord.at<float>(i,k);
        for (int j = 0; j < k; j++)
        {
            enlargedImage.at<cv::Vec3b>(i,j)[0] = inImage.at<cv::Vec3b>(i,j)[0];
            enlargedImage.at<cv::Vec3b>(i,j)[1] = inImage.at<cv::Vec3b>(i,j)[1];
            enlargedImage.at<cv::Vec3b>(i,j)[2] = inImage.at<cv::Vec3b>(i,j)[2];
        }
        enlargedImage.at<cv::Vec3b>(i,k)[0] = int(inImage.at<cv::Vec3b>(i,k-1)[0] * 2 / 3 + inImage.at<cv::Vec3b>(i,k)[0] / 3);
        enlargedImage.at<cv::Vec3b>(i,k)[1] = int(inImage.at<cv::Vec3b>(i,k-1)[1] * 2 / 3 + inImage.at<cv::Vec3b>(i,k)[1] / 3);
        enlargedImage.at<cv::Vec3b>(i,k)[2] = int(inImage.at<cv::Vec3b>(i,k-1)[2] * 2 / 3 + inImage.at<cv::Vec3b>(i,k)[2] / 3);
        
        enlargedImage.at<cv::Vec3b>(i,k+1)[0] = int(inImage.at<cv::Vec3b>(i,k+1)[0] * 2 / 3 + inImage.at<cv::Vec3b>(i,k)[0] / 3);
        enlargedImage.at<cv::Vec3b>(i,k+1)[1] = int(inImage.at<cv::Vec3b>(i,k+1)[1] * 2 / 3 + inImage.at<cv::Vec3b>(i,k)[1] / 3);
        enlargedImage.at<cv::Vec3b>(i,k+1)[2] = int(inImage.at<cv::Vec3b>(i,k+1)[2] * 2 / 3 + inImage.at<cv::Vec3b>(i,k)[2] / 3);
        
        matrecord.at<float>(i,k+1) += 1;
        for (int j = k + 2;j <= inImage.cols + 1; j++)
        {
            enlargedImage.at<cv::Vec3b>(i,j)[0] = inImage.at<cv::Vec3b>(i,j-1)[0];
            enlargedImage.at<cv::Vec3b>(i,j)[1] = inImage.at<cv::Vec3b>(i,j-1)[1];
            enlargedImage.at<cv::Vec3b>(i,j)[2] = inImage.at<cv::Vec3b>(i,j-1)[2];
            matrecord.at<float>(i,j-1) += 1;
        }
    }
    enlargedImage.copyTo(outImage);
}

void enlargeLines_rows(cv::Mat& inImage,cv::Mat&inTrace,cv::Mat& outImage, cv::Mat& matrecord)
{
    cv::Mat reverseY, reverseM(inImage.cols,inImage.rows,inImage.type());
    inImage.copyTo(reverseY);
    for(int i = 0; i < inImage.rows; i++)
        for(int j = 0; j < inImage.cols; j++)
        {
            reverseM.at<cv::Vec3b>(j,i)[0] = reverseY.at<cv::Vec3b>(i,j)[0];
            reverseM.at<cv::Vec3b>(j,i)[1] = reverseY.at<cv::Vec3b>(i,j)[1];
            reverseM.at<cv::Vec3b>(j,i)[2] = reverseY.at<cv::Vec3b>(i,j)[2];
        }
    reverseM.copyTo(inImage);
    
    cv::Mat enlargedImage(inImage.rows,inImage.cols+1,CV_8UC3);  //三通道，8bit
    for (int i = 0; i < inImage.rows; i++)
    {
        int k = inTrace.at<float>(i);
        k = k + matrecord.at<float>(i,k);
        for (int j = 0; j < k; j++)
        {
            enlargedImage.at<cv::Vec3b>(i,j)[0] = inImage.at<cv::Vec3b>(i,j)[0];
            enlargedImage.at<cv::Vec3b>(i,j)[1] = inImage.at<cv::Vec3b>(i,j)[1];
            enlargedImage.at<cv::Vec3b>(i,j)[2] = inImage.at<cv::Vec3b>(i,j)[2];
        }
        enlargedImage.at<cv::Vec3b>(i,k)[0] = int(inImage.at<cv::Vec3b>(i,k-1)[0] * 2 / 3 + inImage.at<cv::Vec3b>(i,k)[0] / 3);
        enlargedImage.at<cv::Vec3b>(i,k)[1] = int(inImage.at<cv::Vec3b>(i,k-1)[1] * 2 / 3 + inImage.at<cv::Vec3b>(i,k)[1] / 3);
        enlargedImage.at<cv::Vec3b>(i,k)[2] = int(inImage.at<cv::Vec3b>(i,k-1)[2] * 2 / 3 + inImage.at<cv::Vec3b>(i,k)[2] / 3);
        
        enlargedImage.at<cv::Vec3b>(i,k+1)[0] = int(inImage.at<cv::Vec3b>(i,k+1)[0] * 2 / 3 + inImage.at<cv::Vec3b>(i,k)[0] / 3);
        enlargedImage.at<cv::Vec3b>(i,k+1)[1] = int(inImage.at<cv::Vec3b>(i,k+1)[1] * 2 / 3 + inImage.at<cv::Vec3b>(i,k)[1] / 3);
        enlargedImage.at<cv::Vec3b>(i,k+1)[2] = int(inImage.at<cv::Vec3b>(i,k+1)[2] * 2 / 3 + inImage.at<cv::Vec3b>(i,k)[2] / 3);
        
        matrecord.at<float>(i,k+1) += 1;
        for (int j = k + 2;j <= inImage.cols + 1; j++)
        {
            enlargedImage.at<cv::Vec3b>(i,j)[0] = inImage.at<cv::Vec3b>(i,j-1)[0];
            enlargedImage.at<cv::Vec3b>(i,j)[1] = inImage.at<cv::Vec3b>(i,j-1)[1];
            enlargedImage.at<cv::Vec3b>(i,j)[2] = inImage.at<cv::Vec3b>(i,j-1)[2];
            matrecord.at<float>(i,j-1) += 1;
        }
    }
    
    cv::Mat reverseY1, reverseM1(enlargedImage.cols,enlargedImage.rows,enlargedImage.type());
    enlargedImage.copyTo(reverseY1);
    for(int i = 0; i < enlargedImage.rows; i++)
        for(int j = 0; j < enlargedImage.cols; j++)//再翻转回来
        {
            reverseM1.at<cv::Vec3b>(j,i)[0] = reverseY1.at<cv::Vec3b>(i,j)[0];
            reverseM1.at<cv::Vec3b>(j,i)[1] = reverseY1.at<cv::Vec3b>(i,j)[1];
            reverseM1.at<cv::Vec3b>(j,i)[2] = reverseY1.at<cv::Vec3b>(i,j)[2];
        }
    reverseM1.copyTo(enlargedImage);
    
    enlargedImage.copyTo(outImage);
}