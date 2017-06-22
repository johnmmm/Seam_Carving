//#include <cv.h>

#include <stdio.h>
#include "Seam_carving.hpp"
#include "Carving.hpp"
#include "Double_carving.hpp"
#include "Enlarge.hpp"
#include "Remove.hpp"

using namespace std;


int main(int argc,char** argv)
{
//    cv::Mat image = cv::imread("/Users/mac/Desktop/programme/program/4st_term/seamcarving/1.jpg");  //加载图像
//    cv::namedWindow("Original Image");
//    cv::imshow("Original Image",image);
//    
//    cv::Mat tmpMat;
//    image.copyTo(tmpMat);
//    
    
    
    cv::Mat outImage;
//
    cv::waitKey(2000);
    
    //进行双向同时裁剪
    //doubleCarving(tmpMat);
    
    //进行放大操作
//    cv::Mat traces[LNUM];
//    cv::Mat largeLines[LNUM];//记录裁剪一列所用的
//    for(int i = 0; i < LNUM; i++)
//    {
//        enlarge_cols(tmpMat, outImage, traces[i], largeLines[i]);
//        outImage.copyTo(tmpMat);
//        cv::waitKey(100);
//    }
//    image.copyTo(tmpMat);
//    
//    cv::Mat matrecord(image.rows,image.cols,CV_32F,cv::Scalar(0));
//    for(int i = 0; i < image.rows; i++)
//        for(int j = 0; j < image.cols; j++)//上来先置0
//            matrecord.at<float>(i,j) = 0;
//    for(int i = 0; i < LNUM; i++)
//    {
//        enlargeLines_cols(tmpMat, traces[i], outImage, matrecord);
//        tmpMat = outImage;
//    }
    //纵向放大
//    cv::Mat traces[LNUM];
//    cv::Mat largeLines[LNUM];//记录裁剪一列所用的
//    for(int i = 0; i < LNUM; i++)
//    {
//        enlarge_rows(tmpMat, outImage, traces[i], largeLines[i]);
//        outImage.copyTo(tmpMat);
//        cv::waitKey(100);
//    }
//    image.copyTo(tmpMat);
//    
//    cv::Mat matrecord(image.cols,image.rows,CV_32F,cv::Scalar(0));
//    for(int i = 0; i < image.cols; i++)
//        for(int j = 0; j < image.rows; j++)//上来先置0
//            matrecord.at<float>(i,j) = 0;
//    for(int i = 0; i < LNUM; i++)
//    {
//        enlargeLines_rows(tmpMat, traces[i], outImage, matrecord);
//        tmpMat = outImage;
//    }
    
    //分开进行不同程度的裁剪
//    for (int i = 0;i < BNUM;i++)
//    {
//        run_a_col(tmpMat,outImage);
//        tmpMat = outImage;
//        cout << i << endl;
//    }
//    for (int i = 0;i < HNUM;i++)
//    {
//        run_a_row(tmpMat,outImage);
//        tmpMat = outImage;
//        cout << i << endl;
//    }
    //显示恢复路径
//    cv::Mat tmpMat2;
//    outImage.copyTo(tmpMat2);
//    for (int i = 0; i < BNUM; i++)
//    {
//    
//        recoverOneLine(tmpMat2,traces[BNUM-i-1],deletedLines[BNUM-i-1],outImage);
//        tmpMat2 = outImage;
//        cv::waitKey(50);
//    }
    
    paintremoving(outImage, 0);
    
    cv::imwrite("/Users/mac/Desktop/programme/program/4st_term/seamcarving/output_1_large.bmp", outImage);
    return 0;
    
}
