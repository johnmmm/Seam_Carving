//#include <cv.h>
#include "opencv/highgui.h"
#include "opencv/cv.h"
#include <stdio.h>
#define BNUM 220
#define HNUM 220//用来管理高度裁剪的

//计算能量曲线
void calculateColEnergy(cv::Mat& srcMat,cv::Mat& dstMat,cv::Mat& traceMat)  //Mat 矩阵
{
    srcMat.copyTo(dstMat);   //不用“=”，防止两个矩阵指向的都是同一个矩阵，现在只需要传里面的数值
    
    for (int i = 1;i < srcMat.rows;i++)  //从第2行开始计算
    {
        //第一列
        if (dstMat.at<float>(i-1,0) <= dstMat.at<float>(i-1,1))  //.at 获取像素
        {
            dstMat.at<float>(i,0) = srcMat.at<float>(i,0) + dstMat.at<float>(i-1,0);
            traceMat.at<float>(i,0) = 1; //traceMat记录当前位置的上一行应取那个位置，上左为0，上中1，上右为2
        }
        else
        {
            dstMat.at<float>(i,0) = srcMat.at<float>(i,0) + dstMat.at<float>(i-1,1);
            traceMat.at<float>(i,0) = 2;
        }
        
        //中间列
        for (int j = 1;j < srcMat.cols-1;j++)  //获取每一个点的能量路径
        {
            float k[3];  //获取上一行对应三个位置的像素作比较
            k[0] = dstMat.at<float>(i-1,j-1);
            k[1] = dstMat.at<float>(i-1,j);
            k[2] = dstMat.at<float>(i-1,j+1);
            
            int index = 0;
            if (k[1] < k[0])
                index = 1;
            if (k[2] < k[index])
                index = 2;
            dstMat.at<float>(i,j) = srcMat.at<float>(i,j) + dstMat.at<float>(i-1,j-1+index);
            traceMat.at<float>(i,j) = index;
            
        }
        
        //最后一列
        if (dstMat.at<float>(i-1,srcMat.cols-1) <= dstMat.at<float>(i-1,srcMat.cols-2))
        {
            dstMat.at<float>(i,srcMat.cols-1) = srcMat.at<float>(i,srcMat.cols-1) + dstMat.at<float>(i-1,srcMat.cols-1);
            traceMat.at<float>(i,srcMat.cols-1) = 1;
        }
        else
        {
            dstMat.at<float>(i,srcMat.cols-1) = srcMat.at<float>(i,srcMat.cols-1) + dstMat.at<float>(i-1,srcMat.cols-2);
            traceMat.at<float>(i,srcMat.cols-1) = 0;
        }
        
    }
}

void calculateRowEnergy(cv::Mat& srcMat,cv::Mat& dstMat,cv::Mat& traceMat)  //Mat 矩阵
{
    srcMat.copyTo(dstMat);   //不用“=”，防止两个矩阵指向的都是同一个矩阵，现在只需要传里面的数值
    
    for (int i = 1;i < srcMat.cols;i++)  //从第2列开始计算
    {
        //第一列
        if (dstMat.at<float>(0,i-1) <= dstMat.at<float>(1,i-1))  //.at 获取像素
        {
            dstMat.at<float>(0,i) = srcMat.at<float>(0,i) + dstMat.at<float>(0,i-1);
            traceMat.at<float>(0,i) = 1; //traceMat记录当前位置的上一行应取那个位置，上左为0，上中1，上右为2
        }
        else
        {
            dstMat.at<float>(0,i) = srcMat.at<float>(0,i) + dstMat.at<float>(1,i-1);
            traceMat.at<float>(0,i) = 2;
        }
        
        //中间列
        for (int j = 1;j < srcMat.rows-1;j++)  //获取每一个点的能量路径
        {
            float k[3];  //获取上一行对应三个位置的像素作比较
            k[0] = dstMat.at<float>(j-1,i-1);
            k[1] = dstMat.at<float>(j-1,i);
            k[2] = dstMat.at<float>(j-1,i+1);
            
            int index = 0;
            if (k[1] < k[0])
                index = 1;
            if (k[2] < k[index])
                index = 2;
            dstMat.at<float>(j,i) = srcMat.at<float>(j,i) + dstMat.at<float>(j-1+index,i-1);
            traceMat.at<float>(j,i) = index;
            
        }
        
        //最后一行
        if (dstMat.at<float>(srcMat.rows-1,i-1) <= dstMat.at<float>(srcMat.rows-2,i-1))
        {
            dstMat.at<float>(srcMat.rows-1,i) = srcMat.at<float>(srcMat.rows-1,i) + dstMat.at<float>(srcMat.rows-1,i-1);
            traceMat.at<float>(srcMat.rows-1,i) = 1;
        }
        else
        {
            dstMat.at<float>(srcMat.rows-1,i) = srcMat.at<float>(srcMat.rows-1,i) + dstMat.at<float>(srcMat.rows-2,i-1);
            traceMat.at<float>(srcMat.rows-1,i) = 0;
        }
        
    }
}

// 找出最小能量线
void getMinColEnergyTrace(const cv::Mat& energyMat,const cv::Mat& traceMat,cv::Mat& minTrace)
{
    int row = energyMat.rows - 1;// 取的是energyMat最后一行的数据，所以行标是rows-1
    
    int index = 0;	// 保存的是最小那条轨迹的最下面点在图像中的列标
    
    // 获得index，即最后那行最小值的位置
    for (int i = 1;i < energyMat.cols;i++)
    {
        if (energyMat.at<float>(row,i) < energyMat.at<float>(row,index))
        {
            index = i;
        } // end if
    } // end for i = ...
    
    // 以下根据traceMat，得到minTrace，minTrace是多行一列矩阵
    {
        minTrace.at<float>(row,0) = index;
        
        int tmpIndex = index;
        
        for (int i = row;i > 0;i--)
        {
            int temp = traceMat.at<float>(i,tmpIndex);// 当前位置traceMat所存的值
            
            if (temp == 0) // 往左走
            {
                tmpIndex = tmpIndex - 1;
            }
            else if (temp == 2) // 往右走
            {
                tmpIndex = tmpIndex + 1;
            } // 如果temp = 1，则往正上走，tmpIndex不需要做修改
            
            minTrace.at<float>(i-1,0) = tmpIndex;
        }
    }
}

void getMinRowEnergyTrace(const cv::Mat& energyMat,const cv::Mat& traceMat,cv::Mat& minTrace)
{
    int col = energyMat.cols - 1;// 取的是energyMat最后一列的数据，所以行标是cols-1
    
    int index = 0;	// 保存的是最小那条轨迹的最下面点在图像中的列标
    
    // 获得index，即最后那行最小值的位置
    for (int i = 1;i < energyMat.rows;i++)
    {
        if (energyMat.at<float>(i,col) < energyMat.at<float>(index,col))
        {
            index = i;
        } // end if
    } // end for i = ...
    
    // 以下根据traceMat，得到minTrace，minTrace是多行一列矩阵
    {
        minTrace.at<float>(0,col) = index;
        
        int tmpIndex = index;
        
        for (int i = col;i > 0;i--)
        {
            int temp = traceMat.at<float>(tmpIndex,i);// 当前位置traceMat所存的值
            
            if (temp == 0) // 往左走
            {
                tmpIndex = tmpIndex - 1;
            }
            else if (temp == 2) // 往右走
            {
                tmpIndex = tmpIndex + 1;
            } // 如果temp = 1，则往正上走，tmpIndex不需要做修改
            
            minTrace.at<float>(0,i-1) = tmpIndex;
        }
    }
}

// 删掉一列
void delOneCol(cv::Mat& srcMat,cv::Mat& dstMat,cv::Mat& minTrace,cv::Mat& beDeletedLine)
{
    for (int i = 0;i < dstMat.rows;i++)  //对目标图像进行操作
    {
        int k = minTrace.at<float>(i,0);
        
        for (int j = 0;j < k;j++)  //删除列前的元素复制
        {
            dstMat.at<cv::Vec3b>(i,j)[0] = srcMat.at<cv::Vec3b>(i,j)[0];  // 8U 类型的 RGB 彩色图像 [i] 存放bgr的值
            dstMat.at<cv::Vec3b>(i,j)[1] = srcMat.at<cv::Vec3b>(i,j)[1];
            dstMat.at<cv::Vec3b>(i,j)[2] = srcMat.at<cv::Vec3b>(i,j)[2];
        }
        for (int j = k;j < dstMat.cols-1;j++)  //复制删除列后的元素 最后一列默认不处理
        {
            if (j == dstMat.cols-1)
                continue;
            dstMat.at<cv::Vec3b>(i,j)[0] = srcMat.at<cv::Vec3b>(i,j+1)[0];
            dstMat.at<cv::Vec3b>(i,j)[1] = srcMat.at<cv::Vec3b>(i,j+1)[1];
            dstMat.at<cv::Vec3b>(i,j)[2] = srcMat.at<cv::Vec3b>(i,j+1)[2];
            
        }
        beDeletedLine.at<cv::Vec3b>(i,0)[0] = srcMat.at<cv::Vec3b>(i,k)[0];
        beDeletedLine.at<cv::Vec3b>(i,0)[1] = srcMat.at<cv::Vec3b>(i,k)[1];
        beDeletedLine.at<cv::Vec3b>(i,0)[2] = srcMat.at<cv::Vec3b>(i,k)[2];
    }
}

//删除一行
void delOneRow(cv::Mat& srcMat,cv::Mat& dstMat,cv::Mat& minTrace,cv::Mat& beDeletedLine)
{
    for (int i = 0;i < dstMat.cols;i++)  //对目标图像进行操作
    {
        int k = minTrace.at<float>(0,i);
        
        for (int j = 0;j < k;j++)  //删除列前的元素复制
        {
            dstMat.at<cv::Vec3b>(j,i)[0] = srcMat.at<cv::Vec3b>(j,i)[0];  // 8U 类型的 RGB 彩色图像 [i] 存放bgr的值
            dstMat.at<cv::Vec3b>(j,i)[1] = srcMat.at<cv::Vec3b>(j,i)[1];
            dstMat.at<cv::Vec3b>(j,i)[2] = srcMat.at<cv::Vec3b>(j,i)[2];
        }
        for (int j = k;j < dstMat.rows-1;j++)  //复制删除列后的元素 最后一列默认不处理
        {
            if (j == dstMat.rows-1)
                continue;
            dstMat.at<cv::Vec3b>(j,i)[0] = srcMat.at<cv::Vec3b>(j+1,i)[0];
            dstMat.at<cv::Vec3b>(j,i)[1] = srcMat.at<cv::Vec3b>(j+1,i)[1];
            dstMat.at<cv::Vec3b>(j,i)[2] = srcMat.at<cv::Vec3b>(j+1,i)[2];
            
        }
        beDeletedLine.at<cv::Vec3b>(0,i)[0] = srcMat.at<cv::Vec3b>(k,i)[0];
        beDeletedLine.at<cv::Vec3b>(0,i)[1] = srcMat.at<cv::Vec3b>(k,i)[1];
        beDeletedLine.at<cv::Vec3b>(0,i)[2] = srcMat.at<cv::Vec3b>(k,i)[2];
    }
}

//跑两种裁剪
void run_a_col(cv::Mat& image,cv::Mat& outImage)
{
    //cv::Mat& outMinTrace,cv::Mat& outDeletedLine
    cv::Mat image_gray(image.rows,image.cols,CV_8U,cv::Scalar(0));  //scalar 初始化为0 cv_8u 单通道阵列
    cv::cvtColor(image,image_gray,CV_BGR2GRAY); //彩色图像转换为灰度图像
    
    cv::Mat gradiant_H(image.rows,image.cols,CV_32F,cv::Scalar(0));//水平梯度矩阵  32bit浮点数
    cv::Mat gradiant_V(image.rows,image.cols,CV_32F,cv::Scalar(0));//垂直梯度矩阵
    
//    cv::Mat kernel_H = (cv::Mat_<float>(3,3) << 0, 0, 0, 0, 1, -1, 0, 0, 0); //求水平梯度所使用的卷积核（赋初始值）
//    cv::Mat kernel_V = (cv::Mat_<float>(3,3) << 0, 0, 0, 0, 1, 0, 0, -1, 0); //求垂直梯度所使用的卷积核（赋初始值） x的转置
//    
//    cv::filter2D(image_gray,gradiant_H,gradiant_H.depth(),kernel_H);  //实现函数的卷积运算  depth()为精度
//    cv::filter2D(image_gray,gradiant_V,gradiant_V.depth(),kernel_V);
    
    int scale = 1; int delta = 0;
    Sobel(image_gray, gradiant_H, gradiant_H.depth(), 1, 0, 3, scale, delta);  //dx dy ksize
    Sobel(image_gray, gradiant_V, gradiant_V.depth(), 0, 1, 3, scale, delta);
    
    cv::Mat gradMag_mat(image.rows,image.rows,CV_32F,cv::Scalar(0));
    cv::add(cv::abs(gradiant_H),cv::abs(gradiant_V),gradMag_mat);//水平与垂直滤波结果的绝对值相加，可以得到近似梯度大小
    
    ////如果要显示梯度大小这个图，因为gradMag_mat深度是CV_32F，所以需要先转换为CV_8U
//    cv::Mat testMat;
//    gradMag_mat.convertTo(testMat,CV_8U,1,0);
//    cv::imshow("Image Show Window2",testMat);
    
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
    
    //删除一列
    cv::Mat image2(image.rows,image.cols-1,image.type());
    cv::Mat beDeletedLine(image.rows,1,CV_8UC3);//记录被删掉的那一列的值
    delOneCol(image,image2,minTrace,beDeletedLine);
    //cv::imshow("Image Show Window",image2);
    
    image2.copyTo(outImage);
    //minTrace.copyTo(outMinTrace);
    //beDeletedLine.copyTo(outDeletedLine);
}

void run_a_row(cv::Mat& image,cv::Mat& outImage)
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
    //cv::imshow("Image Show Window (?)",image);
    
    cv::Mat image_gray(image.rows,image.cols,CV_8U,cv::Scalar(0));  //scalar 初始化为0 cv_8u 单通道阵列
    cv::cvtColor(image,image_gray,CV_BGR2GRAY); //彩色图像转换为灰度图像
    
    cv::Mat gradiant_H(image.rows,image.cols,CV_32F,cv::Scalar(0));//水平梯度矩阵  32bit浮点数
    cv::Mat gradiant_V(image.rows,image.cols,CV_32F,cv::Scalar(0));//垂直梯度矩阵
    
    //    cv::Mat kernel_H = (cv::Mat_<float>(3,3) << 0, 0, 0, 0, 1, -1, 0, 0, 0); //求水平梯度所使用的卷积核（赋初始值）
    //    cv::Mat kernel_V = (cv::Mat_<float>(3,3) << 0, 0, 0, 0, 1, 0, 0, -1, 0); //求垂直梯度所使用的卷积核（赋初始值） x的转置
    //
    //    cv::filter2D(image_gray,gradiant_H,gradiant_H.depth(),kernel_H);  //实现函数的卷积运算  depth()为精度
    //    cv::filter2D(image_gray,gradiant_V,gradiant_V.depth(),kernel_V);
    
    int scale = 1; int delta = 0;
    Sobel(image_gray, gradiant_H, gradiant_H.depth(), 1, 0, 3, scale, delta);  //dx dy ksize
    Sobel(image_gray, gradiant_V, gradiant_V.depth(), 0, 1, 3, scale, delta);
    
    cv::Mat gradMag_mat(image.rows,image.rows,CV_32F,cv::Scalar(0));
    cv::add(cv::abs(gradiant_H),cv::abs(gradiant_V),gradMag_mat);//水平与垂直滤波结果的绝对值相加，可以得到近似梯度大小
    
    ////如果要显示梯度大小这个图，因为gradMag_mat深度是CV_32F，所以需要先转换为CV_8U
    //    cv::Mat testMat;
    //    gradMag_mat.convertTo(testMat,CV_8U,1,0);
    //    cv::imshow("Image Show Window2",testMat);
    
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
    //显示的时候翻转过来
    for(int i = 0; i < image.rows; i++)
        for(int j = 0; j < image.cols; j++)
        {
            reverseY.at<cv::Vec3b>(j,i)[0] = tmpImage.at<cv::Vec3b>(i,j)[0];
            reverseY.at<cv::Vec3b>(j,i)[1] = tmpImage.at<cv::Vec3b>(i,j)[1];
            reverseY.at<cv::Vec3b>(j,i)[2] = tmpImage.at<cv::Vec3b>(i,j)[2];
        }
    
    cv::imshow("Image Show Window (A)",reverseY);  //显示最小能量线，过程
    
    //删除一列
    cv::Mat image2(image.rows,image.cols-1,image.type());
    cv::Mat reverse2(image.cols-1,image.rows,image.type());
    cv::Mat beDeletedLine(image.rows,1,CV_8UC3);//记录被删掉的那一列的值
    delOneCol(image,image2,minTrace,beDeletedLine);
    //cv::imshow("Image Show Window",image2);
    
    
    //结束的时候再翻转回来
    for(int i = 0; i < image2.rows; i++)
        for(int j = 0; j < image2.cols; j++)
        {
            reverse2.at<cv::Vec3b>(j,i)[0] = image2.at<cv::Vec3b>(i,j)[0];
            reverse2.at<cv::Vec3b>(j,i)[1] = image2.at<cv::Vec3b>(i,j)[1];
            reverse2.at<cv::Vec3b>(j,i)[2] = image2.at<cv::Vec3b>(i,j)[2];
        }
    reverse2.copyTo(outImage);
    ///image2.copyTo(outImage);
    //minTrace.copyTo(outMinTrace);
    //beDeletedLine.copyTo(outDeletedLine);
}

//恢复？？
void recoverOneLine(cv::Mat& inImage,cv::Mat&inTrace,cv::Mat& inDeletedLine,cv::Mat& outImage)
{
    
    cv::Mat recorvedImage(inImage.rows,inImage.cols+1,CV_8UC3);  //三通道，8bit
    for (int i = 0; i < inImage.rows; i++)
    {
        int k = inTrace.at<float>(i);
        for (int j = 0; j < k; j++)
        {
            recorvedImage.at<cv::Vec3b>(i,j)[0] = inImage.at<cv::Vec3b>(i,j)[0];
            recorvedImage.at<cv::Vec3b>(i,j)[1] = inImage.at<cv::Vec3b>(i,j)[1];
            recorvedImage.at<cv::Vec3b>(i,j)[2] = inImage.at<cv::Vec3b>(i,j)[2];
        }
        recorvedImage.at<cv::Vec3b>(i,k)[0] = inDeletedLine.at<cv::Vec3b>(i,0)[0];
        recorvedImage.at<cv::Vec3b>(i,k)[1] = inDeletedLine.at<cv::Vec3b>(i,0)[1];
        recorvedImage.at<cv::Vec3b>(i,k)[2] = inDeletedLine.at<cv::Vec3b>(i,0)[2];
        
        for (int j = k + 1;j < inImage.cols + 1; j++)
        {
            recorvedImage.at<cv::Vec3b>(i,j)[0] = inImage.at<cv::Vec3b>(i,j-1)[0];
            recorvedImage.at<cv::Vec3b>(i,j)[1] = inImage.at<cv::Vec3b>(i,j-1)[1];
            recorvedImage.at<cv::Vec3b>(i,j)[2] = inImage.at<cv::Vec3b>(i,j-1)[2];
        }
    }
    
    //显示恢复的轨迹
    cv::Mat tmpImage(recorvedImage.rows,recorvedImage.cols,recorvedImage.type());
    recorvedImage.copyTo(tmpImage);
    for (int i = 0;i < tmpImage.rows;i++)
    {
        int k = inTrace.at<float>(i,0);
        tmpImage.at<cv::Vec3b>(i,k)[0] = 0;
        tmpImage.at<cv::Vec3b>(i,k)[1] = 255;
        tmpImage.at<cv::Vec3b>(i,k)[2] = 0;
    }
    cv::imshow("Image Show Window (B)",tmpImage);
    
    recorvedImage.copyTo(outImage);
}

int main(int argc,char** argv)
{
    cv::Mat image = cv::imread("/Users/mac/Desktop/myself/pictures/my space/1.jpg");  //加载图像
    cv::namedWindow("Original Image");
    cv::imshow("Original Image",image);
    
    cv::Mat tmpMat;
    image.copyTo(tmpMat); 
    
    //cv::Mat traces[BNUM];
    //cv::Mat deletedLines[BNUM];//记录裁剪一列所用的
    
    cv::Mat outImage;
    
    cv::waitKey(2000);
    
    //共同裁剪
    for (int i = 0;i < BNUM;i++)
    {
        run_a_col(tmpMat,outImage);
        tmpMat = outImage;
        //cv::waitKey(50);
        run_a_row(tmpMat,outImage);
        tmpMat = outImage;
        cv::waitKey(50);
    }
    
    //分开进行不同程度的裁剪
//    for (int i = 0;i < BNUM;i++)
//    {
//        run_a_col(tmpMat,outImage);
//        tmpMat = outImage;
//        cv::waitKey(50);
//    }
//    for (int i = 0;i < HNUM;i++)
//    {
//        run_a_row(tmpMat,outImage);
//        tmpMat = outImage;
//        cv::waitKey(50);
//    }
    
    cv::imwrite("/Users/mac/Desktop/myself/pictures/my space/output.bmp", outImage);
    
    //显示恢复路径
//    cv::Mat tmpMat2;
//    outImage.copyTo(tmpMat2);
//     
//    for (int i = 0; i < BNUM; i++)
//    {
//     
//        recoverOneLine(tmpMat2,traces[BNUM-i-1],deletedLines[BNUM-i-1],outImage);
//        tmpMat2 = outImage;
//        cv::waitKey(50);
//    }
    cv::waitKey(115000);
    return 0;
    
}