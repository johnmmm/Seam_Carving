//
//  Double_carving.cpp
//  myfirstopencvpro
//
//  Created by mac on 17/6/22.
//  Copyright © 2017年 mac. All rights reserved.
//

#include "Double_carving.hpp"

using namespace std;

void doubleCarving(cv::Mat& image)//双向裁剪
{
    //双向裁剪，计算T矩阵，最后一个图即为答案！
    long long T[BNUM+1][HNUM+1] = {0};
    long long tmpt = 0;
    cv::Mat rMat,cMat[HNUM+1],tmping,tmping1;
    T[0][0]=0;
    image.copyTo(rMat);
    image.copyTo(tmping);
    image.copyTo(tmping1);
    for(int i = 0; i <= BNUM; i++)
        for(int j = 0; j <= HNUM; j++)
        {
            if(j == 0)
            {
                rMat.copyTo(tmping);//先复制，在进行下一个的推算
                if(i == BNUM)
                    continue;
                rMat.copyTo(tmping1);
                T[i+1][j] = T[i][j] + run_a_col(rMat, tmping1);
                tmping1.copyTo(rMat);
            }
            else//在后面都是从前面获得答案的
            {
                if(i == 0)
                {
                    tmping.copyTo(tmping1);
                    T[i][j] = T[i][j-1] + run_a_row(tmping, tmping1);
                    tmping1.copyTo(cMat[j]);
                    tmping1.copyTo(tmping);
                }
                else
                {
                    cMat[j].copyTo(tmping1);
                    T[i][j] = T[i-1][j] + run_a_col(cMat[j], tmping1);
                    tmping.copyTo(tmping1);
                    tmpt = T[i][j-1] + run_a_row(tmping, tmping1);
                    if(tmpt < T[i][j])//认为现在出现的更好
                    {
                        T[i][j] = tmpt;//改变T的值
                        tmping1.copyTo(cMat[j]);
                        tmping1.copyTo(tmping);
                    }
                    else
                    {
                        cMat[j].copyTo(tmping1);
                        run_a_col(cMat[j], tmping1);//从新裁剪
                        tmping1.copyTo(cMat[j]);
                        tmping1.copyTo(tmping);
                    }
                    
                }
            }
            cout << i << ", " << j << endl;
        }
    cout << T[BNUM][HNUM] << endl;
    cv::imwrite("/Users/mac/Desktop/myself/pictures/my_space/output123.bmp", cMat[HNUM]);
    return;
}