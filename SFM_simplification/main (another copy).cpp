#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/xfeatures2d.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/calib3d.hpp"
#include "opencv2/video/tracking.hpp"
#include "optimizer.hpp"
//#include "opencv2/core/eigen.hpp"

#include <math.h>
#include <fstream>
#include <iostream>

#include<Eigen/Core>




#include <Eigen/SVD>

const int MAX_FEATURES = 1000;
const int RES_X = 1280;
const int RES_Y = 720;
const float GOOD_MATCH_PERCENT = 0.3f;


using namespace cv;
//using namespace cv::xfeatures2d;
using namespace std;


int main(int argc, char* argv[])
{
    int lastAdded=-1;
    double distanceTreshold = 100.;
    vector<Point2f> backupPoints;
    vector<int> backupPointsIndex;
    cv::VideoCapture cap;
    if(argc>1)
    {
        cout<<"Trying to open "<<argv[1]<<endl;
        cap.open(argv[1]);
    }
    else
    {
        cap.open("/media/cstr/Slam_Data/NEM-L51/VID_20181022_151032.mp4");
    }

    //double f=(((double)RES_X)/2)/0.621356021; //width/2 / tan(width_angle/2) //value given by the camera constructor
    double f= 1.0796157060674482e+03;       //value given by opencv calibration

    Point2f pp(((double)RES_X)/2, ((double)RES_Y)/2); //value given by the camera constructor
    //Point2f pp(6.3823992628242706e+02, 3.6216824886639841e+02); //value given by calibration
    Mat cameraMatrix = (Mat1d(3, 3) << f, 0., pp.x, 0., f, pp.y, 0., 0., 1.0);
    Optimizer o =Optimizer(f, pp.x, pp.y);
    while(1)
    {
        static bool init=false;
        static int frame_count=14; //to activate with the condition below to skip frames, and reduce the computation cost;
        static vector<Point2f> points1; //image n-1 points
        static vector<int> pointsIndex;
        vector<Point2f> points2; //image n points
        Mat frame;
        Mat gray;
        // Capture frame-by-frame
        cap >> frame;
        // If the frame is empty, break immediately
        if (frame.empty())
        {
            cout<<"End of file"<<endl;
            break;
        }
        frame_count++;
        if (frame_count<15&&init) //indicate the number of frame to skip, if, we will only treat (1/n)th of the frames
        {

            continue;
        }


        cvtColor(frame, gray, cv::COLOR_RGB2GRAY);
        static Mat prev_gray = gray;

        // Variables to store keypoints and descriptors



        if(init)
        {
            static bool firstLoop=true;
            //KLT matches, using ORB
            if(points1.size()<MAX_FEATURES/2)
            {
                Ptr<Feature2D> orb = ORB::create(MAX_FEATURES/8, 1.2f, 8, 31, 0, 2, ORB::HARRIS_SCORE, 31, 30 );//wher does the /8 come from ? nowhere
                std::vector<KeyPoint> keypoints;
                for (int i=0; i<4; i++)
                {
                    for (int j=0; j<4; j++)
                    {
                        //TODO : correct the 31 pixels problem ***************************************
                        //FAST(prev_gray(Rect((RES_X/4)*i,(RES_Y/4)*j,(RES_X/4),(RES_Y/4))), keypoints, 40, true);
                        int x,y,w,h;
                        x=(RES_X/4)*i-31;
                        y=(RES_Y/4)*j -31;
                        w=(RES_X/4)+62;
                        h=(RES_Y/4)+62;
                        if(i==0)
                            x+=31;
                        if(i==0||i==3)
                            w-=31;
                        if(j==0)
                            y+=31;
                        if(j==0||j==3)
                            h-=31;

                        orb->detect(prev_gray(Rect(x,y,w,h)), keypoints);
                        //goodFeaturesToTrack(prev_gray, points2, MAX_FEATURES, 0.01, 25);
                        std::vector<Point2f> temp;
                        KeyPoint::convert(keypoints, points2, vector<int>());
                        temp.reserve(points1.size()+keypoints.size());
                        Point2f pt = Point2f(x,y);
                        for(int k=0; k<points2.size(); k++)
                        {
                            points2[k]+=pt;
                        }
                        temp.insert(temp.end(), points1.begin(), points1.end());
                        temp.insert(temp.end(), points2.begin(), points2.end());
                        points1=temp;
                    }
                }

                pointsIndex.reserve(points1.size());
                if (points1.size()-pointsIndex.size()>0)
                    pointsIndex.insert(pointsIndex.end(), points1.size()-pointsIndex.size(), -1);
            }


                                   /* for(int l=0; l<points1.size()-1; l++){
                            for(int k=l; k<points1.size(); k++){
                            if(abs(points1[l].x-points1[k].x)+abs(points1[l].y-points1[k].y)<16)
                             points1.erase (points1.begin() + k);
                            }
                        }
*/



            vector<float> err;
            Size winSize=Size(20,20);
            vector<uchar> status;
            TermCriteria termcrit=TermCriteria(TermCriteria::COUNT+TermCriteria::EPS, 15, 0.01);

            calcOpticalFlowPyrLK(prev_gray, gray, points1, points2, status, err, winSize, 3, termcrit, 0, 0.001);
            //getting rid of points for which the KLT tracking failed or those who have gone outside the frame
            int indexCorrection = 0;
            for( int i=0; i<status.size(); i++)
            {
                Point2f pt = points2.at(i- indexCorrection);
                if ((status.at(i) == 0)||(pt.x<0)||(pt.y<0)||err[i]>10)
                {
                    if((pt.x<0)||(pt.y<0))
                    {
                        status.at(i) = 0;
                        //cout<<"Warning : point out of frame"<<endl;
                    }
                    points1.erase (points1.begin() + i - indexCorrection);
                    points2.erase (points2.begin() + i - indexCorrection);
                    pointsIndex.erase (pointsIndex.begin() + i - indexCorrection);
                    indexCorrection++;
                }
            }

            //find the essential matrix using Nister's algorithm
            Mat mask;

            //static Mat Rt;
            //static Mat Post;


            Mat E = cv::findFundamentalMat(points2, points1, RANSAC, 0.5, 0.999, mask);
//recover the pose from the essential matrix


            indexCorrection = 0;
            for( int i=0; i<points1.size(); i++)
            {
                if (!mask.at<bool>(i,0))
                {
                    points1.erase (points1.begin() + i - indexCorrection);
                    points2.erase (points2.begin() + i - indexCorrection);
                    pointsIndex.erase (pointsIndex.begin() + i - indexCorrection);
                    indexCorrection++;
                }
            }

            E = cv::findEssentialMat(points2, points1, f, pp, RANSAC, 0.999, 0.5, mask);

            indexCorrection = 0;
            for( int i=0; i<points1.size(); i++)
            {
                if (!mask.at<bool>(i,0))
                {
                    points1.erase (points1.begin() + i - indexCorrection);
                    points2.erase (points2.begin() + i - indexCorrection);
                    pointsIndex.erase (pointsIndex.begin() + i - indexCorrection);
                    indexCorrection++;
                }
            }
/*
            mask = Mat();
            int n=myrecoverPose(E, points2, points1, cameraMatrix, Rt, Post, distanceTreshold, mask);
            if(n<points1.size()*0.6)
            {
                cout<<"frame dropped"<<endl;
                frame_count-=5;
                points1=backupPoints;
                pointsIndex=backupPointsIndex;
                init=true;
                continue;
            }

            indexCorrection = 0;
            for( int i=0; i<points1.size(); i++)
            {
                if (!mask.at<bool>(i,0))
                {
                    points1.erase (points1.begin() + i - indexCorrection);
                    points2.erase (points2.begin() + i - indexCorrection);
                    pointsIndex.erase (pointsIndex.begin() + i - indexCorrection);
                    indexCorrection++;
                    //cout<<"supprimé à cause du recoverpose"<<endl;
                }
            }*/

            /*
                    if(points1.size()<MAX_FEATURES/10){
                    cout<<points1.size()<<endl;
                      if(firstLoop){
                                      points1=points2;
                    prev_gray=gray;
                      }
                    cout<<"recalcul"<<endl;
                    continue;
                    }
                    */

            if(firstLoop)
                pointsIndex[0]=0;
            for (int i=1; i<pointsIndex.size(); i++)  //insure that only the points that have been observed at least 2 times have an index
            {
                if(pointsIndex[i]==-1)
                    pointsIndex[i]=max(pointsIndex[i-1]+1, lastAdded+1);
            }
            lastAdded=pointsIndex.back();

            Mat Ra, Rb, Post;
            decomposeEssentialMat(E, Ra, Rb, Post);

            vector<Optimizer::information> measures;
            for(int i=0; i<points1.size(); i++)
            {
                Vec3b intensity = frame.at<Vec3b>(points2[i]);
                measures.push_back(Optimizer::information(pointsIndex[i], Point2(points1[i].x,points1[i].y), 65536*intensity.val[2] + 256 * intensity.val[1] + intensity.val[0]));
            }
            //Eigen::Map<Eigen::Matrix<double, 3, 3, Eigen::RowMajor>> Rt_Eigen(Rt.ptr<double>(), 3, 3);
            Eigen::Map<Eigen::Matrix<double, 1,3, Eigen::RowMajor>> Post_Eigen(Post.ptr<double>(), 1, 3);
            Eigen::Map<Eigen::Matrix<double, 3, 3, Eigen::RowMajor>> Ra_Eigen(Ra.ptr<double>(), 3, 3);

            Eigen::Map<Eigen::Matrix<double, 3, 3, Eigen::RowMajor>> Rb_Eigen(Rb.ptr<double>(), 3, 3);






            vector<bool> mask1;
            if(!o.addObservation(measures, Ra_Eigen, Rb_Eigen, Post_Eigen, mask1)){
                            cout<<"frame dropped"<<endl;
                frame_count-=5;
                points1=backupPoints;
                pointsIndex=backupPointsIndex;
                init=true;
                continue;
            }
            indexCorrection=0;
            for(int i=0; i < mask1.size(); i++){
            if (!mask1[i])
                {
                    points1.erase (points1.begin() + i - indexCorrection);
                    points2.erase (points2.begin() + i - indexCorrection);
                    pointsIndex.erase (pointsIndex.begin() + i - indexCorrection);
                    indexCorrection++;
                }
            }
            if (indexCorrection>0)
            cout<<"supprimé à cause du recoverpose : "<<indexCorrection<<endl;

            size_t i;
            for( i = 0; i < points2.size(); i++ )
            {
                cv::circle( frame, points2[i], 2, cv::Scalar(0,255,0), 2, 8);
                cv::circle( frame, points1[i], 2, cv::Scalar(0,0,255), 2, 8);
                cv::line(frame, points1[i], points2[i], cv::Scalar(255,0,0));

            }
            firstLoop=false;
        }

        // Press  ESC on keyboard to exit
        imshow("video", frame);
        char c=(char)waitKey(0);
        if(c==27)
            break;
        frame_count=0;


        //prepare next step
        init=true;
        points1=points2;
        prev_gray=gray;
        backupPoints=points2;
        backupPointsIndex = pointsIndex;

    }
    // When everything done, release the video capture object
    cap.release();

    // Closes all the frames
    destroyAllWindows();

    return 0;
}
