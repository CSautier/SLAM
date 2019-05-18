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
    vector<Point2f> prev_points;//points from the previos keyframe
    vector<Point2f> points1; //image n-1 points
    vector<Point2f> points2; //image n points
    vector<int> pointsIndex;
    vector<Point2f> backup_points1; //image n points
    vector<int> backup_pointsIndex;
    bool init=false;
    int frame_count=14; //to activate with the condition below to skip frames, and reduce the computation cost;
    while(1)
    {
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
        cvtColor(frame, gray, cv::COLOR_RGB2GRAY);
        static Mat prev_gray = gray;

        if(init)
        {
            vector<float> err;
            Size winSize=Size(31,31);
            vector<uchar> status;
            TermCriteria termcrit=TermCriteria(TermCriteria::COUNT+TermCriteria::EPS, 15, 0.01);
            calcOpticalFlowPyrLK(prev_gray, gray, points1, points2, status, err, winSize, 3, termcrit, 0, 0.001);
            //getting rid of points for which the KLT tracking failed or those who have gone outside the frame
            int indexCorrection = 0;
            for( int i=0; i<status.size(); i++)
            {
                Point2f pt = points2.at(i- indexCorrection);
                if ((status.at(i) == 0)||(pt.x<0)||(pt.y<0)||err[i]>8)
                {
                    if((pt.x<0)||(pt.y<0))
                    {
                        status.at(i) = 0;
                        //cout<<"Warning : point out of frame"<<endl;
                    }
                    points1.erase (points1.begin() + i - indexCorrection);
                    points2.erase (points2.begin() + i - indexCorrection);
                    prev_points.erase (prev_points.begin() + i - indexCorrection);
                    pointsIndex.erase (pointsIndex.begin() + i - indexCorrection);
                    indexCorrection++;
                }
            }
            Mat mask(gray.size(), CV_8UC3, cv::Scalar(0,0,0));
        for(int i = 0; i < points2.size(); i++ )
        {
            cv::circle( mask, points2[i], 1, cv::Scalar(0,255,0), 2, 8);
        }

        // Press  ESC on keyboard to exit
        imshow("video", frame+mask);
        char c=(char)waitKey(1);
        if(c==27)
            break;

        points1=points2;
        }



        // Variables to store keypoints and descriptors

        //if(frame_count%15==0)
        if(points1.size()<MAX_FEATURES/2||frame_count==15)
        {
        frame_count=0;
            backup_points1=points1;
            backup_pointsIndex=pointsIndex;
            points1 = prev_points;



                        if(init)
            {
                static bool firstLoop=true;


                //find the essential matrix using Nister's algorithm
                Mat mask;
                cameraMatrix.at<double>(0,0)=o.fx();
                cameraMatrix.at<double>(1,1)=o.fy();
                cameraMatrix.at<double>(0,2)=o.cx();
                cameraMatrix.at<double>(1,2)=o.cy();
                Mat E = cv::findEssentialMat(points2, points1, cameraMatrix, RANSAC, 0.999, 1.0, mask);

                int indexCorrection = 0;
                for( int i=0; i<points1.size(); i++)
                {
                    if (!mask.at<bool>(i,0))
                    {
                        points1.erase (points1.begin() + i - indexCorrection);
                        points2.erase (points2.begin() + i - indexCorrection);
                        //prev_points.erase (prev_points.begin() + i - indexCorrection);
                        pointsIndex.erase (pointsIndex.begin() + i - indexCorrection);
                        indexCorrection++;
                    }
                }
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
                if(!o.addObservation(measures, Ra_Eigen, Rb_Eigen, Post_Eigen, mask1))
                {
                    cout<<"frame dropped"<<endl;
                    frame_count=14;
                    init=true;
                    points1=backup_points1;
                    pointsIndex=backup_pointsIndex;
                    prev_gray=gray;
                    continue;
                }
                indexCorrection=0;
                for(int i=0; i < mask1.size(); i++)
                {
                    if (!mask1[i])
                    {
                        points1.erase (points1.begin() + i - indexCorrection);
                        points2.erase (points2.begin() + i - indexCorrection);
                        //prev_points.erase (prev_points.begin() + i - indexCorrection);
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


                Ptr<Feature2D> orb = points1.size()<MAX_FEATURES ? ORB::create((points1.size())/20, 1.2f, 8, 31, 0, 2, ORB::HARRIS_SCORE, 31, 30 ) : ORB::create(0, 1.2f, 8, 31, 0, 2, ORB::HARRIS_SCORE, 31, 30 );// the 8 is kinda arbitrary *******
                if(!init)
                orb = ORB::create(MAX_FEATURES/8, 1.2f, 8, 31, 0, 2, ORB::HARRIS_SCORE, 31, 30 );
                std::vector<KeyPoint> keypoints;
                for (int i=0; i<4; i++)
                {
                    for (int j=0; j<4; j++)
                    {
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
                        KeyPoint::convert(keypoints, points1, vector<int>());
                        temp.reserve(points2.size()+keypoints.size());
                        Point2f pt = Point2f(x,y);
                        for(int k=0; k<points1.size(); k++)
                        {
                            points1[k]+=pt;
                        }
                        temp.insert(temp.end(), points2.begin(), points2.end());
                        temp.insert(temp.end(), points1.begin(), points1.end());
                        points2=temp;
                    }
                }

                pointsIndex.reserve(points2.size());
                if (points2.size()-pointsIndex.size()>0)
                    pointsIndex.insert(pointsIndex.end(), points2.size()-pointsIndex.size(), -1);
            // Press  ESC on keyboard to exit
            imshow("video", frame);
            char c=(char)waitKey(0);
            if(c==27)
                break;


            //prepare next step
            init=true;
            prev_points = points2;
            points1=prev_points;
        }
        prev_gray=gray;
    }
    // When everything done, release the video capture object
    cap.release();

    // Closes all the frames
    destroyAllWindows();

    return 0;
}
