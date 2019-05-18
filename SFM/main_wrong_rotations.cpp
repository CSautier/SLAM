#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/xfeatures2d.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/calib3d.hpp"
#include "opencv2/video/tracking.hpp"
#include "optimizer.hpp"
#include "opencv2/core/eigen.hpp"

#include <math.h>
#include <fstream>
#include <iostream>

#include<Eigen/Core>

const int MAX_FEATURES = 1000;
const int RES_X = 1280;
const int RES_Y = 720;
const float GOOD_MATCH_PERCENT = 0.3f;


using namespace cv;
//using namespace cv::xfeatures2d;
using namespace std;


int main(int argc, char* argv[])
{
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
    ofstream data ("poses.data");
    ofstream dataPoints ("points.data");
    data << "# X\tY\tZ"<<endl;
    dataPoints << "# X\tY\tZ"<<endl;

    double f=(((double)RES_X)/2)/0.621356021; //width/2 / tan(width_angle/2)
    Point2f pp(((double)RES_X)/2, ((double)RES_Y)/2);
    Mat cameraMatrix = (Mat1d(3, 3) << f, 0, pp.x, 0, f, pp.y, 0, 0, 1.0);
    //Mat relativeProjection = (Mat1d(4, 3) << f, 0, pp.x, 0., 0, f, pp.y, 0. 0, 0, 1.,0);
    Optimizer o =Optimizer(f, pp.x, pp.y);
    while(1)
    {

        //static int frame_count=0; //to activate with the condition below to skip frames, and reduce the computation cost;
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
        /*
        frame_count++;
        if (frame_count<=5){//indicate the number of frame to skip, if, we will only treat (1/n)th of the frames
        continue;
        }
        frame_count=0;
        */

        cvtColor(frame, gray, cv::COLOR_RGB2GRAY);
        static Mat prev_gray = gray;

        // Variables to store keypoints and descriptors

        static bool init=false;

        //***********************************************************************************************
        /*
        // bruteforce matches, empirically less robust
          static std::vector<KeyPoint> * keypoints1_ptr = new std::vector<KeyPoint>();
        static Mat * descriptors1_ptr = new Mat();
          std::vector<KeyPoint> keypoints2;
          Mat descriptors2;
           // Detect ORB features and compute descriptors.
        Ptr<Feature2D> orb = ORB::create(MAX_FEATURES);
        if(!init)
          orb->detectAndCompute(gray, Mat(), *keypoints1_ptr, *descriptors1_ptr);

        orb->detectAndCompute(prev_gray, Mat(), keypoints2, descriptors2);
          // Match features.
        std::vector<DMatch> matches;
        Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create("BruteForce-Hamming");
        matcher->match(*descriptors1_ptr, descriptors2, matches, Mat());
        // Sort matches by score
        std::sort(matches.begin(), matches.end());
         // Remove not so good matches
        const int numGoodMatches = matches.size() * GOOD_MATCH_PERCENT;
        matches.erase(matches.begin()+numGoodMatches, matches.end());

          // Draw top matches
        Mat imMatches;
        drawMatches(gray, *keypoints1_ptr, prev_gray, keypoints2, matches, imMatches);
            // Display the resulting frame
        //    imshow( "Matches", imMatches );

            // Extract location of good matches
        std::vector<Point2f> points1, points2;

        for( size_t i = 0; i < matches.size(); i++ )
        {
          points1.push_back( (*keypoints1_ptr)[ matches[i].queryIdx ].pt );
          points2.push_back( keypoints2[ matches[i].trainIdx ].pt );
        }
          *descriptors1_ptr=descriptors2;
          *keypoints1_ptr=keypoints2;
          */
//*************************************************************************************************


//*************************************************************************************************
        //KLT matches, using ORB
        static vector<Point2f> points1; //image n-1 points
        static vector<int> pointsIndex{0};
        vector<Point2f> points2; //image n points
        if(points1.size()<MAX_FEATURES/2)
        {
            Ptr<Feature2D> orb = ORB::create(MAX_FEATURES/16, 1.2f, 8, 31, 0, 2, ORB::HARRIS_SCORE, 31, 20 );
            std::vector<KeyPoint> keypoints;
            for (int i=0; i<4; i++)
            {
                for (int j=0; j<4; j++)
                {

                    orb->detect(prev_gray(Rect((RES_X/4)*i,(RES_Y/4)*j,(RES_X/4),(RES_Y/4))), keypoints);
                    std::vector<Point2f> temp;

                    KeyPoint::convert(keypoints, points2, vector<int>());
                    temp.reserve(points1.size()+keypoints.size());
                    Point2f pt = Point2f(RES_X/4*i,RES_Y/4*j);
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
            for (int i=pointsIndex.back(); i<=pointsIndex.back()+points1.size()-pointsIndex.size(); ++i)
            {
                pointsIndex.push_back(i);
            }
            //cout<<pointsIndex.back()<<endl;
        }

        vector<float> err;
        Size winSize=Size(21,21);
        vector<uchar> status;
        TermCriteria termcrit=TermCriteria(TermCriteria::COUNT+TermCriteria::EPS, 30, 0.01);

        calcOpticalFlowPyrLK(prev_gray, gray, points1, points2, status, err, winSize, 3, termcrit, 0, 0.001);
        //getting rid of points for which the KLT tracking failed or those who have gone outside the frame
        int indexCorrection = 0;
        for( int i=0; i<status.size(); i++)
        {
            Point2f pt = points2.at(i- indexCorrection);
            if ((status.at(i) == 0)||(pt.x<0)||(pt.y<0))
            {
                if((pt.x<0)||(pt.y<0))	 // not sure if useful
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
        //**********************************************************************************************




        //find the essential matrix using Nister's algorithm
        Mat mask;

        static vector<pair<Point2f, Point2f>> associations; // cette liste contient l'association point 2D de l'image -2 et point 2D de l'image -1

        static Mat R = (Mat1d(3, 3) << 1, 0, 0, 0, 1,0, 0, 0, 1);

        //cout<<R<<endl;
        static Mat Pos=(Mat1d(1, 3) << 0, 0, 0);


        static Mat P;

        if(!init)
            hconcat(R,Pos.t(), P);



        static Mat Proj_1;
        static Mat Proj_2;

        static Mat Rt= (Mat1d(3, 3) << 1, 0, 0, 0, 1,0, 0, 0, 1);
        static Mat Post=(Mat1d(1, 3) << 0, 0, 0);

        if(init)
        {
            Mat E = cv::findEssentialMat(points2, points1, f, pp, RANSAC, 0.999, 1.0, mask);
//recover the pose from the essential matrix
            recoverPose(E, points2, points1, Rt, Post, f, pp, mask);

            //skip the frame if the pose cannot be recovered.
            bool b = false;
            for (int i=0; i<mask.rows; i++)
            {
                if (mask.at<bool>(i,0))
                {
                    b=true;
                    break;
                }
            }
            if (!b)
                continue;


            //cout<<Rt<<"\n\n";



            R = R*Rt;
            if (associations.size()==0)
            {
                Pos+= Post.t()*Rt;
                hconcat(R,Pos.t(), P);

                //cout<<Triangulation<<endl;
                for(int i=0; i<points1.size(); i++)
                {
                    associations.reserve(points1.size());
                    associations.push_back(pair<Point2f, Point2f>(points1[i], points2[i]));

                }
            }
            else
            {
                Point2f p1 = points1[0];//TODO : improve this
                Point2f p2 = points1[1];


                int f1 = -1;
                int f2 = -1;
                for (int i =0; i<associations.size(); i++)
                {
                    if(associations[i].second==p1)
                        f1=i;
                    if(associations[i].second==p2)
                        f2=i;
                    if (f1>=0 && f2 >=0)
                        break;
                }
                if (f1<0||f2<0)
                {
                    cerr<<"The pose is lost"<<endl;
                    return -1;
                }


                double scale;
                Mat Pos_before_scale;
                Mat P_before_scale;
                Pos_before_scale=Pos + Post.t()*Rt;
                hconcat(R,Pos_before_scale.t(), P_before_scale);
                vector<Point2f> list_old;//n-2
                vector<Point2f> list1;//n-1
                vector<Point2f> list_n;//n
                list1.reserve(2);
                list_n.reserve(2);
                list_old.reserve(2);
                list1.insert(list1.end(), points1.begin(), points1.begin()+2);
                list_n.insert(list_n.end(), points2.begin(), points2.begin()+2);
                list_old.insert(list_old.end(), associations[f1].first);
                list_old.insert(list_old.end(), associations[f2].first);
                Mat Triangulation2_1;//between n-2 and n-1
                Mat Triangulation1_0;//between n-1 and n
                triangulatePoints(Proj_2, Proj_1, list_old, list1, Triangulation2_1);
                triangulatePoints(Proj_1, cameraMatrix*P_before_scale, list1, list_n, Triangulation1_0);
                //cout<<Triangulation2_1<<endl;
                //cout<<Triangulation1_0<<endl;
                scale = norm(Point3d(Triangulation2_1.at<float>(0,0)/Triangulation2_1.at<float>(3,0) - Triangulation2_1.at<float>(0,1)/Triangulation2_1.at<float>(3,1),
                                     Triangulation2_1.at<float>(1,0)/Triangulation2_1.at<float>(3,0) - Triangulation2_1.at<float>(1,1)/Triangulation2_1.at<float>(3,1),
                                     Triangulation2_1.at<float>(2,0)/Triangulation2_1.at<float>(3,0)-Triangulation2_1.at<float>(2,1)/Triangulation2_1.at<float>(3,1) )) /
                        norm(Point3d(Triangulation1_0.at<float>(0,0)/Triangulation1_0.at<float>(3,0) - Triangulation1_0.at<float>(0,1)/Triangulation1_0.at<float>(3,1),
                                     Triangulation1_0.at<float>(1,0)/Triangulation1_0.at<float>(3,0) - Triangulation1_0.at<float>(1,1)/Triangulation1_0.at<float>(3,1),
                                     Triangulation1_0.at<float>(2,0)/Triangulation1_0.at<float>(3,0)-Triangulation1_0.at<float>(2,1)/Triangulation1_0.at<float>(3,1) ));
                Post*=scale;
                Pos += Post.t()*Rt;
                hconcat(R,Pos.t(), P);
                associations.clear();
                for(int i=0; i<points1.size(); i++)
                {
                    associations.reserve(points1.size());
                    associations.push_back(pair<Point2f, Point2f>(points1[i], points2[i]));                //cout<<Triangulation.at<float>(0,i)<<endl;

                }
            }
//triangulation
            Mat Triangulation;
            triangulatePoints(Proj_1, cameraMatrix*P, points1, points2, Triangulation);
            static int compteurtemp=0;
            ofstream datatemp("data_"+to_string(compteurtemp)+".data");
            compteurtemp++;
            datatemp<< "# X\tY\tZ"<<endl;
            for (int i=0; i<pointsIndex.size(); i++)
            {
                //o.addPoint(pointsIndex[i], Triangulation.at<float>(0,i)/Triangulation.at<float>(3,i), Triangulation.at<float>(1,i)/Triangulation.at<float>(3,i),Triangulation.at<float>(2,i)/Triangulation.at<float>(3,i));
                datatemp<<Triangulation.at<float>(2,i)/Triangulation.at<float>(3,i)<<"\t"<< Triangulation.at<float>(0,i)/Triangulation.at<float>(3,i)<<"\t"<< -Triangulation.at<float>(1,i)/Triangulation.at<float>(3,i)<<"\t"<<endl;

            }
            datatemp.close();
            //return(0);






            Eigen::Map<Eigen::Matrix<double, 3, 3, Eigen::RowMajor>> Rt_Eigen(Rt.ptr<double>(), 3, 3);
            Eigen::Map<Eigen::Matrix<double, 1,3, Eigen::RowMajor>> Post_Eigen(Post.ptr<double>(), 1, 3);
            //o.addPose(Rt_Eigen, Post_Eigen);

        }
//cout<<R<<"\n\n";

//display results
        //cout<<P<<"\n\n\n";
        data<<Pos.at<double>(0,2)<<"\t"<<Pos.at<double>(0,0)<<"\t"<<-Pos.at<double>(0,1)<<"\t"<<endl;












        //    static vector<pair<cv::Point2f,cv::Point2f>> mask_list;
        size_t i;
        for( i = 0; i < points2.size(); i++ )
        {
            cv::circle( frame, points2[i], 2, cv::Scalar(0,255,0), 2, 8);
            //mask_list.insert(mask_list.begin(),pair<cv::Point2f,cv::Point2f>(points1[i], points1[i]));
        }
        /*
        while(mask_list.size()>6000){
        mask_list.pop_back();
        }
        cv::Mat maskt(gray.size(), CV_8UC3, cv::Scalar(0,0,0));
        for (int i=0; i<mask_list.size();i++){
        cv::line (maskt, mask_list[i].first, mask_list[i].second, cv::Scalar(0,255,0));
        }
        cv::add(frame, maskt, frame);
        */









//cout<<mask<<"\n\n";


        // Press  ESC on keyboard to exit
        imshow("video", frame);
        char c=(char)waitKey(25);
        if(c==27)
            break;


        //prepare next step
        //o.optimize();
        init=true;
        points1=points2;
        prev_gray=gray;
        swap(Proj_2,Proj_1);
        if(associations.size()>0){
        Mat P1 = (Mat1d(3, 4) << f, 0, pp.x, 0., 0, f, pp.y, 0., 0, 0, 1.0,0.);
        Mat P2 = (Mat1d(3, 4) << f, 0, pp.x, -f, 0., f, pp.y, 0, 0, 0., 1.0,0.);//63.71/2 = 31.855   1.609383295  3.504143
        Mat Tri;
        vector<Point2f> p11 = {pp};
        vector<Point2f> p22 = {Point2f(pp.x-f*0.285376476, pp.y)};
        cout<<P1<<endl;

        cout<<P2<<endl;
        triangulatePoints(P1, P2, p11, p22, Tri);
        cout<<Tri<<"\n\n\n";
        return(0);
        }

        Proj_1=cameraMatrix*P;
    }
    //o.graphPrint();
    // When everything done, release the video capture object
    cap.release();

    // Closes all the frames
    destroyAllWindows();

    return 0;
}
