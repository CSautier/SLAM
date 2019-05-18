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

    //double f=(((double)RES_X)/2)/0.621356021; //width/2 / tan(width_angle/2) //value given by the camera constructor
    double f= 1.0796157060674482e+03;       //value given by opencv calibration

    //Point2f pp(((double)RES_X)/2, ((double)RES_Y)/2); //value given by the camera constructor
    Point2f pp(6.3823992628242706e+02, 3.6216824886639841e+02); //value given by calibration
    Mat cameraMatrix = (Mat1d(3, 3) << f, 0, pp.x, 0, f, pp.y, 0, 0, 1.0);
    //Mat relativeProjection = (Mat1d(4, 3) << f, 0, pp.x, 0., 0, f, pp.y, 0. 0, 0, 1.,0);
    Optimizer o =Optimizer(f, pp.x, pp.y);
    while(1)
    {

        static int frame_count=0; //to activate with the condition below to skip frames, and reduce the computation cost;
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
        if (frame_count<=5){//indicate the number of frame to skip, if, we will only treat (1/n)th of the frames
        continue;
        }
        frame_count=0;


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
        static vector<pair<Point2f, Point2f>> associations; // cette liste contient l'association point 2D de l'image -2 et point 2D de l'image -1
        if(points1.size()<MAX_FEATURES/2)
        {
            Ptr<Feature2D> orb = ORB::create(MAX_FEATURES/16, 1.2f, 8, 31, 0, 2, ORB::HARRIS_SCORE, 31, 30 );
            std::vector<KeyPoint> keypoints;
            for (int i=0; i<4; i++)
            {
                for (int j=0; j<4; j++)
                {
                    FAST(prev_gray(Rect((RES_X/4)*i,(RES_Y/4)*j,(RES_X/4),(RES_Y/4))), keypoints, 40, true);
                    //orb->detect(prev_gray(Rect((RES_X/4)*i,(RES_Y/4)*j,(RES_X/4),(RES_Y/4))), keypoints);
                    //goodFeaturesToTrack(prev_gray, points2, MAX_FEATURES, 0.01, 25);
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
            if (points1.size()-pointsIndex.size()>0)
            pointsIndex.insert(pointsIndex.end(), points1.size()-pointsIndex.size(), -1);
            }
        vector<float> err;
        Size winSize=Size(21,21);
        vector<uchar> status;
        TermCriteria termcrit=TermCriteria(TermCriteria::COUNT+TermCriteria::EPS, 15, 0.01);

        calcOpticalFlowPyrLK(prev_gray, gray, points1, points2, status, err, winSize, 3, termcrit, 0, 0.001);
        for(int i=0; i<points1.size(); i++){
        //cout<<points1[i]<<"\n"<<points2[i]<<"\n\n";
        }

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




/*
        if(init){
        if(associations.size()==0)//this odd condition is true only for the second loop
                pointsIndex[0]=0;
        for (int i=1; i<pointsIndex.size(); i++){ //insure that only the points that have been observed at least 2 times have an index
        if(pointsIndex[i]==-1)
        pointsIndex[i]=pointsIndex[i-1]+1;
        }
        }
        */
        //**********************************************************************************************



        //find the essential matrix using Nister's algorithm
        Mat mask;

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
            if(recoverPose(E, points2, points1, Rt, Post, f, pp, mask)==0)
            {
            cout<<"frame dropped"<<endl;
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
            }
        }

        if(associations.size()==0)//this odd condition is true only for the second loop
                pointsIndex[0]=0;
        for (int i=1; i<pointsIndex.size(); i++){ //insure that only the points that have been observed at least 2 times have an index
        if(pointsIndex[i]==-1)
        pointsIndex[i]=pointsIndex[i-1]+1;
        }

            //cout<<mask.t()<<endl;
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
            {
                            continue;
                            cout<<"frame dropped"<<endl;
            }
            if(Post.at<double>(2,0)<0.){
            //cout<<mask.t()<<endl;
            }

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
                //cout<<Post.t()<<endl;
                hconcat(R.t(),(-R.t())*(Pos_before_scale.t()), P_before_scale);
                P_before_scale= cameraMatrix*P_before_scale;
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
                triangulatePoints(Proj_1, P_before_scale, list1, list_n, Triangulation1_0);
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
                    associations.push_back(pair<Point2f, Point2f>(points1[i], points2[i]));

                }
            }
//triangulation
            Mat Triangulation;
            Mat P_after_scale;
            hconcat(R.t(),(-R.t())*(Pos.t()), P_after_scale);
            P_after_scale=cameraMatrix*P_after_scale;
            triangulatePoints(Proj_1, P_after_scale, points1, points2, Triangulation);
            static int compteurtemp=0;
            ofstream datatemp("data_"+to_string(compteurtemp)+".data");
            compteurtemp++;
            datatemp<< "# X\tY\tZ"<<endl;
            for (int i=0; i<pointsIndex.size(); i++)
            {
                o.addPoint(pointsIndex[i], points1[i].x, points1[i].y, Triangulation.at<float>(0,i)/Triangulation.at<float>(3,i), Triangulation.at<float>(1,i)/Triangulation.at<float>(3,i),Triangulation.at<float>(2,i)/Triangulation.at<float>(3,i));
                //o.addPoint(pointsIndex[i], Triangulation.at<float>(0,i)/Triangulation.at<float>(3,i), Triangulation.at<float>(1,i)/Triangulation.at<float>(3,i),Triangulation.at<float>(2,i)/Triangulation.at<float>(3,i));
                datatemp<<Triangulation.at<float>(2,i)/Triangulation.at<float>(3,i)<<"\t"<< Triangulation.at<float>(0,i)/Triangulation.at<float>(3,i)<<"\t"<< -Triangulation.at<float>(1,i)/Triangulation.at<float>(3,i)<<"\t"<<endl;
            }
            datatemp.close();
            //return(0);



            Eigen::Map<Eigen::Matrix<double, 3, 3, Eigen::RowMajor>> Rt_Eigen(Rt.ptr<double>(), 3, 3);
            Eigen::Map<Eigen::Matrix<double, 1,3, Eigen::RowMajor>> Post_Eigen(Post.ptr<double>(), 1, 3);
            o.addPose(Rt_Eigen, Post_Eigen);

        }
        data<<Pos.at<double>(0,2)<<"\t"<<Pos.at<double>(0,0)<<"\t"<<-Pos.at<double>(0,1)<<"\t"<<endl;












        //    static vector<pair<cv::Point2f,cv::Point2f>> mask_list;
        size_t i;
        for( i = 0; i < points2.size(); i++ )
        {
            cv::circle( frame, points2[i], 2, cv::Scalar(0,255,0), 2, 8);
            cv::circle( frame, points1[i], 2, cv::Scalar(0,0,255), 2, 8);
            cv::line(frame, points1[i], points2[i], cv::Scalar(255,0,0));
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




        // Press  ESC on keyboard to exit
        imshow("video", frame);
        char c=(char)waitKey(0);
        if(c==27)
            break;


        //prepare next step
        Eigen::Matrix<double, 3, 3> Rupdate;
        Vector3 tupdate;
        if(o.optimize(Rupdate, tupdate)){
        //eigen2cv(Rupdate, R);
        //eigen2cv(tupdate, Pos);
        //Mat R_opencv(Rupdate.rows(), Rupdate.cols(), CV_32FC1, Rupdate.data());
        cout<<Pos<<endl;
        R=(Mat1d(3, 3) <<Rupdate(0,0),Rupdate(0,1),Rupdate(0,2),Rupdate(1,0),Rupdate(1,1),Rupdate(1,2),Rupdate(2,0),Rupdate(2,1),Rupdate(2,2));
        Pos = (Mat1d(1, 3) <<tupdate(0),tupdate(1),tupdate(2));
                cout<<Pos<<endl;
                //cout<<Pos<<"\n\n"<<endl;
        //R = Mat(Rupdate.rows(), Rupdate.cols(), CV_32FC1, Rupdate.data());
        //Pos = Mat(tupdate.rows(), tupdate.cols(), CV_32FC1, tupdate.data());
        }
        init=true;
        points1=points2;
        prev_gray=gray;
        swap(Proj_2,Proj_1);
        hconcat(R.t(),(-R.t())*(Pos.t()), Proj_1);
        Proj_1=cameraMatrix*Proj_1;
    }
    //o.graphPrint();
    // When everything done, release the video capture object
    cap.release();

    // Closes all the frames
    destroyAllWindows();

    return 0;
}
