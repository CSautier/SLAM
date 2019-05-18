#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/xfeatures2d.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/calib3d.hpp"
#include "opencv2/video/tracking.hpp"
#include "optimizer.cpp"


#include <math.h>
#include <fstream>
#include <iostream>

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
    ofstream data ("position.data");
    data << "# X\tY\tZ"<<endl;

    double f=(((double)RES_X)/2)/0.621356021; //width/2 / tan(width_angle/2)
    Point2f pp(((double)RES_X)/2, ((double)RES_Y)/2);
    Mat cameraMatrix = (Mat1d(3, 3) << f, 0, pp.x, 0, f, pp.y, 0, 0, 1.0);
    //Optimizer(f, pp.x, pp.y);//careful, don't try to add the first pose;
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
            cerr<<"End of file"<<endl;
            return(-1);
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
        vector<Point2f> points2; //image n points
        if(points1.size()<MAX_FEATURES/2) //TODO : ne pas remplacer les pts mais en ajouter de nouveaux pour ne pas perdre le B.A.
        {
        cout<<"recalcul";
        Ptr<Feature2D> orb = ORB::create(MAX_FEATURES/16, 1.2f, 8, 31, 0, 2, ORB::HARRIS_SCORE, 31, 20 );
            std::vector<KeyPoint> keypoints;
                for (int i=0; i<4; i++){
        for (int j=0; j<4; j++){

            orb->detect(prev_gray(Rect((RES_X/4)*i,(RES_Y/4)*j,(RES_X/4),(RES_Y/4))), keypoints);
            std::vector<Point2f> temp;

            KeyPoint::convert(keypoints, points2, vector<int>());
            temp.reserve(points1.size()+keypoints.size());
            Point2f pt = Point2f(RES_X/4*i,RES_Y/4*j);
            for(int k=0; k<points2.size(); k++){
            points2[k]+=pt;
            }
            temp.insert(temp.end(), points1.begin(), points1.end());
            temp.insert(temp.end(), points2.begin(), points2.end());
            points1=temp;
            }
            }

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
                indexCorrection++;
            }

        }
        //**********************************************************************************************




        //find the essential matrix using Nister's algorithm
        Mat mask;

        static vector<pair<Point2f, Point3d>> associations; // cette liste contient l'association point 2D de l'image -1 et point 3D triangulés entre -1 et -2

        static Mat R = (Mat1d(3, 3) << 1, 0, 0, 0, 1,0, 0, 0, 1);

        //cout<<R<<endl;
        static Mat Pos=(Mat1d(1, 3) << 0, 0, 0);


        static Mat P;

        if(!init)
        hconcat(R,Pos.t(), P);


        static Mat Proj;


        if(init)
        {
            Mat E = cv::findEssentialMat(points2, points1, f, pp, RANSAC, 0.999, 1.0, mask);
//recover the pose from the essential matrix
            //continue;
            Mat Rt;
            Mat Post;
            recoverPose(E, points2, points1, Rt, Post, f, pp, mask);

            //skip the frame if the pose cannot be recovered.
            bool b = false;
            for (int i=0; i<mask.rows; i++){
            if (mask.at<bool>(i,0)){
            b=true;
            break;
            }
            }
            if (!b)
            continue;


            //cout<<Rt<<"\n\n";



            R = R*Rt;
            Mat Triangulation;
            if (associations.size()==0){

                Pos+= Post.t()*Rt;
                hconcat(R,Pos.t(), P);
                triangulatePoints(Proj, cameraMatrix*P, points1, points2, Triangulation);

                //cout<<Triangulation<<endl;
                for(int i=0; i<Triangulation.cols; i++){
                associations.reserve(Triangulation.rows);
                associations.push_back(pair<Point2f, Point3d>(points2[i], Point3d(Triangulation.at<float>(0,i)/Triangulation.at<float>(3,i),Triangulation.at<float>(1,i)/Triangulation.at<float>(3,i) ,Triangulation.at<float>(2,i)/Triangulation.at<float>(3,i) )));
                //cout<<Triangulation.at<float>(0,i)<<endl;

                }
                //cout<<Triangulation.at<float>(0,0)<<"\n"<<Triangulation.at<float>(1,0)<<"\n"<<Triangulation.at<float>(2,0)<<"\n"<<Triangulation.at<float>(3,0)<<"\n"<<endl;

            }
            else{
            Point2f p1 = points1[0];//TODO : improve this
            Point2f p2 = points1[1];



            //TODO : correct this
            int f1 = -1;
            int f2 = -1;
            for (int i =0; i<associations.size(); i++){
            if(associations[i].first==p1)
            f1=i;
            if(associations[i].first==p2)
            f2=i;
            if (f1>=0 && f2 >=0)
            break;
            }
            if (f1<0||f2<0){
            cerr<<"The pose is lost"<<endl;
            return -1;
            }







            double scale = norm(associations[f1].second-associations[f2].second);
            Mat Pos_before_scale;
            Mat P_before_scale;
            Pos_before_scale=Pos + Post.t()*Rt;
            hconcat(R,Pos_before_scale.t(), P_before_scale);
            vector<Point2f> list1;
            vector<Point2f> list2;
            list1.reserve(2);
            list2.reserve(2);
            list1.insert(list1.end(), points1.begin(), points1.begin()+2);
            list2.insert(list2.end(), points2.begin(), points2.begin()+2);
            triangulatePoints(Proj, cameraMatrix*P_before_scale, list1, list2, Triangulation);
            //cout<<Triangulation<<endl;
            scale = scale / norm(Point3d(Triangulation.at<float>(0,0)/Triangulation.at<float>(3,0) - Triangulation.at<float>(0,1)/Triangulation.at<float>(3,1),Triangulation.at<float>(1,0)/Triangulation.at<float>(3,0) - Triangulation.at<float>(1,1)/Triangulation.at<float>(3,1),
                    Triangulation.at<float>(2,0)/Triangulation.at<float>(3,0)-Triangulation.at<float>(2,1)/Triangulation.at<float>(3,1) ));
            Pos += scale* Post.t()*Rt;
            hconcat(R,Pos.t(), P);
            triangulatePoints(Proj, cameraMatrix*P_before_scale, points1, points2, Triangulation);
            associations.clear();
            for(int i=0; i<Triangulation.cols; i++){
                associations.reserve(Triangulation.rows);
                associations.push_back(pair<Point2f, Point3d>(points2[i], Point3d(Triangulation.at<float>(0,i)/Triangulation.at<float>(3,i),Triangulation.at<float>(1,i)/Triangulation.at<float>(3,i) ,Triangulation.at<float>(2,i)/Triangulation.at<float>(3,i) )));
                //cout<<Triangulation.at<float>(0,i)<<endl;

                }
            }
        }
//cout<<R<<"\n\n";


//display results
        cout<<P<<"\n\n\n";
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
        init=true;
        points1=points2;
        prev_gray=gray;
        Proj=cameraMatrix*P;
    }

    // When everything done, release the video capture object
    cap.release();

    // Closes all the frames
    destroyAllWindows();

    return 0;
}
