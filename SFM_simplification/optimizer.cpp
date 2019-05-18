#include <iostream>
//#include <gtsam/geometry/Point2.h>
#include <gtsam/inference/Symbol.h>
#include <gtsam/slam/PriorFactor.h>
#include <gtsam/slam/ProjectionFactor.h>
#include <gtsam/slam/GeneralSFMFactor.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/Values.h>
#include <gtsam/geometry/triangulation.h>
#include "optimizer.hpp"
#include <fstream>

#include <gtsam/nonlinear/ISAM2.h>
//#include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>
//#include <gtsam/nonlinear/DoglegOptimizer.h>


using namespace std;
using namespace gtsam;



Optimizer::Optimizer(double f, double cx, double cy)
{
    doglegparameters = ISAM2DoglegParams(1000./*, 1e-3*/);
/*ISAM2GaussNewtonParams gaussnewtonparameters(1e-1);*/
    parameters.optimizationParams = doglegparameters;
    //parameters.optimizationParams = gaussnewtonparameters;
  parameters.relinearizeThreshold = 0.01;
  parameters.relinearizeSkip = 1;
  parameters.enableRelinearization=true;
  parameters.cacheLinearizedFactors = true;
  parameters.enableDetailedResults = true;
  parameters.factorization = gtsam::ISAM2Params::CHOLESKY;
  parameters.print();
    isam = ISAM2(parameters);
    n=0;
    K = Cal3_S2(f,f,0.,cx,cy);
    firstPoint = false;
    // Add a prior on pose x1. This indirectly specifies where the origin is.
    noiseModel::Diagonal::shared_ptr priorNoise = noiseModel::Diagonal::Sigmas((Vector(6) << Vector3::Constant(0.01), Vector3::Constant(0.01)).finished()); // 1cm std on x,y,z 0.01 rad on roll,pitch,yaw
    //noiseModel::Isotropic::shared_ptr measurement_noise = noiseModel::Isotropic::Sigma(2, 2.0); // pixel error in (x,y)
    noiseModel::mEstimator::Huber::shared_ptr measurement_noise = noiseModel::mEstimator::Huber::Create(1.0); //********************************

    noiseModel::Diagonal::shared_ptr cal_noise = noiseModel::Diagonal::Sigmas((Vector(5) << 20, 20, 0.01 /*skew*/, 10, 10).finished());
    graph.emplace_shared<PriorFactor<Pose3> >(Symbol('x', 0), Pose3(), priorNoise); // add directly to graph
    graph.emplace_shared<PriorFactor<Cal3_S2>>(Symbol('K', 0), K, cal_noise);
    initialEstimate.insert(Symbol('x',0),Pose3());
    initialEstimate.insert(Symbol('K', 0), K);
    MEASURE_BEFORE_COMPUTATION=2;
    lastPointadded=-1;
    error=0.;
}


bool Optimizer::addObservation(vector<information>& measures, Matrix Ra,  Matrix Rb, Matrix Post, vector<bool>& mask)
{
    mask = vector<bool>(measures.size(), true);
    if(n==0) //initialization
    {
        for(information p : measures)
        {
            observations.push_back(Observation(p.index, 0, p.pt, p.color));
        }
    }
    else
    {
        Rot3 R;
        Pose3 P;//relative pose from n-1 to n
        pair<vector<int>, vector<Point3>> listPoints3d;
        Point3 T = Point3(Post.transpose());
        Pose3 Pa = Pose3(Rot3(Ra),T);
        Pose3 Pb = Pose3(Rot3(Ra),-T);
        Pose3 Pc = Pose3(Rot3(Rb),T);
        Pose3 Pd = Pose3(Rot3(Rb),-T);
        pair<vector<int>, vector<Point3>> P1;
        pair<vector<int>, vector<Point3>> P2;
        pair<vector<int>, vector<Point3>> P3;
        pair<vector<int>, vector<Point3>> P4;
        vector<Matrix34, Eigen::aligned_allocator<Matrix34>> Ka;
        vector<Matrix34, Eigen::aligned_allocator<Matrix34>> Kb;
        vector<Matrix34, Eigen::aligned_allocator<Matrix34>> Kc;
        vector<Matrix34, Eigen::aligned_allocator<Matrix34>> Kd;
        CameraProjectionMatrix<Cal3_S2> createP(currentEstimate.at<Cal3_S2>(Symbol('K', 0)));
        Ka.push_back(createP(Pose3()));
        Kb.push_back(createP(Pose3()));
        Kc.push_back(createP(Pose3()));
        Kd.push_back(createP(Pose3()));
        Ka.push_back(createP(Pa));
        Kb.push_back(createP(Pb));
        Kc.push_back(createP(Pc));
        Kd.push_back(createP(Pd));

        for(information p : measures) //loop over all the measures
        {
            if(p.index+1>observations.size())
                break;
            Point2Vector m{observations[p.index].measures.back(), p.pt};
            Point3 pt = triangulateDLT(Ka, m, 1e-9);
            if (pt.z()>0 && Pa.transform_to(pt).z()>0)
            {
                P1.second.push_back(Pa.transform_to(pt));
                P1.first.push_back(p.index);
            }
            pt = triangulateDLT(Kb, m, 1e-9);
            if (pt.z()>0 && Pb.transform_to(pt).z()>0)
            {
                P2.second.push_back(Pb.transform_to(pt));
                P2.first.push_back(p.index);
            }
            pt = triangulateDLT(Kc, m, 1e-9);
            if (pt.z()>0 && Pc.transform_to(pt).z()>0)
            {
                P3.second.push_back(Pc.transform_to(pt));
                P3.first.push_back(p.index);
            }
            pt = triangulateDLT(Kd, m, 1e-9);
            if (pt.z()>0 && Pd.transform_to(pt).z()>0)
            {
                P4.second.push_back(Pd.transform_to(pt));
                P4.first.push_back(p.index);
            }
        }

        cout<<"P1 : "<<P1.first.size()<<endl;
        cout<<"P2 : "<<P2.first.size()<<endl;
        cout<<"P3 : "<<P3.first.size()<<endl;
        cout<<"P4 : "<<P4.first.size()<<endl;
        if(P1.first.size()>P2.first.size()&&P1.first.size()>P3.first.size()&&P1.first.size()>P4.first.size())
        {
        double v = P1.first.size();
            if(P1.first.size()<20||(P2.first.size()/v+P3.first.size()/v+P4.first.size()/v)>0.7)
                return false;
            R=Rot3(Ra);
            P=Pa;
            listPoints3d.first.reserve(P1.first.size());
            listPoints3d.second.reserve(P1.first.size());
            for(int i=0; i<P1.first.size(); i++)
            {
                listPoints3d.first.push_back(P1.first[i]);
                listPoints3d.second.push_back(P.transform_from(P1.second[i]));
            }
        }
        else if(P2.first.size()>P1.first.size()&&P2.first.size()>P3.first.size()&&P2.first.size()>P4.first.size())
        {
        double v = P2.first.size();
            if(P2.first.size()<20||(P1.first.size()/v+P3.first.size()/v+P4.first.size()/v)>0.7)
                return false;
            R=Rot3(Ra);
            P=Pb;
            T=-T;
            listPoints3d.first.reserve(P2.first.size());
            listPoints3d.second.reserve(P2.first.size());
            for(int i=0; i<P2.first.size(); i++)
            {
                listPoints3d.first.push_back(P2.first[i]);
                listPoints3d.second.push_back(P.transform_from(P2.second[i]));
            }
        }
        else if(P3.first.size()>P1.first.size()&&P3.first.size()>P2.first.size()&&P3.first.size()>P4.first.size())
        {
        double v = P3.first.size();
            if(P3.first.size()<20||(P1.first.size()/v+P2.first.size()/v+P4.first.size()/v)>0.7)
                return false;
            R=Rot3(Rb);
            P=Pc;
            T=T;
            listPoints3d.first.reserve(P3.first.size());
            listPoints3d.second.reserve(P3.first.size());
            for(int i=0; i<P3.first.size(); i++)
            {
                listPoints3d.first.push_back(P3.first[i]);
                listPoints3d.second.push_back(P.transform_from(P3.second[i]));
            }
        }
        else
        {
        double v = P4.first.size();
            if(P4.first.size()<20||(P1.first.size()/v+P2.first.size()/v+P3.first.size()/v)>0.7)
                return false;
            R=Rot3(Rb);
            P=Pd;
            T=-T;
            listPoints3d.first.reserve(P4.first.size());
            listPoints3d.second.reserve(P4.first.size());
            for(int i=0; i<P4.first.size(); i++)
            {
                listPoints3d.first.push_back(P4.first[i]);
                listPoints3d.second.push_back(P.transform_from(P4.second[i]));
            }
        }


        if(n==1) //particular case, we don't have to compute scale yet
        {
            initialEstimate.insert(Symbol('x',1), P);//P is normalized, it becomes the scale
            //triangulate all points observed twice
            int j=0;
            for(int i=0; i<measures.size(); i++){
            while(j<listPoints3d.first.size()&&listPoints3d.first[j]<measures[i].index){
            j++;
            }
            if(listPoints3d.first[j]==measures[i].index){
                observations[listPoints3d.first[j]].coordinates = listPoints3d.second[j];
                observations[listPoints3d.first[j]].cameraId.push_back(1);//add 1 to the list of the camera that have seen this point
                observations[listPoints3d.first[j]].measures.push_back(measures[i].pt); //then add the measures
            }
            else{
            mask[i]=false;
                                if(measures[i].index+1>observations.size())
                    {
                        observations.push_back(Observation(measures[i].index, 1, measures[i].pt, measures[i].color));
                    }
            }
            if(j==listPoints3d.first.size()-1)
            break;
            }
        }
        else
        {
            Pose3 pose = currentEstimate.exists(Symbol('x',n-1)) ?  currentEstimate.at<Pose3>(Symbol('x',n-1)) : initialEstimate.at<Pose3>(Symbol('x',n-1));
            double scale=0;
            vector<double> scalelist;
            for(int i=0; i<listPoints3d.first.size(); i++)
            {
            //fot now we triangulate only on the optimized points. Could do better ********
                if(currentEstimate.exists(Symbol('p', listPoints3d.first[i])))// or if the point has been measured 2 times ****************************************
                {
                scalelist.push_back((currentEstimate.at<Point3>(Symbol('p', listPoints3d.first[i]))-pose.translation()).norm()/(listPoints3d.second[i]-pose.translation()).norm());
                }
            }
            sort(scalelist.begin(), scalelist.end());
            scale = scalelist[scalelist.size()/2];
            cout<<"scale : "<< scale<<endl;
            //apply scale to initialize value
            initialEstimate.insert(Symbol('x',n), Pose3(R,scale*T)*pose);

            //triangulate all possible points
            int j=0;
            for(int i=0; i<measures.size(); i++){
            while(j<listPoints3d.first.size()&&listPoints3d.first[j]<measures[i].index){
            j++;
            }
            if(listPoints3d.first[j]==measures[i].index){

                            if(observations[listPoints3d.first[j]].measures.size()==1) //if the point has been observed only once
                {
                    try
                    {
                        Point3 pt = triangulatePoint3(vector<Pose3> {pose, initialEstimate.at<Pose3>(Symbol('x',n))}, boost::make_shared<Cal3_S2>(currentEstimate.at<Cal3_S2>(Symbol('K', 0))),
                                                      Point2Vector{observations[listPoints3d.first[j]].measures.back(), measures[i].pt}); //we can now triangulate it's position
                        observations[listPoints3d.first[j]].coordinates = pt;
                        observations[listPoints3d.first[j]].cameraId.push_back(n);
                        observations[listPoints3d.first[j]].measures.push_back(measures[i].pt);
                    }
                    catch(...)
                    {
                        cout<<"Triangulation error on point "<<listPoints3d.first[j]<<endl;
                    }
                }
                else
                {
                    observations[listPoints3d.first[j]].cameraId.push_back(n);
                    observations[listPoints3d.first[j]].measures.push_back(measures[i].pt);
                    if(observations[listPoints3d.first[j]].measures.size()>MEASURE_BEFORE_COMPUTATION)
                    {
                        graph.emplace_shared<GeneralSFMFactor2<Cal3_S2>>(measures[i].pt, measurement_noise, Symbol('x', n), Symbol('p', listPoints3d.first[j]), Symbol('K', 0));
                    }
                }
            }
            else{
            mask[i]=false;
                                         if(measures[i].index+1>observations.size())
                    {
                        observations.push_back(Observation(measures[i].index, 1, measures[i].pt, measures[i].color));
                    }
            }
            if(j==listPoints3d.first.size()-1)
            break;
            }
        }

        for(information p : measures) //add points that could not have been triangulated because it's their first observations
        {
            if(p.index+1>observations.size())
                observations.push_back(Observation(p.index, n, p.pt, p.color));
        }

        for(int i= 0; i<observations.size(); i++)
        {
            if(observations[i].measures.size()==MEASURE_BEFORE_COMPUTATION&&(!currentEstimate.exists(Symbol('p', i)))&&(!initialEstimate.exists(Symbol('p', i))))
            {
                initialEstimate.insert<Point3>(Symbol('p', i), observations[i].coordinates);
                for(int j=0; j<MEASURE_BEFORE_COMPUTATION; j++)
                {
                    graph.emplace_shared<GeneralSFMFactor2<Cal3_S2>>(observations[i].measures[j], measurement_noise, Symbol('x', observations[i].cameraId[j]), Symbol('p', i), Symbol('K', 0));
                }
                if(!firstPoint) //if no point has a prior
                {
                    Point3 pt = observations[i].coordinates;
                    noiseModel::Isotropic::shared_ptr point_noise = noiseModel::Isotropic::Sigma(3, max(max(abs(pt.x()),abs(pt.y())),abs(pt.z())));
                    graph.emplace_shared<PriorFactor<Point3>>(Symbol('p', i), pt, point_noise);
                    firstPoint=true;
                }
                lastPointadded=i;
            }
        }
    }
    if (this->optimize()){
    n++;
    return true;
    }
    return false;
}




bool Optimizer::optimize()
{
try{
    isam.update(graph, initialEstimate);
}
catch(...){
cout<<"********************"<<endl;
cout<<"Echec de l'itération"<<endl;
cout<<"********************"<<endl;
}

        currentEstimate = isam.calculateEstimate();
        initialEstimate.clear();
        graph  = NonlinearFactorGraph();

    currentEstimate.at<Cal3_S2>(Symbol('K',0)).print();

     int i=0;
     ofstream data ("points_computed.data");
     data << "# X\tY\tZ\trgb"<<endl;
    ofstream posesdata ("poses_computed.data");
     posesdata << "# X\tY\tZ"<<endl;
     while(i<observations.size())
     {
         if(currentEstimate.exists(Symbol('p',i)))
             data<<currentEstimate.at<Point3>(Symbol('p',i)).z()<<"\t"<<-currentEstimate.at<Point3>(Symbol('p',i)).x()<<"\t"<<-currentEstimate.at<Point3>(Symbol('p',i)).y()<<"\t"<<observations[i].color<<endl;
         i++;
     }
     for(int i=0; i<=n; i++){
     posesdata<<currentEstimate.at<Pose3>(Symbol('x',i)).z()<<"\t"<<-currentEstimate.at<Pose3>(Symbol('x',i)).x()<<"\t"<<-currentEstimate.at<Pose3>(Symbol('x',i)).y()<<"\t"<<endl;
     }

            return true;
}


void Optimizer::graphPrint()
{
    graph.print("\nFactor Graph:\n");
}

    double Optimizer::cx(){
    if (currentEstimate.exists(Symbol('K',0)))
    return currentEstimate.at<Cal3_S2>(Symbol('K',0)).px();
    return -1;
    }
    double Optimizer::cy(){
        if (currentEstimate.exists(Symbol('K',0)))
    return currentEstimate.at<Cal3_S2>(Symbol('K',0)).py();
    return -1;
    }
    double Optimizer::fx(){
        if (currentEstimate.exists(Symbol('K',0)))
    return currentEstimate.at<Cal3_S2>(Symbol('K',0)).fx();
        return -1;
    }
    double Optimizer::fy(){
        if (currentEstimate.exists(Symbol('K',0)))
    return currentEstimate.at<Cal3_S2>(Symbol('K',0)).fy();
        return -1;
    }

/* comments:

there is one scenario in which the points that did not pass the check are still saved, that is when they are at the end of the points lists, and then, they will be considered not triangulated because new, and not false, which they are
this problem seems innocuous in 99.99% of the case, and even beneficial most of the time


*/

