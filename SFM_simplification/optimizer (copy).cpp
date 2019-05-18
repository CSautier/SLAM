#include <iostream>
//#include <gtsam/geometry/Point2.h>
#include <gtsam/inference/Symbol.h>
#include <gtsam/slam/PriorFactor.h>
#include <gtsam/slam/ProjectionFactor.h>
#include <gtsam/slam/GeneralSFMFactor.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/Values.h>
//#include <gtsam/nonlinear/NonlinearISAM.h>
#include <gtsam/geometry/triangulation.h>
#include "optimizer.hpp"
#include <fstream>


#include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>
#include <gtsam/nonlinear/DoglegOptimizer.h>


using namespace std;
using namespace gtsam;



Optimizer::Optimizer(double f, double cx, double cy)
{
    //relinearizeInterval=1;
    //isam = NonlinearISAM(relinearizeInterval);
    n=0;
    K = Cal3_S2(f,f,0.,cx,cy);
    firstPoint = false;
    // Add a prior on pose x1. This indirectly specifies where the origin is.
    noiseModel::Diagonal::shared_ptr priorNoise = noiseModel::Diagonal::Sigmas((Vector(6) << Vector3::Constant(0.01), Vector3::Constant(0.01)).finished()); // 1cm std on x,y,z 0.01 rad on roll,pitch,yaw
    noiseModel::Isotropic::shared_ptr measurement_noise = noiseModel::Isotropic::Sigma(2, 2.0); // pixel error in (x,y)
    noiseModel::Diagonal::shared_ptr cal_noise = noiseModel::Diagonal::Sigmas((Vector(5) << 10, 10, 0.01 /*skew*/, 20, 20).finished());
    graph.emplace_shared<PriorFactor<Pose3> >(Symbol('x', 0), Pose3(), priorNoise); // add directly to graph
    graph.emplace_shared<PriorFactor<Cal3_S2>>(Symbol('K', 0), K, cal_noise);
    //initialEstimate.insert(Symbol('x',0),Pose3());
    //initialEstimate.insert(Symbol('K', 0), K);
    //currentEstimate.insert(Symbol('K', 0), K);
    Estimate.insert(Symbol('x',0),Pose3());
    Estimate.insert(Symbol('K', 0), K);
    MEASURE_BEFORE_COMPUTATION=2;
    lastPointadded=-1;
}


void Optimizer::addObservation(vector<information>& measures, Matrix Rt, Matrix Post)
{
/* TODO : Take the essential matrix
from that compute the 4 possible poses with a SVD (eigen)
compute all the triangulation with the 4 poses.
keep the one that had the least cheirality exception (or better, rewrite the triangulation, so that we can avoid slow exception
keep those triangulation as values to calculate the scale.
*/
    Rot3 R = Rot3(Rt);
    Point3 T = Point3(Post.transpose());
    Pose3 P = Pose3(R,T);
    if(n==0) //initialization
    {
        for(information p : measures)
        {
            observations.push_back(Observation(p.index, 0, p.pt, p.color));
        }
    }
    else
    {
        if(n==1) //particular case, we don't have to compute scale, and the currentEstimates is not usable yet
        {
            Estimate.insert(Symbol('x',1), P*Estimate.at<Pose3>(Symbol('x',0)));//P is normalized, it becomes the scale
            //triangulate all points observed twice
            for(information p : measures) //loop over all the measures
            {
                if(p.index+1>observations.size()) //if this has never been observed (not sure if useful)
                {
                    observations.push_back(Observation(p.index, 1, p.pt, p.color));//then create the index observation
                }
                else
                {
                    try
                    {
                        Point3 pt = triangulatePoint3(vector<Pose3> {Estimate.at<Pose3>(Symbol('x',0)), P}, boost::make_shared<Cal3_S2>(Estimate.at<Cal3_S2>(Symbol('K', 0))), Point2Vector{observations[p.index].measures.back(), p.pt}); //project the point
                        observations[p.index].coordinates = pt;
                        observations[p.index].cameraId.push_back(1);//add 1 to the list of the camera that have seen this point
                        observations[p.index].measures.push_back(p.pt); //then add the measures
                    }
                    catch(...)
                    {
                    cout<<"triangulation failed"<<endl;
                    }
                }
            }
        }
        else
        {
            Pose3 pose = Estimate.at<Pose3>(Symbol('x',n-1));
            double scale;
            int iterations;
            for(int i=0; i<measures.size(); i++)
            {
                if(observations.size()>measures[i].index&&Estimate.exists(Symbol('p', measures[i].index)))
                {
                    try
                    {
                    scale+=(Estimate.at<Point3>(Symbol('p', measures[i].index))-Estimate.at<Pose3>(Symbol('x',n-1)).translation()).norm()/(triangulatePoint3(vector<Pose3> {pose, P*pose}, boost::make_shared<Cal3_S2>(Estimate.at<Cal3_S2>(Symbol('K', 0))),
                                                    Point2Vector{observations[measures[i].index].measures.back(), measures[i].pt})-Estimate.at<Pose3>(Symbol('x',n-1)).translation()).norm();
                    iterations++;
                    }
                    catch(...){

                    }
                }
            }
            scale/=iterations;
            cout<<"scale : "<< scale<<endl;
            //apply scale to initialize value
            Estimate.insert(Symbol('x',n), Pose3(R,scale*T)*pose);
            //triangulate all possible points
            for(information p : measures) //loop over all the measures
            {
                if(p.index==861)
                    cout<<observations[p.index].measures.size()<<endl;
                if(p.index+1>observations.size()) //if this has never been observed
                {
                    observations.push_back(Observation(p.index, n, p.pt, p.color));
                }
                else if(observations[p.index].measures.size()==1) //if the point has been observed only once
                {
                    try
                    {
                        Point3 pt = triangulatePoint3(vector<Pose3> {pose, Estimate.at<Pose3>(Symbol('x',n))}, boost::make_shared<Cal3_S2>(Estimate.at<Cal3_S2>(Symbol('K', 0))), Point2Vector{observations[p.index].measures.back(), p.pt}); //we can now triangulate it's position
                        observations[p.index].coordinates = pt;
                        observations[p.index].cameraId.push_back(n);
                        observations[p.index].measures.push_back(p.pt);
                    }
                    catch(...)
                    {
                    cout<<"Triangulation error on point "<<p.index<<endl;
                    }
                }
                else
                {
                    observations[p.index].cameraId.push_back(n);
                    observations[p.index].measures.push_back(p.pt);
                    if(observations[p.index].measures.size()>MEASURE_BEFORE_COMPUTATION)
                    {
                        graph.emplace_shared<GeneralSFMFactor2<Cal3_S2>>(p.pt, measurement_noise, Symbol('x', n), Symbol('p', p.index), Symbol('K', 0));
                    }
                }

            }
        }
        for(int i= 0; i<observations.size(); i++)
        {
            if(observations[i].measures.size()==MEASURE_BEFORE_COMPUTATION&&!Estimate.exists(Symbol('p', i)))
            {
                Estimate.insert<Point3>(Symbol('p', i), observations[i].coordinates);
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
    this->optimize();
    n++;

}

void Optimizer::optimize()
{
    cout << "initial error = " << graph.error(Estimate) << endl;
    //Estimate.print();
    //graphPrint();
    bool b =false;
    double precision = 1e-4;
    while(!b)
    {
        try
        {
            if(precision<1){
            DoglegParams params;
            params.relativeErrorTol = precision;
            params.absoluteErrorTol = precision;
            params.verbosityDL = DoglegParams::VerbosityDL::SILENT;
            //Estimate = LevenbergMarquardtOptimizer(graph, Estimate).optimize();
            Estimate = DoglegOptimizer(graph, Estimate, params).optimize();
            b=true;
            }
            else{
            Estimate = LevenbergMarquardtOptimizer(graph, Estimate).optimize();
            }

        }
        catch(...)
        {
            precision*=10;
            cout<<"relative error is now "<< precision<<endl;
        }


    }
    //graph.print();
    //Estimate.print("results");
    cout << "final error = " << graph.error(Estimate) << endl;
    int i=0;
    ofstream data ("points_computed.data");
    data << "# X\tY\tZ\trgb"<<endl;
    while(i<observations.size())
    {
        if(Estimate.exists(Symbol('p',i)))
            data<<Estimate.at<Point3>(Symbol('p',i)).z()<<"\t"<<Estimate.at<Point3>(Symbol('p',i)).x()<<"\t"<<-Estimate.at<Point3>(Symbol('p',i)).y()<<"\t"<<observations[i].color<<endl;
        i++;
    }

    /*
            int i=0;
        ofstream data ("points_computed.data");
        data << "# X\tY\tZ\trgb"<<endl;
        while(i<observations.size())
        {
            if(initialEstimate.exists(Symbol('p',i)))
                data<<initialEstimate.at<Point3>(Symbol('p',i)).z()<<"\t"<<initialEstimate.at<Point3>(Symbol('p',i)).x()<<"\t"<<-initialEstimate.at<Point3>(Symbol('p',i)).y()<<"\t"<<observations[i].color<<endl;
            i++;
        }

        isam.update(graph, initialEstimate);
        cout<<"optimisé"<<endl;
        initialEstimate.clear();
        currentEstimate = isam.estimate();
        currentEstimate.print();
        graph  = NonlinearFactorGraph();
        */
}


void Optimizer::graphPrint()
{
    graph.print("\nFactor Graph:\n");
}



