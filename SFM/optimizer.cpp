#include <iostream>
#include <gtsam/geometry/Point2.h>
//#include <gtsam/geometry/Pose3.h>
#include <gtsam/inference/Symbol.h>
#include <gtsam/slam/PriorFactor.h>
#include <gtsam/slam/ProjectionFactor.h>
#include <gtsam/slam/GeneralSFMFactor.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/Values.h>
//#include <gtsam/slam/BetweenFactor.h>
#include <gtsam/nonlinear/NonlinearISAM.h>
#include "optimizer.hpp"
#include <fstream>

using namespace std;
using namespace gtsam;



Optimizer::Optimizer(double f, double cx, double cy)
{
    relinearizeInterval=3;
    isam = NonlinearISAM(relinearizeInterval);
    n=0;
    lastPointComputed=-1;
    lastPointadded=-1;
    K = Cal3_S2(f,f,0.,cx,cy);
    optimizable=true;
    optimized=false;
    firstPoint = false;
    // Add a prior on pose x1. This indirectly specifies where the origin is.
    noiseModel::Diagonal::shared_ptr priorNoise = noiseModel::Diagonal::Sigmas((Vector(6) << Vector3::Constant(0.01), Vector3::Constant(0.01)).finished()); // 1cm std on x,y,z 0.01 rad on roll,pitch,yaw
    noiseModel::Isotropic::shared_ptr measurement_noise = noiseModel::Isotropic::Sigma(2, 3.0); // pixel error in (x,y)
     noiseModel::Diagonal::shared_ptr cal_noise = noiseModel::Diagonal::Sigmas((Vector(5) << 20, 20, 0.01 /*skew*/, 10, 10).finished());
    graph.emplace_shared<PriorFactor<Pose3> >(Symbol('x', 0), Pose3(), priorNoise); // add directly to graph
    graph.emplace_shared<PriorFactor<Cal3_S2>>(Symbol('K', 0), K, cal_noise);
    initialEstimate.insert(Symbol('x',0),Pose3());
    initialEstimate.insert(Symbol('K', 0), K);
    //currentEstimate.insert(Symbol('x',0),Pose3());
}

//méthodes
void Optimizer::addPose(Matrix Rt, Matrix Post)
{
    Rot3 R = Rot3(Rt);
    Point3 T = Point3(Post.transpose());
    Pose3 P = Pose3(R,T);

    //graph.emplace_shared<BetweenFactor<Pose3> >(Symbol('x',n), Symbol('x',n+1), P, poseNoise);
    if(optimized){
        initialEstimate.insert(Symbol('x',n+1), P*currentEstimate.at<Pose3>(Symbol('x',n)));
            optimized=false;
    }
    else{
        initialEstimate.insert(Symbol('x',n+1), P*initialEstimate.at<Pose3>(Symbol('x',n)));
    }
    n++;
}

void Optimizer::addPoint(int indexPoint, double u, double v,double x, double y,double z)
{
    Optimizer::addPoint(indexPoint, n, u, v, x, y, z);
}
void Optimizer::addPoint(int indexPoint, int indexCamera, double u, double v, double x, double y, double z) //careful, we need world coordinates x,y,z
{
    if(indexPoint>lastPointComputed)
    {
    if(indexPoint-lastPointComputed>1)
    cout<<"je ne sais pas coder"<<endl;
    observations.push_back(Observation(indexPoint, indexCamera, Point3(x,y,z), Point2(u,v)));
        lastPointComputed = indexPoint;
    }
    else if (indexPoint>lastPointadded&&observations[indexPoint].measures.size()<3){
        if(indexPoint>=observations.size())
            cout<<"je ne sais pas coder bis"<<endl;
    observations[indexPoint].cameraId.push_back(indexCamera);
    observations[indexPoint].measures.push_back( Point2(u,v));
    }
    else if(indexPoint>lastPointadded&&observations[indexPoint].measures.size()==3){
    for (int i=0; i<3; i++){
    graph.emplace_shared<GeneralSFMFactor2<Cal3_S2>>(observations[indexPoint].measures[i], measurement_noise, Symbol('x', observations[indexPoint].cameraId[i]), Symbol('p', indexPoint), Symbol('K', 0));
    }
    initialEstimate.insert<Point3>(Symbol('p', indexPoint), Point3(x,y,z));
    if(!firstPoint){
                        noiseModel::Isotropic::shared_ptr point_noise = noiseModel::Isotropic::Sigma(3, max(max(abs(x),abs(y)),abs(z)));
                    graph.emplace_shared<PriorFactor<Point3>>(Symbol('p', indexPoint), Point3(x,y,z), point_noise);
                    firstPoint=true;
    }
    optimizable=true;
    lastPointadded=indexPoint;
    }
    else{
        graph.emplace_shared<GeneralSFMFactor2<Cal3_S2>>(Point2(u,v), measurement_noise, Symbol('x', indexCamera), Symbol('p', indexPoint), Symbol('K', 0));
        optimizable = true;
    }
}

bool Optimizer::optimize(Eigen::Matrix<double, 3, 3>& R, Vector3& t){
    if(optimizable){
        isam.update(graph, initialEstimate);
        cout<<"optimisé"<<endl;
    initialEstimate.clear();
    currentEstimate = isam.estimate();
    R = currentEstimate.at<Pose3>(Symbol('x',n)).rotation().matrix();
    t = currentEstimate.at<Pose3>(Symbol('x',n)).translation().vector();
    int i=0;
     ofstream data ("points_computed.data");
    data << "# X\tY\tZ"<<endl;
    while(i<lastPointComputed){
    if(currentEstimate.exists(Symbol('p',i)))
    data<<currentEstimate.at<Point3>(Symbol('p',i)).z()<<"\t"<<currentEstimate.at<Point3>(Symbol('p',i)).x()<<"\t"<<-currentEstimate.at<Point3>(Symbol('p',i)).y()<<endl;
    i++;
    }

    //currentEstimate.print();
    graph  = NonlinearFactorGraph();
    optimized=true;
    optimizable=false;
    return true;
    }
    return false;
}


void Optimizer::graphPrint()
{
    graph.print("\nFactor Graph:\n");
}


// - add with order
// - add with new points (?)
// - add without point order
// - compute
//

