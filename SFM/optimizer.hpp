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
#include <fstream>

using namespace std;
using namespace gtsam;

class Optimizer
{
public:
    // Create a factor graph
    NonlinearFactorGraph graph;
    NonlinearISAM isam;
    Cal3_S2 K;
    int relinearizeInterval;
    noiseModel::Diagonal::shared_ptr poseNoise;
    noiseModel::Diagonal::shared_ptr observationNoise;
    Values initialEstimate;
    Values currentEstimate;
    int n;
    int lastPointComputed;
    int lastPointadded;
    noiseModel::Isotropic::shared_ptr measurement_noise;
    bool optimizable;
        bool optimized;
        bool firstPoint;
        struct Observation {
    int index;
    Point3 coordinates;
    vector<Point2> measures;
    vector<int> cameraId;
    //Observation(int i, Point3 p, Point2 p2d);

    Observation(int i, int c, Point3 p, Point2 p2d) {
            index=i;//this information is redundant for now but might be useful some day
            coordinates=p;
             measures = {p2d};
             cameraId = {c};
            }
    };
        vector<Observation> observations;
    Optimizer(double f, double cx, double cy);

//méthodes
    void addPose(Matrix Rt, Matrix Post);
    void addPoint(int indexPoint, double u, double v,double x, double y,double z);
    void addPoint(int indexPoint, int indexCamera, double u, double v,double x, double y,double z);
    bool optimize(Eigen::Matrix<double, 3, 3>& R, Vector3& t);//return the success of the optimization
    void graphPrint();
// - add with order
// - add with new points (?)
// - add without point order
// - compute
//

};
