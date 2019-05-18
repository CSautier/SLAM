//#include <iostream>
//#include <gtsam/geometry/Point2.h>
//#include <gtsam/inference/Symbol.h>
//#include <gtsam/slam/PriorFactor.h>
#include <gtsam/slam/ProjectionFactor.h>
#include <gtsam/slam/GeneralSFMFactor.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/Values.h>
//#include <gtsam/nonlinear/NonlinearISAM.h>
//#include <gtsam/geometry/triangulation.h>
//#include <fstream>

#include <gtsam/nonlinear/ISAM2.h>
//#include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>
//#include <gtsam/nonlinear/DoglegOptimizer.h>
using namespace std;
using namespace gtsam;

class Optimizer
{
public:
    // Create a factor graph
    NonlinearFactorGraph graph;
    ISAM2 isam;
    ISAM2Params parameters;
    ISAM2DoglegParams doglegparameters;
    Cal3_S2 K;
    int relinearizeInterval;
    noiseModel::Diagonal::shared_ptr poseNoise;
    noiseModel::Diagonal::shared_ptr observationNoise;
    Values initialEstimate;
    Values currentEstimate;
    //Values Estimate;
    Values safe_Estimate;
    NonlinearFactorGraph safe_graph;
    double error;
    int n;
    int lastPointComputed;
    int lastPointadded;
    int MEASURE_BEFORE_COMPUTATION;
    noiseModel::Isotropic::shared_ptr measurement_noise;
    bool firstPoint;
    struct Observation
    {
        int index;
        Point3 coordinates;
        vector<Point2> measures;
        vector<int> cameraId;
        int color;
        Observation(int i, int c, Point2 p2d, int colour)
        {
            index=i;//this information is redundant for now but might be useful some day
            measures = {p2d};
            cameraId = {c};
            color=colour;
        }
    };
    struct information{
    int index;
    Point2 pt;
    int color;
    information(int i, Point2 p2d, int colour){
    index=i;
    pt=p2d;
    color=colour;
    }
    };
    vector<Observation> observations;
    Optimizer(double f, double cx, double cy);

//méthodes
    bool addObservation(vector<information>& measures,  Matrix Ra,  Matrix Rb, Matrix Post, vector<bool>& mask);//return the success of the observation
    bool optimize();//return the success of the optimization
    void graphPrint();
    double cx();
    double cy();
    double fx();
    double fy();

};
