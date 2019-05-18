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


//#include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>
//#include <gtsam/nonlinear/DoglegOptimizer.h>
using namespace std;
using namespace gtsam;

class Optimizer
{
public:
    // Create a factor graph
    NonlinearFactorGraph graph;
    //NonlinearISAM isam;
    Cal3_S2 K;
    int relinearizeInterval;
    noiseModel::Diagonal::shared_ptr poseNoise;
    noiseModel::Diagonal::shared_ptr observationNoise;
 //   Values initialEstimate;
 //   Values currentEstimate;
    Values Estimate;
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

        /*Observation(int i, int c, Point3 p, Point2 p2d) {
                index=i;//this information is redundant for now but might be useful some day
                coordinates=p;
                 measures = {p2d};
                 cameraId = {c};
                }*/
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
    void addObservation(vector<information>& measures, Matrix R, Matrix t);
    void optimize();//return the success of the optimization
    void graphPrint();

};
