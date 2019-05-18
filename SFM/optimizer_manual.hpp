#include <iostream>
#include <gtsam/geometry/Point2.h>
//#include <gtsam/geometry/Pose3.h>
#include <gtsam/inference/Symbol.h>
#include <gtsam/slam/PriorFactor.h>
#include <gtsam/slam/ProjectionFactor.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/Values.h>
#include <gtsam/slam/BetweenFactor.h>
#include <gtsam/nonlinear/NonlinearISAM.h>

using namespace std;
using namespace gtsam;


class Optimizer
{
public:
    // Create a factor graph
    NonlinearFactorGraph graph;
    int relinearizeInterval;
    noiseModel::Isotropic::shared_ptr measurementNoise;
    noiseModel::Diagonal::shared_ptr poseNoise;
    Cal3_S2 K;
    int n;

    Optimizer(double f, double u, double v);

//méthodes
    void add(Matrix Rt, Matrix Post);

    void graphPrint();
// - add with order
// - add with new points (?)
// - add without point order
// - compute
//

};
