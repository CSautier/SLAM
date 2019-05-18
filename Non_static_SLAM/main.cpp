// Each variable in the system (poses and landmarks) must be identified with a unique key.
// We can either use simple integer keys (1, 2, 3, ...) or symbols (X1, X2, L1).
// Here we will use Symbols
#include <gtsam/inference/Symbol.h>

// In GTSAM, measurement functions are represented as 'factors'. Several common factors
// have been provided with the library for solving robotics/SLAM/Bundle Adjustment problems.
// Here we will use a RangeBearing factor for the range-bearing measurements to identified
// landmarks, and Between factors for the relative motion described by odometry measurements.
// Also, we will initialize the robot at the origin using a Prior factor.
#include <gtsam/slam/PriorFactor.h>
#include <gtsam/slam/BetweenFactor.h>

// When the factors are created, we will add them to a Factor Graph. As the factors we are using
// are nonlinear factors, we will need a Nonlinear Factor Graph.
#include <gtsam/nonlinear/NonlinearFactorGraph.h>


// Once the optimized values have been calculated, we can also calculate the marginal covariance
// of desired variables
#include <gtsam/nonlinear/Marginals.h>

// The nonlinear solvers within GTSAM are iterative solvers, meaning they linearize the
// nonlinear functions around an initial linearization point, then solve the linear system
// to update the linearization point. This happens repeatedly until the solver converges
// to a consistent set of variable values. This requires us to specify an initial guess
// for each variable, held in a Values container.
#include <gtsam/nonlinear/Values.h>
#include <gtsam/nonlinear/ISAM2.h>
#include <iostream>
#include <fstream>
#include "objects.h"
#include "maxmixture_factor.h"
#include <math.h>
#include <ctime>
#define Max_ellipsis 300

using namespace std;
using namespace gtsam;

ofstream tempfile ("temp");
ofstream plot ("SLAM.p");

vector<Odometry> odometryList;
vector<Measure> measureList;
double ks = 1;//speed constant in m/s
double kw = 0.1;//angular speed constant in rad/s
const double sqrt3 = sqrt(3);
int number_of_odometries = 0;

class Settings
{
public:
    string odometryPath;
    string measurePath;
    double gaussnewtonparameter;
    double relinearizeThreshold;
    int relineariseSkip;
    bool cacheLinearizedFactors;
    bool enableDetailedResults;
    bool useQRfactorization;
    bool useMaxMixture;
    bool saveResults;
    double priorTranslationStd;
    double priorRotationStd;
    double odometryTranslationStd;
    double odometryRotationStd;
    double measuremenTranslationStd;
    double measurementRotationStd;
    double movementThreshold;
    int widthResolution;
    int heightResolution;
    string xmin;
    string xmax;
    string ymin;
    string ymax;


    string s;

    Settings(string inputSettingsFile) //
    {
        ifstream is(inputSettingsFile);
        if (!is.is_open())
        {
            cout << "Could not open the configuration file: \"" << inputSettingsFile << "\"" << endl;
            throw -1;
        }
        is >> s >> odometryPath;
        is >> s >> measurePath;
        is >> s >> gaussnewtonparameter;
        is >> s >> relinearizeThreshold;
        is >> s >> relineariseSkip;
        is >> s >> cacheLinearizedFactors;
        is >> s >> enableDetailedResults;
        is >> s >> useQRfactorization;
        is >> s >> useMaxMixture;
        is >> s >> saveResults;
        is >> s >> priorTranslationStd;
        is >> s >> priorRotationStd;
        is >> s >> odometryTranslationStd;
        is >> s >> odometryRotationStd;
        is >> s >> measuremenTranslationStd;
        is >> s >> measurementRotationStd;
        is >> s >> movementThreshold;
        is >> s >> widthResolution;
        is >> s >> heightResolution;
        is >> s >> xmin;
        is >> s >> xmax;
        is >> s >> ymin;
        is >> s >> ymax;
        is.close();
    }

};


class ModifiableISAM2 : public ISAM2
{
public:
    void replaceParams (ISAM2Params parameters)
    {
        this->params_=parameters;
        this->params_.print();
    }
    ModifiableISAM2 (ISAM2Params parameters) : ISAM2 (parameters) {}
};

void objectsSetup(string objectPath){
 ifstream is(objectPath);
 string s;
 int id;
 string t;
while (is >> s >> id >> s >> t )
    {
if(object_map.count(id)==0)
        {
 try{
 object_map.insert({id, Object(stod(t))});//we can indicate a half-life here. If not it will be infinity
 cout<<"Item n°"<<id<<" added with half-life="<<stod(t)<<endl;
    }
    catch(...){}
        }
    }
    is.clear(); /* clears the end-of-file and error flags */
    is.close();
}

//read the odometry and measures files. Not useful in real time
void reader(string odometryPath, string measuresPath)
{
    ifstream is(odometryPath);
    double t, x, y, z, rx, ry, rz, rw;
    while (is >> t >> x >> y >> z >> rx >> ry >> rz >> rw)
    {
        odometryList.push_back(Odometry(t, Pose3(Rot3::quaternion(rw, rx, ry, rz), Point3(x,y,z))));
        number_of_odometries++;
    }
    is.clear(); /* clears the end-of-file and error flags */
    is.close();

    is=ifstream (measuresPath);
    int id;

    while (is >> t >> id >> x >> y >> z >> rx >> ry >> rz >> rw)
    {
        measureList.push_back(Measure(t, id, Pose3(Rot3::RzRyRx(-M_PI/2, 0, -M_PI/2), Point3())*Pose3(Rot3::quaternion(rw, rx, ry, rz), Point3(x,y,z/*z,-x,-y*/))));//apply transform from camera axis to world axis
        if(object_map.count(id)==0)
        {
                object_map.insert({id, Object()});//we can indicate a half-life here. If not it will be infinity
                cout<<"***************************\nItem n°"<<id<<" non declared, added with half-life=infinity\n***************************\n";
        }
    }
    is.clear(); //clears the end-of-file and error flags
    is.close();
}

int main(int argc, char** argv)
{
    const string inputSettingsFile = argc > 1 ? argv[1] : "default.txt";
    const string inputObjectsFile = argc > 2 ? argv[2] : "objects.txt";
    //read the parameters
    Settings settings(inputSettingsFile);
    objectsSetup(inputObjectsFile);
    // Create a factor graph
    NonlinearFactorGraph graph;

    //create an ISAM2 optimizer, with the parameters indicated in the file
    ISAM2Params parameters;
    ISAM2GaussNewtonParams gaussnewtonparameters = ISAM2GaussNewtonParams(settings.gaussnewtonparameter);
    parameters.optimizationParams = gaussnewtonparameters;
    parameters.relinearizeThreshold = settings.relinearizeThreshold;
    parameters.relinearizeSkip = settings.relineariseSkip;
    parameters.cacheLinearizedFactors = settings.cacheLinearizedFactors;
    parameters.enableDetailedResults = settings.enableDetailedResults;
    if(settings.useQRfactorization)
    {
        parameters.factorization = gtsam::ISAM2Params::QR;
    }
    else
    {
        parameters.factorization = gtsam::ISAM2Params::CHOLESKY;
    }
    parameters.print();
    ModifiableISAM2 isam(parameters);
    cout<<"**************************\nResolution: "<<settings.widthResolution<<"x"<<settings.heightResolution<<"\n**************************\n";

    //create the noisemodels
    noiseModel::Diagonal::shared_ptr odometryNoise = noiseModel::Diagonal::Sigmas((Vector(6) << Vector3::Constant(settings.odometryTranslationStd), Vector3::Constant(settings.odometryRotationStd)).finished()); // 0.3cm std on x,y, 0.01 rad on theta
    noiseModel::Diagonal::shared_ptr measurementNoise = noiseModel::Diagonal::Sigmas((Vector(6) << Vector3::Constant(settings.measuremenTranslationStd), Vector3::Constant(settings.measurementRotationStd)).finished()); // 0.1 rad std on bearing, 0.1cm on range
    noiseModel::Diagonal::shared_ptr zeroMovementNoise = noiseModel::Diagonal::Sigmas((Vector(6) << Vector3::Constant(settings.measuremenTranslationStd/1000), Vector3::Constant(settings.measurementRotationStd/1000)).finished()); // 0.1 rad std on bearing, 0.1cm on range

    Values initialEstimate;
    Values currentEstimate;

    reader(settings.odometryPath, settings.measurePath);
    double xmin,ymin=INFINITY;
    double xmax,ymax=-INFINITY;

    double prev_t=odometryList[0].t;
    int measureindex = 0;

    int n =0;
    double elapsed_secs;
    bool optimize=true;
    for(int i=0; i<odometryList.size(); i++) //this can be replaced if necessary by a Ros handler of the odometry information
    {
        //verify the time for the mesures is correct this should normally be used only if the first measures come before the first odometry, or if the measures come as the robot hasn't moved
        while(measureindex<measureList.size()&&measureList[measureindex].t<prev_t)
        {
            measureindex++;
        }

        if(i!=0)//not the first loop
        {
            Pose3 p3 = odometryList[i-1].pose.between(odometryList[i].pose);//relative pose between the 2 odometries

            if(currentEstimate.exists(Symbol('x',n-1))){
                initialEstimate.insert(Symbol('x',n), currentEstimate.at<Pose3>(Symbol('x',n-1))*(p3));
            }
            else
            {
                initialEstimate.insert(Symbol('x',n), initialEstimate.at<Pose3>(Symbol('x',n-1))*(p3));
            }

            if((p3.translation()).norm()<settings.movementThreshold)//if the motion is almost zero, the noise must be very low
            {
                graph.emplace_shared<BetweenFactor<Pose3> >(Symbol('x',n-1), Symbol('x',n), p3, zeroMovementNoise);
            }
            else{
             graph.emplace_shared<BetweenFactor<Pose3> >(Symbol('x',n-1), Symbol('x',n), p3, odometryNoise);
            }

        }
        else //the first loop
        {
        //add a prior
            initialEstimate.insert(Symbol('x',0), odometryList[i].pose);
            noiseModel::Diagonal::shared_ptr priorNoise = noiseModel::Diagonal::Sigmas((Vector(6) << Vector3::Constant(settings.priorTranslationStd), Vector3::Constant(settings.priorRotationStd)).finished()); // 0.3cm std on x,y, 0.01 rad on theta
            graph.emplace_shared<PriorFactor<Pose3> >(Symbol('x',0), odometryList[0].pose, priorNoise);
            cout<<"Initialization: Successful"<<endl;
        }
        while(measureindex<measureList.size()&&measureList[measureindex].t<odometryList[i].t)//loop over all the measures, between 2 odometries
        {
            optimize=true;
            double dt=measureList[measureindex].t-object_map.at(measureList[measureindex].id).last_t;
            double p = exp(-dt/object_map.at(measureList[measureindex].id).tau);
            if(object_map.at(measureList[measureindex].id).last_index==0)  //first observation of a landmark, it's position will be the measurement
            {
                object_map.at(measureList[measureindex].id).last_index++;
                int ind=measureList[measureindex].id*10*ClosestPow10(object_map.at(measureList[measureindex].id).last_index)+object_map.at(measureList[measureindex].id).last_index;
                initialEstimate.insert(Symbol('l',ind), initialEstimate.at<Pose3>(Symbol('x',n))*(measureList[measureindex].pose));
                graph.emplace_shared<BetweenFactor<Pose3> >(Symbol('x',n), Symbol('l',ind), measureList[measureindex].pose, measurementNoise);
            }
            else //not the first observation, it's position will be the previous position of the landmark
            {
                if(p!=1)
                object_map.at(measureList[measureindex].id).last_index++;
                int ind=measureList[measureindex].id*10*ClosestPow10(object_map.at(measureList[measureindex].id).last_index)+object_map.at(measureList[measureindex].id).last_index;
                graph.emplace_shared<BetweenFactor<Pose3> >(Symbol('x',n), Symbol('l',ind), measureList[measureindex].pose, measurementNoise);

                //the noise is a maxmixture of a peak gaussian around the position of the previously measured landmark, and a broad gaussian estimating where the landmark might have moved.
                if(p!=1)
                {
                if(currentEstimate.exists(Symbol('l',measureList[measureindex].id*10*ClosestPow10(object_map.at(measureList[measureindex].id).last_index-1)+object_map.at(measureList[measureindex].id).last_index-1)))
                {
                initialEstimate.insert(Symbol('l',ind),currentEstimate.at<Pose3>(
                                Symbol('l',measureList[measureindex].id*10*ClosestPow10(object_map.at(measureList[measureindex].id).last_index-1)+object_map.at(measureList[measureindex].id).last_index-1)));
                }
                else
                {
                initialEstimate.insert(Symbol('l',ind),initialEstimate.at<Pose3>(
                                Symbol('l',measureList[measureindex].id*10*ClosestPow10(object_map.at(measureList[measureindex].id).last_index-1)+object_map.at(measureList[measureindex].id).last_index-1)));
                }

                    if(settings.useMaxMixture)
                    {
                        vector<double> weights {p, 1-p};
                        vector<BetweenFactor<Pose3>> between_factors;
                        noiseModel::Diagonal::shared_ptr noise1 = noiseModel::Diagonal::Sigmas((Vector(6) << Vector3::Constant(ks*dt/2000),Vector3::Constant(kw*dt/2000)).finished());
                        noiseModel::Diagonal::shared_ptr noise2 = noiseModel::Diagonal::Sigmas((Vector(6) << Vector3::Constant(ks*dt/sqrt3),Vector3::Constant(kw*dt/sqrt3)).finished());
                        between_factors.push_back(BetweenFactor<Pose3>(Symbol('l',measureList[measureindex].id*10*ClosestPow10(object_map.at(measureList[measureindex].id).last_index-1)+object_map.at(measureList[measureindex].id).last_index-1),
                                                  Symbol('l',ind), Pose3(), noise1));
                        between_factors.push_back(BetweenFactor<Pose3>(Symbol('l',measureList[measureindex].id*10*ClosestPow10(object_map.at(measureList[measureindex].id).last_index-1)+object_map.at(measureList[measureindex].id).last_index-1),
                                                  Symbol('l',ind), Pose3(), noise2));
                        graph.emplace_shared<maxmixture::MaxMixtureFactor<BetweenFactor<Pose3>>>(between_factors, weights);

                    }
                    else
                    {
                        //this is the old model, using a simple gaussian and not a maxmixture
                        noiseModel::Diagonal::shared_ptr timeNoise = noiseModel::Diagonal::Sigmas((Vector(6) << Vector3::Constant(ks*dt*(1-exp(-dt/object_map.at(measureList[measureindex].id).tau))),
                                Vector3::Constant(kw*dt*(1-exp(-dt/object_map.at(measureList[measureindex].id).tau)))).finished());
                        graph.emplace_shared<BetweenFactor<Pose3> >(Symbol('l',measureList[measureindex].id*10*ClosestPow10(object_map.at(measureList[measureindex].id).last_index-1)+object_map.at(measureList[measureindex].id).last_index-1),
                                Symbol('l',ind), Pose3(), timeNoise);
                    }

                }
            }
            object_map.at(measureList[measureindex].id).last_t=measureList[measureindex].t;
            measureindex++;
        }

        if(optimize)
        {
        clock_t begin = clock();
        isam.update(graph, initialEstimate);
        currentEstimate = isam.calculateEstimate();
        initialEstimate.clear();
        graph  = NonlinearFactorGraph();
        elapsed_secs += double(clock()-begin);
        optimize = false;
        }
        prev_t=odometryList[i].t;
        n++;

        if(i%20==0)
        {
            cout<<"Optimization: "<<i<<" / "<<number_of_odometries<<endl;
        }
            int last_index=1;

//**********************************************************************Visualization


        int max = ((n+5)/10)*10;
         for(int count=10; count<=min(Max_ellipsis,max); count+=10)
            {
                Matrix m;
                double x;
                double y;
                if(currentEstimate.exists(Symbol('x',max-count)))
                {
                m = isam.marginalCovariance(Symbol('x',max-count)).block<2,2>(0,0);
                x=currentEstimate.at<Pose3>(Symbol('x',max-count)).x();
                y=currentEstimate.at<Pose3>(Symbol('x',max-count)).y();
                Eigen::SelfAdjointEigenSolver<Matrix> eigensolver(m);
                double angle = atan2(eigensolver.eigenvectors()(0,1), eigensolver.eigenvectors()(0,0));
                if(angle < 0)
                    angle += 6.28318530718;
                angle = 180*angle/3.14159265359;
                double axe1 = 2*sqrt(5.991*eigensolver.eigenvalues()(0));
                double axe2 = 2*sqrt(5.991*eigensolver.eigenvalues()(0));

                tempfile <<  "set object "<<last_index++<<" ellipse center "<<x<<","<< y <<" size "<<axe1<<","<<axe2<<" angle "<<angle<<" front fillstyle empty border lt 4 -1"<<"\n";
                if (x<xmin&&isfinite(x))
                    xmin=x;
                if (x>xmax&&isfinite(x))
                    xmax=x;
                if (y<ymin&&isfinite(y))
                    ymin=y;
                if (y>ymax&&isfinite(y))
                    ymax=y;
                }
            }
            for(std::_Rb_tree_iterator<std::pair<const int, Object> > it = object_map.begin(); it!=object_map.end(); it++)
            {
                if(it->second.last_index!=0)
                {
                    int ind=it->first*10*ClosestPow10(it->second.last_index)+it->second.last_index;
                    Matrix m;
                    double x, y;
                    m = isam.marginalCovariance(Symbol('l',ind)).block<2,2>(0,0);
                    x=currentEstimate.at<Pose3>(Symbol('l',ind)).x();
                    y=currentEstimate.at<Pose3>(Symbol('l',ind)).y();
                    Eigen::SelfAdjointEigenSolver<Matrix> eigensolver(m);
                    double angle = atan2(eigensolver.eigenvectors()(0,1), eigensolver.eigenvectors()(0,0));
                    if(angle < 0)
                        angle += 6.28318530718;
                    angle = 180*angle/3.14159265359;
                    double axe1 = 2*sqrt(5.991*eigensolver.eigenvalues()(0));
                    double axe2 = 2*sqrt(5.991*eigensolver.eigenvalues()(0));
                    tempfile <<  "set object "<<Max_ellipsis+it->first<<" ellipse center "<<x<<","<< y<<" size "<<axe1<<","<<axe2<<" angle "<<angle<<" front fillstyle empty border lt "<<it->first<<" -1"<<endl;
                    if (x<xmin&&isfinite(x))
                        xmin=x;
                    if (x>xmax&&isfinite(x))
                        xmax=x;
                    if (y<ymin&&isfinite(y))
                        ymin=y;
                    if (y>ymax&&isfinite(y))
                        ymax=y;
                }
            }
            tempfile<<"plot '-' ls 7 with lines"<<"\n";
            for(int count=0; count+1<=n; count++)
            {
            double x;
            double y;
                if(currentEstimate.exists(Symbol('x',count)))
                {
                x=currentEstimate.at<Pose3>(Symbol('x',count)).x();
                y=currentEstimate.at<Pose3>(Symbol('x',count)).y();
                }
                else
                {
                x=initialEstimate.at<Pose3>(Symbol('x',count)).x();
                y=initialEstimate.at<Pose3>(Symbol('x',count)).y();
                }
                tempfile<<x<<" "<<y<<"\n";
                    if (x<xmin&&isfinite(x))
                        xmin=x;
                    if (x>xmax&&isfinite(x))
                        xmax=x;
                    if (y<ymin&&isfinite(y))
                        ymin=y;
                    if (y>ymax&&isfinite(y))
                        ymax=y;

            }
            tempfile <<  "e\n";
    }


    //double elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;
    cout<<"the optimization process took: "<<elapsed_secs/ CLOCKS_PER_SEC<<"s"<<endl;

    tempfile.close();
    ifstream temp ("temp");
    string s;

    xmin-= max(xmax - xmin,ymax-ymin) *0.1;
    xmax+=max(xmax - xmin,ymax-ymin)*0.1;
    ymin-=max(xmax - xmin,ymax-ymin)*0.1;
    ymax+=max(xmax - xmin,ymax-ymin)*0.1;
    try{
    xmin=stod(settings.xmin);
    }
    catch(...){}
        try{
    xmax=stod(settings.xmax);
    }
    catch(...){}
        try{
    ymin=stod(settings.ymin);
    }
    catch(...){}
        try{
    ymax=stod(settings.ymax);
    }
    catch(...){}

    plot << """reset\nset terminal gif animate delay 3 size "<<settings.widthResolution<<","<<settings.heightResolution<<"\nset output 'Slam.gif'\nset xrange ["<<xmin<<":"<<xmax<<
         "]\nset yrange ["<<ymin<<":"<<ymax<<"]\nset border 0\nunset tics\nunset key\nset tics scale 0.5\nset format xy ""\n""";

    while (getline(temp, s, '\n'))
    {
        plot << s << "\n";
    }
    temp.close();
    plot.close();

    if(settings.saveResults)
    {
ofstream results ("Results.txt");


ifstream is(settings.odometryPath);
    int i=0;
    double t, x, y, z, rx, ry, rz, rw;
    while (is >> t >> x >> y >> z >> rx >> ry >> rz >> rw)
    {
    if(currentEstimate.exists(Symbol('x',i)))
    {
            Vector rotation = (currentEstimate.at<Pose3>(Symbol('x',i)).rotation()).quaternion();
        results<<fixed<<t<<"\t"<<currentEstimate.at<Pose3>(Symbol('x',i)).x()<<"\t"<<currentEstimate.at<Pose3>(Symbol('x',i)).y()<<"\t"<<currentEstimate.at<Pose3>(Symbol('x',i)).z()<<"\t"<<rotation[0]<<"\t"<<rotation[1]<<"\t"<<rotation[2]<<"\t"<<rotation[3]<<endl;
    }
    else
    {
            Vector rotation = (initialEstimate.at<Pose3>(Symbol('x',i)).rotation()).quaternion();
        results<<fixed<<t<<"\t"<<initialEstimate.at<Pose3>(Symbol('x',i)).x()<<"\t"<<initialEstimate.at<Pose3>(Symbol('x',i)).y()<<"\t"<<initialEstimate.at<Pose3>(Symbol('x',i)).z()<<"\t"<<rotation[0]<<"\t"<<rotation[1]<<"\t"<<rotation[2]<<"\t"<<rotation[3]<<endl;
    }
      i++;
    }
    is.clear(); /* clears the end-of-file and error flags */
    is.close();
    }


}
