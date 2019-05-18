#include <gtsam/geometry/Pose3.h>
#include <cmath>

using namespace gtsam;
struct observation{

};

struct Object {
double half_life;
double tau;
int last_index;
double last_t;

Object(double half_life){
this->half_life=half_life;
tau=half_life/log(2);
last_index=0;
};
Object(){
tau=INFINITY;
half_life=INFINITY;
last_index=0;
};
};

struct Odometry {
double t;
Pose3 pose;
Odometry(double time, Pose3 p){
pose=p;
t=time;
};
};

struct Measure {
double t;
Pose3 pose;
int id;
Measure(double time, int index, Pose3 p){
pose=p;
t=time;
id=index;
};
};

std::map<int, Object> object_map = {
    //{ 1, Object() }
};

int ClosestPow10(int v){
int n=0;
while(v>=1){
v/=10;
n++;
}
return pow(10,n);
}
