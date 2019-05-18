
#include <iostream>
#include <fstream>
using namespace std;


int main(int argc, char** argv)
{
    ifstream is("./Data/odometry_corentin.txt");
    double prev_t=0;
    double total=0;
    int n=0;
    double t, x, y, z, rx, ry, rz, rw;
    while (is >> t >> x >> y >> z >> rx >> ry >> rz >> rw)
    {
    if(prev_t!=0)
    {
            cout<<"t = "<<t<<"fps : "<<1/(t-prev_t)<<endl;
        total+=(t-prev_t);
    n++;
    }
    prev_t=t;
    }
    cout<<"total = "<<n/total<<endl;
    is.clear(); /* clears the end-of-file and error flags */
    is.close();
}
