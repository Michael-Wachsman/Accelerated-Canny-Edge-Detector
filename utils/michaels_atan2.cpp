#include <cstdint>
#include <algorithm>
#include <cmath>

// source https://nghiaho.com/?p=997 and https://math.stackexchange.com/questions/1098487/atan2-faster-approximation
// error +- .09 degrees
double fast_atan2(float x,float y){
    // since atan2(x,y) == atan(y/x)
    x = y/x/(1020)
    return M_PI_4*x - x*(std::fabs(x) - 1)*(0.2447 + 0.0663*std::fabs(x));

}

// made by chat and verified by me lol
//only gets the cardinal directions and the diagonals
double superFastAtan2(double x, double y){
    if (dy > 0) {
        if (dx == 0) return 0; // North
        if (dx > 0) return 45; // North-East
        if (dx < 0) return 315; // North-West
    } else if (dy < 0) {
        if (dx == 0) return 180; // South
        if (dx > 0) return 135; // South-East
        if (dx < 0) return 225; // South-West
    } else {
        if (dx > 0) return 90; // East
        if (dx < 0) return 270; // West
    }
}
// Alternative... use an array like a hash table 


// made by chat and verified by me lol
//only gets the cardinal directions and the diagonals THAT MATTER
double superSimpleFastAtan2(double x, double y){
    if (dy > 0) {
        if (dx == 0) return 0; // North
        if (dx > 0) return 45; // North-East
        if (dx < 0) return 135; // North-West
    } else if (dy < 0) {
        if (dx == 0) return 0; // South
        if (dx > 0) return 135; // South-East
        if (dx < 0) return 45; // South-West
    } else {
        if (dx > 0) return 90; // East
        if (dx < 0) return 90; // West
    }
}
// Alternative... use an array like a hash table 