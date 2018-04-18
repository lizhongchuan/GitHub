#ifndef HITABLEH
#define HITABLEH 

#include "ray.h"

class material;



struct hit_record
{
    float t;  
    vec3 p;
    vec3 normal; 
    material *mat_ptr;
};

class hitable  {

};

#endif




