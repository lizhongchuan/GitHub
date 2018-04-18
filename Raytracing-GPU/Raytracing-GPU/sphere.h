#ifndef SPHEREH
#define SPHEREH

#include "hitable.h"

class sphere: public hitable  {
    public:
		sphere() {}
        sphere(vec3 cen, float r, material *m) : center(cen), radius(r), mat_ptr(m)  {};

        vec3 center;
        float radius;
        material *mat_ptr;
};


#endif



