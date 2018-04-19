#ifndef SPHEREH
#define SPHEREH

#include "vec3.h"
#include "material.h"

struct hit_record
{
	float t;
	vec3 p;
	vec3 normal;
	material *mat_ptr;
};

class sphere
{
public:
	sphere() {}
	sphere(vec3 cen, float r, material *m) : center(cen), radius(r), mat_ptr(m) {};

	vec3 center;
	float radius;
	material *mat_ptr;
};


#endif



