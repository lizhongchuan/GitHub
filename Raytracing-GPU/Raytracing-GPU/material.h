#ifndef MATERIALH
#define MATERIALH 


class material
{
public:
	int matType;
	vec3 albedo;
	float ratio;
};

class lambertian : public material
{

public:
	lambertian(const vec3& a)
	{
		albedo = a;
		matType = 0;
		ratio = 0;
	}
};

class metal : public material 
{
public:
	metal(const vec3& a, float f)
	{
		albedo = a;
		if (f < 1)
			ratio = f;
		else ratio = 1;
		matType = 1;
	}
};

class dielectric : public material 
{
public:
	dielectric(float ri)
	{
		matType = 2;
		albedo = vec3(1.0, 1.0, 1.0);
		ratio = ri;
	}
};

#endif




