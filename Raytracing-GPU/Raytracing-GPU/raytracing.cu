
#ifndef _RAY_TRACING_CU_
#define _RAY_TRACING_CU_

#include <helper_cuda.h>
#include <helper_math.h>
#include <curand_kernel.h>

#include "camera.h"
#include "material.h"
#include "sphere.h"

typedef unsigned int  uint;
typedef unsigned char uchar;



struct Ray
{
	float3 o;   // origin
	float3 d;   // direction
};

struct cameraCuda 
{
	float3 origin;
	float3 lower_left_corner;
	float3 horizontal;
	float3 vertical;
	float3 u, v, w;
	float lens_radius;
};

__constant__ cameraCuda d_camera[1]; //取地址符&拷贝有错误,只能使用数组?
__constant__ uint d_sphere_count[1] = { 0 };

uchar *d_output = 0;


struct sphereCuda
{
	float3 center;
	float radius;

	int matType;
	float3 albedo;
	float ratio; // for lambertian no use; for metal 0-1 fuzz; for dielectric ref_idx 
};

sphereCuda *d_sphere_list = 0;

struct hit_recordCuda
{
	float t;
	float3 p;
	float3 normal;

	int matType;
	float3 albedo;
	float ratio; // for lambertian no use; for metal 0-1 fuzz; for dielectric ref_idx 

};

__device__ float d_schlick(float cosine, float ref_idx) {
	float r0 = (1.0f - ref_idx) / (1.0f + ref_idx);
	r0 = r0*r0;
	return r0 + (1.0f - r0)*powf((1.0f - cosine), 5);
}

__device__ bool d_refract(const float3& v, const float3& n, float ni_over_nt, float3& refracted) {
	float3 uv = normalize(v);
	float dt = dot(uv, n);
	float discriminant = 1.0 - ni_over_nt*ni_over_nt*(1 - dt*dt);
	if (discriminant > 0) {
		refracted = ni_over_nt*(uv - n*dt) - n*sqrt(discriminant);
		return true;
	}
	else
		return false;
}

__device__ float3 d_reflect(const float3& v, const float3& n) {
	return v - 2 * dot(v, n)*n;
}

__device__ float d_drand48(unsigned long long seed)  // to do
{
	curandState devStates;
	curand_init(seed, 0, 0, &devStates);
	float RANDOM = curand_uniform(&devStates);
	RANDOM = RANDOM * 0.5 + 0.25;// 0.25 -> 0.75
	return RANDOM;
}

__device__ float3 d_random_in_unit_disk(unsigned long long seed)
{
	float3 p;
	float3 one3 = { 1.0f, 1.0f, 0.0f };
	//do {
		float3 rand3 = { d_drand48(seed), d_drand48(seed + 30), 0};
		p = 2.0f * rand3 - one3;
	//} while (dot(p, p) >= 1.0);

	return p;
}

__device__ float3 d_random_in_unit_sphere(unsigned long long seed) {
	float3 p;
	float3 one3 = { 1.0f, 1.0f, 1.0f };
	//do {
		float3 rand3 = { d_drand48(seed), d_drand48(seed + 10), d_drand48(seed + 20) };
		p = 2.0 * rand3 - one3;
	//} while (dot(p, p) >= 1.0f);
	return p;
}

__device__ bool d_scatter_lambertian(const Ray& r_in, const hit_recordCuda& rec, float3& attenuation, Ray& scattered)
{
	float3 target = rec.p + rec.normal + d_random_in_unit_sphere(r_in.d.x * r_in.d.y * 100000000.0);
	scattered = { rec.p, target - rec.p };
	attenuation = rec.albedo;
	return true;
}

__device__ bool d_scatter_metal(const Ray& r_in, const hit_recordCuda& rec, float3& attenuation, Ray& scattered)
{
	float3 reflected = d_reflect(normalize(r_in.d), rec.normal);
	scattered = { rec.p, reflected + rec.ratio*d_random_in_unit_sphere(r_in.d.x * r_in.d.y * 100000000.0) };//rec.ratio .ie fuzz
	attenuation = rec.albedo;
	return (dot(scattered.d, rec.normal) > 0);
}

__device__ bool d_scatter_dielectric(const Ray& r_in, const hit_recordCuda& rec, float3& attenuation, Ray& scattered)
{
		
	float3 outward_normal;
	float3 reflected = d_reflect(r_in.d, rec.normal);
	float ni_over_nt;
	attenuation = { 1.0f, 1.0f, 1.0f };
	float3 refracted;
	float reflect_prob;
	float cosine;
	if (dot(r_in.d, rec.normal) > 0) 
	{
		outward_normal = { -rec.normal.x, -rec.normal.y, -rec.normal.z };
		ni_over_nt = rec.ratio; //rec.ratio .ie ref_idx

		cosine = dot(r_in.d, rec.normal) / length(r_in.d);
		cosine = sqrtf(1.0f - rec.ratio*rec.ratio*(1.0f - cosine*cosine)); //rec.ratio .ie ref_idx
	}
	else 
	{
		outward_normal = rec.normal;
		ni_over_nt = 1.0f / rec.ratio; //rec.ratio .ie ref_idx
		cosine = -dot(r_in.d, rec.normal) / length(r_in.d);
	}
	if (d_refract(r_in.d, outward_normal, ni_over_nt, refracted))
		reflect_prob = d_schlick(cosine, rec.ratio); //rec.ratio .ie ref_idx
	else
		reflect_prob = 1.0;
	if (d_drand48(r_in.d.x * r_in.d.y * 100000000.0) < reflect_prob)
		scattered = { rec.p, reflected };
	else
		scattered = { rec.p, refracted };
	return true;
}

__device__ bool d_hit(const Ray& r, sphereCuda* pS, float t_min, float t_max, hit_recordCuda& rec)
{ 
	if (!pS) return false;

	float3 oc = r.o - pS->center;
	float a = dot(r.d, r.d);
	float b = dot(oc, r.d);
	float c = dot(oc, oc) - pS->radius*pS->radius;
	float discriminant = b*b - a*c;
	if (discriminant > 0)
	{
		float temp = (-b - sqrt(discriminant)) / a;
		if (temp < t_max && temp > t_min) 
		{
			rec.t = temp;
			rec.p = r.o + temp * r.d;
			rec.normal = (rec.p - pS->center) / pS->radius;
			rec.matType = pS->matType;
			rec.albedo = pS->albedo;
			rec.ratio = pS->ratio;
			return true;
		}
		temp = (-b + sqrt(discriminant)) / a;
		if (temp < t_max && temp > t_min) 
		{
			rec.t = temp;
			rec.p = r.o + temp * r.d;
			rec.normal = (rec.p - pS->center) / pS->radius;
			rec.matType = pS->matType;
			rec.albedo = pS->albedo;
			rec.ratio = pS->ratio;
			return true;
		}
	}
	return false;
}


__device__ bool d_hit_all(const Ray& r, sphereCuda* pSphereList, float t_min, float t_max, hit_recordCuda& rec)
{
	hit_recordCuda temp_rec;
	bool hit_anything = false;
	double closest_so_far = t_max;
	for (int i = 0; i < d_sphere_count[0]; i++)
	{
		if (d_hit(r, pSphereList + i, t_min, closest_so_far, temp_rec))
		{
			hit_anything = true;
			closest_so_far = temp_rec.t;
			rec = temp_rec;
		}
	}
	return hit_anything;
}

__device__ float3 d_color(Ray &r_in, sphereCuda* pSphereList, int& depth, bool& bContinue)
{
	depth++;
	hit_recordCuda rec;
	if (d_hit_all(r_in, pSphereList, 0.001f, FLT_MAX, rec))
	{
		Ray scattered;
		float3 attenuation;

		bool bscatter = false;
		switch (rec.matType)
		{
		case 0: // lambertian
			bscatter = d_scatter_lambertian(r_in, rec, attenuation, scattered);
			break;
		case 1: // metal
			bscatter = d_scatter_metal(r_in, rec, attenuation, scattered);
			break;
		case 2: // dielctric
			bscatter = d_scatter_dielectric(r_in, rec, attenuation, scattered);
			break;
		default:
			bscatter = d_scatter_lambertian(r_in, rec, attenuation, scattered);
			break;
		}
		r_in = scattered;
		if (depth < 50 && bscatter) 
		{
			bContinue = true;
			return attenuation;// *d_color(scattered, pSphereList, depth + 1);
		}
		else 
		{
			bContinue = false;
			float3 zero3 = { 0, 0, 0 };
			return zero3;
		}
	}
	else 
	{
		bContinue = false;
		float3 unit_direction = normalize(r_in.d);
		float t = 0.5*(unit_direction.y + 1.0);
		float3 white3 = { 1.0, 1.0, 1.0 };
		float3 blue3 = { 0.5, 0.7, 1.0 };
		return (1.0 - t) * white3 + t * blue3;
	}
}

__device__ Ray d_get_ray(float s, float t)
{
	float3 rd = d_camera[0].lens_radius * d_random_in_unit_disk(s * t * 10000000000.0);
	float3 offset = d_camera[0].u * rd.x + d_camera[0].v * rd.y;
	Ray ray =
	{
		d_camera[0].origin + offset,
		d_camera[0].lower_left_corner + s*d_camera[0].horizontal +
		t*d_camera[0].vertical - d_camera[0].origin - offset
	};
	return ray;
}

__global__ void d_ray_tracing(uchar *d_output, sphereCuda* pSphereList, uint nx, uint ny, uint ns)
{
	if (!d_output) return;

	uint x = blockIdx.x*blockDim.x + threadIdx.x;
	uint y = blockIdx.y*blockDim.y + threadIdx.y;

	if ((x >= nx) || (y >= ny)) return;

	float3 col = { 0.0f, 0.0f, 0.0f };
	
	for (int s = 0; s < ns; s++) 
	{
		float u = float(x + d_drand48(y * nx + x + s + 10)) / float(nx);
		float v = float(y + d_drand48(y * nx + x + s + 20)) / float(ny);
		Ray r = d_get_ray(u, v);

		int depth = 0;
		bool bCont = true;

		float3 rt = { 1.0f, 1.0f, 1.0f };
		while (bCont)
		{
			rt *= d_color(r, pSphereList, depth, bCont);
		}

		col += rt;//d_color(r, pSphereList, 0);
	}
	col /= float(ns);
	col = { sqrtf(col.x), sqrtf(col.y), sqrtf(col.z) };

	// test
	//col = normalize(r.o);

	//curandState devStates;
	//curand_init(y * nx + x, 0, 0, &devStates);
	//float RANDOM = curand_uniform(&devStates);// +0.2;

	//uchar ir = uchar(255.99*RANDOM);
	//uchar ig = uchar(255.99*RANDOM);
	//uchar ib = uchar(255.99*RANDOM);

	uchar ir = uchar(255.99*col.x);
	uchar ig = uchar(255.99*col.y);
	uchar ib = uchar(255.99*col.z);

	d_output[3 * (y * nx + x)] = ir;
	d_output[3 * (y * nx + x) + 1] = ig;
	d_output[3 * (y * nx + x) + 2] = ib;

}

extern "C" void initRayTracingCuda(camera *h_camera, sphere** pObjList, int spCount, int w, int h)
{
	if (!h_camera || !pObjList) return;

	cudaDeviceReset();

	cameraCuda* pCam = new cameraCuda;

	pCam->origin = {h_camera->origin.x(), h_camera->origin.y(), h_camera->origin.z()};
	pCam->lower_left_corner = { h_camera->lower_left_corner.x(), h_camera->lower_left_corner.y(), h_camera->lower_left_corner.z() };
	pCam->horizontal = { h_camera->horizontal.x(), h_camera->horizontal.y(), h_camera->horizontal.z() };
	pCam->vertical = { h_camera->vertical.x(), h_camera->vertical.y(), h_camera->vertical.z() };
	pCam->u = { h_camera->u.x(), h_camera->u.y(), h_camera->u.z() };
	pCam->v = { h_camera->v.x(), h_camera->v.y(), h_camera->v.z() };
	pCam->w = { h_camera->w.x(), h_camera->w.y(), h_camera->w.z() };
	pCam->lens_radius = h_camera->lens_radius;

	cudaError_t err = cudaGetLastError();

	err = cudaMemcpyToSymbol(d_sphere_count, &spCount, sizeof(uint));
	err = cudaMemcpyToSymbol(d_camera, pCam, sizeof(cameraCuda));

	cudaMalloc((void**)&d_output, sizeof(uchar) * w * h * 3);

	sphereCuda* pSphereHost = new sphereCuda[spCount];
	for (int i = 0; i < spCount; i++)
	{
		pSphereHost[i].center = { pObjList[i]->center.x(), pObjList[i]->center.y(), pObjList[i]->center.z()};
		pSphereHost[i].radius = pObjList[i]->radius;

		pSphereHost[i].matType = pObjList[i]->mat_ptr->matType;
		pSphereHost[i].albedo = { pObjList[i]->mat_ptr->albedo.x(), 
			pObjList[i]->mat_ptr->albedo.y(), pObjList[i]->mat_ptr ->albedo.z() };
		pSphereHost[i].ratio = pObjList[i]->mat_ptr->ratio;;
	}

	cudaMalloc((void**)&d_sphere_list, sizeof(sphereCuda) * spCount);
	cudaMemcpy(d_sphere_list, pSphereHost, sizeof(sphereCuda) * spCount, cudaMemcpyHostToDevice);
	

	delete pCam;
	pCam = NULL;
}

extern "C" void cleanCuda()
{
	cudaFree(d_output);
	cudaFree(d_sphere_list);
}

extern "C" void ray_tracing_kernel(dim3 gridSize, dim3 blockSize, uchar *d_output, uint nx, uint ny, uint ns)
{
	d_ray_tracing<< <gridSize, blockSize >> >(d_output, d_sphere_list, nx, ny, ns);
}

int iDivUp(int a, int b)
{
	return (a % b != 0) ? (a / b + 1) : (a / b);
}

extern "C" void DoRayTracing(int w, int h, int ns, unsigned char* pChar)
{
	if (!d_output || !pChar) return;

	dim3 blockSize(16, 16);
	dim3 gridSize = dim3(iDivUp(w, blockSize.x), iDivUp(h, blockSize.y));

	ray_tracing_kernel(gridSize, blockSize, d_output, w, h, ns);

	cudaMemcpy(pChar, d_output, sizeof(uchar) * w * h * 3, cudaMemcpyDeviceToHost);
}



#endif
