#include <fstream>
#include "sphere.h"
#include "float.h"
#include "camera.h"
#include "material.h"

extern "C" void initRayTracingCuda(camera *h_camera, sphere** pObjList, int spCount, int w, int h);
extern "C" void DoRayTracing(int w, int h, int ns, unsigned char*);

float drand48()
{
	return (rand() % (100) / (float)(100));
}

sphere **random_scene(int* num ) {
    int n = 500;
	sphere **list = new sphere*[n+1];
    list[0] =  new sphere(vec3(0,-1000,0), 1000, new lambertian(vec3(0.5, 0.5, 0.5)));
    int i = 1;
    for (int a = -11; a < 11; a++) {
        for (int b = -11; b < 11; b++) {
            float choose_mat = drand48();
            vec3 center(a+0.9*drand48(),0.2,b+0.9*drand48()); 
            if ((center-vec3(4,0.2,0)).length() > 0.9) { 
                if (choose_mat < 0.8) {  // diffuse
                    list[i++] = new sphere(center, 0.2, new lambertian(vec3(drand48()*drand48(), drand48()*drand48(), drand48()*drand48())));
                }
                else if (choose_mat < 0.95) { // metal
                    list[i++] = new sphere(center, 0.2,
                            new metal(vec3(0.5*(1 + drand48()), 0.5*(1 + drand48()), 0.5*(1 + drand48())),  0.5*drand48()));
                }
                else {  // glass
                    list[i++] = new sphere(center, 0.2, new dielectric(1.5));
                }
            }
        }
    }

    list[i++] = new sphere(vec3(0, 1, 0), 1.0, new dielectric(1.5));
    list[i++] = new sphere(vec3(-4, 1, 0), 1.0, new lambertian(vec3(0.4, 0.2, 0.1)));
    list[i++] = new sphere(vec3(4, 1, 0), 1.0, new metal(vec3(0.7, 0.6, 0.5), 0.0));

	*num = i;

    return list;
}

int main() {

    int nx = 800;
    int ny = 450;
    int ns = 100;

	std::ofstream outfile("mytest.ppm", std::ios_base::out);
	outfile << "P3" << std::endl;
	outfile << nx << " " << ny << std::endl;
	outfile << "255" << std::endl;

	int cou = 0;
	sphere **world = random_scene(&cou);

    vec3 lookfrom(13,2,3);
    vec3 lookat(0,0,0);
    float dist_to_focus = 10.0;
    float aperture = 0.1;

    camera cam(lookfrom, lookat, vec3(0,1,0), 20, float(nx)/float(ny), aperture, dist_to_focus);

	initRayTracingCuda(&cam, world, cou, nx, ny);

	unsigned char* pChar = new unsigned char[nx * ny * 3];
	memset(pChar,0, sizeof(unsigned char) * nx * ny * 3);

	DoRayTracing(nx, ny, ns, pChar);

	for (int j = ny-1; j >= 0; j--) 
	{
		for (int i = 0; i < nx; i++)
		{
			outfile << (int)pChar[3 * (j * nx + i)] << " " 
				<< (int)pChar[3 * (j * nx + i) + 1] << " "
				<< (int)pChar[3 * (j * nx + i) + 2] << std::endl;
		}
	}
	delete []pChar;
	pChar = NULL;

}



