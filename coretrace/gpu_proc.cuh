#include "types.h"

__global__
void ray_kernel(double MinXSun,
                double MaxXSun,
                double MinYSun,
                double MaxYSun,
                double *d_PosSunStage,
                double *d_Sun_RLocToRef,
                double *d_Origin,
                double *d_Glob_RLocToRef,
                GlobalRay *d_IncomingRays,
                st_uint_t NumberOfRays);
