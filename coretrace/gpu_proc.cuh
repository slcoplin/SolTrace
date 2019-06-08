#include "types.h"

#define gpuErrChk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true)
{
    if (code != cudaSuccess) {
        fprintf(stderr, "GPUassert: %s %s %d\n",
            cudaGetErrorString(code), file, line);
        exit(code);
    }
}

/* Defined in gpu_mathproc.cu */
__device__
void MatrixVectorMultGPU(double M[9], double V[3], double MxV[3]);

__device__
void TransformToLocalGPU(double PosRef[3], double CosRef[3], double Origin[3],
    double RRefToLoc[9],
    double PosLoc[3], double CosLoc[3]);

__device__
void TransformToReferenceGPU(double PosLoc[3], double CosLoc[3], double Origin[3],
    double RLocToRef[9],
    double PosRef[3], double CosRef[3]);

__device__
inline void CopyVec3GPU( double dest[3], double src[3] )
{
    dest[0] = src[0];
    dest[1] = src[1];
    dest[2] = src[2];
}

__device__
double DOT_GPU(double A[3], double B[3]);


/////
/* The following are for intersecitons and
 * are defined in their respective files */
///


__device__
void Surface(
            double PosXYZ[3],
            ElementInfo *Element,
            double *FXYZ,
            double DFXYZ[3],
            int *ErrorFlag );
            
__device__
int intri(double x1, double y1,
                 double x2, double y2,
                 double x3, double y3,
                 double xt, double yt);

__device__
int inquad(double x1, double y1,
                 double x2, double y2,
                 double x3, double y3,
                 double x4, double y4,
                 double xt, double yt);

__device__
void Intersect( double PosLoc[3],
            double CosLoc[3],
            ElementInfo *Element,
            double PosXYZ[3],
            double CosKLM[3],
            double DFXYZ[3],
            double *PathLength,
            int *ErrorFlag );

__device__
void SurfaceZatXYPair(
            double PosXYZ[3],
            ElementInfo *Element,
            double *FXYZ,
            int *ErrorFlag );

__device__
void QuadricSurfaceClosedForm(
            ElementInfo *Element,
            double PosLoc[3],
            double CosLoc[3],
            double PosXYZ[3],
            double DFXYZ[3],
            double *PathLength,
            int *ErrorFlag);

__device__
void DetermineElementIntersectionNew(
            ElementInfo *Element,
            double PosRayIn[3],
            double CosRayIn[3],
            double PosRayOut[3],
            double CosRayOut[3],
            double DFXYZ[3],
            double *PathLength,
            int *ErrorFlag,
            int *Intercept,
            int *BacksideFlag );
