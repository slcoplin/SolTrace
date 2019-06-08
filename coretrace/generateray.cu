
#include <math.h>

#include "types.h"

#include "gpu_proc.cuh"

// For cudaMalloc
#include <cuda_runtime.h>

// For host calls to curand_init which sets up rn
#include <curand.h>
// For device calls to curand_uniform. (0.0, 1.0]
// And defninition of curandState
#include <curand_kernel.h>

#define SEED 179

// Lecture slides http://cs.brown.edu/courses/cs195v/lecture/week11.pdf
// curand documentation https://docs.nvidia.com/cuda/curand/device-api-overview.html#device-api-overview
// TODO wrap calls?


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
    st_uint_t NumberOfRays) {

    uint thread_id = threadIdx.x + (blockIdx.x * blockDim.x);

    curandState rand_state;
    // Create random number generator
    curand_init(SEED, thread_id, 0, &rand_state);

    // Make sure sufficient rays are traced
    // Update the thread index to the stide of the number of threads
    uint stride = gridDim.x * blockDim.x;
    for (uint ray_id = thread_id; ray_id < NumberOfRays; ray_id += stride) {
        double XRaySun, YRaySun, ZRaySun;

        //ZRaySun := 0.0;  //Origin of rays in xy plane of sun coord system.
        ZRaySun = -10000.0;  //changed 5/1/00.  rays originate from well bebind the sun coordinate system xy
                                // plane which has been translated to primary stage origin.         This value has been reduced signficantly because of numerical issues in tracing rays from sun
                                // to the closer form solution for a cylinder.  It used to 1e6 and has been reduced to 1e4, which should still be sufficient.   10-26-09 Wendelin

      //{Generate random rays inside of region of interest}

        //following changed on 09/26/05 to more efficiently generate rays relative to element center of mass in primary stage
        /*{XRaySun := 2.0*MaxRad*ran3(Seed) - MaxRad;  //ran3 produces results independent of platform.
        YRaySun := 2.0*MaxRad*ran3(Seed) - MaxRad;
        if (XRaySun*XRaySun + YRaySun*YRaySun) > MaxRad*MaxRad then goto GENRAY;
        XRaySun := Xcm + XRaySun;  //adjust location of generated rays about element center of mass
        YRaySun := Ycm + YRaySun;}*/

        XRaySun = MinXSun + (MaxXSun - MinXSun) * curand_uniform_double(&rand_state);     //uses a rectangular region of interest about the primary
        YRaySun = MinYSun + (MaxYSun - MinYSun) * curand_uniform_double(&rand_state);     //stage. Added 09/26/05



        //{Offload ray location and direction cosines into sun array}
        double PosRaySun[3];
        PosRaySun[0] = XRaySun;
        PosRaySun[1] = YRaySun;
        PosRaySun[2] = ZRaySun;

        double CosRaySun[3];
        CosRaySun[0] = 0.0;
        CosRaySun[1] = 0.0;
        CosRaySun[2] = 1.0;

        double PosRayStage[3];
        double CosRayStage[3];

        //{Transform ray locations and dir cosines into Stage system}
        TransformToReferenceGPU(PosRaySun, CosRaySun, d_PosSunStage, d_Sun_RLocToRef, PosRayStage, CosRayStage);

        //{Transform ray locations and dir cosines into global system}
        // Puts the result into IncomingRays
        TransformToReferenceGPU(PosRayStage, CosRayStage, d_Origin, d_Glob_RLocToRef, d_IncomingRays[thread_id].Pos, d_IncomingRays[thread_id].Cos);
    }
}


void generate_rays(TSystem *System,
                   GlobalRay *IncomingRays,
                   st_uint_t NumberOfRays){

  // Space for output
  GlobalRay *d_IncomingRays;
  gpuErrChk(cudaMalloc(&d_IncomingRays, NumberOfRays * sizeof(GlobalRay)));

  // Transfer Sun info into device memory
  double *d_PosSunStage;
  double *d_Origin;
  gpuErrChk(cudaMalloc(&d_PosSunStage, 3 * sizeof(double)));
  gpuErrChk(cudaMalloc(&d_Origin, 3 * sizeof(double)));
  gpuErrChk(cudaMemcpy(d_PosSunStage, System->Sun.PosSunStage, 3 * sizeof(double), cudaMemcpyHostToDevice));
  gpuErrChk(cudaMemcpy(d_Origin, System->StageList[0]->Origin, 3 * sizeof(double), cudaMemcpyHostToDevice));

  // Transfer transform matrices
  double *d_Glob_RLocToRef;
  gpuErrChk(cudaMalloc(&d_Glob_RLocToRef, 9 * sizeof(double *)));
  for (uint i = 0; i < 3; i++) {
      gpuErrChk(cudaMemcpy(d_Glob_RLocToRef + 3 * i, System->StageList[0]->RLocToRef[i], 3 * sizeof(double), cudaMemcpyHostToDevice));
  }
  double *d_Sun_RLocToRef;
  gpuErrChk(cudaMalloc(&d_Sun_RLocToRef, 9 * sizeof(double *)));
  for (uint i = 0; i < 3; i++) {
      gpuErrChk(cudaMemcpy(d_Sun_RLocToRef + 3 * i, System->Sun.RLocToRef[i], 3 * sizeof(double), cudaMemcpyHostToDevice));
  }

  // Call ray generating kernel
  uint blocks = 512;
  uint threads_per_block = 512;
  ray_kernel<<<blocks, threads_per_block >>>
      (System->Sun.MinXSun, System->Sun.MaxXSun,
      System->Sun.MinYSun, System->Sun.MaxYSun,
      d_PosSunStage, d_Sun_RLocToRef, d_Origin, d_Glob_RLocToRef,
      d_IncomingRays, NumberOfRays
      );
  gpuErrChk(cudaPeekAtLastError());

  // Copy output into IncomingRays
  gpuErrChk(cudaMemcpy(IncomingRays, d_IncomingRays, NumberOfRays * sizeof(GlobalRay), cudaMemcpyDeviceToHost));

  System->SunRayCount = NumberOfRays;
}
