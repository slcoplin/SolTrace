
/*******************************************************************************************************
*  Copyright 2018 Alliance for Sustainable Energy, LLC
*
*  NOTICE: This software was developed at least in part by Alliance for Sustainable Energy, LLC
*  ("Alliance") under Contract No. DE-AC36-08GO28308 with the U.S. Department of Energy and the U.S.
*  The Government retains for itself and others acting on its behalf a nonexclusive, paid-up,
*  irrevocable worldwide license in the software to reproduce, prepare derivative works, distribute
*  copies to the public, perform publicly and display publicly, and to permit others to do so.
*
*  Redistribution and use in source and binary forms, with or without modification, are permitted
*  provided that the following conditions are met:
*
*  1. Redistributions of source code must retain the above copyright notice, the above government
*  rights notice, this list of conditions and the following disclaimer.
*
*  2. Redistributions in binary form must reproduce the above copyright notice, the above government
*  rights notice, this list of conditions and the following disclaimer in the documentation and/or
*  other materials provided with the distribution.
*
*  3. The entire corresponding source code of any redistribution, with or without modification, by a
*  research entity, including but not limited to any contracting manager/operator of a United States
*  National Laboratory, any institution of higher learning, and any non-profit organization, must be
*  made publicly available under this license for as long as the redistribution is made available by
*  the research entity.
*
*  4. Redistribution of this software, without modification, must refer to the software by the same
*  designation. Redistribution of a modified version of this software (i) may not refer to the modified
*  version by the same designation, or by any confusingly similar designation, and (ii) must refer to
*  the underlying software originally provided by Alliance as "SolTrace". Except to comply with the
*  foregoing, the term "SolTrace", or any confusingly similar designation may not be used to refer to
*  any modified version of this software or any modified version of the underlying software originally
*  provided by Alliance without the prior written consent of Alliance.
*
*  5. The name of the copyright holder, contributors, the United States Government, the United States
*  Department of Energy, or any of their employees may not be used to endorse or promote products
*  derived from this software without specific prior written permission.
*
*  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR
*  IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND
*  FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER,
*  CONTRIBUTORS, UNITED STATES GOVERNMENT OR UNITED STATES DEPARTMENT OF ENERGY, NOR ANY OF THEIR
*  EMPLOYEES, BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
*  DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
*  DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER
*  IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF
*  THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*******************************************************************************************************/


#include <math.h>

#include "types.h"
#include "procs.h"


// For cudaMalloc
#include <cuda_runtime.h>
// For host calls to curand_init which sets up rn
#include <curand.h>
// For device calls to curand_uniform. (0.0, 1.0]
// And defninition of curandState
#include <curand_kernel.h>

#include "generateray.cuh"

// Lecture slides http://cs.brown.edu/courses/cs195v/lecture/week11.pdf
// curand documentation https://docs.nvidia.com/cuda/curand/device-api-overview.html#device-api-overview
// TODO wrap calls?


#define RANGEN myrng

__global__
void ray_kernel(TSystem *System,
                GlobalRay *IncomingRays,
                st_uint_t NumberOfRays){

}

void generate_rays(TSystem *System,
                   GlobalRay *IncomingRays,
                   st_uint_t NumberOfRays){
  curandState *d_state;
  cudaMalloc(&d_state, nThreads * nBlocks);

  GlobalRay *d_IncomingRays;
  cudaMalloc(&d_IncomingRays, NumberOfRays * sizeof(GlobalRay));

  /*
  for (st_uint_t RayIndex = 0; RayIndex < NumberOfRays; RayIndex++) {
      double PosRaySun[3]; // Unused. Was for sun hash
      GenerateRay(myrng, System->Sun.PosSunStage, System->StageList[0]->Origin,
          System->StageList[0]->RLocToRef, &System->Sun,
          IncomingRays[RayIndex].Pos, IncomingRays[RayIndex].Cos, PosRaySun);
      System->SunRayCount++;
  }
  */
}

void GenerateRay(
            MTRand &myrng,
            double PosSunStage[3],
            double Origin[3],
            double RLocToRef[3][3],
            TSun *Sun,
            double PosRayGlobal[3],
            double CosRayGlobal[3],
            double PosRaySun[3]
            )
{
/*{This procedure generates a randomly located ray in the x-y plane of the sun coordinate system in
 the z direction of the sun coord. system, checks to see that the ray is within the region of interest
 defined by the spatial extent of the elements of Stage as seen from the sun
 and ultimately transforms that ray to the global coord. system.   The z-axis of the sun coord. system points
 towards the Stage coord. system origin.

 Input
       - Seed = Seed for random number generator
       - Sun = Sun data record of type TSun
       - Origin = Primary Stage origin
       - RLocToRef = transformation matrix from local to reference frame
 Output
       - PosRayGlobal = Position of ray in Global coordinate system
       - CosRayGlobal = Direction cosines of ray in Global coordinate system} */

    double XRaySun = 0.0, YRaySun = 0.0, ZRaySun = 0.0;
    double CosRaySun[3] = { 0.0, 0.0, 0.0 };
    double PosRayStage[3] = { 0.0, 0.0, 0.0 };
    double CosRayStage[3] = { 0.0, 0.0, 0.0 };
    PosRaySun[0] = 0.;
    PosRaySun[1] = 0.;
    PosRaySun[2] = 0.;

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

    XRaySun = Sun->MinXSun + (Sun->MaxXSun - Sun->MinXSun)*RANGEN();     //uses a rectangular region of interest about the primary
    YRaySun = Sun->MinYSun + (Sun->MaxYSun - Sun->MinYSun)*RANGEN();     //stage. Added 09/26/05


    //{Offload ray location and direction cosines into sun array}
    PosRaySun[0] = XRaySun;
    PosRaySun[1] = YRaySun;
    PosRaySun[2] = ZRaySun;
    CosRaySun[0] = 0.0;
    CosRaySun[1] = 0.0;
    CosRaySun[2] = 1.0;

    //{Transform ray locations and dir cosines into Stage system}
    TransformToReference(PosRaySun, CosRaySun, PosSunStage, Sun->RLocToRef, PosRayStage, CosRayStage);

    //{Transform ray locations and dir cosines into global system}
    TransformToReference(PosRayStage, CosRayStage, Origin, RLocToRef, PosRayGlobal, CosRayGlobal);

}
//End of Procedure--------------------------------------------------------------
