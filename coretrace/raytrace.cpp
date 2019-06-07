
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



#include <vector>
#include <string>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <algorithm>
#include <ctime>

#include "types.h"
#include "procs.h"
#include "treemesh.h"

#include "generateray->cuh"
#include "check_intersection.cuh"

#define ZeroVec(x) x[0]=x[1]=x[2]=0.0

//structure to store element address and projected polar coordinate size
struct eprojdat
{
    TElement* el_addr;
    double d_proj;
    double az;
    double zen;

    eprojdat(){};
    eprojdat(TElement* e, double d, double a, double z)
    {
        el_addr = e;
        d_proj = d;
        az = a;
        zen = z;
    };
};

//Comparison function for sorting vector of eprojdat
static bool eprojdat_compare(const eprojdat &A, const eprojdat &B)
{
    return A.d_proj > B.d_proj;
};

bool check_input(st_uint_t NumberOfRays, TSystem *System){
    if (NumberOfRays < 1)
    {
        System->errlog("invalid number of rays: %d", NumberOfRays);
        return false;
    }

    if (System->StageList.size() < 1)
    {
        System->errlog("no stages defined.");
        return false;
    }

    return true;
}

bool Trace(TSystem *System, unsigned int seed,
           st_uint_t NumberOfRays,
           st_uint_t MaxNumberOfRays,
           bool IncludeSunShape,
           bool IncludeErrors,
           bool AsPowerTower,
           int (*callback)(st_uint_t ntracedtotal, st_uint_t ntraced, st_uint_t ntotrace, st_uint_t curstage, st_uint_t nstages, void *data),
           void *cbdata,
           std::vector< std::vector< double > > *st0data,
           std::vector< std::vector< double > > *st1in,
           bool save_st_data) // FALSE, st0data and st1in are null.
{
    ZeroVec(System->Sun.PosSunStage);

    //bool aspowertower_ok = false;

    try
    {
        TOpticalProperties *optics=NULL;

        int k = 0;
        TElement *optelm = 0;

        TStage *Stage;

        System->SunRayCount = 0;
        MTRand myrng(seed);

        assert(check_input(NumberOfRays, System));

        if (!SunToPrimaryStage(System, System->StageList[0], &System->Sun, System->Sun.PosSunStage)){
            return false;
        }

        // Rays that get passed stage to stage
        GlobalRay *IncomingRays = (GlobalRay *) malloc(NumberOfRays * sizeof(GlobalRay));
        assert(IncomingRays);

        generate_rays(System, IncomingRays, NumberOfRays);


        for (st_uint_t cur_stage_i=0;cur_stage_i<System->StageList.size();cur_stage_i++)
        {

            Stage = System->StageList[cur_stage_i];

            // List of Rays that will have their information filled to complete
            // the tracing loop.
            // Starts all variables at 0.
            Ray *AllRays = (Ray *) calloc(NumberOfRays, sizeof(Ray));
            assert(AllRays);

            // loop through rays within each stage
            for (st_uint_t RayIndex = 0; RayIndex < NumberOfRays; RayIndex++) {

                // Initialize ray variables
                Ray *ray = &AllRays[RayIndex]; // Same as AllRays + RayIndex

                // Load the ray
                // TODO: Add handler for i > 0 stage : DONE
                CopyVec3( ray->PosRayGlob, IncomingRays[RayIndex].Pos );
                CopyVec3( ray->CosRayGlob, IncomingRays[RayIndex].Cos );

                // transform the global incoming ray to local stage coordinates
                TransformToLocal(ray->PosRayGlob, ray->CosRayGlob,
                    Stage->Origin, Stage->RRefToLoc,
                    ray->PosRayStage, ray->CosRayStage);


                // Start ray tracing

                // Getting list of elements to check for intersection


                // Find number of elements to check intersections with, and set element_list
                st_uint_t nintelements = Stage->ElementList.size();
                std::vector<TElement*> element_list = Stage->ElementList;

                // Check for ray intersections
                check_intersection_in_stage(element_list, nintelements, ray);

        // If the ray hits something, handle it
                if (ray->StageHit)
                {
                    // time for optics

                    // {Otherwise trace ray through interaction}
                    // {Determine if backside or frontside properties should be used}

                    // trace through the interaction
                    optelm = Stage->ElementList[ ray->LastElementNumber - 1 ];
                    optics = 0;

                    if (ray->LastHitBackSide)
                        optics = &optelm->Optics->Back;
                    else
                        optics = &optelm->Optics->Front;

                    // Does the interaction with the element collided with, and
                    // converts the ray into the global reference frame

                    k = ray->LastElementNumber - 1;

                    // Do the ray interaction (reflect, etc)
                    if (IncludeSunShape && cur_stage_i == 0)
                    {
                        // change to account for first hit only in primary stage 8-11-31. (Only does first hit, not future ones)
                        // Apply sunshape to UNPERTURBED ray at intersection point
                        //only apply sunshape error once for primary stage

                        double CosIn[3];
                        double CosOut[3] = { 0.0, 0.0, 0.0 };

                        CopyVec3(CosIn, ray->LastCosRaySurfElement);
                        Errors(myrng, CosIn, 1, &System->Sun,
                               Stage->ElementList[k], optics, CosOut, ray->LastDFXYZ);  //sun shape
                        CopyVec3(ray->LastCosRaySurfElement, CosOut);
                    }

                    //{Determine interaction at surface and direction of perturbed ray}
                    int ErrorFlag = 0;
                    double PosRayOutElement[3] = { 0.0, 0.0, 0.0 };
                    double CosRayOutElement[3] = { 0.0, 0.0, 0.0 };
                    Interaction( myrng, ray->LastPosRaySurfElement, ray->LastCosRaySurfElement, ray->LastDFXYZ,
                        Stage->ElementList[k]->InteractionType, optics, 630.0,
                        PosRayOutElement, CosRayOutElement, &ErrorFlag);

                    // { Transform ray back to stage coord system and trace through stage again}

                    TransformToReference(PosRayOutElement, CosRayOutElement,
                            Stage->ElementList[k]->Origin, Stage->ElementList[k]->RLocToRef,
                            ray->PosRayStage, ray->CosRayStage);
                    TransformToReference(ray->PosRayStage, ray->CosRayStage,
                            Stage->Origin, Stage->RLocToRef,
                            ray->PosRayGlob, ray->CosRayGlob);
                }
                else { // No collision happened
                    // This might be wrong in the stage 0 case, because of the way we no longer
                    // regenerate rays if they don't hit something in stage 0
                    CopyVec3(ray->LastPosRaySurfStage, ray->PosRayStage);
                    CopyVec3(ray->LastCosRaySurfStage, ray->CosRayStage);
                }

                // append ray data
                // Records the point of collision
                // If we allow multiple collisions, I'm not sure what will happen
                TRayData::ray_t *p_ray = 0;
                p_ray = Stage->RayData.Append( ray->LastPosRaySurfStage,
                                      ray->LastCosRaySurfStage,
                                      ray->LastElementNumber,
                                      cur_stage_i+1,
                                      RayIndex );
                if (!p_ray)
                {
                    System->errlog("Failed to save ray data at index %d", Stage->RayData.Count()-1);
                    return false;
                }

                // Save ray data
                CopyVec3(IncomingRays[RayIndex].Pos, ray->PosRayGlob);
                CopyVec3(IncomingRays[RayIndex].Cos, ray->CosRayGlob);
            }

        }

        return true;
    }
    catch( const std::exception &e )
    {
        System->errlog("trace error: %s", e.what());
        return false;
    }
}
