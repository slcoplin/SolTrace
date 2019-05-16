
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
//#define WITH_DEBUG_TIMER
#ifdef WITH_DEBUG_TIMER
    #include <chrono>    //comment out for production
#endif

#include "types.h"
#include "procs.h"
#include "treemesh.h"


void time(const char *message, ofstream *fout)
{
    (*fout) << message << chrono::duration_cast< chrono::milliseconds >( chrono::system_clock::now().time_since_epoch() ).count() << "\n";

}

inline void CopyVec3( double dest[3], const std::vector<double> &src )
{
    dest[0] = src[0];
    dest[1] = src[1];
    dest[2] = src[2];
}

inline void CopyVec3( std::vector<double> &dest, double src[3] )
{
    dest[0] = src[0];
    dest[1] = src[1];
    dest[2] = src[2];
}

inline void CopyVec3( double dest[3], double src[3] )
{
    dest[0] = src[0];
    dest[1] = src[1];
    dest[2] = src[2];
}

#define ZeroVec(x) x[0]=x[1]=x[2]=0.0

class GlobalRay
{
public:
    GlobalRay() {
        Num = 0;
        for (int i=0;i<3;i++) Pos[i]=Cos[i]=0.0;
    }

    double Pos[3];
    double Cos[3];
    st_uint_t Num;
};

struct Ray
{
    double PosRayOutElement[3] = { 0.0, 0.0, 0.0 };
    double CosRayOutElement[3] = { 0.0, 0.0, 0.0 };
    double CosIn[3] = { 0.0, 0.0, 0.0 };
    double CosOut[3] = { 0.0, 0.0, 0.0 };
    double PosRayGlob[3] = { 0.0, 0.0, 0.0 };
    double CosRayGlob[3] = { 0.0, 0.0, 0.0 };
    double PosRayStage[3] = { 0.0, 0.0, 0.0 };
    double CosRayStage[3] = { 0.0, 0.0, 0.0 };
    double PosRayElement[3] = { 0.0, 0.0, 0.0 };
    double CosRayElement[3] = { 0.0, 0.0, 0.0 };
    double PosRaySurfElement[3] = { 0.0, 0.0, 0.0 };
    double CosRaySurfElement[3] = { 0.0, 0.0, 0.0 };
    double LastPosRaySurfElement[3] = { 0.0, 0.0, 0.0 };
    double LastCosRaySurfElement[3] = { 0.0, 0.0, 0.0 };
    double LastPosRaySurfStage[3] = { 0.0, 0.0, 0.0 }; // Position the ray collides with an element, in the reference frame of the stage
    double LastCosRaySurfStage[3] = { 0.0, 0.0, 0.0 };
    double PosRaySurfStage[3] = { 0.0, 0.0, 0.0 };
    double CosRaySurfStage[3] = { 0.0, 0.0, 0.0 };
    double DFXYZ[3] = { 0.0, 0.0, 0.0 };
    double LastDFXYZ[3] = { 0.0, 0.0, 0.0 };
    int ErrorFlag = 0, InterceptFlag = 0, HitBackSide = 0, LastHitBackSide = 0;
    double PathLength = 0.0;
    double LastPathLength = 0.0;
    bool StageHit = false;

    // TODO: This is only used for the first loop through the first stage so it might be a waste of space
    double PosRaySun[3] = { 0.0, 0.0, 0.0 };
};

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

/*
 * Finds the elements (and # of elements) corresponding to the current stage
 * that the ray might hit.
 *
 * Inputs: Variables as defined in Trace.
 *
 * Modified inputs: Output: nintelements and element_list_ptr
 *
 */
void get_elements_in_stage(TStage *Stage,
						   std::vector<TElement*> *&element_list_ptr,
						   st_uint_t &nintelements
						   ){
	// Check all elements
	nintelements = Stage->ElementList.size();

	// Set element_list to Stage->ElementList
	element_list_ptr = &((std::vector<TElement*>) Stage->ElementList);
}


/*
 * Check the ray for intersections with all elements in element_list_ptr.
 * Modifies ray, LastElementNumber.
 *
 * Inputs: Variables as defined in Trace.
 *
 */
void check_intersection_in_stage(std::vector<TElement*> *element_list_ptr,
		   	   	   	             st_uint_t nintelements,
								 st_uint_t cur_stage_i,
							     Ray &ray,
								 st_uint_t &LastElementNumber){
	for (st_uint_t j = 0; j < nintelements; j++)
	{
		TElement *Element;
		Element = (*element_list_ptr)[j];

		if (!Element->Enabled)
			continue;

		//  {Transform ray to element[j] coord system of Stage[i]}
		TransformToLocal(ray.PosRayStage, ray.CosRayStage,
			Element->Origin, Element->RRefToLoc,
			ray.PosRayElement, ray.CosRayElement);

		ray.ErrorFlag = 0;
		ray.HitBackSide = 0;
		ray.InterceptFlag = 0;

		// increment position by tiny amount to get off the element if tracing to the same element
		for (int i = 0; i < 3; i++) {
			ray.PosRayElement[i] = ray.PosRayElement[i] + 1.0e-5*ray.CosRayElement[i];
		}

		// {Determine if ray intersects element[j]; if so, Find intersection point with surface of element[j] }
		DetermineElementIntersectionNew(Element, ray.PosRayElement, ray.CosRayElement,
			ray.PosRaySurfElement, ray.CosRaySurfElement, ray.DFXYZ,
			&ray.PathLength, &ray.ErrorFlag, &ray.InterceptFlag, &ray.HitBackSide);



		if (ray.InterceptFlag)
		{
			//{If hit multiple elements, this loop determines which one hit first.
			//Also makes sure that correct part of closed surface is hit. Also, handles wavy, but close to flat zernikes and polynomials correctly.}
			//if (PathLength < LastPathLength) and (PosRaySurfElement[2] <= Element->ZAperture) then
			if (ray.PathLength < ray.LastPathLength)
			{
				if (ray.PosRaySurfElement[2] <= Element->ZAperture
					|| Element->SurfaceIndex == 'm'
					|| Element->SurfaceIndex == 'M'
					|| Element->SurfaceIndex == 'r'
					|| Element->SurfaceIndex == 'R')
				{
					ray.StageHit = true;
					ray.LastPathLength = ray.PathLength;
					CopyVec3(ray.LastPosRaySurfElement, ray.PosRaySurfElement);
					CopyVec3(ray.LastCosRaySurfElement, ray.CosRaySurfElement);
					CopyVec3(ray.LastDFXYZ, ray.DFXYZ);
					LastElementNumber = j + 1;    //mjw change from j index to element id
					TransformToReference(ray.PosRaySurfElement, ray.CosRaySurfElement,
						Element->Origin, Element->RLocToRef,
						ray.PosRaySurfStage, ray.CosRaySurfStage);

					CopyVec3(ray.LastPosRaySurfStage, ray.PosRaySurfStage);
					CopyVec3(ray.LastCosRaySurfStage, ray.CosRaySurfStage);
					ray.LastHitBackSide = ray.HitBackSide;
				}
			}
		}
	}
}

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
    st_uint_t LastElementNumber = 0;

	ZeroVec(System->Sun.PosSunStage)

    // Initialize ray variables
    Ray ray;

    std::vector<GlobalRay> IncomingRays;
    st_uint_t StageDataArrayIndex=0;

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

        try
        {
            IncomingRays.resize( NumberOfRays );
        } catch (std::exception &e) {
            System->errlog("Incoming rays resize exception: %d, '%s'", NumberOfRays, e.what());
            return false;
        }


        if (!SunToPrimaryStage(System, System->StageList[0], &System->Sun, System->Sun.PosSunStage)){
            return false;
        }

#ifdef WITH_DEBUG_TIMER
        ofstream fout("C:\\Users\\mwagner\\Documents\\NREL\\Dev\\SolTraceWX\\log.txt");
        fout.clear();
#else
        ofstream fout;

        time("Starting stage calculations:\t", &fout);

        fout.close();
#endif

        for (st_uint_t cur_stage_i=0;cur_stage_i<System->StageList.size();cur_stage_i++)
        {

            Stage = System->StageList[cur_stage_i];

            LastElementNumber = 0;
            ray.LastHitBackSide = 0;

            StageDataArrayIndex = 0;

			// PB: good place for adding loop. IncomingRays already has a vector of size NumberOfRays
			// IncomingRays may have elements from previous stage 1 data
			// but our testing case, no rays has been traced yet

			// loop through rays within each stage
			for (int PreviousStageDataArrayIndex = 0; PreviousStageDataArrayIndex < NumberOfRays; PreviousStageDataArrayIndex++) {

				// Load the ray and trace it.
				// First stage. Generate a ray.
				if (cur_stage_i == 0)
				{

					// we are in the first stage, so
					// generate a new sun ray in global coords
					GenerateRay(myrng, System->Sun.PosSunStage, Stage->Origin,
						Stage->RLocToRef, &System->Sun,
						ray.PosRayGlob, ray.CosRayGlob, ray.PosRaySun);
					System->SunRayCount++;


					if (System->SunRayCount > MaxNumberOfRays)
					{
						System->errlog("generated sun rays reached maximum count: %d", MaxNumberOfRays);
						return false;
					}

				}
				// TODO: Add handler for i > 0 stage : DONE
				// Other stages. Load the ray from the previous stage.
				else
				{
					// we are in a subsequent stage, so trace using an incoming ray
					// saved from the previous stages
					CopyVec3( ray.PosRayGlob, IncomingRays[StageDataArrayIndex].Pos );
					CopyVec3( ray.CosRayGlob, IncomingRays[StageDataArrayIndex].Cos );
					StageDataArrayIndex++;

				}

				// transform the global incoming ray to local stage coordinates
				TransformToLocal(ray.PosRayGlob, ray.CosRayGlob,
					Stage->Origin, Stage->RRefToLoc,
					ray.PosRayStage, ray.CosRayStage);


				// Start ray tracing

				// Getting list of elements to check for intersection

				ray.LastPathLength = 1e99;
				ray.StageHit = false;

				// Find number of elements to check intersections with, and set element_list
				st_uint_t nintelements;
                std::vector<TElement*> *element_list_ptr;
				get_elements_in_stage(Stage, element_list_ptr, nintelements);


				// Check for ray intersections
				check_intersection_in_stage(element_list_ptr, nintelements,
					cur_stage_i,
					ray,
					LastElementNumber);

                // If the ray hits something, handle that case
				if (ray.StageHit)
				{
    				// now all the rays have been traced
    				// time for optics

    				// {Otherwise trace ray through interaction}
    				// {Determine if backside or frontside properties should be used}

    				// trace through the interaction
    				optelm = Stage->ElementList[ LastElementNumber - 1 ];
    				optics = 0;

    				if (ray.LastHitBackSide)
    					optics = &optelm->Optics->Back;
    				else
    					optics = &optelm->Optics->Front;

                    // Does the interaction with the element collided with, and
                    // converts the ray into the global reference frame

    				k = abs( LastElementNumber ) - 1;

    				// Do the ray interaction (reflect, etc)
    				if (IncludeSunShape && cur_stage_i == 0)
    				{
                        // change to account for first hit only in primary stage 8-11-31. (Only does first hit, not future ones)
    					// Apply sunshape to UNPERTURBED ray at intersection point
    					//only apply sunshape error once for primary stage
    					CopyVec3(ray.CosIn, ray.LastCosRaySurfElement);
    					Errors(myrng, ray.CosIn, 1, &System->Sun,
    						   Stage->ElementList[k], optics, ray.CosOut, ray.LastDFXYZ);  //sun shape
    					CopyVec3(ray.LastCosRaySurfElement, ray.CosOut);
    				}

    				//{Determine interaction at surface and direction of perturbed ray}
    				ray.ErrorFlag = 0;

    				Interaction( myrng, ray.LastPosRaySurfElement, ray.LastCosRaySurfElement, ray.LastDFXYZ,
    					Stage->ElementList[k]->InteractionType, optics, 630.0,
    					ray.PosRayOutElement, ray.CosRayOutElement, &ray.ErrorFlag);

    				// { Transform ray back to stage coord system and trace through stage again}
    				TransformToReference(ray.PosRayOutElement, ray.CosRayOutElement,
    						Stage->ElementList[k]->Origin, Stage->ElementList[k]->RLocToRef,
    						ray.PosRayStage, ray.CosRayStage);
    				TransformToReference(ray.PosRayStage, ray.CosRayStage,
    						Stage->Origin, Stage->RLocToRef,
    						ray.PosRayGlob, ray.CosRayGlob);
                }
                else { // No collision happened
                    // This might be wrong in the stage 0 case, because of the way we no longer
                    // regenerate rays if they don't hit something in stage 0
                    CopyVec3(ray.LastPosRaySurfStage, ray.PosRayStage);
                    CopyVec3(ray.LastCosRaySurfStage, ray.CosRayStage);
                }

                // append ray data
                // Records the point of collision
                // If we allow multiple collisions, I'm not sure what will happen
                TRayData::ray_t *p_ray = 0;
                p_ray = Stage->RayData.Append( ray.LastPosRaySurfStage,
                                      ray.LastCosRaySurfStage,
                                      LastElementNumber,
                                      cur_stage_i+1,
                                      PreviousStageDataArrayIndex );
                if (!p_ray)
                {
                    System->errlog("Failed to save ray data at index %d", Stage->RayData.Count()-1);
                    return false;
                }

                // Save ray data
                CopyVec3(IncomingRays[PreviousStageDataArrayIndex].Pos, ray.PosRayGlob);
                CopyVec3(IncomingRays[PreviousStageDataArrayIndex].Cos, ray.CosRayGlob);
                IncomingRays[PreviousStageDataArrayIndex].Num = PreviousStageDataArrayIndex + 1;
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
