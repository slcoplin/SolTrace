
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
#ifdef WITH_DEBUG_TIMER
    (*fout) << message << chrono::duration_cast< chrono::milliseconds >( chrono::system_clock::now().time_since_epoch() ).count() << "\n";
#endif
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
    double LastPosRaySurfStage[3] = { 0.0, 0.0, 0.0 };
    double LastCosRaySurfStage[3] = { 0.0, 0.0, 0.0 };
    double PosRaySurfStage[3] = { 0.0, 0.0, 0.0 };
    double CosRaySurfStage[3] = { 0.0, 0.0, 0.0 };
    double DFXYZ[3] = { 0.0, 0.0, 0.0 };
    double LastDFXYZ[3] = { 0.0, 0.0, 0.0 };
    double IncidentAngle = 0.0;
    double UnitLastDFXYZ[3] = { 0.0, 0.0, 0.0 };
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
 * Modified inputs: Output: nintelements and element_list_ptr,
 * 				    Sometimes modifies sunint_elements, reflint_elements.
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
 * Modifies ray, LastElementNumber, LastRayNumber.
 *
 * Inputs: Variables as defined in Trace.
 *
 */
void check_intersection_in_stage(std::vector<TElement*> *element_list_ptr,
		   	   	   	             st_uint_t nintelements,
								 st_uint_t cur_stage_i,
							     Ray &ray,
								 st_uint_t &LastElementNumber,
								 st_uint_t &LastRayNumber,
								 st_uint_t RayNumber){
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
					// TODO: The lastRayNumber=RayNumber might be reduntant, and is the only place these variables
					// are used in this fuction
					LastRayNumber = RayNumber;
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




/*
 * Does the end of stage wrap-up of rays.
 * Warn: Changes LastRayNumberInPreviousStage
 *
 * Input: Variables as defined in Trace.
 *        LastRayNumberInPreviousStage is a reference to LastRayNumberInPreviousStage and will be modified.
 *
 * Return: Boolean: false - error; true - ended as expected
 *
 */
bool end_stage(TSystem *System,
			   std::vector< std::vector< double > > *st0data,
		       std::vector< std::vector< double > > *st1in,
			   bool save_st_data,
		       TStage *Stage,
			   st_uint_t cur_stage_i,
			   std::vector<GlobalRay> &IncomingRays,
			   st_uint_t StageDataArrayIndex,
			   bool PreviousStageHasRays,
			   st_uint_t PreviousStageDataArrayIndex,
			   st_uint_t &LastRayNumberInPreviousStage
			   ){
	if(cur_stage_i==0 && save_st_data)
	{
		//if flagged save the stage 0 incoming rays data
		TRayData *raydat = &Stage->RayData;
		st_uint_t nray0 = raydat->Count();

		for(st_uint_t ii=0; ii<nray0; ii++)
		{
			TRayData::ray_t *rr = raydat->Index(ii,false);

			std::vector<double> ray(8);
			for(int j=0; j<3; j++)
				ray[j] = rr->pos[j];
			for(int j=0; j<3; j++)
				ray[j+3] = rr->cos[j];
			ray[6] = rr->element;
			ray[7] = rr->raynum;
			st0data->push_back(ray);
		}
	}

	if(cur_stage_i==1 && save_st_data)
	{
		//if flagged, save the stage 1 incoming rays data to the data structure passed into the algorithm
		for(int ir=0; ir<StageDataArrayIndex; ir++)
		{
			st1in->push_back(std::vector<double>(7));
			for(int jr=0; jr<3; jr++)
			{
				st1in->back().at(jr) = IncomingRays[ir].Pos[jr];
				st1in->back().at(jr+3) = IncomingRays[ir].Cos[jr];
			}
			st1in->back().at(6) = IncomingRays[ir].Num;
		}
	}


	if (!PreviousStageHasRays)
	{
		// no rays to carry forward
		LastRayNumberInPreviousStage = 0;
	}
	else if (PreviousStageDataArrayIndex < IncomingRays.size())
	{
		LastRayNumberInPreviousStage = IncomingRays[PreviousStageDataArrayIndex].Num;
		if (LastRayNumberInPreviousStage == 0)
		{
			size_t pp = IncomingRays[PreviousStageDataArrayIndex-1].Num;
			System->errlog("LastRayNumberInPreviousStage=0, stage %d, PrevIdx=%d, CurIdx=%d, pp=%d", cur_stage_i+1,
								PreviousStageDataArrayIndex, StageDataArrayIndex, pp);
			return false;
		}
	}
	else
	{
		System->errlog("Invalid PreviousStageDataArrayIndex: %u, @ stage %d",
					   PreviousStageDataArrayIndex, cur_stage_i+1);
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
           bool save_st_data)
{
    bool load_st_data = st0data != 0 && st1in != 0;
    if(load_st_data)
    {
        load_st_data = st0data->size() > 0 && st1in->size() > 0;
    }

    st_uint_t LastElementNumber = 0, LastRayNumber = 0;
    st_uint_t MultipleHitCount = 0;


	for (int i = 0; i < 3; i++) {
		System->Sun.PosSunStage[i] = 0.0;
	}

    // Initialize ray variables
    Ray ray;

    std::vector<GlobalRay> IncomingRays;
    st_uint_t StageDataArrayIndex=0;
    bool PreviousStageHasRays = false;
    st_uint_t PreviousStageDataArrayIndex = 0;
    st_uint_t LastRayNumberInPreviousStage = NumberOfRays;

    ZeroVec( ray.LastPosRaySurfStage );
    ZeroVec( ray.LastCosRaySurfStage );

    //bool aspowertower_ok = false;
    bool in_multi_hit_loop = false;


    try
    {
        TOpticalProperties *optics=NULL;
        PreviousStageHasRays = false;

        int k = 0;
        TElement *optelm = 0;
        TRayData::ray_t *p_ray = 0;
        TStage *Stage;

        System->SunRayCount=0;
        st_uint_t RayNumber = 1;
        MTRand myrng(seed);

        if( load_st_data && st0data->size() < 1)
        {
            System->errlog("empty stage 0 data array provided to Trace()");
            return false;
        }
        if( load_st_data && st1in->size() < 1 )
        {
            System->errlog("empty stage 1 data array provided to Trace()");
            return false;
        }

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

        try
        {
            IncomingRays.resize( NumberOfRays );
        } catch (std::exception &e) {
            System->errlog("Incoming rays resize exception: %d, '%s'", NumberOfRays, e.what());
            return false;
        }


        if (!SunToPrimaryStage(System, System->StageList[0], &System->Sun, System->Sun.PosSunStage))
            return false;

#ifdef WITH_DEBUG_TIMER
        ofstream fout("C:\\Users\\mwagner\\Documents\\NREL\\Dev\\SolTraceWX\\log.txt");
        fout.clear();
#else
        ofstream fout;
#endif

        time("Initialize:\t", &fout);


//#define WRITE_NODE_FILE
#ifdef WRITE_NODE_FILE
        {
        //Write out to a file for debugging
        ofstream fout2("C:\\Users\\mwagner\\Documents\\NREL\\Dev\\SolTraceWX\\meshxy.txt");
        fout2.clear();

        fout2 << "node,xmin,xmax,ymin,ymax\n";

        vector<st_opt_element>* all_nodes = sun_hash.get_all_nodes();
        for(int i=0; i<all_nodes->size(); i++)
        {
            double* xr = all_nodes->at(i).get_xr();
            double* yr = all_nodes->at(i).get_yr();

            fout2 << all_nodes->at(i).get_address() << "," << xr[0] << "," << xr[1] << "," << yr[0] << "," << yr[1] << "\n";
        }

        fout2.close();

        ofstream fout3("C:\\Users\\mwagner\\Documents\\NREL\\Dev\\SolTraceWX\\meshpolar.txt");
        fout3.clear();

        fout3 << "node,xmin,xmax,ymin,ymax\n";

        all_nodes = rec_hash.get_all_nodes();
        for(int i=0; i<all_nodes->size(); i++)
        {
            double* xr = all_nodes->at(i).get_xr();
            double* yr = all_nodes->at(i).get_yr();

            fout3 << all_nodes->at(i).get_address() << "," << xr[0] << "," << xr[1] << "," << yr[0] << "," << yr[1] << "\n";
        }

        fout3.close();
        }
#endif


        //declare items used within the loop
        vector<TElement*> sunint_elements;
        vector<TElement*> reflint_elements;
        std::vector<TElement*> *element_list_ptr;

        time("Starting stage calculations:\t", &fout);
#ifdef WITH_DEBUG_TIMER
        fout.close();
#endif

        //use the callbacks based on elapsed time rather than fixed rays processed.

        clock_t startTime = clock();     //start timer
        int rays_per_callback_estimate = 50;    //starting rough estimate for how often to check the clock

        for (st_uint_t cur_stage_i=0;cur_stage_i<System->StageList.size();cur_stage_i++)
        {

            if (cur_stage_i > 0 && PreviousStageHasRays == false)
            {
                // no rays to pass through from previous stage
                // so nothing to trace in this stage
                goto Label_EndStageLoop;
            }

            Stage = System->StageList[cur_stage_i];

            LastElementNumber = 0;
            LastRayNumber = 0;
            ray.LastHitBackSide = 0;

            StageDataArrayIndex = 0;
            PreviousStageDataArrayIndex = 0;

			// TODO: Unflagged loading data
#ifdef FLAGGED_FOR_NO_LOADING_DATA
            //if loading stage 0 data, construct appropriate arrays here
            if(cur_stage_i==0 && load_st_data)
            {
                double rpos[3],rcos[3];
                //Stage 0 data
                for(int j=0; j<st0data->size(); j++)
                {

                    LoadExistingStage0Ray(j, st0data,
                        rpos, rcos,
                        LastElementNumber, LastRayNumber);


                    p_ray = Stage->RayData.Append(
                        rpos, rcos,
                        LastElementNumber, 1,
                        LastRayNumber );

                }

                //stage 1 data
                for(int j=0; j<st1in->size(); j++)
                {
                    int rnum;
                    LoadExistingStage1Ray(j, st1in, rpos, rcos, rnum);
                    CopyVec3(IncomingRays[j].Pos, rpos);
                    CopyVec3(IncomingRays[j].Cos, rcos);
                    IncomingRays[j].Num = rnum;
                }

                PreviousStageHasRays = true;
                PreviousStageDataArrayIndex = st1in->size()-1;
                System->SunRayCount = LastRayNumber;
                goto Label_EndStageLoop;
            }

#endif


			// PB: good place for adding loop. IncomingRays already has a vector of size NumberOfRays
			// IncomingRays may have elements from previous stage 1 data
			// but our testing case, no rays has been traced yet

			// loop through rays within each stage
			for (int PreviousStageDataArrayIndex = 0; PreviousStageDataArrayIndex <= NumberOfRays; PreviousStageDataArrayIndex++) {

				// Formerly Label_StartRayLoop:
				MultipleHitCount = 0;
				sunint_elements.clear();


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
					RayNumber = IncomingRays[StageDataArrayIndex].Num;
					CopyVec3( ray.PosRayGlob, IncomingRays[StageDataArrayIndex].Pos );
					CopyVec3( ray.CosRayGlob, IncomingRays[StageDataArrayIndex].Cos );
					StageDataArrayIndex++;

				}

				// transform the global incoming ray to local stage coordinates
				TransformToLocal(ray.PosRayGlob, ray.CosRayGlob,
					Stage->Origin, Stage->RRefToLoc,
					ray.PosRayStage, ray.CosRayStage);


				// Start ray tracing

				in_multi_hit_loop = false;

			Label_MultiHitLoop:
				// Getting list of elements to check for intersection

				ray.LastPathLength = 1e99;
				ray.StageHit = false;

				// Find number of elements to check intersections with, and set element_list
				st_uint_t nintelements;
				get_elements_in_stage(Stage, element_list_ptr, nintelements);


				// Check for ray intersections
				check_intersection_in_stage(element_list_ptr, nintelements,
					cur_stage_i,
					ray,
					LastElementNumber, LastRayNumber, RayNumber);


			Label_StageHitLogic:
				if (!ray.StageHit)
				{
					if (cur_stage_i == 0) // first stage only
					{
						// at least one hit on stage, so move on to next ray
						CopyVec3(IncomingRays[PreviousStageDataArrayIndex].Pos, ray.PosRayGlob);
						CopyVec3(IncomingRays[PreviousStageDataArrayIndex].Cos, ray.CosRayGlob);
						IncomingRays[PreviousStageDataArrayIndex].Num = RayNumber;

						if (RayNumber == NumberOfRays)
							// TODO: change to end_stage()
							goto Label_EndStageLoop;

						// PreviousStageDataArrayIndex++;
						PreviousStageHasRays = true;

						// continue ray loop

						RayNumber++;
						continue;
						// goto Label_StartRayLoop;
					}

					else
					{
						// stages beyond first stage
						if (Stage->TraceThrough || MultipleHitCount > 0)
						{

							CopyVec3(IncomingRays[PreviousStageDataArrayIndex].Pos, ray.PosRayGlob);
							CopyVec3(IncomingRays[PreviousStageDataArrayIndex].Cos, ray.CosRayGlob);
							IncomingRays[PreviousStageDataArrayIndex].Num = RayNumber;

							if (RayNumber == LastRayNumberInPreviousStage)
								goto Label_EndStageLoop;

							// PreviousStageDataArrayIndex++;
							PreviousStageHasRays = true;

							if (MultipleHitCount != 0){
								continue;
							}
						}
						LastElementNumber = 0;
						LastRayNumber = RayNumber;
						CopyVec3(ray.LastPosRaySurfStage, ray.PosRayStage);
						CopyVec3(ray.LastCosRaySurfStage, ray.CosRayStage);
					}

				} // end of not stagehit logic

				// append ray data
				p_ray = Stage->RayData.Append( ray.LastPosRaySurfStage,
									  ray.LastCosRaySurfStage,
									  LastElementNumber,
									  cur_stage_i+1,
									  LastRayNumber );

				if (!p_ray)
				{
					System->errlog("Failed to save ray data at index %d", Stage->RayData.Count()-1);
					return false;
				}

				if (LastElementNumber == 0) // {If missed all elements}
				{
					if (RayNumber == LastRayNumberInPreviousStage)
					{
						if ( !Stage->TraceThrough )
						{
							PreviousStageHasRays = false;
							if (PreviousStageDataArrayIndex > 0)
							{
								// TODO: double check PreviousStageDataArrayIndex
								PreviousStageHasRays = true;
								PreviousStageDataArrayIndex--; // last ray was previous one
							}
						}
						goto Label_EndStageLoop;
					}
					else
					{
						// if this the first stage
						// generate new sun ray but pass the current one to the next stage
						// otherwise, go to the next stage directly
						// sine RayNumber in stage i > 0 is read from IncomingRays[]
						if (cur_stage_i == 0)
						{
							RayNumber++;
						}
						continue;
					}
				}

				MultipleHitCount++;

				// TODO: Virtual stage? What does this do?
				if ( Stage->Virtual )
				{
					CopyVec3(ray.PosRayOutElement, ray.LastPosRaySurfElement);
					CopyVec3(ray.CosRayOutElement, ray.LastCosRaySurfElement);
					goto Label_TransformBackToGlobal;
				}

				// now all the rays have been traced
				// time for optics

				// {Otherwise trace ray through interaction}
				// {Determine if backside or frontside properties should be used}

				// trace through the interaction
				optelm = Stage->ElementList[ p_ray->element - 1 ];
				optics = 0;

				// TODO: Function: TestValue = return_optics_test_value(ray, optelm, optics)
				if (ray.LastHitBackSide)
					optics = &optelm->Optics->Back;
				else
					optics = &optelm->Optics->Front;


				double TestValue;
				switch(optelm->InteractionType )
				{
					case 1: // refraction
						TestValue = optics->Transmissivity;
						break;
					case 2: // reflection

						if ( optics->UseReflectivityTable )
						{
							int npoints = optics->ReflectivityTable.size();
							int m = 0;
							for (int i = 0; i < 3; i++) {
								ray.UnitLastDFXYZ[i] = -ray.LastDFXYZ[i]/sqrt(DOT(ray.LastDFXYZ, ray.LastDFXYZ));
							}

							ray.IncidentAngle = acos(DOT(ray.LastCosRaySurfElement, ray.UnitLastDFXYZ));
							if (ray.IncidentAngle >= optics->ReflectivityTable[ npoints-1 ].angle )
							{
								TestValue = optics->ReflectivityTable[ npoints-1 ].refl;
							}
							else
							{
								while ( optics->ReflectivityTable[m].angle < ray.IncidentAngle )
									m++;

								if (m == 0)
									TestValue = optics->ReflectivityTable[m].refl;
								else
									TestValue = (optics->ReflectivityTable[m].refl + optics->ReflectivityTable[m-1].refl)/2.0;
							}
						}
						else
							TestValue = optics->Reflectivity;
						break;
					default:
						System->errlog("Bad optical interaction type = %d (stage %d)",cur_stage_i,optelm->InteractionType);
						return false;
				}

				// Monte Carlo for absorption
				// {Apply MonteCarlo probability of absorption. Limited for now, but can make more complex later on if desired}
				if (TestValue <= myrng())
				{
					// ray was fully absorbed, so indicate by negating the element number
					p_ray->element = 0 - p_ray->element;

					if (RayNumber == LastRayNumberInPreviousStage)
					{
						PreviousStageHasRays = false;
						if (PreviousStageDataArrayIndex > 0)
						{
							PreviousStageDataArrayIndex--;
							PreviousStageHasRays = true;
						}
						goto Label_EndStageLoop;
					}
					else
					{
						// if all the rays have been traced, then go to next stage
						// otherwise, increase RayNumber and generate a new sun ray
						if (cur_stage_i == 0)
						{
							if (RayNumber == NumberOfRays)
								goto Label_EndStageLoop;
							else
								RayNumber++;
						}

						continue;
					}
				}

// Formerly Label_TransformBackToGlobal
				// TODO: Function: transform_to_global()
				Label_TransformBackToGlobal:
				k = abs( p_ray->element ) - 1;

				// Do the ray interaction (reflect, etc)
				if ( !Stage->Virtual )
				{
					if (IncludeSunShape && cur_stage_i == 0 && MultipleHitCount == 1)//change to account for first hit only in primary stage 8-11-31
					{
						// Apply sunshape to UNPERTURBED ray at intersection point
						//only apply sunshape error once for primary stage
						CopyVec3(ray.CosIn, ray.LastCosRaySurfElement);
						Errors(myrng, ray.CosIn, 1, &System->Sun,
							   Stage->ElementList[k], optics, ray.CosOut, ray.LastDFXYZ);  //sun shape
						CopyVec3(ray.LastCosRaySurfElement, ray.CosOut);
					}

					//{Determine interaction at surface and direction of perturbed ray}
					ray.ErrorFlag = 0;

					// {Apply surface normal errors to surface normal before interaction ray at intersection point - Wendelin 11-23-09}
					if( IncludeErrors )
					{
						CopyVec3( ray.CosIn, ray.CosRayOutElement );
						SurfaceNormalErrors(myrng, ray.LastDFXYZ, optics, ray.CosOut);  //surface normal errors
						CopyVec3( ray.LastDFXYZ, ray.CosOut );
					}

					Interaction( myrng, ray.LastPosRaySurfElement, ray.LastCosRaySurfElement, ray.LastDFXYZ,
						Stage->ElementList[k]->InteractionType, optics, 630.0,
						ray.PosRayOutElement, ray.CosRayOutElement, &ray.ErrorFlag);

					// {Apply specularity optical error to PERTURBED (i.e. after interaction) ray at intersection point}
					if( IncludeErrors )
					{
						CopyVec3(ray.CosIn, ray.CosRayOutElement);
						Errors(myrng, ray.CosIn, 2, &System->Sun,
							   Stage->ElementList[k], optics, ray.CosOut, ray.LastDFXYZ);  //optical errors
						CopyVec3(ray.CosRayOutElement, ray.CosOut);
					}
				}

				// { Transform ray back to stage coord system and trace through stage again}
				TransformToReference(ray.PosRayOutElement, ray.CosRayOutElement,
						Stage->ElementList[k]->Origin, Stage->ElementList[k]->RLocToRef,
						ray.PosRayStage, ray.CosRayStage);
				TransformToReference(ray.PosRayStage, ray.CosRayStage,
						Stage->Origin, Stage->RLocToRef,
						ray.PosRayGlob, ray.CosRayGlob);

				// Check if allow multiple stage hits
				// TODO: switch gotos to while true
				if (!Stage->MultiHitsPerRay)
				{
					ray.StageHit = false;
					goto Label_StageHitLogic;
				}
				else
				{
					in_multi_hit_loop = true;
					goto Label_MultiHitLoop;
				}
// TODO: Change Label_EndStageLoop to end_stage()
Label_EndStageLoop:
				bool no_error;
				no_error = end_stage(System, st0data, st1in, save_st_data,
									 Stage, cur_stage_i, IncomingRays,
									 StageDataArrayIndex,
									 PreviousStageHasRays, PreviousStageDataArrayIndex, LastRayNumberInPreviousStage);
				if (!no_error) {
					return false;
				}
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
