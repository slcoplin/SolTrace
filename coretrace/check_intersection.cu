
#include "check_intersection.cuh"
#include "procs.h"

/*
 * Check the ray for intersections with all elements in element_list.
 * Modifies ray.
 *
 * Inputs: Variables as defined in Trace.
 *
 */
void check_intersection_in_stage(std::vector<TElement*> &element_list,
                                          st_uint_t nintelements,
                                           Ray *ray){
    ray->StageHit = false;
    for (st_uint_t j = 0; j < nintelements; j++)
    {
        TElement *Element;
        Element = element_list[j];

        if (!Element->Enabled)
            continue;

        double LastPathLength = 1e99;
        double PathLength;
        double DFXYZ[3];
        double PosRaySurfElement[3];
        double CosRaySurfElement[3];
        double PosRaySurfStage[3];
        double CosRaySurfStage[3];
        double PosRayElement[3];
        double CosRayElement[3];

        //  {Transform ray to element[j] coord system of Stage[i]}
        TransformToLocal(ray->PosRayStage, ray->CosRayStage,
            Element->Origin, Element->RRefToLoc,
            PosRayElement, CosRayElement);

        // increment position by tiny amount to get off the element if tracing to the same element
        for (int i = 0; i < 3; i++) {
            PosRayElement[i] = PosRayElement[i] + 1.0e-5*CosRayElement[i];
        }

        int ErrorFlag = 0;
        int HitBackSide = 0;
        int InterceptFlag = 0;

        // {Determine if ray intersects element[j]; if so, Find intersection point with surface of element[j] }
        DetermineElementIntersectionNew(Element, PosRayElement, CosRayElement,
            PosRaySurfElement, CosRaySurfElement, DFXYZ,
            &PathLength, &ErrorFlag, &InterceptFlag, &HitBackSide);

        if (InterceptFlag)
        {
            //{If hit multiple elements, this loop determines which one hit first.
            //Also makes sure that correct part of closed surface is hit. Also, handles wavy, but close to flat zernikes and polynomials correctly.}
            //if (PathLength < LastPathLength) and (PosRaySurfElement[2] <= Element->ZAperture) then
            if (PathLength < LastPathLength)
            {
                if (PosRaySurfElement[2] <= Element->ZAperture
                    || Element->SurfaceIndex == 'm'
                    || Element->SurfaceIndex == 'M'
                    || Element->SurfaceIndex == 'r'
                    || Element->SurfaceIndex == 'R')
                {
                    ray->StageHit = true;
                    LastPathLength = PathLength;
                    CopyVec3(ray->LastPosRaySurfElement, PosRaySurfElement);
                    CopyVec3(ray->LastCosRaySurfElement, CosRaySurfElement);
                    CopyVec3(ray->LastDFXYZ, DFXYZ);
                    ray->LastElementNumber = j + 1;    //mjw change from j index to element id
                    TransformToReference(PosRaySurfElement, CosRaySurfElement,
                        Element->Origin, Element->RLocToRef,
                        PosRaySurfStage, CosRaySurfStage);

                    CopyVec3(ray->LastPosRaySurfStage, PosRaySurfStage);
                    CopyVec3(ray->LastCosRaySurfStage, CosRaySurfStage);
                    ray->LastHitBackSide = HitBackSide;
                }
            }
        }
    }
}

void check_intersection_for_all_rays(TStage *Stage, Ray *AllRays, st_uint_t NumberOfRays){
    // Check for an intersection for all rays in AllRays
    for (st_uint_t RayIndex = 0; RayIndex < NumberOfRays; RayIndex++) {

        Ray *ray = &AllRays[RayIndex];

        // Find number of elements to check intersections with, and set element_list
        st_uint_t nintelements = Stage->ElementList.size();
        std::vector<TElement*> element_list = Stage->ElementList;

        // Check for ray intersections
        check_intersection_in_stage(element_list, nintelements, ray);
    }
}
