
#include "check_intersection.cuh"
#include "gpu_proc.cuh"
#include "types.h"

/*
 * Check the ray for intersections with all elements in element_list.
 * Modifies ray.
 *
 * Inputs: Variables as defined in Trace.
 *
 */
 __global__
void check_for_intersection_kernel(ElementInfo *element_list, st_uint_t nintelements,
                                   Ray *AllRays, st_uint_t NumberOfRays){
    uint thread_id = threadIdx.x + (blockIdx.x * blockDim.x);

    // Make sure sufficient rays are traced
    // Update the thread index to the stide of the number of threads
    uint stride = gridDim.x * blockDim.x;
    for (uint ray_id = thread_id; ray_id < NumberOfRays; ray_id += stride) {
        Ray *ray = &AllRays[ray_id];
        ray->StageHit = false;
        for (st_uint_t j = 0; j < nintelements; j++)
        {
            ElementInfo *Element;
            Element = &element_list[j];

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
            TransformToLocalGPU(ray->PosRayStage, ray->CosRayStage,
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
                        CopyVec3GPU(ray->LastPosRaySurfElement, PosRaySurfElement);
                        CopyVec3GPU(ray->LastCosRaySurfElement, CosRaySurfElement);
                        CopyVec3GPU(ray->LastDFXYZ, DFXYZ);
                        ray->LastElementNumber = j + 1;    //mjw change from j index to element id
                        TransformToReferenceGPU(PosRaySurfElement, CosRaySurfElement,
                            Element->Origin, Element->RLocToRef,
                            PosRaySurfStage, CosRaySurfStage);

                        CopyVec3GPU(ray->LastPosRaySurfStage, PosRaySurfStage);
                        CopyVec3GPU(ray->LastCosRaySurfStage, CosRaySurfStage);
                        ray->LastHitBackSide = HitBackSide;
                    }
                }
            }
        }
    }
}

void check_intersection_for_all_rays(TStage *Stage, Ray *AllRays, st_uint_t NumberOfRays){
    // Find number of elements to check intersections with, and set element_list
    std::vector<TElement*> element_list = Stage->ElementList;
    st_uint_t nintelements = element_list.size();

    // Copy releavant Element info over
    ElementInfo *elements_info = (ElementInfo *) malloc(nintelements * sizeof(ElementInfo));
    assert(elements_info);
	for (uint i = 0; i < element_list.size(); i++){
        ElementInfo *elm = &elements_info[i];
        elm->Enabled = element_list[i]->Enabled;

        elm->ZAperture = element_list[i]->ZAperture;
        elm->SurfaceIndex = element_list[i]->SurfaceIndex;

        elm->ShapeIndex = element_list[i]->ShapeIndex;
        elm->ParameterA = element_list[i]->ParameterA;
        elm->ParameterB = element_list[i]->ParameterB;
        elm->ParameterC = element_list[i]->ParameterC;
        elm->ParameterD = element_list[i]->ParameterD;
        elm->ParameterE = element_list[i]->ParameterE;
        elm->ParameterF = element_list[i]->ParameterF;
        elm->ParameterG = element_list[i]->ParameterG;
        elm->ParameterH = element_list[i]->ParameterH;

        elm->SurfaceType = element_list[i]->SurfaceType;

        elm->Kappa = element_list[i]->Kappa;
        elm->VertexCurvX = element_list[i]->VertexCurvX;
        elm->VertexCurvY = element_list[i]->VertexCurvY;
        elm->AnnularRadius = element_list[i]->AnnularRadius;
        elm->CrossSectionRadius = element_list[i]->CrossSectionRadius;
        elm->ConeHalfAngle = element_list[i]->ConeHalfAngle;
        elm->CurvOfRev = element_list[i]->CurvOfRev;

        // Transfer arrays and multidimensional arrays
        for (uint idx = 0; idx < 3; idx++) {
            for (uint idx_i = 0; idx_i < 3; idx_i++) {
                elm->RRefToLoc[idx * 3 + idx_i] = element_list[i]->RRefToLoc[idx][idx_i];
                elm->RLocToRef[idx * 3 + idx_i] = element_list[i]->RLocToRef[idx][idx_i];
            }
            elm->Origin[idx] = element_list[i]->Origin[idx];
        }
        for (uint idx = 0; idx < 5; idx++) {
            elm->Alpha[idx] = element_list[i]->Alpha[idx];
        }
	}


    // Transfer input into device memory
    Ray *d_AllRays;
    ElementInfo *d_element_list;
    gpuErrChk(cudaMalloc(&d_AllRays, NumberOfRays * sizeof(Ray)));
    gpuErrChk(cudaMalloc(&d_element_list, nintelements * sizeof(ElementInfo)));
    gpuErrChk(cudaMemcpy(d_AllRays, AllRays, NumberOfRays * sizeof(Ray), cudaMemcpyHostToDevice));
    gpuErrChk(cudaMemcpy(d_element_list, elements_info, nintelements * sizeof(ElementInfo), cudaMemcpyHostToDevice));

    // Call kernel
    // Check for an intersection for all rays in AllRays
    uint blocks = 512;
    uint threads_per_block = 512;
    check_for_intersection_kernel << <blocks, threads_per_block >> >
        (d_element_list, nintelements, d_AllRays, NumberOfRays);
    gpuErrChk(cudaPeekAtLastError());

    // Copy output into back
    gpuErrChk(cudaMemcpy(AllRays, d_AllRays, NumberOfRays * sizeof(Ray), cudaMemcpyDeviceToHost));

}
