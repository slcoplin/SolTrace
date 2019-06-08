
#include "check_intersection.cuh"

#define gpuErrChk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line,
    bool abort = true)
{
    if (code != cudaSuccess) {
        fprintf(stderr,"GPUassert: %s %s %d\n",
            cudaGetErrorString(code), file, line);
        exit(code);
    }
}

__device__
inline void CopyVec3_GPU( double dest[3], double src[3] )
{
    dest[0] = src[0];
    dest[1] = src[1];
    dest[2] = src[2];
}

__device__
double DOT_GPU(double A[3], double B[3])
{
//{Purpose: To compute the DOT_GPU product of 2 N-dimensional vectors, A and B
  //        Input -
  //              A(N) = First input vector
  //              B(N) = Second input vector
  //              N = dimension of vectors
  //        Output -
  //             Result of DOT_GPU = DOT_GPU product of A and B}

    return (A[0]*B[0] + A[1]*B[1] + A[2]*B[2]);
}

__device__
void MatrixVectorMult_GPU(double M[9], double V[3], double MxV[3])
{
    /*{Purpose: To perform multiplication of a matrix (3,3 in row-major) and
                a vector (3) to result in new vector (3)

              Input -
                    Matrix = Matrix
                    Vector = Vector
              Output -
                     MxV = Matrix*Vector = new vector}*/

    MxV[0] = M[0] * V[0] + M[1] * V[1] + M[2] * V[2];
    MxV[1] = M[3] * V[0] + M[4] * V[1] + M[5] * V[2];
    MxV[2] = M[6] * V[0] + M[7] * V[1] + M[8] * V[2];
}


__device__
void TransformToLocal_GPU(double PosRef[3], double CosRef[3], double Origin[3],
    double RRefToLoc[9],
    double PosLoc[3], double CosLoc[3])
{
    /*{Purpose:  To perform coordinate transformation from reference system to local
               system.
               Input -
                     PosRef = X,Y,Z coordinates of ray point in reference system
                     CosRef = Direction cosines of ray in reference system
                     Origin = X,Y,Z coordinates of origin of local system as measured
                              in reference system
                     RRefToLoc = Rotation matrices required for coordinate transform
                                 from reference to local
               Output -
                     PosLoc = X,Y,Z coordinates of ray point in local system
                     CosLoc = Direction cosines of ray in local system }*/

    double PosDum[3];

    /*{Multiply the position vector and the direction cosine vector by the transformation
     matrix to get the new vectors in the local system.  The position vector is first
     referenced to the origin of the local frame.}*/
    for (int i = 0; i < 3; i++)
        PosDum[i] = PosRef[i] - Origin[i];

    MatrixVectorMult_GPU(RRefToLoc, PosDum, PosLoc);
    MatrixVectorMult_GPU(RRefToLoc, CosRef, CosLoc);
}
//end of procedure--------------------------------------------------------------

__device__
void TransformToReference_GPU(double PosLoc[3], double CosLoc[3], double Origin[3],
    double RLocToRef[9],
    double PosRef[3], double CosRef[3])
{
    /*{Purpose:  To perform coordinate transformation from local system to reference
               system.
               Input -
                     PosLoc = X,Y,Z coordinates of ray point in local system
                     CosLoc = Direction cosines of ray in Loc system
                     Origin = X,Y,Z coordinates of origin of local system as measured
                              in reference system
                     RLocToRef = Rotation matrices required for coordinate transform
                                 from local to reference (inverse of reference to
                                 local transformation)
               Output -
                     PosRef = X,Y,Z coordinates of ray point in reference system
                     CosRef = Direction cosines of ray in reference system}*/

    double PosDum[3];

    /*{Use previously calculated RLocToRef matrix (in TransformToLocal_GPU) to obtain the
     inverse transformation back to Reference system.}*/
    MatrixVectorMult_GPU(RLocToRef, PosLoc, PosDum);
    MatrixVectorMult_GPU(RLocToRef, CosLoc, CosRef);

    for (int i = 0; i < 3; i++)
        PosRef[i] = PosDum[i] + Origin[i];
}

/* The information needed to find intersections */
/* Variables the same as TElement */
struct ElementInfo
{
    // 42 doubles, 1 bool, 2 char, 1 int
    // >300 bytes long
    bool Enabled;
    double Origin[3];
    double RRefToLoc[9];
    double RLocToRef[9];
    double ZAperture;
    char SurfaceIndex;

    char ShapeIndex;
    double ParameterA;
    double ParameterB;
    double ParameterC;
    double ParameterD;
    double ParameterE;
    double ParameterF;
    double ParameterG;
    double ParameterH;

    int SurfaceType;

    double Kappa;
    double Alpha[5];
    double VertexCurvX;
    double VertexCurvY;
    double AnnularRadius;
    double CrossSectionRadius;
    double ConeHalfAngle;
    double CurvOfRev;
};

__device__
void Surface(
            double PosXYZ[3],
            ElementInfo *Element,
            double *FXYZ,
            double DFXYZ[3],
            int *ErrorFlag );

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
            TransformToLocal_GPU(ray->PosRayStage, ray->CosRayStage,
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
                        CopyVec3_GPU(ray->LastPosRaySurfElement, PosRaySurfElement);
                        CopyVec3_GPU(ray->LastCosRaySurfElement, CosRaySurfElement);
                        CopyVec3_GPU(ray->LastDFXYZ, DFXYZ);
                        ray->LastElementNumber = j + 1;    //mjw change from j index to element id
                        TransformToReference_GPU(PosRaySurfElement, CosRaySurfElement,
                            Element->Origin, Element->RLocToRef,
                            PosRaySurfStage, CosRaySurfStage);

                        CopyVec3_GPU(ray->LastPosRaySurfStage, PosRaySurfStage);
                        CopyVec3_GPU(ray->LastCosRaySurfStage, CosRaySurfStage);
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


#define sqr(x) (x*x)
#define   Order 3
#define   NumIterations 20
#define   Epsilon 0.000001

#define sign(x) (x>=0)


// barycentric technique for triangles (7 jul 2010)
// http://www.blackpawn.com/texts/pointinpoly/default.html
/*
    // Compute vectors
    v0 = C - A
    v1 = B - A
    v2 = P - A

    // Compute DOT_GPU products
    DOT_GPU00 = DOT_GPU(v0, v0)
    DOT_GPU01 = DOT_GPU(v0, v1)
    DOT_GPU02 = DOT_GPU(v0, v2)
    DOT_GPU11 = DOT_GPU(v1, v1)
    DOT_GPU12 = DOT_GPU(v1, v2)

    // Compute barycentric coordinates
    invDenom = 1 / (DOT_GPU00 * DOT_GPU11 - DOT_GPU01 * DOT_GPU01)
    u = (DOT_GPU11 * DOT_GPU02 - DOT_GPU01 * DOT_GPU12) * invDenom
    v = (DOT_GPU00 * DOT_GPU12 - DOT_GPU01 * DOT_GPU02) * invDenom

    // Check if point is in triangle
    return (u > 0) && (v > 0) && (u + v < 1)
*/
__device__
int intri(double x1, double y1,
                 double x2, double y2,
                 double x3, double y3,
                 double xt, double yt)
{
    double a = (x1 - xt)*(y2 - yt) - (x2 - xt)*(y1 - yt);
    double b = (x2 - xt)*(y3 - yt) - (x3 - xt)*(y2 - yt);
    double c = (x3 - xt)*(y1 - yt) - (x1 - xt)*(y3 - yt);
    return (sign(a) == sign(b) && sign(b) == sign( c));
}

__device__
int inquad(double x1, double y1,
                 double x2, double y2,
                 double x3, double y3,
                 double x4, double y4,
                 double xt, double yt)
{
    return intri(x1,y1,x2,y2,x3,y3,xt,yt)
        || intri(x1,y1,x3,y3,x4,y4,xt,yt);
}

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
            int *BacksideFlag )
{
    double r = 0.0, Ro = 0.0, Ri = 0.0, XL = 0.0, x = 0.0, y = 0.0;
    double /*SLOP30 = 0.0, SLOP60 = 0.0,*/ Y1 = 0.0, Y2 = 0.0, Y3 = 0.0, Y4 = 0.0;
    double P1x = 0.0, P1y = 0.0, P2x = 0.0, P2y = 0.0, P3x = 0.0, P3y = 0.0, P4x = 0.0, P4y = 0.0;
    //double Tn;
    int in_quad = 0;
   //ZAperPlane: real;

    *ErrorFlag = 0;
    double SLOP60 = 1.7320508075688767; //tan(60.0*(acos(-1.0)/180.0));

    //AperturePlane(Element);           <------- calculated now in ODConcentrator
    //ZAperPlane = Element->ZAperture;

    //find intersection with surface first
    Intersect(PosRayIn, CosRayIn, Element, PosRayOut, CosRayOut, DFXYZ, PathLength, ErrorFlag);
    if (*ErrorFlag > 0 || *PathLength < 0)
    {
        *Intercept = false;
        PosRayOut[0] = 0.0;
        PosRayOut[1] = 0.0;
        PosRayOut[2] = 0.0;
        CosRayOut[0] = 0.0;
        CosRayOut[1] = 0.0;
        CosRayOut[2] = 0.0;
        DFXYZ[0] = 0.0;
        DFXYZ[1] = 0.0;
        DFXYZ[2] = 0.0;
        *BacksideFlag = false;
        *PathLength = 0.0;
        goto Label_100;
    }

    x = PosRayOut[0];
    y = PosRayOut[1];
    r = sqrt(x*x + y*y);



    switch (Element->ShapeIndex)
    {
    case 'c':
    case 'C': //circular aperture
            Ro = Element->ParameterA/2.0;

            if (r > Ro) //ray falls outsideside circular aperture
            {
               *Intercept = false;
               PosRayOut[0] = 0.0;
               PosRayOut[1] = 0.0;
               PosRayOut[2] = 0.0;
               CosRayOut[0] = 0.0;
               CosRayOut[1] = 0.0;
               CosRayOut[2] = 0.0;
               DFXYZ[0] = 0.0;
               DFXYZ[1] = 0.0;
               DFXYZ[2] = 0.0;
               *PathLength = 0.0;
               *ErrorFlag = 0;
               *BacksideFlag = false;
               goto Label_100;
            }
            else
            {
                if (DOT_GPU(CosRayIn, DFXYZ) < 0)
                    *BacksideFlag = false;
                else
                    *BacksideFlag = true;
                *Intercept = true;
                goto Label_100;
            }

    case 'h':
    case 'H': //hexagonal aperture
            Ro = Element->ParameterA/2.0;

            if (r > Ro) //ray falls outside circular circumference aperture
            {
               *Intercept = false;
               PosRayOut[0] = 0.0;
               PosRayOut[1] = 0.0;
               PosRayOut[2] = 0.0;
               CosRayOut[0] = 0.0;
               CosRayOut[1] = 0.0;
               CosRayOut[2] = 0.0;
               DFXYZ[0] = 0.0;
               DFXYZ[1] = 0.0;
               DFXYZ[2] = 0.0;
               *PathLength = 0.0;
               *ErrorFlag = 0;
               *BacksideFlag = false;
               goto Label_100;
            }
            Ri = Ro*cos(30.0*(ACOSM1O180));

            if ( r <= Ri ) //ray falls inside inscribed circle
            {
                if ( DOT_GPU(CosRayIn, DFXYZ) < 0 )
                    *BacksideFlag = false;
                else
                    *BacksideFlag = true;
                *Intercept = true;
                goto Label_100;
            }

            XL = sqrt(Ro*Ro - Ri*Ri); //otherwise break hexagon into 3 sections
            if ( (x <= Ro) && (x > XL) )  //1st section
            {
                Y1 = SLOP60*(x-Ro);
                Y2 = -Y1;
                if ( (y >= Y1) && (y <= Y2) )
                {
                    if ( DOT_GPU(CosRayIn, DFXYZ) < 0 )
                        *BacksideFlag = false;
                    else
                        *BacksideFlag = true;

                    *Intercept = true;
                    goto Label_100;
                }
                *Intercept = false;
                PosRayOut[0] = 0.0;
                PosRayOut[1] = 0.0;
                PosRayOut[2] = 0.0;
                CosRayOut[0] = 0.0;
                CosRayOut[1] = 0.0;
                CosRayOut[2] = 0.0;
                DFXYZ[0] = 0.0;
                DFXYZ[1] = 0.0;
                DFXYZ[2] = 0.0;
                *PathLength = 0.0;
                *BacksideFlag = false;
                *ErrorFlag = 0;
                goto Label_100;
            }

            if ( (x <= XL) && (x >= -XL) )    //2nd section
            {
                if ( (y >= -Ri) && (y <= Ri) )
                {
                    if ( DOT_GPU(CosRayIn, DFXYZ) < 0 )
                        *BacksideFlag = false;
                    else
                        *BacksideFlag = true;
                    *Intercept = true;
                    goto Label_100;
                }
                *Intercept = false;
                PosRayOut[0] = 0.0;
                PosRayOut[1] = 0.0;
                PosRayOut[2] = 0.0;
                CosRayOut[0] = 0.0;
                CosRayOut[1] = 0.0;
                CosRayOut[2] = 0.0;
                DFXYZ[0] = 0.0;
                DFXYZ[1] = 0.0;
                DFXYZ[2] = 0.0;
                *PathLength = 0.0;
                *BacksideFlag = false;
                *ErrorFlag = 0;
                goto Label_100;
            }

            if ( (x < -XL) && (x >= -Ro) )    //3rd section
            {
                Y3 = SLOP60*(x+Ro);
                Y4 = -Y3;
                if ( (y >= Y4) && (y <= Y3) )
                {
                    if ( DOT_GPU(CosRayIn, DFXYZ) < 0 )
                        *BacksideFlag = false;
                    else
                        *BacksideFlag = true;
                    *Intercept = true;
                    goto Label_100;
                }
                *Intercept = false;
                PosRayOut[0] = 0.0;
                PosRayOut[1] = 0.0;
                PosRayOut[2] = 0.0;
                CosRayOut[0] = 0.0;
                CosRayOut[1] = 0.0;
                CosRayOut[2] = 0.0;
                DFXYZ[0] = 0.0;
                DFXYZ[1] = 0.0;
                DFXYZ[2] = 0.0;
                *PathLength = 0.0;
                *BacksideFlag = false;
                *ErrorFlag = 0;
                goto Label_100;
            }
        break;

    case 't':
    case 'T': //Triangular aperture
            Ro = Element->ParameterA/2.0;

            if ( r > Ro ) //ray falls outside circular circumference aperture
            {
                *Intercept = false;
                PosRayOut[0] = 0.0;
                PosRayOut[1] = 0.0;
                PosRayOut[2] = 0.0;
                CosRayOut[0] = 0.0;
                CosRayOut[1] = 0.0;
                CosRayOut[2] = 0.0;
                DFXYZ[0] = 0.0;
                DFXYZ[1] = 0.0;
                DFXYZ[2] = 0.0;
                *PathLength = 0.0;
                *BacksideFlag = false;
                *ErrorFlag = 0;
                goto Label_100;
            }
            Ri = Ro*sin(30.0*(ACOSM1O180));

            if ( r <= Ri )  //ray falls inside inscribed circle
            {
                if ( DOT_GPU(CosRayIn, DFXYZ) < 0 )
                    *BacksideFlag = false;
                else
                    *BacksideFlag = true;
                *Intercept = true;
                goto Label_100;
            }

            if ( (x <= Ro) && (x > 0.0) )  //1st section
            {
                Y1 = -SLOP60*(x-Ri/cos(30.0*(ACOSM1O180)));
                Y2 = -Ri;
                if ( (y <= Y1) && (y >= Y2) )
                {
                    if ( DOT_GPU(CosRayIn, DFXYZ) < 0 )
                        *BacksideFlag = false;
                    else
                        *BacksideFlag = true;
                    *Intercept = true;
                    goto Label_100;
                }
                *Intercept = false;
                PosRayOut[0] = 0.0;
                PosRayOut[1] = 0.0;
                PosRayOut[2] = 0.0;
                CosRayOut[0] = 0.0;
                CosRayOut[1] = 0.0;
                CosRayOut[2] = 0.0;
                DFXYZ[0] = 0.0;
                DFXYZ[1] = 0.0;
                DFXYZ[2] = 0.0;
                *PathLength = 0.0;
                *BacksideFlag = false;
                *ErrorFlag = 0;
                goto Label_100;
            }
            if ( (x >= -Ro) && (x <= 0.0) )  //2nd section
            {
                Y3 = SLOP60*(x+Ri/cos(30.0*(ACOSM1O180)));
                Y4 = -Ri;
                if ( (y >= Y4) && (y <= Y3) )
                {
                    if ( DOT_GPU(CosRayIn, DFXYZ) < 0 )
                        *BacksideFlag = false;
                    else
                        *BacksideFlag = true;
                    *Intercept = true;
                    goto Label_100;
                }
                *Intercept = false;
                PosRayOut[0] = 0.0;
                PosRayOut[1] = 0.0;
                PosRayOut[2] = 0.0;
                CosRayOut[0] = 0.0;
                CosRayOut[1] = 0.0;
                CosRayOut[2] = 0.0;
                DFXYZ[0] = 0.0;
                DFXYZ[1] = 0.0;
                DFXYZ[2] = 0.0;
                *PathLength = 0.0;
                *BacksideFlag = false;
                *ErrorFlag = 0;
                goto Label_100;
            }
        break;

    case 'r':
    case 'R': //Rectangular aperture

            if ( (x > Element->ParameterA/2.0) || (x < -Element->ParameterA/2.0) )
            {
                *Intercept = false;
                PosRayOut[0] = 0.0;
                PosRayOut[1] = 0.0;
                PosRayOut[2] = 0.0;
                CosRayOut[0] = 0.0;
                CosRayOut[1] = 0.0;
                CosRayOut[2] = 0.0;
                DFXYZ[0] = 0.0;
                DFXYZ[1] = 0.0;
                DFXYZ[2] = 0.0;
                *PathLength = 0.0;
                *BacksideFlag = false;
                *ErrorFlag = 0;
                goto Label_100;
            }

            if ( (y > Element->ParameterB/2.0) || (y < -Element->ParameterB/2.0) )
            {
                *Intercept = false;
                PosRayOut[0] = 0.0;
                PosRayOut[1] = 0.0;
                PosRayOut[2] = 0.0;
                CosRayOut[0] = 0.0;
                CosRayOut[1] = 0.0;
                CosRayOut[2] = 0.0;
                DFXYZ[0] = 0.0;
                DFXYZ[1] = 0.0;
                DFXYZ[2] = 0.0;
                *PathLength = 0.0;
                *BacksideFlag = false;
                *ErrorFlag = 0;
                goto Label_100;
            }

            if ( DOT_GPU(CosRayIn, DFXYZ) < 0 ) //successfully falls on rectangle
                *BacksideFlag = false;
            else
                *BacksideFlag = true;

            *Intercept = true;
            goto Label_100;

    case 'a':
    case 'A': //Annulus or torus contour

            if ( (Element->ParameterA == 0.0) && (Element->ParameterB == 0.0) ) goto Label_5; //torus

            if ( (r < Element->ParameterA) || (r > Element->ParameterB) )
            {
                *Intercept = false;
                PosRayOut[0] = 0.0;
                PosRayOut[1] = 0.0;
                PosRayOut[2] = 0.0;
                CosRayOut[0] = 0.0;
                CosRayOut[1] = 0.0;
                CosRayOut[2] = 0.0;
                DFXYZ[0] = 0.0;
                DFXYZ[1] = 0.0;
                DFXYZ[2] = 0.0;
                *PathLength = 0.0;
                *BacksideFlag = false;
                *ErrorFlag = 0;
                goto Label_100;
            }
Label_5:
            if ( x >= 0.0 )
            {
                if ( (asin(y/r) > Element->ParameterC*(ACOSM1O180)/2.0) || (asin(y/r) < -Element->ParameterC*(ACOSM1O180)/2.0) )
                {
                    *Intercept = false;
                    PosRayOut[0] = 0.0;
                    PosRayOut[1] = 0.0;
                    PosRayOut[2] = 0.0;
                    CosRayOut[0] = 0.0;
                    CosRayOut[1] = 0.0;
                    CosRayOut[2] = 0.0;
                    DFXYZ[0] = 0.0;
                    DFXYZ[1] = 0.0;
                    DFXYZ[2] = 0.0;
                    *PathLength = 0.0;
                    *BacksideFlag = false;
                    *ErrorFlag = 0;
                    goto Label_100;
                }

                if ( DOT_GPU(CosRayIn, DFXYZ) < 0 ) //successfully falls on annular section
                    *BacksideFlag = false;
                else
                    *BacksideFlag = true;
                *Intercept = true;
                goto Label_100;
            }

            if ( x < 0.0 )
            {
                if ( (y >= 0) && ((acos(y/r)+M_PI/2.0) > Element->ParameterC*(ACOSM1O180)/2.0) )
                {
                    *Intercept = false;
                    PosRayOut[0] = 0.0;
                    PosRayOut[1] = 0.0;
                    PosRayOut[2] = 0.0;
                    CosRayOut[0] = 0.0;
                    CosRayOut[1] = 0.0;
                    CosRayOut[2] = 0.0;
                    DFXYZ[0] = 0.0;
                    DFXYZ[1] = 0.0;
                    DFXYZ[2] = 0.0;
                    *PathLength = 0.0;
                    *BacksideFlag = false;
                    *ErrorFlag = 0;
                    goto Label_100;
                }
                else if ( (y < 0) && ((-acos(-y/r)-M_PI/2.0) < -Element->ParameterC*(ACOSM1O180)/2.0) )
                {
                    *Intercept = false;
                    PosRayOut[0] = 0.0;
                    PosRayOut[1] = 0.0;
                    PosRayOut[2] = 0.0;
                    CosRayOut[0] = 0.0;
                    CosRayOut[1] = 0.0;
                    CosRayOut[2] = 0.0;
                    DFXYZ[0] = 0.0;
                    DFXYZ[1] = 0.0;
                    DFXYZ[2] = 0.0;
                    *PathLength = 0.0;
                    *BacksideFlag = false;
                    *ErrorFlag = 0;
                    goto Label_100;
                }

                if ( DOT_GPU(CosRayIn, DFXYZ) < 0 ) //successfully falls on annular section
                    *BacksideFlag = false;
                else
                    *BacksideFlag = true;
                *Intercept = true;
                goto Label_100;
            }
        break;

    case 'l':
    case 'L': //off axis aperture section of line focus trough  or cylinder
            if ( (Element->ParameterA == 0.0) && (Element->ParameterB == 0.0) ) goto Label_10; //for cylinder, only need to check for limits on y

            if ( (x < Element->ParameterA) || (x > Element->ParameterB) )
            {
                *Intercept = false;
                PosRayOut[0] = 0.0;
                PosRayOut[1] = 0.0;
                PosRayOut[2] = 0.0;
                CosRayOut[0] = 0.0;
                CosRayOut[1] = 0.0;
                CosRayOut[2] = 0.0;
                DFXYZ[0] = 0.0;
                DFXYZ[1] = 0.0;
                DFXYZ[2] = 0.0;
                *PathLength = 0.0;
                *BacksideFlag = false;
                *ErrorFlag = 0;
                goto Label_100;
            }

Label_10:
            if ( (y < -Element->ParameterC/2.0) || (y > Element->ParameterC/2.0) )
            {
                *Intercept = false;
                PosRayOut[0] = 0.0;
                PosRayOut[1] = 0.0;
                PosRayOut[2] = 0.0;
                CosRayOut[0] = 0.0;
                CosRayOut[1] = 0.0;
                CosRayOut[2] = 0.0;
                DFXYZ[0] = 0.0;
                DFXYZ[1] = 0.0;
                DFXYZ[2] = 0.0;
                *PathLength = 0.0;
                *BacksideFlag = false;
                *ErrorFlag = 0;
                goto Label_100;
            }

            if ( DOT_GPU(CosRayIn, DFXYZ) < 0 ) //successfully falls on line focus or cylindrical section
                *BacksideFlag = false;
            else
                *BacksideFlag = true;

            *Intercept = true;
            goto Label_100;
    
    case 'i':
    case 'I': //irregular triangle
            P1x = Element->ParameterA;
            P1y = Element->ParameterB;
            P2x = Element->ParameterC;
            P2y = Element->ParameterD;
            P3x = Element->ParameterE;
            P3y = Element->ParameterF;

            if (!intri( P1x, P1y, P2x, P2y, P3x, P3y, x, y ))
            {
                *Intercept = false;
                PosRayOut[0] = 0.0;
                PosRayOut[1] = 0.0;
                PosRayOut[2] = 0.0;
                CosRayOut[0] = 0.0;
                CosRayOut[1] = 0.0;
                CosRayOut[2] = 0.0;
                DFXYZ[0] = 0.0;
                DFXYZ[1] = 0.0;
                DFXYZ[2] = 0.0;
                *PathLength = 0.0;
                *ErrorFlag = 0;
                *BacksideFlag = false;
                goto Label_100;
            }

            if ( DOT_GPU(CosRayIn, DFXYZ) < 0 )
                *BacksideFlag = false;
            else
                *BacksideFlag = true;
            *Intercept = true;
            goto Label_100;
     

    case 'q':
    case 'Q': //irregular quadrilateral
            P1x = Element->ParameterA;
            P1y = Element->ParameterB;
            P2x = Element->ParameterC;
            P2y = Element->ParameterD;
            P3x = Element->ParameterE;
            P3y = Element->ParameterF;
            P4x = Element->ParameterG;
            P4y = Element->ParameterH;

            in_quad = inquad(P1x, P1y, P2x, P2y, P3x, P3y, P4x, P4y, x, y);

            if (!in_quad)
            {
                *Intercept = false;
                PosRayOut[0] = 0.0;
                PosRayOut[1] = 0.0;
                PosRayOut[2] = 0.0;
                CosRayOut[0] = 0.0;
                CosRayOut[1] = 0.0;
                CosRayOut[2] = 0.0;
                DFXYZ[0] = 0.0;
                DFXYZ[1] = 0.0;
                DFXYZ[2] = 0.0;
                *PathLength = 0.0;
                *ErrorFlag = 0;
                *BacksideFlag = false;
                goto Label_100;
            }

            if ( DOT_GPU(CosRayIn, DFXYZ) < 0 )
                *BacksideFlag = false;
            else
                *BacksideFlag = true;
            *Intercept = true;
            goto Label_100;
        
    } //end select case

Label_100:
    if ( *BacksideFlag )   //if hit on backside of element then slope of surface is reversed
    {
        DFXYZ[0] = -DFXYZ[0];
        DFXYZ[1] = -DFXYZ[1];
        DFXYZ[2] = -DFXYZ[2];
    }
}
//End of Procedure--------------------------------------------------------------

__device__
void QuadricSurfaceClosedForm(
            ElementInfo *Element,
            double PosLoc[3],
            double CosLoc[3],
            double PosXYZ[3],
            double DFXYZ[3],
            double *PathLength,
            int *ErrorFlag)
{
    double Xdelta = 0.0,Ydelta = 0.0,Zdelta = 0.0;
    double Xc=0,Yc=0,Zc=0,Kx=0,Ky=0,Kz=0;
    double r = 0.0,r2 = 0.0,a2=0,b2=0,c2=0;
    double t1 = 0.0,t2 = 0.0,A=0,B=0,C=0,slopemag = 0.0;

    *ErrorFlag = 0;

    switch( Element->SurfaceIndex )
    {
    case 's':
    case 'S': // sphere
            a2 = 1;
            b2 = 1;
            c2 = 1;
            Kx = 1;
            Ky = 1;
            Kz = 1;
            r = 1.0/Element->VertexCurvX;
            r2 = r*r;
            Xc = 0.0;
            Yc = 0.0;
            Zc = r;

            Xdelta = PosLoc[0] - Xc;
            Ydelta = PosLoc[1] - Yc;
            Zdelta = PosLoc[2] - Zc;

            A = CosLoc[0]*CosLoc[0]*Kx/a2 + CosLoc[1]*CosLoc[1]*Ky/b2 + CosLoc[2]*CosLoc[2]*Kz/c2;
            B = 2.0*(Kx*Xdelta*CosLoc[0]/a2 + Ky*Ydelta*CosLoc[1]/b2 + Kz*Zdelta*CosLoc[2]/c2);
            C = Kx*Xdelta*Xdelta/a2 + Ky*Ydelta*Ydelta/a2 + Kz*Zdelta*Zdelta/c2 - r2;
        break;

    case 'p':
    case 'P':   //parabola
            a2 = 4.0*(1.0/Element->VertexCurvX)/2.0;
            b2 = a2;
            c2 = 1.0;
            Xc = 0.0;
            Yc = 0.0;
            Zc = 0.0;

            Xdelta = PosLoc[0] - Xc;
            Ydelta = PosLoc[1] - Yc;
            Zdelta = PosLoc[2] - Zc;

            A = sqr(CosLoc[0])/a2 + sqr(CosLoc[1])/b2;
            B = 2.0*CosLoc[0]*Xdelta/a2 + 2.0*CosLoc[1]*Ydelta/b2 - CosLoc[2]/c2;
            C = sqr(Xdelta)/a2 + sqr(Ydelta)/b2 - Zdelta/c2;
        break;

    case 'o':
    case 'O':   //other
        break;

    case 't':
    case 'T':   //cylinder
            a2 = 1;
            b2 = 1;
            c2 = 1;
            Kx = 1;
            Ky = 0;
            Kz = 1;
            r = 1.0/Element->CurvOfRev;
            r2 = r*r;
            Xc = 0.0;
            Yc = 0.0;
            Zc = r;

            Xdelta = PosLoc[0] - Xc;
            Ydelta = PosLoc[1] - Yc;
            Zdelta = PosLoc[2] - Zc;

            A = CosLoc[0]*CosLoc[0]*Kx/a2 + CosLoc[1]*CosLoc[1]*Ky/b2 + CosLoc[2]*CosLoc[2]*Kz/c2;
            B = 2.0*(Kx*Xdelta*CosLoc[0]/a2 + Ky*Ydelta*CosLoc[1]/b2 + Kz*Zdelta*CosLoc[2]/c2);
            C = Kx*Xdelta*Xdelta/a2 + Ky*Ydelta*Ydelta/a2 + Kz*Zdelta*Zdelta/c2 - r2;
        break;

    case 'c':
    case 'C':   //cone
        break;

    case 'f':
    case 'F':   //flat
        break;
    }

    if (sqr(B) > 4.0*A*C)
    {
        t1 = (-B + sqrt(sqr(B)-4.0*A*C))/(2.0*A);
        t2 = (-B - sqrt(sqr(B)-4.0*A*C))/(2.0*A);
        if (t2 > 0)    //initial ray location outside surface
        {
            PosXYZ[0] = PosLoc[0] + t2*CosLoc[0];
            PosXYZ[1] = PosLoc[1] + t2*CosLoc[1];
            PosXYZ[2] = PosLoc[2] + t2*CosLoc[2];
            *PathLength = t2;

            //*************************************************************************************************************
            //makes sure to get shortest ray path on valid side of surface; 10-05-10    for open surface of parabola
            //if cylinder, then PosXYZ[3] will always be less than or equal to Element.Zaperture so never passes this test.
            // Test for  cylinder follows below.
            if (PosXYZ[2] > Element->ZAperture)
            {
                PosXYZ[0] = PosLoc[0] + t1*CosLoc[0];
                PosXYZ[1] = PosLoc[1] + t1*CosLoc[1];
                PosXYZ[2] = PosLoc[2] + t1*CosLoc[2];
                *PathLength = t1;
            }

            // Remember at this point, intersection is being found on an INFINITELY long cylinder.
            //if 1st intersection on INFINITELY long cylinder is from the outside, t2, check to make sure
            //intersection is within the finite
            // length of the actual cylinder geometry, if not then 2nd intersection on the inside, t1,
            //is valid one to use.  This means ray could
            // enter from the open end  of the cylinder and hit on the inside.  The final test for this is in the
            //calling routine:  DetermineElementIntersectionNew
            // Wendelin 10-05-10
            if ((Element->SurfaceIndex == 't') || (Element->SurfaceIndex == 'T'))
            {
                if ((PosXYZ[1] < -Element->ParameterC/2.0) || (PosXYZ[1] > Element->ParameterC/2.0))
                {
                    PosXYZ[0] = PosLoc[0] + t1*CosLoc[0];
                    PosXYZ[1] = PosLoc[1] + t1*CosLoc[1];
                    PosXYZ[2] = PosLoc[2] + t1*CosLoc[2];
                    *PathLength = t1;
                }
            }
           //***********************************************************************************************************

            goto Label_100;
        }
        if (t2 == 0)   //initial ray location at surface
        {
            PosXYZ[0] = PosLoc[0] + t1*CosLoc[0];
            PosXYZ[1] = PosLoc[1] + t1*CosLoc[1];
            PosXYZ[2] = PosLoc[2] + t1*CosLoc[2];
            *PathLength = t1;
            goto Label_100;
        }
        if (t2 < 0 && t1 > 0)     //initial ray location inside surface
        {
            PosXYZ[0] = PosLoc[0] + t1*CosLoc[0];
            PosXYZ[1] = PosLoc[1] + t1*CosLoc[1];
            PosXYZ[2] = PosLoc[2] + t1*CosLoc[2];
            *PathLength = t1;
            goto Label_100;
        }
        if (t1 <= 0)
        {
            *PathLength = t1; //ray heading away from surface
            *ErrorFlag = 1;
            return;
        }
    }
    else
    {
        *PathLength = 0.0; //ray tangent or missed
        *ErrorFlag = 1;
        return;
    }

Label_100:
    slopemag = sqrt(sqr(2.0*Kx*(PosXYZ[0] - Xc)/a2)+sqr(2.0*Ky*(PosXYZ[1] - Yc)/b2)+sqr(2.0*Kz*(PosXYZ[2] - Zc)/c2));
    DFXYZ[0] = -(2.0*Kx*(PosXYZ[0] - Xc)/a2)/slopemag;
    DFXYZ[1] = -(2.0*Ky*(PosXYZ[1] - Yc)/b2)/slopemag;
    DFXYZ[2] = -(2.0*Kz*(PosXYZ[2] - Zc)/c2)/slopemag;
}
//end of procedure--------------------------------------------------------------
__device__
void SurfaceZatXYPair(
            double PosXYZ[3],
            ElementInfo *Element,
            double *FXYZ,
            int *ErrorFlag )
{
/*{Purpose: To compute the Z value of the surface equation at an X,Y pair.
    Input - PosXYZ[3] = X, Y, Z coordinate position
            Element.SurfaceType = Surface type flag
                          = 1 for rotationally symmetric surfaces
                          = 2 for torics and cylinders
                          = 3 for plane surfaces
                          = 4 for surface interpolated from finite element data points
                          = 5 for surface interpolated from VSHOT data points
                          = 6 for surface described by Zernike monomials
                          = 7 for single axis parabolic curvature surfaces
                          = 8 for rotationally symmetric polynomial description
                          = 9 for       "          "     cubic spline interpolation
            Element.Alpha = Sensitivity coefficients which specify deviation from conic
                    of revolution
            Element.VertexCurvX = Vertex Curvature of surface
            Element.Kappa = Surface specifier
                 < 0         ==> Hyperboloid
                 = 0         ==> Paraboloid
                 > 0 and < 1 ==> Hemelipsoid of revolution about major axis
                 = 1         ==> Hemisphere
                 > 1         ==> Hemelipsoid of revolution about minor axis
            Element.ConeHalfAngle = Half-angle of cone for cones or revolution or axicons
            Element.CurvOfRev = Curvature of revolution

    Output - FXYZ = Z value of Surface equation
             ErrorFlag = Error Flag
                         = 0  ==> no errors
                         > 0  ==> interpolation error
}*/

    double X=0.0,Y=0.0;
    double Rho2=0.0;

     //Initialize variables
    X = PosXYZ[0];
    Y = PosXYZ[1];
    *ErrorFlag = 0;

//===SurfaceType = 1, 7  Rotationally Symmetric surfaces and single axis curvature sections===========================
    if (Element->SurfaceType == 1 || Element->SurfaceType == 7)
    {
        if (Element->SurfaceType == 1)
            Rho2 = X*X + Y*Y;    //rotationally symmetric
        else
            Rho2 = X*X;         //single axis curvature depends only on x

        if (Element->ConeHalfAngle != 0.0) goto Label_160;

        //wendelin 5-18-11

        //if (Element->Kappa*Element->VertexCurvX*Element->VertexCurvX*Rho2 > 1.0)  //xy pair cannot be found on closed surface   06-10-07
        if ( Element->Kappa*(Element->VertexCurvX*Element->VertexCurvX*X*X+Element->VertexCurvY*Element->VertexCurvY*Y*Y) > 1.0 )  //xy pair cannot be found on closed surface   06-10-07
        {
            *FXYZ = 0.0;
            return;
        }

        //wendelin 5-18-11
        // *FXYZ = Element->VertexCurvX*Rho2/(1.0+sqrt(1.0-Element->Kappa*Element->VertexCurvX*Element->VertexCurvX*Rho2));
        *FXYZ = (Element->VertexCurvX*X*X+Element->VertexCurvY*Y*Y)
                / (1.0+sqrt(1.0-Element->Kappa*(Element->VertexCurvX*Element->VertexCurvX*X*X+Element->VertexCurvY*Y*Y)));

/*        for (i=0;i<5;i++)
             if (Element->Alpha[i] != 0.0) goto Label_130;
             */

        return;

Label_160:
        *FXYZ = sqrt(Rho2)/tan(Element->ConeHalfAngle*(ACOSM1O180));
        return;
    }
}

__device__
void Intersect( double PosLoc[3],
            double CosLoc[3],
            ElementInfo *Element,
            double PosXYZ[3],
            double CosKLM[3],
            double DFXYZ[3],
            double *PathLength,
            int *ErrorFlag )
{
/*{Purpose: To compute intersection point and direction numbers for surface normal
at intersection point of ray and surface. Path length is also computed.  From Spencer & Murty paper pg. 674
   Input - PosLoc[3] = Initial position of ray in local coordinate system.
           CosLoc[3] = Initial direction cosines of ray in local system.
           Element.SurfaceType = Surface type flag
                         = 1 for rotationally symmetric surfaces
                         = 2 for torics and cylinders
                         = 3 for plane surfaces
                         = 4 for finite element data surface
                         = 5 for VSHOT data surface
                         = 6 for Zernike Monomial description
                         = 7 for single axis curvature surfaces
                         = 8 for rotationally symmetric polynomial description
                         = 9 for      "          "         cubic spline interpolation
                         =10 for torus
           Element.Alpha = Sensitivity coefficients which specify deviation from conic
                   of revolution. For plane p = kx+ly+mz, Alpha[1] = p, Alpha{2..4] = k,l,m
           Element.VertexCurvX = Vertex Curvature of surface
           Element.Kappa = Surface specifier
                 < 0         ==> Hyperboloid
                 = 0         ==> Paraboloid
                 > 0 and < 1 ==> Hemelipsoid of revolution about major axis
                 = 1         ==> Hemisphere
                 > 1         ==> Hemielipsoid of revolution about minor axis
           Element.ConeHalfAngle = Half-angle of cone for cones or revolution or axicons
           Element.CurvOfRev = Curvature of revolution

   Output - PosXYZ[3] = X, Y, Z coordinate of ray/surface intersection
            CosKLM[3] = direction cosines of ray
            DFXYZ[3]  = direction numbers for the surface normal at the
                        intersection point (partial derivatives with respect to
                        X, Y, Z of surface equation).
            PathLength = Path length
            ErrorFlag  = Error flag
                         = 0 for no errors
                         = 1 for Newton-Raphson iteration failed to converge
                         = 2 for interpolation error in SURFACE procedure} */
    int i = 0;
    double S0 = 0.0;
    double X1 = 0.0,x = 0.0,y = 0.0,r = 0.0;
    double Y1 = 0.0;
    double SJ = 0.0;
    double SJ1 = 0.0;
    double DFDXYZ = 0.0;
    double FXYZ = 0.0;
    int OKFlag = 0;
    double ZStart = 0.0;
    double S0Aperture = 0.0;
    double Ro = 0.0, Ri = 0.0, XL = 0.0;
    bool ZAInterceptInsideAperture = false;
    double Y2 = 0.0,Y3 = 0.0,Y4 = 0.0;
    double SLOP60 = 0.0, FXY = 0.0;
    double P1x = 0.0,P1y = 0.0,P2x = 0.0,P2y = 0.0,P3x = 0.0,P3y = 0.0,P4x = 0.0,P4y = 0.0;
    int in_quad = 0;

    *ErrorFlag = 0;
    for (i=0;i<3;i++)
    {
        PosXYZ[i] = PosLoc[i];
        CosKLM[i] = CosLoc[i];
    }

    //Closed form solutions used for closed surfaces (could use Newton-Raphson also,but would have to
    //pick the correct starting point (i.e. the initial point itself) to converge on first intersection
    //chose closed for cylinder
    if (Element->SurfaceType == 2) // cylinder
    {
        QuadricSurfaceClosedForm(Element, PosLoc, CosLoc, PosXYZ, DFXYZ, PathLength, ErrorFlag);
        return;
    }


    // wendelin 5-26-11 chose not use closed form solution for sphere.
    // this solves for a full spheroid, but can build a full spheroid from two hemispheres with iterative solution
    if ((Element->SurfaceType == 1) && (Element->SurfaceIndex == 's' || Element->SurfaceIndex == 'S')) //sphere
    {
        QuadricSurfaceClosedForm(Element, PosLoc, CosLoc, PosXYZ, DFXYZ, PathLength, ErrorFlag);
        return;
    }

    //--------end of closed form solutions-------------
    //  {If not doing closed form solution, proceed to iterative solution}


    //start of new block for determining starting plane for Newton-Raphson   03-11-03

    /*{First, find starting plane.  The correct choice depends on the z-direction of the ray and the original
    position of the ray relative to the element surface.  First step is to find the intersection point
    of ray with  the element aperture plane and determine if it is inside or outside the aperture.
    Next, find z value of surface at x,y coords of original position.
    This determines which side of the surface equation the original position is. Then proceed through conditionals
    to determine the correct starting plane for Newton-Raphson.} */

    if (Element->ZAperture - PosXYZ[2] == 0.0) //numerical fix? 11-16-06 Tim Wendelin
        S0Aperture = 0.0;
    else
        S0Aperture = (Element->ZAperture - PosXYZ[2])/(CosKLM[2] + 0.00000000001); //numerical fix? tim wendelin 11-20-06

    x = PosXYZ[0]+CosKLM[0]*S0Aperture;               //x,y and radial position in aperture plane
    y = PosXYZ[1]+CosKLM[1]*S0Aperture;
    r = sqrt(x*x + y*y);

    //Determine if intersection point of ray with aperture plane falls inside element aperture
    SLOP60 = 1.7320508075688767; // tan(60.0*(ACOSM1O180));

    ZAInterceptInsideAperture=false;

    switch (Element->ShapeIndex)
    {
    case 'c':
    case 'C': // Circular aperture
            Ro = Element->ParameterA/2.0;
            if (r > Ro) //ray falls outsideside circular aperture
                ZAInterceptInsideAperture = false;
            else
                ZAInterceptInsideAperture = true;
        break;

    case 'h':
    case 'H': //hexagonal aperture
            Ro = Element->ParameterA/2.0;

            if (r > Ro) //ray falls outside circular circumference aperture
            {
                ZAInterceptInsideAperture = false;
                goto Label_5;
            }

            Ri = Ro*cos(30.0*(ACOSM1O180));

            if (r <= Ri) //ray falls inside inscribed circle
            {
                ZAInterceptInsideAperture = true;
                goto Label_5;
            }

            XL = sqrt(Ro*Ro - Ri*Ri); //otherwise break hexagon into 3 sections
            if (x <= Ro && x > XL)  //1st section
            {
                Y1 = SLOP60*(x-Ro);
                Y2 = -Y1;
                if (y >= Y1 && y <= Y2)
                {
                    ZAInterceptInsideAperture = true;
                    goto Label_5;
                }

                ZAInterceptInsideAperture = false;
                goto Label_5;
            }

            if (x <= XL && x >= -XL) //2nd section
            {
                if (y >= -Ri && y <= Ri)
                {
                    ZAInterceptInsideAperture = true;
                    goto Label_5;
                }

                ZAInterceptInsideAperture = false;
                goto Label_5;
            }

            if (x < -XL && x >= -Ro) //3rd section
            {
                Y3 = SLOP60*(x+Ro);
                Y4 = -Y3;
                if (y >= Y4 && y <= Y3)
                {
                    ZAInterceptInsideAperture = true;
                    goto Label_5;
                }

                ZAInterceptInsideAperture = false;
                goto Label_5;
            }
        break;

    case 't':
    case 'T': //Triangular aperture
            Ro = Element->ParameterA/2.0;

            if (r > Ro) //ray falls outside circular circumference aperture
            {
                ZAInterceptInsideAperture = false;
                goto Label_5;
            }

            Ri = Ro*sin(30.0*(ACOSM1O180));

            if (r <= Ri)  //ray falls inside inscribed circle
            {
                ZAInterceptInsideAperture = true;
                goto Label_5;
            }

            if (x <= Ro && x > 0.0) //1st section
            {
                Y1 = -SLOP60*(x-Ri/cos(30.0*(ACOSM1O180)));
                Y2 = -Ri;
                if (y <= Y1 && y >= Y2)
                    ZAInterceptInsideAperture = true;
                else
                    ZAInterceptInsideAperture = false;
                goto Label_5;
            }

            if (x >= -Ro && x <= 0.0) //2nd section
            {
                Y3 = SLOP60*(x+Ri/cos(30.0*(ACOSM1O180)));
                Y4 = -Ri;
                if (y >= Y4 && y <= Y3)
                    ZAInterceptInsideAperture = true;
                else
                    ZAInterceptInsideAperture = false;

                goto Label_5;
            }
        break;

    case 'r':
    case 'R': // Rectangular aperture

            if (x > Element->ParameterA/2.0 && x < -Element->ParameterA/2.0)
            {
                ZAInterceptInsideAperture = false;
                goto Label_5;
            }

            if (y > Element->ParameterB/2.0 && y < -Element->ParameterB/2.0)
            {
                ZAInterceptInsideAperture = false;
                goto Label_5;
            }

            ZAInterceptInsideAperture = true;
            goto Label_5;

    case 'a':
    case 'A'://Annulus

            if (r < Element->ParameterA || r > Element->ParameterB)
            {
                ZAInterceptInsideAperture = false;
                goto Label_5;
            }

            if (x >= 0.0)
            {
                if ( (asin(y/r) > Element->ParameterC*(ACOSM1O180)/2.0)
                        || (asin(y/r) < -Element->ParameterC*(ACOSM1O180)/2.0) )
                    ZAInterceptInsideAperture = false;
                else
                    ZAInterceptInsideAperture = true;
                goto Label_5;
            }

            if (x < 0.0)
            {
                if ( (y >= 0) && ((acos(y/r)+M_PI/2.0) > Element->ParameterC*(ACOSM1O180)/2.0) )
                {
                    ZAInterceptInsideAperture = false;
                    goto Label_5;
                }
                else if ((y < 0) && ((-acos(-y/r)-M_PI/2.0) < -Element->ParameterC*(ACOSM1O180)/2.0) )
                {
                    ZAInterceptInsideAperture = false;
                    goto Label_5;
                }

                ZAInterceptInsideAperture = true;
                goto Label_5;
            }

        break;

    case 'l':
    case 'L': //off axis aperture section of line focus trough  or cylinder

            if (Element->ParameterA == 0.0 && Element->ParameterB == 0.0) goto Label_4; //for cylinder, only need to check for limits on y

            if (x < Element->ParameterA || x > Element->ParameterB)
            {
                ZAInterceptInsideAperture = false;
                goto Label_5;
            }

Label_4:
            if (y < -Element->ParameterC/2.0 || y > Element->ParameterC/2.0)
            {
                ZAInterceptInsideAperture = false;
                goto Label_5;
            }

            ZAInterceptInsideAperture = true;
            goto Label_5;


    case 'i':
    case 'I': //irregular triangle
            P1x = Element->ParameterA;
            P1y = Element->ParameterB;
            P2x = Element->ParameterC;
            P2y = Element->ParameterD;
            P3x = Element->ParameterE;
            P3y = Element->ParameterF;

            if (!intri( P1x, P1y, P2x, P2y, P3x, P3y, x, y ))
            {
                ZAInterceptInsideAperture = false;
                goto Label_5;
            }

            ZAInterceptInsideAperture = true;
            goto Label_5;

    case 'q':
    case 'Q'://irregular quadrilateral

            P1x = Element->ParameterA;
            P1y = Element->ParameterB;
            P2x = Element->ParameterC;
            P2y = Element->ParameterD;
            P3x = Element->ParameterE;
            P3y = Element->ParameterF;
            P4x = Element->ParameterG;
            P4y = Element->ParameterH;

            in_quad = inquad(P1x, P1y, P2x, P2y, P3x, P3y, P4x, P4y, x, y);

            if (!in_quad)
            {
                ZAInterceptInsideAperture = false;
                goto Label_5;
            }

            ZAInterceptInsideAperture = true;
            goto Label_5;
    default:
        break;
    } // end switch


Label_5:
//    if (in_quad && !ZAInterceptInsideAperture)
//        printf("ERROR\n");

    ZStart = 0.0;    //default for all surfacetypes

    if ( Element->SurfaceType != 3
         && Element->SurfaceType != 4
         && Element->SurfaceType != 9)
    {
        SurfaceZatXYPair(PosXYZ, Element, &FXY, ErrorFlag);    //find z value of surface at x,y

        if (PosXYZ[2] <= 0.0 && CosKLM[2] > 0.0)     //if ray position below z=0 and pointing up then
        {                                                     //ZStart should be z=0 plane.
            ZStart = 0.0;
            goto Label_10;
        }

        if (PosXYZ[2] <= FXY && CosKLM[2] > 0.0)     //if ray position is below surface equation and pointing up
        {                                                //then ZStart should be z=0 plane.
            ZStart = 0.0;
            goto Label_10;
        };

        if ( PosXYZ[2] <= FXY
                && CosKLM[2] < 0.0
                && PosXYZ[2] > Element->ZAperture
                && ZAInterceptInsideAperture )
        {                                                 //if ray position is below surface equation, above the aperture
            ZStart = 0.0;                                      //plane and pointing down
            goto Label_10;                                            //and the interception point with aperture plane is inside of
        }                                                  //aperture, then ZStart should be z=0 plane.

        if (PosXYZ[2] <= FXY && CosKLM[2] < 0.0)      //if ray position is below surface equation, pointing down
        {                                                 //and hits surface below aperture plane then ZStart should be
            ZStart = Element->ZAperture;                        //aperture plane.
            goto Label_10;
        }

        if (PosXYZ[2]  > FXY && CosKLM[2] < 0.0)      //if ray position is above surface and pointing in negative z
        {                                                 //direction then ZStart should be z=0 plane
            ZStart = 0.0;
            goto Label_10;
        }

        if (PosXYZ[2] > FXY && CosKLM[2] > 0.0)
             ZStart = Element->ZAperture;  //if ray position is above the surface and
    }                                                           //pointing up then ZStart should be

Label_10:
    if (ZStart-PosXYZ[2] == 0.0)   //numerical fix? 11-16-06 Tim Wendelin
        S0 = 0.0;
    else
        S0 = (ZStart-PosXYZ[2])/(CosKLM[2] + 0.00000000001); //numerical fix? tim wendelin 11-20-06;   //SO is the pathlength from the initial ray position to the Newton-Raphson starting plane

    X1 = PosXYZ[0] + CosKLM[0]*S0;      // from this we calculate the x,y position on ZStart starting plane
    Y1 = PosXYZ[1] + CosKLM[1]*S0;

    SJ1 = 0.0;

    i = 0;
//Begin the Newton-Raphson Iteration
    while ( i++ < NumIterations)
    {
        SJ = SJ1;
        PosXYZ[0] = X1 + CosKLM[0]*SJ;
        PosXYZ[1] = Y1 + CosKLM[1]*SJ;
        PosXYZ[2] = ZStart + CosKLM[2]*SJ;

        Surface(PosXYZ, Element, &FXYZ, DFXYZ, &OKFlag);

        if (OKFlag == 0) goto Label_40;

        *ErrorFlag = 2;  //Interpolation error in Surface procedure
        goto Label_100;

Label_40:
        DFDXYZ = DOT_GPU(DFXYZ, CosKLM);
        if ( fabs(FXYZ) <= Epsilon*fabs(DFDXYZ) ) goto Label_100;

        SJ1 = SJ - FXYZ/DFDXYZ;
    }
    *ErrorFlag = 1;   //Failed to converge

Label_100:
    *PathLength = S0 + SJ;
}

__device__
void Surface(
            double PosXYZ[3],
            ElementInfo *Element,
            double *FXYZ,
            double DFXYZ[3],
            int *ErrorFlag )
{
/*{Purpose: To compute the surface equation and it's derivatives for various
geometric surfaces.
    Input - PosXYZ[3] = X, Y, Z coordinate position
            Element.SurfaceType = Surface type flag
                          = 1 for rotationally symmetric surfaces
                          = 2 for torics and cylinders
                          = 3 for plane surfaces
                          = 4 for surface interpolated from finite element data points
                          = 5 for surface interpolated from VSHOT data points
                          = 6 for surface described by Zernike monomials
                          = 7 for single axis parabolic curvature surfaces
                          = 8 for rotationally symmetric polynomial description
                          = 9 for       "          "     cubic spline interpolation
            Element.Alpha = Sensitivity coefficients which specify deviation from conic
                    of revolution
            Element.VertexCurvX = Vertex Curvature of surface
            Element.Kappa = Surface specifier
                 < 0         ==> Hyperboloid
                 = 0         ==> Paraboloid
                 > 0 and < 1 ==> Hemelipsoid of revolution about major axis
                 = 1         ==> Hemisphere
                 > 1         ==> Hemelipsoid of revolution about minor axis
            Element.ConeHalfAngle = Half-angle of cone for cones or revolution or axicons
            Element.CurvOfRev = Curvature of revolution

    Output - FXYZ = Surface equation
             DFXYZ[3] = Derivatives of surface equation
             ErrorFlag = Error Flag
                         = 0  ==> no errors
                         > 0  ==> interpolation error
}*/

    int i=0;
    double X=0.0,Y=0.0,Z=0.0;
    double Rho2=0.0;
    double DFDX=0, DFDY=0, DFDZ=0;
    double Sum1=0.0, Sum2=0.0, Term=0.0;
    double Y2=0.0, Y2J=0.0;
    double FY=0.0;

//label 100, 130,160, 850, 990;

    //Initialize variables
    X = PosXYZ[0];
    Y = PosXYZ[1];
    Z = PosXYZ[2];
    *ErrorFlag = 0;

    //===SurfaceType = 1, 7  Rotationally Symmetric surfaces and single axis curvature sections===========================
    if (Element->SurfaceType == 1 || Element->SurfaceType == 7)
    {
        if (Element->SurfaceType == 1)
            Rho2 = X*X + Y*Y;    //rotationally symmetric
        else
            Rho2 = X*X;         //single axis curvature depends only on x

        if (Element->ConeHalfAngle != 0.0)
            goto Label_160;

        //wendelin 5-18-11 changes to allow different vertex curvature in the x and y directions for the parabola; this block of code
        //is a subset of the more general form below therefore it has been commented out.  It also assumes VertexCurvY = either VertexCurvX or zero
        //and doesn't allow different nonzero values for the parabolic case.   Not using the alpha parameters for the general case for now.
/*
        for (i=0;i<5;i++)
            if (Element->Alpha[i] != 0.0)
                goto Label_130;

        *FXYZ = Z - 0.5*Element->VertexCurvX*(Rho2 + Element->Kappa*Z*Z);
        DFDX = -Element->VertexCurvX*X;
        DFDY = -Element->VertexCurvY*Y; //VertexCurvY = VertexCurvX if rotationally symmetric or 0 if single axis curved
        DFDZ = 1.0 - Element->Kappa*Element->VertexCurvX*Z;
        goto Label_990;
*/
        Sum1 = 0.0;
        Sum2 = 0.0;

        // wendelin 5-18-11
        /*
        for (i=0;i<5;i++)
        {
            Sum1 = i*Element->Alpha[i]*Rho2i + Sum1;
            Rho2i = Rho2i*Rho2;
            Sum2 = Element->Alpha[i]*Rho2i + Sum2;
        }*/

        //wendelin 5-18-11 changes to allow different vertex curvature in the x and y directions for the parabola only
        // Term = sqrt(1.0 - Element->Kappa*Element->VertexCurvX*Element->VertexCurvX*Rho2);
        Term = sqrt(1.0 - Element->Kappa*(Element->VertexCurvX*Element->VertexCurvX*X*X+Element->VertexCurvY*Element->VertexCurvY*Y*Y));   //new
        //*FXYZ = Z - Element->VertexCurvX*Rho2/(1.0 + Term) - Sum2;
        *FXYZ = Z - (Element->VertexCurvX*X*X+Element->VertexCurvY*Y*Y)/(1.0 + Term) - Sum2;   //new

        DFDX = -X*(Element->VertexCurvX/Term + 2.0*Sum1);
        DFDY = -Y*(Element->VertexCurvY/Term + 2.0*Sum1); //VertexCurvY = VertexCurvX if rotationally symmetric or 0 if single axis curved
        DFDZ = 1.0;
        goto Label_990;

Label_160:
        *FXYZ = Z - sqrt(Rho2)/tan(Element->ConeHalfAngle*(ACOSM1O180));
        DFDX = -X/(sqrt(Rho2)*tan(Element->ConeHalfAngle*(ACOSM1O180)));
        DFDY = -Y/(sqrt(Rho2)*tan(Element->ConeHalfAngle*(ACOSM1O180)));
        DFDZ = 1.0;
        goto Label_990;
    }

//===SurfaceType = 2, Toroidal or Cylindrical surfaces========================== //not currently used
    if (Element->SurfaceType == 2)
    {
        Sum1 = 0.0;
        Sum2 = 0.0;
        Y2 = Y*Y;
        Y2J = 1.0;

        for (i=0;i<5;i++)
        {
            Sum1 = i*Element->Alpha[i]*Y2J*Y + Sum1;
            Y2J = Y2J*Y2;
            Sum2 = Element->Alpha[i]*Y2J + Sum2;
        }

        Term = sqrt(1.0 - Element->Kappa*Element->VertexCurvX*Element->VertexCurvX*Y2);
        FY = Element->VertexCurvX*Y2/(1.0 + Term) + Sum2;
        *FXYZ = Z - FY - 0.5*Element->CurvOfRev*(X*X + Z*Z - FY*FY);
        DFDX = -Element->CurvOfRev*X;
        DFDY = (Element->CurvOfRev*FY - 1.0)*(Element->VertexCurvX*Y/Term + 2.0*Sum1);
        DFDZ = 1.0 - Element->CurvOfRev*Z;
        goto Label_990;
    }

//===SurfaceType = 3, Plane Surfaces============================================
/*     {The equation of a plane is: kx + ly + mz = p,  where k,l,m are the direction
     cosines of the normal to the plane, and p is the distance from the origin
     to the plane.  In this case, these parameters are contained in the Alpha array.}*/
    if (Element->SurfaceType == 3)
    {
        DFDX = Element->Alpha[0];
        DFDY = Element->Alpha[1];
        DFDZ = Element->Alpha[2];
        *FXYZ = DFDX*X + DFDY*Y + DFDZ*Z - Element->Alpha[3];
        goto Label_990;
    }

//{****Offload derivatives of the surface equation}
Label_990:
    DFXYZ[0] = DFDX;
    DFXYZ[1] = DFDY;
    DFXYZ[2] = DFDZ;

}
//end of procedure--------------------------------------------------------------
