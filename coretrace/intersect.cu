
#include <stdlib.h>
#include <math.h>

#include "types.h"
#include "gpu_proc.cuh"

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

    // Compute dot products
    dot00 = dot(v0, v0)
    dot01 = dot(v0, v1)
    dot02 = dot(v0, v2)
    dot11 = dot(v1, v1)
    dot12 = dot(v1, v2)

    // Compute barycentric coordinates
    invDenom = 1 / (dot00 * dot11 - dot01 * dot01)
    u = (dot11 * dot02 - dot01 * dot12) * invDenom
    v = (dot00 * dot12 - dot01 * dot02) * invDenom

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

            /*
            //Side 1
            Tn = (P2y-P1y)*(P1x-x)-(P2x-P1x)*(P1y-y);
            if (Tn < 0.0)
            {
                ZAInterceptInsideAperture = false;
                goto Label_5;
            }

            //side 2
            Tn = (P3y-P2y)*(P2x-x)-(P3x-P2x)*(P2y-y);
            if (Tn < 0.0)
            {
                ZAInterceptInsideAperture = false;
                goto Label_5;
            }

            //side 3
            Tn = (P1y-P3y)*(P3x-x)-(P1x-P3x)*(P3y-y);
            if (Tn < 0.0)
            {
                ZAInterceptInsideAperture = false;
                goto Label_5;
            }
            */

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

            /*
            //Side 1
            Tn = (P2y-P1y)*(P1x-x)-(P2x-P1x)*(P1y-y);
            if (Tn < 0.0)
            {
                ZAInterceptInsideAperture = false;
                goto Label_5;
            }

            //side 2
            Tn = (P3y-P2y)*(P2x-x)-(P3x-P2x)*(P2y-y);
            if (Tn < 0.0)
            {
                ZAInterceptInsideAperture = false;
                goto Label_5;
            }

            //side 3
            Tn = (P4y-P3y)*(P3x-x)-(P4x-P3x)*(P3y-y);
            if (Tn < 0.0)
            {
                ZAInterceptInsideAperture = false;
                goto Label_5;
            }

            //side 4
            Tn =  (P1y-P4y)*(P4x-x)-(P1x-P4x)*(P4y-y);
            if (Tn < 0.0)
            {
                ZAInterceptInsideAperture = false;
                goto Label_5;
            } */

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
