
#include <math.h>

#include "types.h"
#include "gpu_proc.cuh"

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
    int i=0;
    double X=0.0,Y=0.0,Z=0.0;
    double Rho2=0.0, Rho=0.0;
    double Sum1=0.0, ZZ=0.0,  zm=0.0;

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

        Rho = sqrt(Rho2);

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
