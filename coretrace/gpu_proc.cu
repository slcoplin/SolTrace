
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

#include "types.h"
#include "gpu_proc.cuh"

__device__
void MatrixVectorMultGPU(double M[9], double V[3], double MxV[3])
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
void TransformToLocalGPU(double PosRef[3], double CosRef[3], double Origin[3],
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

    MatrixVectorMultGPU(RRefToLoc, PosDum, PosLoc);
    MatrixVectorMultGPU(RRefToLoc, CosRef, CosLoc);
}
//end of procedure--------------------------------------------------------------

__device__
void TransformToReferenceGPU(double PosLoc[3], double CosLoc[3], double Origin[3],
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

    /*{Use previously calculated RLocToRef matrix (in TransformToLocal) to obtain the
     inverse transformation back to Reference system.}*/
    MatrixVectorMultGPU(RLocToRef, PosLoc, PosDum);
    MatrixVectorMultGPU(RLocToRef, CosLoc, CosRef);

    for (int i = 0; i < 3; i++)
        PosRef[i] = PosDum[i] + Origin[i];
}

