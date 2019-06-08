#include "types.h"

__device__
void MatrixVectorMultGPU(double M[9], double V[3], double MxV[3]);

__device__
void TransformToLocalGPU(double PosRef[3], double CosRef[3], double Origin[3],
    double RRefToLoc[9],
    double PosLoc[3], double CosLoc[3]);

__device__
void TransformToReferenceGPU(double PosLoc[3], double CosLoc[3], double Origin[3],
    double RLocToRef[9],
    double PosRef[3], double CosRef[3]);