#include "types.h"

/*
 * Fills IncomingRays with NumberOfRays random rays,
 * with distribution given by the System.
 * Generate rays for a given system
 * @param System       A System. Stage 0 has sun. Defines Sun.
 * @param IncomingRays An allocated array of rays (in global position).
 *                     Modified to contain generated rays.
 * @param NumberOfRays Number of rays to generate.
 *                     IncomingRays must have space for NumberOfRays
 */
void generate_rays(TSystem *System,
    GlobalRay *IncomingRays,
    st_uint_t NumberOfRays);