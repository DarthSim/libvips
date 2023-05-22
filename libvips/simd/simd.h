#ifndef VIPS_SIMD_H
#define VIPS_SIMD_H

#ifdef __cplusplus
extern "C" {
#endif /*__cplusplus*/

#if HAVE_SSE

#define HAVE_SIMD 1
#include "simd_sse.h"

#elif HAVE_NEON

#define HAVE_SIMD 1
#include "simd_neon.h"

#else

#define HAVE_SIMD 0

#endif

#ifdef __cplusplus
}
#endif /*__cplusplus*/

#endif  /*VIPS_SIMD_H*/
