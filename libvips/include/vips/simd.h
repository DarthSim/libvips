#ifndef VIPS_SIMD_H
#define VIPS_SIMD_H

#ifdef __cplusplus
extern "C" {
#endif /*__cplusplus*/

VIPS_API
void vips_simd_init(void);
VIPS_API
gboolean vips_simd_isenabled(void);
VIPS_API
void vips_simd_set_enabled(gboolean enabled);
VIPS_API
gboolean vips_simd_avx2_issupported(void);

#ifdef __cplusplus
}
#endif /*__cplusplus*/

#endif /*VIPS_SIMD_H*/
