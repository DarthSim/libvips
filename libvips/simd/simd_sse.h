#ifndef VIPS_SIMD_SSE_H
#define VIPS_SIMD_SSE_H

#ifdef __cplusplus
extern "C" {
#endif /*__cplusplus*/

#include <glib/gi18n-lib.h>
#include <smmintrin.h>

#ifndef float32_t
typedef float float32_t;
#endif
#ifndef float64_t
typedef double float64_t;
#endif

#define VIPS_SSE_INLINE \
	static inline __attribute__((__gnu_inline__, __always_inline__))

typedef __m128i VipsSimdInt32x4;
typedef __m128 VipsSimdFloat32x4;
typedef __m128d VipsSimdFloat64x2;

VIPS_SSE_INLINE VipsSimdInt32x4
vips_simd_new_int32x4( 
	const int32_t v1, const int32_t v2, const int32_t v3, const int32_t v4 )
{
	return _mm_set_epi32( v4, v3, v2, v1 );
}

VIPS_SSE_INLINE VipsSimdInt32x4
vips_simd_new_int32x4_const1( const int32_t v )
{
	return _mm_set1_epi32( v );
}

VIPS_SSE_INLINE VipsSimdFloat32x4
vips_simd_new_float32x4( 
	const float32_t v1, const float32_t v2,
	const float32_t v3, const float32_t v4 )
{
	return _mm_set_ps( v4, v3, v2, v1 );
}

VIPS_SSE_INLINE VipsSimdFloat32x4
vips_simd_new_float32x4_const1( const float32_t v )
{
	return _mm_set1_ps( v );
}

VIPS_SSE_INLINE VipsSimdFloat64x2
vips_simd_new_float64x2( const float64_t v1, const float64_t v2 )
{
	return _mm_set_pd( v2, v1 );
}

VIPS_SSE_INLINE VipsSimdFloat64x2
vips_simd_new_float64x2_const1( const float64_t v )
{
	return _mm_set1_pd( v );
}

VIPS_SSE_INLINE VipsSimdInt32x4
vips_simd_zero_int32x4()
{
	return _mm_setzero_si128();
}

VIPS_SSE_INLINE VipsSimdFloat32x4
vips_simd_zero_float32x4()
{
	return _mm_setzero_ps();
}

VIPS_SSE_INLINE VipsSimdFloat64x2
vips_simd_zero_float64x2()
{
	return _mm_setzero_pd();
}

VIPS_SSE_INLINE VipsSimdInt32x4
vips_simd_load_int32x4( const int32_t *ptr )
{
	return _mm_loadu_si128( (const __m128i *) ptr );
}

/* Loads four chars or shorts to VipsSimdInt32x4.
 * Set `size` to 1 for chars or 2 for shorts.
 * Set `issigned` to TRUE for signed formats.
 */
VIPS_SSE_INLINE VipsSimdInt32x4
vips_simd_load_cvt_int32x4( const void *ptr, size_t size, int issigned )
{
	g_assert( size == 1 || size == 2 );

	if( size == 1 ) {
		if( issigned )
			return _mm_cvtepi8_epi32(
				_mm_cvtsi32_si128( *(int32_t *) ptr ) );
		else
			return _mm_cvtepu8_epi32(
				_mm_cvtsi32_si128( *(int32_t *) ptr ) );
	} else {
		if( issigned )
			return _mm_cvtepi16_epi32(
				_mm_loadu_si64( ptr ) );
		else
			return _mm_cvtepu16_epi32(
				_mm_loadu_si64( ptr ) );
	}
}

VIPS_SSE_INLINE VipsSimdFloat32x4
vips_simd_load_float32x4( const float32_t *ptr )
{
	return _mm_loadu_ps( ptr );
}

VIPS_SSE_INLINE VipsSimdFloat64x2
vips_simd_load_float64x2( const float64_t *ptr )
{
	return _mm_loadu_pd( ptr );
}

VIPS_SSE_INLINE void
vips_simd_store_int32x4( int32_t *ptr, const VipsSimdInt32x4 a )
{
	return _mm_storeu_si128( (__m128i *) ptr, a );
}

/* Stores VipsSimdInt32x4 as four chars or shorts.
 * Set `size` to 1 for chars or 2 for shorts.
 * Set `issigned` to TRUE for signed formats.
 */
VIPS_SSE_INLINE void
vips_simd_store_cvt_int32x4(
	const void *ptr, const VipsSimdInt32x4 a, size_t size, int issigned )
{
	g_assert( size == 1 || size == 2 );

	if( size == 1 ) {
		const __m128i b = _mm_packs_epi32( a, a );
		const __m128i c = issigned ?
			_mm_packs_epi16( b, b ) : _mm_packus_epi16( b, b );
		*(int32_t*) ptr = _mm_cvtsi128_si32( c );
	} else {
		const __m128i b = issigned ?
			_mm_packs_epi32( a, a ) : _mm_packus_epi32( a, a );
		*(int64_t*) ptr = _mm_cvtsi128_si64( b );
	}
}

VIPS_SSE_INLINE void
vips_simd_store_float32x4( float32_t *ptr, const VipsSimdFloat32x4 a )
{
	return _mm_storeu_ps( ptr, a );
}

VIPS_SSE_INLINE void
vips_simd_store_float64x2( float64_t *ptr, const VipsSimdFloat64x2 a )
{
	return _mm_storeu_pd( ptr, a );
}

VIPS_SSE_INLINE VipsSimdInt32x4
vips_simd_add_int32x4( const VipsSimdInt32x4 a, const VipsSimdInt32x4 b )
{
	return _mm_add_epi32( a, b );
}

VIPS_SSE_INLINE VipsSimdFloat32x4
vips_simd_add_float32x4( const VipsSimdFloat32x4 a, const VipsSimdFloat32x4 b )
{
	return _mm_add_ps( a, b );
}

VIPS_SSE_INLINE VipsSimdFloat64x2
vips_simd_add_float64x2( const VipsSimdFloat64x2 a, const VipsSimdFloat64x2 b )
{
	return _mm_add_pd( a, b );
}

VIPS_SSE_INLINE VipsSimdInt32x4
vips_simd_mul_int32x4( const VipsSimdInt32x4 a, const VipsSimdInt32x4 b )
{
	return _mm_mullo_epi32( a, b );
}

VIPS_SSE_INLINE VipsSimdFloat32x4
vips_simd_mul_float32x4( const VipsSimdFloat32x4 a, const VipsSimdFloat32x4 b )
{
	return _mm_mul_ps( a, b );
}

VIPS_SSE_INLINE VipsSimdFloat64x2
vips_simd_mul_float64x2( const VipsSimdFloat64x2 a, const VipsSimdFloat64x2 b )
{
	return _mm_mul_pd( a, b );
}

VIPS_SSE_INLINE VipsSimdInt32x4
vips_simd_muladd_int32x4(
	const VipsSimdInt32x4 sum, const VipsSimdInt32x4 a,
	const VipsSimdInt32x4 b )
{
	return _mm_add_epi32( sum, _mm_mullo_epi32( a, b ) );
}

VIPS_SSE_INLINE VipsSimdFloat32x4
vips_simd_muladd_float32x4(
	const VipsSimdFloat32x4 sum, const VipsSimdFloat32x4 a,
	const VipsSimdFloat32x4 b )
{
	return _mm_add_ps( sum, _mm_mul_ps( a, b ) );
}

VIPS_SSE_INLINE VipsSimdFloat64x2
vips_simd_muladd_float64x2(
	const VipsSimdFloat64x2 sum, const VipsSimdFloat64x2 a,
	const VipsSimdFloat64x2 b )
{
	return _mm_add_pd( sum, _mm_mul_pd( a, b ) );
}

VIPS_SSE_INLINE VipsSimdInt32x4
vips_simd_muladd_int32x4_const1(
	const VipsSimdInt32x4 sum, const VipsSimdInt32x4 a, const int32_t b )
{
	return _mm_add_epi32( sum, _mm_mullo_epi32( a, _mm_set1_epi32( b ) ) );
}

VIPS_SSE_INLINE VipsSimdFloat32x4
vips_simd_muladd_float32x4_const1(
	const VipsSimdFloat32x4 sum, const VipsSimdFloat32x4 a,
	const float32_t b )
{
	return _mm_add_ps( sum, _mm_mul_ps( a, _mm_set1_ps( b ) ) );
}

VIPS_SSE_INLINE VipsSimdFloat64x2
vips_simd_muladd_float64x2_const1(
	const VipsSimdFloat64x2 sum, const VipsSimdFloat64x2 a,
	const float64_t b )
{
	return _mm_add_pd( sum, _mm_mul_pd( a, _mm_set1_pd( b ) ) );
}

VIPS_SSE_INLINE VipsSimdInt32x4
vips_simd_shr_int32x4( const VipsSimdInt32x4 a, const int32_t b )
{
	return _mm_srai_epi32( a, b );
}

VIPS_SSE_INLINE VipsSimdInt32x4
vips_simd_shl_int32x4( const VipsSimdInt32x4 a, const int32_t b )
{
	return _mm_slli_epi32( a, b );
}

VIPS_SSE_INLINE VipsSimdInt32x4
vips_simd_clip_int32x4(
	const VipsSimdInt32x4 min, const VipsSimdInt32x4 a,
	const VipsSimdInt32x4 max )
{
	return _mm_max_epi32( min, _mm_min_epi32(max, a) );
}

VIPS_SSE_INLINE VipsSimdFloat32x4
vips_simd_clip_float32x4(
	const VipsSimdFloat32x4 min, const VipsSimdFloat32x4 a,
	const VipsSimdFloat32x4 max )
{
	return _mm_max_ps( min, _mm_min_ps(max, a) );
}

VIPS_SSE_INLINE VipsSimdFloat64x2
vips_simd_clip_float64x2(
	const VipsSimdFloat64x2 min, const VipsSimdFloat64x2 a,
	const VipsSimdFloat64x2 max )
{
	return _mm_max_pd( min, _mm_min_pd(max, a) );
}

#ifdef __cplusplus
}
#endif /*__cplusplus*/

#endif  /*VIPS_SIMD_SSE_H*/
