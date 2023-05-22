#ifndef VIPS_SIMD_NEON_H
#define VIPS_SIMD_NEON_H

#ifdef __cplusplus
extern "C" {
#endif /*__cplusplus*/

#include <glib/gi18n-lib.h>
#include <arm_neon.h>

#define VIPS_NEON_INLINE \
	static inline __attribute__((__gnu_inline__, __always_inline__))

typedef int32x4_t VipsSimdInt32x4;
typedef float32x4_t VipsSimdFloat32x4;
typedef float64x2_t VipsSimdFloat64x2;

VIPS_NEON_INLINE VipsSimdInt32x4
vips_simd_new_int32x4( 
	const int32_t v1, const int32_t v2, const int32_t v3, const int32_t v4 )
{
	return (int32x4_t){ v1, v2, v3, v4};
}

VIPS_NEON_INLINE VipsSimdInt32x4
vips_simd_new_int32x4_const1( const int32_t v )
{
	return vdupq_n_s32( v );
}

VIPS_NEON_INLINE VipsSimdFloat32x4
vips_simd_new_float32x4( 
	const float32_t v1, const float32_t v2,
	const float32_t v3, const float32_t v4 )
{
	return (float32x4_t){ v1, v2, v3, v4};
}

VIPS_NEON_INLINE VipsSimdFloat32x4
vips_simd_new_float32x4_const1( const float32_t v )
{
	return vdupq_n_f32( v );
}

VIPS_NEON_INLINE VipsSimdFloat64x2
vips_simd_new_float64x2( const float64_t v1, const float64_t v2 )
{
	return (float64x2_t){ v1, v2 };
}

VIPS_NEON_INLINE VipsSimdFloat64x2
vips_simd_new_float64x2_const1( const float64_t v )
{
	return vdupq_n_f64( v );
}

VIPS_NEON_INLINE VipsSimdInt32x4
vips_simd_zero_int32x4()
{
	return vdupq_n_s32( 0 );
}

VIPS_NEON_INLINE VipsSimdFloat32x4
vips_simd_zero_float32x4()
{
	return vdupq_n_f32( 0 );
}

VIPS_NEON_INLINE VipsSimdFloat64x2
vips_simd_zero_float64x2()
{
	return vdupq_n_f64( 0 );
}

VIPS_NEON_INLINE VipsSimdInt32x4
vips_simd_load_int32x4( const int32_t *ptr )
{
	return vld1q_s32( ptr );
}

/* Loads four chars or shorts to VipsSimdInt32x4.
 * Set `size` to 1 for chars or 2 for shorts.
 * Set `issigned` to TRUE for signed formats.
 */
VIPS_NEON_INLINE VipsSimdInt32x4
vips_simd_load_cvt_int32x4( const void *ptr, size_t size, int issigned )
{
	g_assert( size == 1 || size == 2 );

	if( size == 1 ) {
		const uint32x2_t a = vdup_n_u32( *(int32_t *) ptr );

		if( issigned ) {
			const int8x8_t b = vreinterpret_s8_u32( a );
			const int16x4_t c = vget_low_s16( vmovl_s8( b ) );
			return vmovl_s16( c );
		} else {
			const uint8x8_t b = vreinterpret_u8_u32( a );
			const uint16x4_t c = vget_low_u16( vmovl_u8( b ) );
			return vreinterpretq_s32_u32( vmovl_u16( c ) );
		}
	} else {
		if( issigned )
			return vmovl_s16( vld1_s16( (const int16_t *) ptr ) );
		else
			return vreinterpretq_s32_u32( vmovl_u16( vld1_u16(
				(const uint16_t *) ptr ) ) );
	}
}

VIPS_NEON_INLINE VipsSimdFloat32x4
vips_simd_load_float32x4( const float32_t *ptr )
{
	return vld1q_f32( ptr );
}

VIPS_NEON_INLINE VipsSimdFloat64x2
vips_simd_load_float64x2( const float64_t *ptr )
{
	return vld1q_f64( ptr );
}

VIPS_NEON_INLINE void
vips_simd_store_int32x4( int32_t *ptr, const VipsSimdInt32x4 a )
{
	return vst1q_s32( ptr, a );
}

/* Stores VipsSimdInt32x4 as four chars or shorts.
 * Set `size` to 1 for chars or 2 for shorts.
 * Set `issigned` to TRUE for signed formats.
 */
VIPS_NEON_INLINE void
vips_simd_store_cvt_int32x4(
	const void *ptr, const VipsSimdInt32x4 a, size_t size, int issigned )
{
	g_assert( size == 1 || size == 2 );

	if( size == 1 ) {
		const int16x4_t b = vqmovn_s32( a );

		if ( issigned ) {
			const int8x8_t c = vqmovn_s16( vcombine_s16( b, b ) );
			*(int32_t *) ptr = vget_lane_s32(
				vreinterpret_s32_s8( c ), 0 );
		} else {
			const uint8x8_t c = vqmovun_s16( vcombine_s16( b, b ) );
			*(uint32_t *) ptr = vget_lane_u32(
				vreinterpret_u32_u8( c ), 0 );
		}
	} else {
		if(issigned )
			vst1_s16( (int16_t *) ptr, vqmovn_s32( a ) );
		else
			vst1_u16( (uint16_t *) ptr, vqmovun_s32( a ) );
	}
}

VIPS_NEON_INLINE void
vips_simd_store_float32x4( float32_t *ptr, const VipsSimdFloat32x4 a )
{
	return vst1q_f32( ptr, a );
}

VIPS_NEON_INLINE void
vips_simd_store_float64x2( float64_t *ptr, const VipsSimdFloat64x2 a )
{
	return vst1q_f64( ptr, a );
}

VIPS_NEON_INLINE VipsSimdInt32x4
vips_simd_add_int32x4( const VipsSimdInt32x4 a, const VipsSimdInt32x4 b )
{
	return vaddq_s32( a, b );
}

VIPS_NEON_INLINE VipsSimdFloat32x4
vips_simd_add_float32x4( const VipsSimdFloat32x4 a, const VipsSimdFloat32x4 b )
{
	return vaddq_f32( a, b );
}

VIPS_NEON_INLINE VipsSimdFloat64x2
vips_simd_add_float64x2( const VipsSimdFloat64x2 a, const VipsSimdFloat64x2 b )
{
	return vaddq_f64( a, b );
}

VIPS_NEON_INLINE VipsSimdInt32x4
vips_simd_mul_int32x4( const VipsSimdInt32x4 a, const VipsSimdInt32x4 b )
{
	return vmulq_s32( a, b );
}

VIPS_NEON_INLINE VipsSimdFloat32x4
vips_simd_mul_float32x4( const VipsSimdFloat32x4 a, const VipsSimdFloat32x4 b )
{
	return vmulq_f32( a, b );
}

VIPS_NEON_INLINE VipsSimdFloat64x2
vips_simd_mul_float64x2( const VipsSimdFloat64x2 a, const VipsSimdFloat64x2 b )
{
	return vmulq_f64( a, b );
}

VIPS_NEON_INLINE VipsSimdInt32x4
vips_simd_muladd_int32x4(
	const VipsSimdInt32x4 sum, const VipsSimdInt32x4 a,
	const VipsSimdInt32x4 b )
{
	return vmlaq_s32( sum, a, b );
}

VIPS_NEON_INLINE VipsSimdFloat32x4
vips_simd_muladd_float32x4(
	const VipsSimdFloat32x4 sum, const VipsSimdFloat32x4 a,
	const VipsSimdFloat32x4 b )
{
	return vmlaq_f32( sum, a, b );
}

VIPS_NEON_INLINE VipsSimdFloat64x2
vips_simd_muladd_float64x2(
	const VipsSimdFloat64x2 sum, const VipsSimdFloat64x2 a,
	const VipsSimdFloat64x2 b )
{
	return vmlaq_f64( sum, a, b );
}

VIPS_NEON_INLINE VipsSimdInt32x4
vips_simd_muladd_int32x4_const1(
	const VipsSimdInt32x4 sum, const VipsSimdInt32x4 a, const int32_t b )
{
	return vmlaq_n_s32( sum, a, b );
}

VIPS_NEON_INLINE VipsSimdFloat32x4
vips_simd_muladd_float32x4_const1(
	const VipsSimdFloat32x4 sum, const VipsSimdFloat32x4 a,
	const float32_t b )
{
	return vmlaq_n_f32( sum, a, b );
}

VIPS_NEON_INLINE VipsSimdFloat64x2
vips_simd_muladd_float64x2_const1(
	const VipsSimdFloat64x2 sum, const VipsSimdFloat64x2 a,
	const float64_t b )
{
	return vmlaq_f64( sum, a, vdupq_n_f64(b) );
}

VIPS_NEON_INLINE VipsSimdInt32x4
vips_simd_shr_int32x4( const VipsSimdInt32x4 a, const int32_t b )
{
	return vshrq_n_s32( a, b );
}

VIPS_NEON_INLINE VipsSimdInt32x4
vips_simd_shl_int32x4( const VipsSimdInt32x4 a, const int32_t b )
{
	return vshlq_n_s32( a, b );
}

VIPS_NEON_INLINE VipsSimdInt32x4
vips_simd_clip_int32x4(
	const VipsSimdInt32x4 min, const VipsSimdInt32x4 a,
	const VipsSimdInt32x4 max )
{
	return vmaxq_s32( min, vminq_s32(max, a) );
}

VIPS_NEON_INLINE VipsSimdFloat32x4
vips_simd_clip_float32x4(
	const VipsSimdFloat32x4 min, const VipsSimdFloat32x4 a,
	const VipsSimdFloat32x4 max )
{
	return vmaxq_f32( min, vminq_f32(max, a) );
}

VIPS_NEON_INLINE VipsSimdFloat64x2
vips_simd_clip_float64x2(
	const VipsSimdFloat64x2 min, const VipsSimdFloat64x2 a,
	const VipsSimdFloat64x2 max )
{
	return vmaxq_f64( min, vminq_f64(max, a) );
}

#ifdef __cplusplus
}
#endif /*__cplusplus*/

#endif  /*VIPS_SIMD_NEON_H*/
