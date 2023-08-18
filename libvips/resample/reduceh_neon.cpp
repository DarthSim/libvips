#ifdef HAVE_CONFIG_H
#include <config.h>
#endif /*HAVE_CONFIG_H*/

#include <stdlib.h>

#include <vips/vips.h>
#include <vips/internal.h>

#include "presample.h"

#if HAVE_NEON

#include <arm_neon.h>

#define VMADDQ_S16(P, C) vpaddq_s32( \
	vmull_s16( vget_low_s16( P ), C ), \
	vmull_s16( vget_high_s16( P ), C ) )

void
reduceh_uchar_simd_4bands( VipsPel *pout, VipsPel *pin, int32_t n_point,
	int32_t width, int16_t * restrict c, int32_t * restrict bounds )
{
	int32_t x, i, c32;
	uint32_t p32;

	const int32x4_t initial = vdupq_n_s32( VIPS_INTERPOLATE_SCALE >> 1 );

	//  0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15
	// r0 g0 b0 a0 r1 g1 b1 a1 r2 g2 b2 a2 r3 g3 b3 a3
	const uint8x16_t tbl16_lo = {
		0, 16, 4, 16, 1, 16, 5, 16,
		2, 16, 6, 16, 3, 16, 7, 16 };
	const uint8x16_t tbl16_hi = {
		 8, 16, 12, 16,  9, 16, 13, 16,
		10, 16, 14, 16, 11, 16, 15, 16 };
	const uint8x8_t tbl8 = { 0, 4, 1, 5, 2, 6, 3, 7 };
	const uint8x8_t tbl4 = { 0, 8, 1, 8, 2, 8, 3, 8 };

	for( x = 0; x < width; x++ ) {
		const int left = bounds[0];
		const int right = bounds[1];
		const int32_t n = right - left;

		uint8_t * restrict p = pin + left * 4;
		uint8_t * restrict q = pout + x * 4;

		uint8x16_t line16;
		uint8x8_t line8;
		int16x8_t pix;
		int16x4_t pix4, vc, vc_lo, vc_hi;

		int32x4_t sum = initial;

		i = 0;

		for( ; i <= n - 4; i += 4 ) {
			/* Load four coeffs
			 */
			memcpy(&c32, &c[i], sizeof(int32_t));
			vc_lo = vreinterpret_s16_s32( vdup_n_s32( c32 ) );
			memcpy(&c32, &c[i + 2], sizeof(int32_t));
			vc_hi = vreinterpret_s16_s32( vdup_n_s32( c32 ) );

			line16 = vld1q_u8( p );
			p += 16;

			pix = vreinterpretq_s16_u8(
				vqtbl1q_u8( line16, tbl16_lo ) );
			sum = vaddq_s32( sum, VMADDQ_S16( pix, vc_lo ) );

			pix = vreinterpretq_s16_u8(
				vqtbl1q_u8( line16, tbl16_hi ) );
			sum = vaddq_s32( sum, VMADDQ_S16( pix, vc_hi ) );
		}

		for( ; i <= n - 2; i += 2 ) {
			memcpy(&c32, &c[i], sizeof(int32_t));
			vc = vreinterpret_s16_s32( vdup_n_s32( c32 ) );

			line8 = vld1_u8( p );
			p += 8;

			pix = vreinterpretq_s16_u16(
				vmovl_u8( vtbl1_u8( line8, tbl8 ) ) );
			sum = vaddq_s32( sum, VMADDQ_S16( pix, vc ) );
		}

		for( ; i < n; i++ ) {
			vc = vdup_n_s16( c[i] );

			memcpy(&p32, p, sizeof(uint32_t));
			line8 = vreinterpret_u8_u32( vdup_n_u32( p32 ) );
			p += 4;

			pix4 = vreinterpret_s16_u8( vtbl1_u8( line8, tbl4 ) );

			sum = vaddq_s32( sum, vmulq_s32(
				vmovl_s16( pix4 ), vmovl_s16( vc ) ) );
		}

		sum = vshrq_n_s32( sum, VIPS_INTERPOLATE_SHIFT );
		
		const int16x4_t sum_16 = vqmovn_s32( sum );
		const uint8x8_t sum_8 = vqmovun_s16(
			vcombine_s16( sum_16, sum_16 ) );
		*(uint32_t *) q = vget_lane_u32(
			vreinterpret_u32_u8( sum_8 ), 0 );

		c += n_point;
		bounds += 2;
	}
}

void
reduceh_uchar_simd_3bands( VipsPel *pout, VipsPel *pin, int32_t n_point,
	int32_t width, int16_t * restrict c, int32_t * restrict bounds )
{
	int32_t x, i, c32;
	uint32_t p32;

	const int32x4_t initial = vdupq_n_s32( VIPS_INTERPOLATE_SCALE >> 1 );

	//  0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15
	// r0 g0 b0 r1 g1 b1 r2 g2 b2 r3 g3 b3
	const uint8x16_t tbl16_lo = {
		0, 16, 3, 16,  1, 16,  4, 16,
		2, 16, 5, 16, 16, 16, 16, 16 };
	const uint8x16_t tbl16_hi = {
		6, 16,  9, 16,  7, 16, 10, 16,
		8, 16, 11, 16, 16, 16, 16, 16 };
	const uint8x8_t tbl8 = { 0, 3, 1, 4, 2, 5, 8, 8 };
	const uint8x8_t tbl4 = { 0, 8, 1, 8, 2, 8, 8, 8 };

	x = 0;

	/* We need to load 4-byte aligned groups but we have a 3-band image.
	 * So when we load or save data, we load or save a few redundant bytes.
	 * We can safely do it until width-1 since we won't get out of
	 * buffers range.
	 */
	for( ; x < width - 1; x++ ) {
		const int left = bounds[0];
		const int right = bounds[1];
		const int32_t n = right - left;

		uint8_t * restrict p = pin + left * 3;
		uint8_t * restrict q = pout + x * 3;

		uint8x16_t line16;
		uint8x8_t line8;
		int16x8_t pix;
		int16x4_t pix4, vc, vc_lo, vc_hi;

		int32x4_t sum = initial;

		i = 0;

		for( ; i <= n - 4; i += 4 ) {
			/* Load four coeffs
			 */
			memcpy(&c32, &c[i], sizeof(int32_t));
			vc_lo = vreinterpret_s16_s32( vdup_n_s32( c32 ) );
			memcpy(&c32, &c[i + 2], sizeof(int32_t));
			vc_hi = vreinterpret_s16_s32( vdup_n_s32( c32 ) );

			line16 = vld1q_u8( p );
			p += 12;

			pix = vreinterpretq_s16_u8(
				vqtbl1q_u8( line16, tbl16_lo ) );
			sum = vaddq_s32( sum, VMADDQ_S16( pix, vc_lo ) );

			pix = vreinterpretq_s16_u8(
				vqtbl1q_u8( line16, tbl16_hi ) );
			sum = vaddq_s32( sum, VMADDQ_S16( pix, vc_hi ) );
		}

		for( ; i <= n - 2; i += 2 ) {
			memcpy(&c32, &c[i], sizeof(int32_t));
			vc = vreinterpret_s16_s32( vdup_n_s32( c32 ) );

			line8 = vld1_u8( p );
			p += 6;

			pix = vreinterpretq_s16_u16(
				vmovl_u8( vtbl1_u8( line8, tbl8 ) ) );
			sum = vaddq_s32( sum, VMADDQ_S16( pix, vc ) );
		}

		for( ; i < n; i++ ) {
			vc = vdup_n_s16( c[i] );

			memcpy(&p32, p, sizeof(uint32_t));
			line8 = vreinterpret_u8_u32( vdup_n_u32( p32 ) );
			p += 3;

			pix4 = vreinterpret_s16_u8( vtbl1_u8( line8, tbl4 ) );

			sum = vaddq_s32( sum, vmulq_s32(
				vmovl_s16( pix4 ), vmovl_s16( vc ) ) );
		}

		sum = vshrq_n_s32( sum, VIPS_INTERPOLATE_SHIFT );
		
		const int16x4_t sum_16 = vqmovn_s32( sum );
		const uint8x8_t sum_8 = vqmovun_s16(
			vcombine_s16( sum_16, sum_16 ) );
		*(uint32_t *) q = vget_lane_u32(
			vreinterpret_u32_u8( sum_8 ), 0 );

		c += n_point;
		bounds += 2;
	}

	/* Less optimal but safe approach for the last x.
	 * We can't load nor save 4 bytes anymore since we'll get out of
	 * buffers range. So for the last x, we carefully load 3 bytes and
	 * carefully save 3 bytes.
	 */
	for( ; x < width; x++ ) {
		const int left = bounds[0];
		const int right = bounds[1];
		const int32_t n = right - left;

		uint8_t * restrict p = pin + left * 3;
		uint8_t * restrict q = pout + x * 3;

		int32x4_t pix, vc;

		int32x4_t sum = initial;

		for( i = 0; i < n; i++ ) {
			vc = vdupq_n_s32( c[i] );

			pix = (int32x4_t) { p[0], p[1], p[2], 0 };
			p += 3;

			sum = vaddq_s32( sum, vmulq_s32( pix, vc ) );
		}

		sum = vshrq_n_s32( sum, VIPS_INTERPOLATE_SHIFT );

		q[0] = vgetq_lane_s32( sum, 0 );
		q[1] = vgetq_lane_s32( sum, 1 );
		q[2] = vgetq_lane_s32( sum, 2 );

		c += n_point;
		bounds += 2;
	}
}

void
reduceh_uchar_simd( VipsPel *pout, VipsPel *pin, int32_t n_point,
	int32_t bands, int32_t width, int16_t * restrict c,
	int32_t * restrict bounds ) {

	switch( bands ) {
	case 4:
		return reduceh_uchar_simd_4bands(
			pout, pin, n_point, width, c, bounds );
	case 3:
		return reduceh_uchar_simd_3bands(
			pout, pin, n_point, width, c, bounds );
	default:
		g_assert_not_reached();
	}
}

#endif /*HAVE_NEON*/
