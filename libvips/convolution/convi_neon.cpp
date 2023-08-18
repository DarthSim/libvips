#ifdef HAVE_CONFIG_H
#include <config.h>
#endif /*HAVE_CONFIG_H*/

#include <stdlib.h>

#include <vips/vips.h>
#include <vips/internal.h>

#include "pconvolution.h"

#if HAVE_NEON

#include <arm_neon.h>

#define VMADDQ_S16(P, C) vpaddq_s32( \
	vmull_s16( vget_low_s16( P ), C ), \
	vmull_s16( vget_high_s16( P ), C ) )

#define VMADD_S16(P, C) vpadd_s32( \
	vget_low_s32( vmull_s16( P, C ) ), \
	vget_high_s32( vmull_s16( P, C ) ) )

void
convi_uchar_simd( VipsPel *pout, VipsPel *pin,
	int32_t n, int32_t ne, int32_t offset, int32_t * restrict offsets,
	int16_t * restrict mant, int32_t exp )
{
	int32_t x, i, mant32;

    	const int32x4_t initial = vdupq_n_s32( 1 << (exp - 1) );
    	const int32x4_t voffset = vdupq_n_s32( offset );
	const uint8x8_t zero_uint8x8 = vdup_n_u8( 0 );
    	const uint8x16_t zero_uint8x16 = vdupq_n_u8( 0 );

	x = 0;

	for( ; x <= ne - 16; x += 16 ) {
		uint8_t * restrict p = (uint8_t *) pin + x;
		uint8_t * restrict q = (uint8_t *) pout + x;

		uint8x16_t source, line1, line2;
		int16x8_t pix;
		int16x4_t vc;

		int32x4_t sum0 = initial;
		int32x4_t sum1 = initial;
		int32x4_t sum2 = initial;
		int32x4_t sum3 = initial;

		i = 0;

		for( ; i <= n - 2; i += 2 ) {
			/* Load two coeffs
			 */
			memcpy(&mant32, &mant[i], sizeof(int32_t));
			vc = vreinterpret_s16_s32( vdup_n_s32( mant32 ) );

			line1 = vld1q_u8( p + offsets[i] );
			line2 = vld1q_u8( p + offsets[i + 1] );

			source = vzip1q_u8( line1, line2 );
			pix = vreinterpretq_s16_u16(
				vmovl_u8( vget_low_u8( source ) ) );
			sum0 = vaddq_s32( sum0, VMADDQ_S16( pix, vc ) );
			pix = vreinterpretq_s16_u16(
				vmovl_u8( vget_high_u8( source ) ) );
			sum1 = vaddq_s32( sum1, VMADDQ_S16( pix, vc ) );

			source = vzip2q_u8( line1, line2 );
			pix = vreinterpretq_s16_u16(
				vmovl_u8( vget_low_u8( source ) ) );
			sum2 = vaddq_s32( sum2, VMADDQ_S16( pix, vc ) );
			pix = vreinterpretq_s16_u16(
				vmovl_u8( vget_high_u8( source ) ) );
			sum3 = vaddq_s32( sum3, VMADDQ_S16( pix, vc ) );
		}

		for( ; i < n; i++ ) {
			vc = vdup_n_s16( mant[i] );

			line1 = vld1q_u8( p + offsets[i] );

			source = vzip1q_u8( line1, zero_uint8x16 );
			pix = vreinterpretq_s16_u16(
				vmovl_u8( vget_low_u8( source ) ) );
			sum0 = vaddq_s32( sum0, VMADDQ_S16( pix, vc ) );
			pix = vreinterpretq_s16_u16(
				vmovl_u8( vget_high_u8( source ) ) );
			sum1 = vaddq_s32( sum1, VMADDQ_S16( pix, vc ) );

			source = vzip2q_u8( line1, zero_uint8x16 );
			pix = vreinterpretq_s16_u16(
				vmovl_u8( vget_low_u8( source ) ) );
			sum2 = vaddq_s32( sum2, VMADDQ_S16( pix, vc ) );
			pix = vreinterpretq_s16_u16(
				vmovl_u8( vget_high_u8( source ) ) );
			sum3 = vaddq_s32( sum3, VMADDQ_S16( pix, vc ) );
		}

		sum0 = vshrq_n_s32( sum0, exp );
		sum1 = vshrq_n_s32( sum1, exp );
		sum2 = vshrq_n_s32( sum2, exp );
		sum3 = vshrq_n_s32( sum3, exp );

		sum0 = vaddq_s32( sum0, voffset );
		sum1 = vaddq_s32( sum1, voffset );
		sum2 = vaddq_s32( sum2, voffset );
		sum3 = vaddq_s32( sum3, voffset );

		const int16x8_t sum01 = vqmovn_high_s32(
			vqmovn_s32( sum0 ), sum1 );
		const int16x8_t sum23 = vqmovn_high_s32(
			vqmovn_s32( sum2 ), sum3 );
		const uint8x16_t sum = vqmovun_high_s16(
			vqmovun_s16( sum01 ), sum23 );

		vst1q_u8( q, sum );
	}

	for( ; x <= ne - 8; x += 8 ) {
		uint8_t * restrict p = (uint8_t *) pin + x;
		uint8_t * restrict q = (uint8_t *) pout + x;

		uint8x8_t line1, line2;
		uint8x16_t source;
		int16x8_t pix;
		int16x4_t vc;

		int32x4_t sum0 = initial;
		int32x4_t sum1 = initial;

		i = 0;

		for( ; i <= n - 2; i += 2 ) {
			/* Load two coeffs
			 */
			memcpy(&mant32, &mant[i], sizeof(int32_t));
			vc = vreinterpret_s16_s32( vdup_n_s32( mant32 ) );

			line1 = vld1_u8( p + offsets[i] );
			line2 = vld1_u8( p + offsets[i + 1] );

			source = vzip1q_u8( vcombine_u8( line1, line1 ),
				vcombine_u8( line2, line2 ) );
			pix = vreinterpretq_s16_u16(
				vmovl_u8( vget_low_u8( source ) ) );
			sum0 = vaddq_s32( sum0, VMADDQ_S16( pix, vc ) );
			pix = vreinterpretq_s16_u16(
				vmovl_u8( vget_high_u8( source ) ) );
			sum1 = vaddq_s32( sum1, VMADDQ_S16( pix, vc ) );
		}

		for( ; i < n; i++ ) {
			vc = vdup_n_s16( mant[i] );

			line1 = vld1_u8( p + offsets[i] );

			source = vzip1q_u8( vcombine_u8( line1, line1 ),
				zero_uint8x16 );
			pix = vreinterpretq_s16_u16(
				vmovl_u8( vget_low_u8( source ) ) );
			sum0 = vaddq_s32( sum0, VMADDQ_S16( pix, vc ) );
			pix = vreinterpretq_s16_u16(
				vmovl_u8( vget_high_u8( source ) ) );
			sum1 = vaddq_s32( sum1, VMADDQ_S16( pix, vc ) );
		}

		sum0 = vshrq_n_s32( sum0, exp );
		sum1 = vshrq_n_s32( sum1, exp );

		sum0 = vaddq_s32( sum0, voffset );
		sum1 = vaddq_s32( sum1, voffset );

		const int16x8_t sum = vqmovn_high_s32(
			vqmovn_s32( sum0 ), sum1 );

		vst1_u8( q, vqmovun_s16( sum ) );
	}

	for( ; x <= ne - 4; x += 4 ) {
		uint8_t * restrict p = (uint8_t *) pin + x;
		uint8_t * restrict q = (uint8_t *) pout + x;

		uint8x8_t line1, line2;
		int16x8_t pix;
		int16x4_t vc;

		int32x4_t sum = initial;

		i = 0;

		for( ; i <= n - 2; i += 2 ) {
			/* Load two coeffs
			 */
			memcpy(&mant32, &mant[i], sizeof(int32_t));
			vc = vreinterpret_s16_s32( vdup_n_s32( mant32 ) );

			line1 = vreinterpret_u8_u32( vdup_n_u32(
				*(uint32_t*) (p + offsets[i]) ) );

			line2 = vreinterpret_u8_u32( vdup_n_u32(
				*(uint32_t*) (p + offsets[i + 1]) ) );

			pix = vreinterpretq_s16_u16(
				vmovl_u8( vzip1_u8( line1, line2 ) ) );
			sum = vaddq_s32( sum, VMADDQ_S16( pix, vc ) );
		}

		for( ; i < n; i++ ) {
			vc = vdup_n_s16( mant[i] );

			line1 = vreinterpret_u8_u32( vdup_n_u32(
				*(uint32_t*) (p + offsets[i]) ) );

			pix = vreinterpretq_s16_u16(
				vmovl_u8( vzip1_u8( line1, zero_uint8x8 ) ) );
			sum = vaddq_s32( sum, VMADDQ_S16( pix, vc ) );
		}

		sum = vshrq_n_s32( sum, exp );

		sum = vaddq_s32( sum, voffset );

		const int16x4_t sum_16 = vqmovn_s32( sum );
		const uint8x8_t sum_8 = vqmovun_s16(
			vcombine_s16( sum_16, sum_16 ) );
		*(uint32_t *) q = vget_lane_u32(
			vreinterpret_u32_u8( sum_8 ), 0 );
	}

	for( ; x < ne; x++ ) {
		uint8_t * restrict p = (uint8_t *) pin + x;
		uint8_t * restrict q = (uint8_t *) pout + x;

		int16x4_t line1, line2, pix, vc;

		int32x2_t sum0 = vget_low_s32( initial );

		i = 0;

		for( ; i <= n - 2; i += 2 ) {
			/* Load two coeffs
			 */
			memcpy(&mant32, &mant[i], sizeof(int32_t));
			vc = vreinterpret_s16_s32( vdup_n_s32( mant32 ) );

			line1 = vdup_n_s16( p[offsets[i]] );
			line2 = vdup_n_s16( p[offsets[i + 1]] );

			pix = vzip1_s16( line1, line2 );
			sum0 = vadd_s32( sum0, VMADD_S16( pix, vc ) );
		}

		int32_t sum = vget_lane_s32( sum0, 0 );

		if( i < n ) {
			sum += (int32_t)(p[offsets[i]]) * mant[i];
		}

		*q = (uint8_t) ( sum >> exp ) + offset;
	}
}

#endif /*HAVE_NEON*/
