#ifdef HAVE_CONFIG_H
#include <config.h>
#endif /*HAVE_CONFIG_H*/

#include <stdlib.h>
#include <stdint.h>

#include <vips/vips.h>
#include <vips/internal.h>

#include "presample.h"

#if HAVE_SSE

#include <smmintrin.h>

void
reduceh_uchar_simd_4bands( VipsPel *pout, VipsPel *pin, int32_t n_point,
	int32_t width, int16_t * restrict c, int32_t * restrict bounds )
{
	int32_t x, i, c32, p32;

	const __m128i initial = _mm_set1_epi32( VIPS_INTERPOLATE_SCALE >> 1 );

	//  0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15
	// r0 g0 b0 a0 r1 g1 b1 a1 r2 g2 b2 a2 r3 g3 b3 a3
	const __m128i tbl_lo = _mm_set_epi8(
		-1, 7, -1, 3, -1, 6, -1, 2,
		-1, 5, -1, 1, -1, 4, -1, 0 );
	const __m128i tbl_hi = _mm_set_epi8(
		-1, 15, -1, 11, -1, 14, -1, 10,
		-1, 13, -1,  9, -1, 12, -1,  8 );
	const __m128i tbl4 = _mm_set_epi8(
		-1, -1, -1, 3, -1, -1, -1, 2,
		-1, -1, -1, 1, -1, -1, -1, 0 );

	for( x = 0; x < width; x++ ) {
		const int left = bounds[0];
		const int right = bounds[1];
		const int32_t n = right - left;

		uint8_t * restrict p = pin + left * 4;
		uint8_t * restrict q = pout + x * 4;

		__m128i line, pix;
		__m128i vc_lo, vc_hi;

		__m128i sum = initial;

		i = 0;

		for( ; i <= n - 4; i += 4 ) {
			/* Load four coeffs
			 */
			memcpy(&c32, &c[i], sizeof(int32_t));
			vc_lo = _mm_set1_epi32( c32 );
			memcpy(&c32, &c[i + 2], sizeof(int32_t));
			vc_hi = _mm_set1_epi32( c32 );

			line = _mm_loadu_si128( (__m128i *) p );
			p += 16;

			pix = _mm_shuffle_epi8( line, tbl_lo );
			sum = _mm_add_epi32(
				sum, _mm_madd_epi16( pix, vc_lo ) );

			pix = _mm_shuffle_epi8( line, tbl_hi );
			sum = _mm_add_epi32(
				sum, _mm_madd_epi16( pix, vc_hi ) );
		}

		for( ; i <= n - 2; i += 2 ) {
			/* Load two coeffs
			 */
			memcpy(&c32, &c[i], sizeof(int32_t));
			vc_lo = _mm_set1_epi32( c32 );

			line = _mm_loadu_si64( p );
			p += 8;

			pix = _mm_shuffle_epi8( line, tbl_lo );
			sum = _mm_add_epi32(
				sum, _mm_madd_epi16( pix, vc_lo ) );
		}

		for( ; i < n; i++ ) {
			vc_lo = _mm_set1_epi16( c[i] );

			memcpy(&p32, p, sizeof(int32_t));
			line = _mm_cvtsi32_si128( p32 );
			p += 4;

			pix = _mm_shuffle_epi8( line, tbl4 );
			sum = _mm_add_epi32(
				sum, _mm_madd_epi16( pix, vc_lo ) );
		}

		sum = _mm_srai_epi32( sum, VIPS_INTERPOLATE_SHIFT );
		
		const __m128i sum_16 = _mm_packs_epi32( sum, sum );
		const __m128i sum_8 = _mm_packus_epi16( sum_16, sum_16 );
		*(uint32_t *) q = _mm_cvtsi128_si32( sum_8 );

		c += n_point;
		bounds += 2;
	}
}

void
reduceh_uchar_simd_3bands( VipsPel *pout, VipsPel *pin, int32_t n_point,
	int32_t width, int16_t * restrict c, int32_t * restrict bounds )
{
	int32_t x, i, c32, p32;

	const __m128i initial = _mm_set1_epi32( VIPS_INTERPOLATE_SCALE >> 1 );

	//  0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15
	// r0 g0 b0 r1 g1 b1 r2 g2 b2 r3 g3 b3
	const __m128i tbl_lo = _mm_set_epi8(
		-1, -1, -1, -1, -1, 5, -1, 2,
		-1,  4, -1,  1, -1, 3, -1, 0 );
	const __m128i tbl_hi = _mm_set_epi8(
		-1, -1, -1, -1, -1, 11, -1, 8,
		-1, 10, -1,  7, -1,  9, -1, 6 );
	const __m128i tbl4 = _mm_set_epi8(
		-1, -1, -1, -1, -1, -1, -1, 2,
		-1, -1, -1,  1, -1, -1, -1, 0 );

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

		__m128i line, pix;
		__m128i vc_lo, vc_hi;

		__m128i sum = initial;

		i = 0;

		for( ; i <= n - 4; i += 4 ) {
			/* Load four coeffs
			 */
			memcpy(&c32, &c[i], sizeof(int32_t));
			vc_lo = _mm_set1_epi32( c32 );
			memcpy(&c32, &c[i + 2], sizeof(int32_t));
			vc_hi = _mm_set1_epi32( c32 );

			line = _mm_loadu_si128( (__m128i *) p );
			p += 12;

			pix = _mm_shuffle_epi8( line, tbl_lo );
			sum = _mm_add_epi32(
				sum, _mm_madd_epi16( pix, vc_lo ) );

			pix = _mm_shuffle_epi8( line, tbl_hi );
			sum = _mm_add_epi32(
				sum, _mm_madd_epi16( pix, vc_hi ) );
		}

		for( ; i <= n - 2; i += 2 ) {
			/* Load two coeffs
			 */
			memcpy(&c32, &c[i], sizeof(int32_t));
			vc_lo = _mm_set1_epi32( c32 );

			line = _mm_loadu_si64( p );
			p += 6;

			pix = _mm_shuffle_epi8( line, tbl_lo );
			sum = _mm_add_epi32(
				sum, _mm_madd_epi16( pix, vc_lo ) );
		}

		for( ; i < n; i++ ) {
			vc_lo = _mm_set1_epi16( c[i] );

			memcpy(&p32, p, sizeof(int32_t));
			line = _mm_cvtsi32_si128( p32 );
			p += 3;

			pix = _mm_shuffle_epi8( line, tbl4 );
			sum = _mm_add_epi32(
				sum, _mm_madd_epi16( pix, vc_lo ) );
		}

		sum = _mm_srai_epi32( sum, VIPS_INTERPOLATE_SHIFT );
		
		const __m128i sum_16 = _mm_packs_epi32( sum, sum );
		const __m128i sum_8 = _mm_packus_epi16( sum_16, sum_16 );
		*(uint32_t *) q = _mm_cvtsi128_si32( sum_8 );

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

		__m128i pix, vc;

		__m128i sum = initial;

		for( i = 0; i < n; i++ ) {
			vc = _mm_set1_epi32( c[i] );

			pix = _mm_set_epi32( 0, p[2], p[1], p[0] );
			p += 3;

			/* Since we use only the first 16 bits of each number,
			 * it's safe to use _mm_madd_epi16 here
			 */
			sum = _mm_add_epi32( sum, _mm_madd_epi16( pix, vc ) );
		}

		sum = _mm_srai_epi32( sum, VIPS_INTERPOLATE_SHIFT );

		const __m128i sum_16 = _mm_packs_epi32( sum, sum );
		const __m128i sum_8 = _mm_packus_epi16( sum_16, sum_16 );
		const int32_t qq = _mm_cvtsi128_si32( sum_8 );

		q[0] = ((uint8_t *)(&qq))[0];
		q[1] = ((uint8_t *)(&qq))[1];
		q[2] = ((uint8_t *)(&qq))[2];

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

#endif /*HAVE_SSE*/
