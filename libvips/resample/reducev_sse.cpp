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
reducev_uchar_simd( VipsPel *pout, VipsPel *pin,
	int32_t n, int32_t ne, int32_t lskip, const int16_t * restrict c )
{
	int32_t x, i;
	__m128i source, line1, line2, pix, vc;

	const __m128i initial = _mm_set1_epi32( VIPS_INTERPOLATE_SCALE >> 1 );
	const __m128i zero = _mm_setzero_si128();

	x = 0;

	for( ; x <= ne - 16; x += 16 ) {
		uint8_t * restrict p = (uint8_t *) pin + x;
		uint8_t * restrict q = (uint8_t *) pout + x;

		__m128i sum0 = initial;
		__m128i sum1 = initial;
		__m128i sum2 = initial;
		__m128i sum3 = initial;

		i = 0;

		for( ; i <= n - 2; i += 2 ) {
			/* Load two coeffs
			 */
			vc = _mm_set1_epi32( *(int32_t *) &c[i] );

			line1 = _mm_loadu_si128( (__m128i *) p );
			p += lskip;

			line2 = _mm_loadu_si128( (__m128i *) p );
			p += lskip;

			source = _mm_unpacklo_epi8( line1, line2 );
			pix = _mm_unpacklo_epi8( source, zero );
			sum0 = _mm_add_epi32( sum0, _mm_madd_epi16( pix, vc ) );
			pix = _mm_unpackhi_epi8( source, zero );
			sum1 = _mm_add_epi32( sum1, _mm_madd_epi16( pix, vc ) );

			source = _mm_unpackhi_epi8( line1, line2 );
			pix = _mm_unpacklo_epi8( source, zero );
			sum2 = _mm_add_epi32( sum2, _mm_madd_epi16( pix, vc ) );
			pix = _mm_unpackhi_epi8( source, zero );
			sum3 = _mm_add_epi32( sum3, _mm_madd_epi16( pix, vc ) );
		}

		for( ; i < n; i++ ) {
			vc = _mm_set1_epi16( c[i] );

			line1 = _mm_loadu_si128( (__m128i *) p );
			p += lskip;

			source = _mm_unpacklo_epi8( line1, zero );
			pix = _mm_unpacklo_epi8( source, zero );
			sum0 = _mm_add_epi32( sum0, _mm_madd_epi16( pix, vc ) );
			pix = _mm_unpackhi_epi8( source, zero );
			sum1 = _mm_add_epi32( sum1, _mm_madd_epi16( pix, vc ) );

			source = _mm_unpackhi_epi8( line1, zero );
			pix = _mm_unpacklo_epi8( source, zero );
			sum2 = _mm_add_epi32( sum2, _mm_madd_epi16( pix, vc ) );
			pix = _mm_unpackhi_epi8( source, zero );
			sum3 = _mm_add_epi32( sum3, _mm_madd_epi16( pix, vc ) );
		}

		sum0 = _mm_srai_epi32( sum0, VIPS_INTERPOLATE_SHIFT );
		sum1 = _mm_srai_epi32( sum1, VIPS_INTERPOLATE_SHIFT );
		sum2 = _mm_srai_epi32( sum2, VIPS_INTERPOLATE_SHIFT );
		sum3 = _mm_srai_epi32( sum3, VIPS_INTERPOLATE_SHIFT );

		const __m128i sum01 = _mm_packs_epi32( sum0, sum1 );
		const __m128i sum23 = _mm_packs_epi32( sum2, sum3 );
		const __m128i sum = _mm_packus_epi16( sum01, sum23 );

		_mm_storeu_si128( (__m128i *) q, sum );
	}

	for( ; x <= ne - 8; x += 8 ) {
		uint8_t * restrict p = (uint8_t *) pin + x;
		uint8_t * restrict q = (uint8_t *) pout + x;

		__m128i sum0 = initial;
		__m128i sum1 = initial;

		i = 0;

		for( ; i <= n - 2; i += 2 ) {
			/* Load two coeffs
			 */
			vc = _mm_set1_epi32( *(int32_t *) &c[i] );

			line1 = _mm_loadu_si64( p );
			p += lskip;

			line2 = _mm_loadu_si64( p );
			p += lskip;

			source = _mm_unpacklo_epi8( line1, line2 );
			pix = _mm_unpacklo_epi8( source, zero );
			sum0 = _mm_add_epi32( sum0, _mm_madd_epi16( pix, vc ) );
			pix = _mm_unpackhi_epi8( source, zero );
			sum1 = _mm_add_epi32( sum1, _mm_madd_epi16( pix, vc ) );
		}

		for( ; i < n; i++ ) {
			vc = _mm_set1_epi16( c[i] );

			line1 = _mm_loadu_si64( p );
			p += lskip;

			source = _mm_unpacklo_epi8( line1, zero );
			pix = _mm_unpacklo_epi8( source, zero );
			sum0 = _mm_add_epi32( sum0, _mm_madd_epi16( pix, vc ) );
			pix = _mm_unpackhi_epi8( source, zero );
			sum1 = _mm_add_epi32( sum1, _mm_madd_epi16( pix, vc ) );
		}

		sum0 = _mm_srai_epi32( sum0, VIPS_INTERPOLATE_SHIFT );
		sum1 = _mm_srai_epi32( sum1, VIPS_INTERPOLATE_SHIFT );

		const __m128i sum = _mm_packs_epi32( sum0 , sum1 );

		_mm_storeu_si64( q, _mm_packus_epi16( sum, sum ) );
	}

	for( ; x <= ne - 4; x += 4 ) {
		uint8_t * restrict p = (uint8_t *) pin + x;
		uint8_t * restrict q = (uint8_t *) pout + x;

		__m128i sum = initial;

		i = 0;

		for( ; i <= n - 2; i += 2 ) {
			/* Load two coeffs
			 */
			vc = _mm_set1_epi32( *(int32_t *) &c[i] );

			line1 = _mm_set1_epi32( *(int32_t *) p );
			p += lskip;

			line2 = _mm_set1_epi32( *(int32_t *) p );
			p += lskip;

			pix = _mm_unpacklo_epi8(
				_mm_unpacklo_epi8( line1, line2 ), zero );
			sum = _mm_add_epi32( sum, _mm_madd_epi16( pix, vc ) );
		}

		for( ; i < n; i++ ) {
			vc = _mm_set1_epi16( c[i] );

			line1 = _mm_set1_epi32( *(int32_t *) p );
			p += lskip;

			pix = _mm_unpacklo_epi8(
				_mm_unpacklo_epi8( line1, zero ), zero );
			sum = _mm_add_epi32( sum, _mm_madd_epi16( pix, vc ) );
		}

		sum = _mm_srai_epi32( sum, VIPS_INTERPOLATE_SHIFT );

		const __m128i sum_16 = _mm_packs_epi32( sum, sum );
		const __m128i sum_8 = _mm_packus_epi16( sum_16, sum_16 );
		*(uint32_t *) q = _mm_cvtsi128_si32( sum_8 );
	}

	for( ; x < ne; x++ ) {
		uint8_t * restrict p = (uint8_t *) pin + x;
		uint8_t * restrict q = (uint8_t *) pout + x;

		__m128i sum0 = initial;

		i = 0;

		for( ; i <= n - 2; i += 2 ) {
			/* Load two coeffs
			 */
			vc = _mm_set1_epi32( *(int32_t *) &c[i] );

			line1 = _mm_set1_epi16( *p );
			p += lskip;

			line2 = _mm_set1_epi16( *p );
			p += lskip;

			pix = _mm_unpacklo_epi16( line1, line2 );
			sum0 = _mm_add_epi32( sum0, _mm_madd_epi16( pix, vc ) );
		}

		int32_t sum = _mm_cvtsi128_si32( sum0 );

		if( i < n ) {
			sum += (int32_t) (*p) * c[i];
		}

		*q = (uint8_t) ( sum >> VIPS_INTERPOLATE_SHIFT );
	}
}

#endif /*HAVE_SSE*/
