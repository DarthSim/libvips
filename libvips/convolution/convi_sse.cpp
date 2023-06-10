#ifdef HAVE_CONFIG_H
#include <config.h>
#endif /*HAVE_CONFIG_H*/

#include <stdlib.h>
#include <stdint.h>

#include <vips/vips.h>
#include <vips/internal.h>

#include "pconvolution.h"

#if HAVE_SSE

#include <smmintrin.h>

void
convi_uchar_simd( VipsPel *pout, VipsPel *pin,
	size_t n, size_t ne, int32_t offset, int32_t * restrict offsets,
	int16_t * restrict mant, int32_t exp )
{
	size_t x, i;
	__m128i source, line1, line2, pix, vc;

    	const __m128i initial = _mm_set1_epi32( 1 << (exp - 1) );
    	const __m128i voffset = _mm_set1_epi32( offset );
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
			vc = _mm_set1_epi32( *(int32_t *) &mant[i] );

			line1 = _mm_loadu_si128( (__m128i *) (p + offsets[i]) );
			line2 = _mm_loadu_si128(
				(__m128i *) (p + offsets[i + 1]) );

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
			vc = _mm_set1_epi16( mant[i] );

			line1 = _mm_loadu_si128( (__m128i *) (p + offsets[i]) );

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

		sum0 = _mm_srai_epi32( sum0, exp );
		sum1 = _mm_srai_epi32( sum1, exp );
		sum2 = _mm_srai_epi32( sum2, exp );
		sum3 = _mm_srai_epi32( sum3, exp );

		sum0 = _mm_add_epi32( sum0, voffset );
		sum1 = _mm_add_epi32( sum1, voffset );
		sum2 = _mm_add_epi32( sum2, voffset );
		sum3 = _mm_add_epi32( sum3, voffset );

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
			vc = _mm_set1_epi32( *(int32_t *) &mant[i] );

			line1 = _mm_loadu_si64( p + offsets[i] );
			line2 = _mm_loadu_si64( p + offsets[i + 1] );

			source = _mm_unpacklo_epi8( line1, line2 );
			pix = _mm_unpacklo_epi8( source, zero );
			sum0 = _mm_add_epi32( sum0, _mm_madd_epi16( pix, vc ) );
			pix = _mm_unpackhi_epi8( source, zero );
			sum1 = _mm_add_epi32( sum1, _mm_madd_epi16( pix, vc ) );
		}

		for( ; i < n; i++ ) {
			vc = _mm_set1_epi16( mant[i] );

			line1 = _mm_loadu_si64( p + offsets[i] );

			source = _mm_unpacklo_epi8( line1, zero );
			pix = _mm_unpacklo_epi8( source, zero );
			sum0 = _mm_add_epi32( sum0, _mm_madd_epi16( pix, vc ) );
			pix = _mm_unpackhi_epi8( source, zero );
			sum1 = _mm_add_epi32( sum1, _mm_madd_epi16( pix, vc ) );
		}

		sum0 = _mm_srai_epi32( sum0, exp );
		sum1 = _mm_srai_epi32( sum1, exp );

		sum0 = _mm_add_epi32( sum0, voffset );
		sum1 = _mm_add_epi32( sum1, voffset );

		const __m128i sum = _mm_packs_epi32( sum0 , sum1 );

		_mm_storeu_si64( q, _mm_packus_epi16( sum, sum ) );
	}

	for( ; x < ne; x++ ) {
		uint8_t * restrict p = (uint8_t *) pin + x;
		uint8_t * restrict q = (uint8_t *) pout + x;

		__m128i sum0 = initial;

		i = 0;

		for( ; i <= n - 2; i += 2 ) {
			/* Load two coeffs
			 */
			vc = _mm_set1_epi32( *(int32_t *) &mant[i] );

			line1 = _mm_set1_epi16( p[offsets[i]] );
			line2 = _mm_set1_epi16( p[offsets[i + 1]] );

			pix = _mm_unpacklo_epi16( line1, line2 );
			sum0 = _mm_add_epi32( sum0, _mm_madd_epi16( pix, vc ) );
		}

		int32_t sum = _mm_cvtsi128_si32( sum0 );

		for( ; i < n; i++ ) {
			sum += (int32_t)(p[offsets[i]]) * mant[i];
		}

		*q = (uint8_t) ( sum >> exp ) + offset;
	}
}

#endif /*HAVE_SSE*/
