#ifdef HAVE_CONFIG_H
#include <config.h>
#endif /*HAVE_CONFIG_H*/

#include <stdlib.h>
#include <stdint.h>

#include <vips/vips.h>
#include <vips/internal.h>

#include "presample.h"

#if HAVE_AVX2

#include <immintrin.h>

int
reducev_uchar_avx2(VipsPel *pout, VipsPel *pin,
	int32_t n, int32_t ne, int32_t lskip, const int16_t *restrict c)
{
	int32_t x, i;

	__m256i line8, line8_hi, line8_lo, line16, vc;
	__m256i sum0, sum1, sum2, sum3;

	const __m256i initial = _mm256_set1_epi32(VIPS_INTERPOLATE_SCALE >> 1);
	const __m256i zero = _mm256_setzero_si256();

	const __m256i tbl0 = _mm256_set_epi8(
		-1, -1, -1, 7, -1, -1, -1, 6,
		-1, -1, -1, 5, -1, -1, -1, 4,
		-1, -1, -1, 3, -1, -1, -1, 2,
		-1, -1, -1, 1, -1, -1, -1, 0);
	const __m256i tbl1 = _mm256_set_epi8(
		-1, -1, -1, 15, -1, -1, -1, 14,
		-1, -1, -1, 13, -1, -1, -1, 12,
		-1, -1, -1, 11, -1, -1, -1, 10,
		-1, -1, -1, 9, -1, -1, -1, 8);
	const __m256i tbl2 = _mm256_set_epi8(
		-1, -1, -1, 23, -1, -1, -1, 22,
		-1, -1, -1, 21, -1, -1, -1, 20,
		-1, -1, -1, 19, -1, -1, -1, 18,
		-1, -1, -1, 17, -1, -1, -1, 16);
	const __m256i tbl3 = _mm256_set_epi8(
		-1, -1, -1, 31, -1, -1, -1, 30,
		-1, -1, -1, 29, -1, -1, -1, 28,
		-1, -1, -1, 27, -1, -1, -1, 26,
		-1, -1, -1, 25, -1, -1, -1, 24);

	for (x = 0; x <= ne - 32; x += 32) {
		uint8_t *restrict p = (uint8_t *) pin + x;
		uint8_t *restrict q = (uint8_t *) pout + x;

		sum0 = initial;
		sum1 = initial;
		sum2 = initial;
		sum3 = initial;

		for (i = 0; i <= n - 2; i += 2) {
			/* Load two coeffs
			 */
			vc = _mm256_set1_epi32(*(int32_t *) &c[i]);

			line8_hi = _mm256_loadu_si256((__m256i *) p);
			p += lskip;

			line8_lo = _mm256_loadu_si256((__m256i *) p);
			p += lskip;

			line8 = _mm256_unpacklo_epi8(line8_hi, line8_lo);

			line16 = _mm256_unpacklo_epi8(line8, zero);
			sum0 = _mm256_add_epi32(sum0, _mm256_madd_epi16(line16, vc));
			line16 = _mm256_unpackhi_epi8(line8, zero);
			sum1 = _mm256_add_epi32(sum1, _mm256_madd_epi16(line16, vc));

			line8 = _mm256_unpackhi_epi8(line8_hi, line8_lo);

			line16 = _mm256_unpacklo_epi8(line8, zero);
			sum2 = _mm256_add_epi32(sum2, _mm256_madd_epi16(line16, vc));
			line16 = _mm256_unpackhi_epi8(line8, zero);
			sum3 = _mm256_add_epi32(sum3, _mm256_madd_epi16(line16, vc));
		}

		if (i < n) {
			vc = _mm256_set1_epi16(c[i]);

			line8 = _mm256_loadu_si256((__m256i *) p);

			line16 = _mm256_shuffle_epi8(line8, tbl0);
			sum0 = _mm256_add_epi32(sum0, _mm256_madd_epi16(line16, vc));
			line16 = _mm256_shuffle_epi8(line8, tbl1);
			sum1 = _mm256_add_epi32(sum1, _mm256_madd_epi16(line16, vc));
			line16 = _mm256_shuffle_epi8(line8, tbl2);
			sum2 = _mm256_add_epi32(sum2, _mm256_madd_epi16(line16, vc));
			line16 = _mm256_shuffle_epi8(line8, tbl3);
			sum3 = _mm256_add_epi32(sum3, _mm256_madd_epi16(line16, vc));
		}

		sum0 = _mm256_srai_epi32(sum0, VIPS_INTERPOLATE_SHIFT);
		sum1 = _mm256_srai_epi32(sum1, VIPS_INTERPOLATE_SHIFT);
		sum2 = _mm256_srai_epi32(sum2, VIPS_INTERPOLATE_SHIFT);
		sum3 = _mm256_srai_epi32(sum3, VIPS_INTERPOLATE_SHIFT);

		sum0 = _mm256_packs_epi32(sum0, sum1);
		sum2 = _mm256_packs_epi32(sum2, sum3);
		sum0 = _mm256_packus_epi16(sum0, sum2);

		_mm256_storeu_si256((__m256i *) q, sum0);
	}

	return x;
}


#endif /*HAVE_AVX2*/
