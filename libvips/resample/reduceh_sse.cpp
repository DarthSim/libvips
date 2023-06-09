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
reduceh_uchar_simd(VipsPel *pout, VipsPel *pin, int32_t bands,
	int32_t n, int32_t width,
	int16_t *restrict cs[VIPS_TRANSFORM_SCALE + 1],
	double Xstart, double Xstep)
{
	int32_t x, i;
	double X = Xstart;

	__m128i line8, line16, vc_lo, vc_hi;
	__m128i sum;

	const __m128i initial = _mm_set1_epi32(VIPS_INTERPOLATE_SCALE >> 1);

	//  0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15
	// r0 g0 b0 a0 r1 g1 b1 a1 r2 g2 b2 a2 r3 g3 b3 a3
	alignas(16) static constexpr int8_t tbl4_lo[16] = {
		0, -1, 4, -1, 1, -1, 5, -1,
		2, -1, 6, -1, 3, -1, 7, -1
	};
	alignas(16) static constexpr int8_t tbl4_hi[16] = {
		8, -1, 12, -1, 9, -1, 13, -1,
		10, -1, 14, -1, 11, -1, 15, -1
	};
	alignas(16) static constexpr int8_t tbl4_4[16] = {
		0, -1, -1, -1, 1, -1, -1, -1,
		2, -1, -1, -1, 3, -1, -1, -1
	};

	//  0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15
	// r0 g0 b0 r1 g1 b1 r2 g2 b2 r3 g3 b3
	alignas(16) static constexpr int8_t tbl3_lo[16] = {
		0, -1, 3, -1, 1, -1, 4, -1,
		2, -1, 5, -1, -1, -1, -1, -1
	};
	alignas(16) static constexpr int8_t tbl3_hi[16] = {
		6, -1, 9, -1, 7, -1, 10, -1,
		8, -1, 11, -1, -1, -1, -1, -1
	};
	alignas(16) static constexpr int8_t tbl3_4[16] = {
		0, -1, -1, -1, 1, -1, -1, -1,
		2, -1, -1, -1, -1, -1, -1, -1
	};

	const __m128i tbl_lo = _mm_loadu_si128(
		(__m128i *) (bands == 3 ? tbl3_lo : tbl4_lo));
	const __m128i tbl_hi = _mm_loadu_si128(
		(__m128i *) (bands == 3 ? tbl3_hi : tbl4_hi));
	const __m128i tbl_4 = _mm_loadu_si128(
		(__m128i *) (bands == 3 ? tbl3_4 : tbl4_4));

	for (x = 0; x < width; x++) {
		const int ix = (int) X;
		const int sx = X * VIPS_TRANSFORM_SCALE * 2;
		const int six = sx & (VIPS_TRANSFORM_SCALE * 2 - 1);
		const int tx = (six + 1) >> 1;
		const int16_t *c = cs[tx];

		uint8_t *restrict p = pin + ix * bands;
		uint8_t *restrict q = pout + x * bands;

		sum = initial;

		for (i = 0; i <= n - 4; i += 4) {
			/* Load four coeffs
			 */
			vc_lo = _mm_set1_epi32(*(int32_t *) &c[i]);
			vc_hi = _mm_set1_epi32(*(int32_t *) &c[i + 2]);

			line8 = _mm_loadu_si128((__m128i *) p);
			p += bands * 4;

			line16 = _mm_shuffle_epi8(line8, tbl_lo);
			sum = _mm_add_epi32(sum, _mm_madd_epi16(line16, vc_lo));
			line16 = _mm_shuffle_epi8(line8, tbl_hi);
			sum = _mm_add_epi32(sum, _mm_madd_epi16(line16, vc_hi));
		}

		for (; i <= n - 2; i += 2) {
			/* Load two coeffs
			 */
			vc_lo = _mm_set1_epi32(*(int32_t *) &c[i]);

			line8 = _mm_loadu_si64(p);
			p += bands * 2;

			line16 = _mm_shuffle_epi8(line8, tbl_lo);
			sum = _mm_add_epi32(sum, _mm_madd_epi16(line16, vc_lo));
		}

		if (i < n) {
			vc_lo = _mm_set1_epi16(c[i]);

			line8 = _mm_loadu_si64(p);
			p += bands;

			line16 = _mm_shuffle_epi8(line8, tbl_4);
			sum = _mm_add_epi32(sum, _mm_madd_epi16(line16, vc_lo));
		}

		sum = _mm_srai_epi32(sum, VIPS_INTERPOLATE_SHIFT);

		sum = _mm_packs_epi32(sum, sum);
		_mm_storeu_si64(q, _mm_packus_epi16(sum, sum));

		X += Xstep;
	}
}

#endif /*HAVE_SSE*/
