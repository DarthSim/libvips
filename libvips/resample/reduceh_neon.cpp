#ifdef HAVE_CONFIG_H
#include <config.h>
#endif /*HAVE_CONFIG_H*/

#include <stdlib.h>

#include <vips/vips.h>
#include <vips/internal.h>

#include "presample.h"

#if HAVE_NEON

#include <arm_neon.h>

void
reduceh_uchar_simd(VipsPel *pout, VipsPel *pin, int32_t width, int32_t bands,
	int16_t *restrict c, int32_t c_stride, int32_t *restrict bounds)
{
	int32_t x, i;

	uint8x16_t line8;
	int32x4_t line32;
	int32x4_t sum;
	int16x4_t sum_16;

	const int32x4_t initial = vdupq_n_s32(VIPS_INTERPOLATE_SCALE >> 1);

	//  0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15
	// r0 g0 b0 a0 r1 g1 b1 a1 r2 g2 b2 a2 r3 g3 b3 a3
	alignas(16) static constexpr uint8_t tbl4_0[16] = {
		0, 16, 16, 16, 1, 16, 16, 16,
		2, 16, 16, 16, 3, 16, 16, 16
	};
	alignas(16) static constexpr uint8_t tbl4_1[16] = {
		4, 16, 16, 16, 5, 16, 16, 16,
		6, 16, 16, 16, 7, 16, 16, 16
	};
	alignas(16) static constexpr uint8_t tbl4_2[16] = {
		8, 16, 16, 16, 9, 16, 16, 16,
		10, 16, 16, 16, 11, 16, 16, 16
	};
	alignas(16) static constexpr uint8_t tbl4_3[16] = {
		12, 16, 16, 16, 13, 16, 16, 16,
		14, 16, 16, 16, 15, 16, 16, 16
	};

	//  0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15
	// r0 g0 b0 r1 g1 b1 r2 g2 b2 r3 g3 b3
	alignas(16) static constexpr uint8_t tbl3_0[16] = {
		0, 16, 16, 16, 1, 16, 16, 16,
		2, 16, 16, 16, 16, 16, 16, 16
	};
	alignas(16) static constexpr uint8_t tbl3_1[16] = {
		3, 16, 16, 16, 4, 16, 16, 16,
		5, 16, 16, 16, 16, 16, 16, 16
	};
	alignas(16) static constexpr uint8_t tbl3_2[16] = {
		6, 16, 16, 16, 7, 16, 16, 16,
		8, 16, 16, 16, 16, 16, 16, 16
	};
	alignas(16) static constexpr uint8_t tbl3_3[16] = {
		9, 16, 16, 16, 10, 16, 16, 16,
		11, 16, 16, 16, 16, 16, 16, 16
	};

	const uint8x16_t tbl_0 = vld1q_u8(bands == 3 ? tbl3_0 : tbl4_0);
	const uint8x16_t tbl_1 = vld1q_u8(bands == 3 ? tbl3_1 : tbl4_1);
	const uint8x16_t tbl_2 = vld1q_u8(bands == 3 ? tbl3_2 : tbl4_2);
	const uint8x16_t tbl_3 = vld1q_u8(bands == 3 ? tbl3_3 : tbl4_3);

	for (x = 0; x < width; x++) {
		const int left = bounds[0];
		const int right = bounds[1];
		const int32_t n = right - left;

		uint8_t *restrict p = pin + left * bands;
		uint8_t *restrict q = pout + x * bands;

		sum = initial;

		for (i = 0; i <= n - 4; i += 4) {
			line8 = vld1q_u8(p);
			p += bands * 4;

			line32 = vreinterpretq_s32_u8(vqtbl1q_u8(line8, tbl_0));
			sum = vmlaq_n_s32(sum, line32, c[i]);
			line32 = vreinterpretq_s32_u8(vqtbl1q_u8(line8, tbl_1));
			sum = vmlaq_n_s32(sum, line32, c[i + 1]);
			line32 = vreinterpretq_s32_u8(vqtbl1q_u8(line8, tbl_2));
			sum = vmlaq_n_s32(sum, line32, c[i + 2]);
			line32 = vreinterpretq_s32_u8(vqtbl1q_u8(line8, tbl_3));
			sum = vmlaq_n_s32(sum, line32, c[i + 3]);
		}

		for (; i <= n - 2; i += 2) {
			line8 = vld1q_u8(p);
			p += bands * 2;

			line32 = vreinterpretq_s32_u8(vqtbl1q_u8(line8, tbl_0));
			sum = vmlaq_n_s32(sum, line32, c[i]);
			line32 = vreinterpretq_s32_u8(vqtbl1q_u8(line8, tbl_1));
			sum = vmlaq_n_s32(sum, line32, c[i + 1]);
		}

		if (i < n) {
			line8 = vld1q_u8(p);
			p += bands;

			line32 = vreinterpretq_s32_u8(vqtbl1q_u8(line8, tbl_0));
			sum = vmlaq_n_s32(sum, line32, c[i]);
		}

		sum = vshrq_n_s32(sum, VIPS_INTERPOLATE_SHIFT);

		sum_16 = vqmovn_s32(sum);
		vst1_u8(q, vqmovun_s16(vcombine_s16(sum_16, sum_16)));

		c += c_stride;
		bounds += 2;
	}
}

#endif /*HAVE_NEON*/
