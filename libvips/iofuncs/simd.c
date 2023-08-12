#ifdef HAVE_CONFIG_H
#include <config.h>
#endif /*HAVE_CONFIG_H*/

#include <vips/vips.h>
#include <vips/simd.h>
#include <vips/internal.h>

gboolean vips__simd_enabled = TRUE;

gboolean vips__ssse3_supported = FALSE;
gboolean vips__avx2_supported = FALSE;
gboolean vips__neon_supported = FALSE;

#if defined(__aarch64__) || defined(_M_ARM64)

void
vips_check_simd_support(void)
{
	vips__neon_supported = TRUE;
}

#elif (defined(__GNUC__) || defined(__APPLE_CC__)) && defined(__x86_64__)

#include <cpuid.h>

void
vips_check_simd_support(void)
{
	unsigned int eax, ebx, ecx, edx;

	eax = ebx = ecx = edx = 0;
	__cpuid(0, eax, ebx, ecx, edx);
	const int max_leaf = eax;

	if (max_leaf >= 1) {
		eax = ebx = ecx = edx = 0;
		__cpuid(1, eax, ebx, ecx, edx);
		if (ecx & (1 << 9))
			vips__ssse3_supported = TRUE;
	}

	if (max_leaf >= 7) {
		eax = ebx = ecx = edx = 0;
		__cpuid_count(7, 0, eax, ebx, ecx, edx);
		if (ebx & (1 << 5))
			vips__avx2_supported = TRUE;
	}
}

#elif defined(_MSC_VER) && defined(_M_X64)

#include <intrin.h>

void
vips_check_simd_support(void)
{
	int data[4];

	data[0] = data[1] = data[2] = data[3] = 0;
	__cpuid(data, 0);

	const int max_leaf = data[0];

	if (max_leaf >= 1) {
		data[0] = data[1] = data[2] = data[3] = 0;
		__cpuidex(data, 1, 0);
		if (data[2] & (1 << 9))
			vips__ssse3_supported = TRUE;
	}

	if (max_leaf >= 7) {
		data[0] = data[1] = data[2] = data[3] = 0;
		__cpuidex(data, 7, 0);
		if (data[1] & (1 << 5))
			vips__avx2_supported = TRUE;
	}
}

#else

void
vips_check_simd_support(void)
{
	// Can't detect SIMD support
}

#endif

void
vips_simd_init(void)
{
#if HAVE_SIMD
	vips_check_simd_support();

	if (g_getenv("VIPS_NOSIMD"))
		vips__simd_enabled = FALSE;
#endif /*HAVE_SIMD*/
}

gboolean
vips_simd_isenabled(void)
{
#if HAVE_SSE
	return vips__simd_enabled && vips__ssse3_supported;
#elif HAVE_NEON /*HAVE_SSE*/
	return vips__simd_enabled && vips__neon_supported;
#else /*HAVE_NEON*/
	return FALSE;
#endif /*HAVE_NEON*/
}

void
vips_simd_set_enabled(gboolean enabled)
{
	vips__simd_enabled = enabled;
}

gboolean
vips_simd_avx2_issupported(void)
{
#if HAVE_AVX2
	return vips__avx2_supported;
#else /*HAVE_AVX2*/
	return FALSE;
#endif /*HAVE_AVX2*/
}
