resample_sources = files(
    'thumbnail.c',
    'mapim.c',
    'affine.c',
    'quadratic.c',
    'resample.c',
    'similarity.c',
    'resize.c',
    'shrink.c',
    'shrinkh.c',
    'shrinkv.c',
    'reduce.c',
    'reduceh.cpp',
    'reducev.cpp',
    'interpolate.c',
    'transform.c',
    'bicubic.cpp',
    'lbb.cpp',
    'nohalo.cpp',
    'vsqbs.cpp',
)

resample_headers = files(
    'presample.h',
    'templates.h',
)

resample_components = []

libvips_sources += resample_sources

if simd_type == simd_type_sse or simd_type == simd_type_avx2
    resample_sse_sources = files(
        'reduceh_sse.cpp',
        'reducev_sse.cpp',
    )

    resample_components += static_library('resample_sse',
        resample_sse_sources,
        cpp_args: '-mssse3',
        dependencies: libvips_deps,
        gnu_symbol_visibility: 'hidden',
    )

    libvips_sources += resample_sse_sources
endif

if simd_type == simd_type_avx2
    resample_avx2_sources = files(
        # 'reduceh_avx2.cpp',
        'reducev_avx2.cpp',
    )

    resample_components += static_library('resample_avx2',
        resample_avx2_sources,
        cpp_args: '-mavx2',
        dependencies: libvips_deps,
        gnu_symbol_visibility: 'hidden',
    )

    libvips_sources += resample_avx2_sources
endif

if simd_type == simd_type_neon
    resample_neon_sources = files(
        'reduceh_neon.cpp',
        'reducev_neon.cpp',
    )

    resample_components += static_library('resample_neon',
        resample_neon_sources,
        dependencies: libvips_deps,
        gnu_symbol_visibility: 'hidden',
    )

    libvips_sources += resample_neon_sources
endif

resample_lib = static_library('resample',
    resample_sources,
    resample_headers,
    link_whole: resample_components,
    dependencies: libvips_deps,
    gnu_symbol_visibility: 'hidden',
)

libvips_components += resample_lib
