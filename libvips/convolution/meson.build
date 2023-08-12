convolution_sources = files(
    'canny.c',
    'edge.c',
    'convolution.c',
    'correlation.c',
    'conv.c',
    'conva.c',
    'convf.c',
    'convi.c',
    'convasep.c',
    'convsep.c',
    'compass.c',
    'fastcor.c',
    'spcor.c',
    'sharpen.c',
    'gaussblur.c',
)

convolution_headers = files(
    'pconvolution.h',
    'correlation.h',
)

convolution_components = []

libvips_sources += convolution_sources

if simd_type == simd_type_sse or simd_type == simd_type_avx2
    convolution_sse_sources = files(
        'convi_sse.cpp',
    )

    convolution_components += static_library('convolution_sse',
        convolution_sse_sources,
        cpp_args: '-mssse3',
        dependencies: libvips_deps,
        gnu_symbol_visibility: 'hidden',
    )

    libvips_sources += convolution_sse_sources
endif

if simd_type == simd_type_neon
    convolution_neon_sources = files(
        'convi_neon.cpp',
    )

    convolution_components += static_library('convolution_neon',
        convolution_neon_sources,
        dependencies: libvips_deps,
        gnu_symbol_visibility: 'hidden',
    )

    libvips_sources += convolution_neon_sources
endif

convolution_lib = static_library('convolution',
    convolution_sources,
    convolution_headers,
    link_whole: convolution_components,
    dependencies: libvips_deps,
    gnu_symbol_visibility: 'hidden',
)

libvips_components += convolution_lib
