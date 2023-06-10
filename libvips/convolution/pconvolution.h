/* base class for all convolution operations
 */

/*

    Copyright (C) 1991-2005 The National Gallery

    This library is free software; you can redistribute it and/or
    modify it under the terms of the GNU Lesser General Public
    License as published by the Free Software Foundation; either
    version 2.1 of the License, or (at your option) any later version.

    This library is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU 
    Lesser General Public License for more details.

    You should have received a copy of the GNU Lesser General Public
    License along with this library; if not, write to the Free Software
    Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA
    02110-1301  USA

 */

/*

    These files are distributed with VIPS - http://www.vips.ecs.soton.ac.uk

 */

#ifndef VIPS_PCONVOLUTION_H
#define VIPS_PCONVOLUTION_H

#ifdef __cplusplus
extern "C" {
#endif /*__cplusplus*/

#include <vips/vector.h>

#define VIPS_TYPE_CONVOLUTION (vips_convolution_get_type())
#define VIPS_CONVOLUTION( obj ) \
	(G_TYPE_CHECK_INSTANCE_CAST( (obj), \
		VIPS_TYPE_CONVOLUTION, VipsConvolution ))
#define VIPS_CONVOLUTION_CLASS( klass ) \
	(G_TYPE_CHECK_CLASS_CAST( (klass), \
		VIPS_TYPE_CONVOLUTION, VipsConvolutionClass))
#define VIPS_IS_CONVOLUTION( obj ) \
	(G_TYPE_CHECK_INSTANCE_TYPE( (obj), VIPS_TYPE_CONVOLUTION ))
#define VIPS_IS_CONVOLUTION_CLASS( klass ) \
	(G_TYPE_CHECK_CLASS_TYPE( (klass), VIPS_TYPE_CONVOLUTION ))
#define VIPS_CONVOLUTION_GET_CLASS( obj ) \
	(G_TYPE_INSTANCE_GET_CLASS( (obj), \
		VIPS_TYPE_CONVOLUTION, VipsConvolutionClass ))

typedef struct _VipsConvolution VipsConvolution;

struct _VipsConvolution {
	VipsOperation parent_instance;

	VipsImage *in;
	VipsImage *out;
	VipsImage *mask;

	/* @mask cast ready for processing.
	 */
	VipsImage *M;
};

typedef struct _VipsConvolutionClass {
	VipsOperationClass parent_class;

} VipsConvolutionClass;

GType vips_convolution_get_type( void );

#if HAVE_SIMD
void convi_uchar_simd(VipsPel *pout, VipsPel *pin,
	int32_t n, int32_t ne, int32_t offset, int32_t *restrict offsets,
	int16_t *restrict mant, int32_t exp);
#endif /*HAVE_SIMD*/

#ifdef __cplusplus
}
#endif /*__cplusplus*/

#endif /*VIPS_PCONVOLUTION_H*/


