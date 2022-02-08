#pragma once

#include "../deps/sutil/vec_math.h"

SUTIL_INLINE SUTIL_HOSTDEVICE float2 fabsf(const float2 &v)
{ return make_float2(fabsf(v.x), fabsf(v.y)); }

SUTIL_INLINE SUTIL_HOSTDEVICE float3 fabsf(const float3 &v)
{ return make_float3(fabsf(v.x), fabsf(v.y), fabsf(v.z)); }

/// GLSL mod (https://docs.gl/sl4/mod)
SUTIL_INLINE SUTIL_HOSTDEVICE float mod(float x, float y)
{ return x - y * floorf(x / y); }

SUTIL_INLINE SUTIL_HOSTDEVICE float3 mod(const float3 &v, float a)
{ return make_float3(mod(v.x, a), mod(v.y, a), mod(v.z, a)); }