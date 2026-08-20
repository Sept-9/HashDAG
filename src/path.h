#pragma once

#include "typedefs.h"
#include "cuda_math.h"

struct Path
{
public:
	uint3 path;
	
	HOST_DEVICE Path(uint3 path)
		: path(path)
	{
	}
	HOST_DEVICE Path(uint32 x, uint32 y, uint32 z)
		: path(make_uint3(x, y, z))
	{
	}

	HOST_DEVICE void ascend(uint32 levels)
	{
		path.x >>= levels;
		path.y >>= levels;
		path.z >>= levels;
	}
	HOST_DEVICE void descend(uint8 child)
	{
		path.x <<= 1;
		path.y <<= 1;
		path.z <<= 1;
		path.x |= (child & 0x4u) >> 2;
		path.y |= (child & 0x2u) >> 1;
		path.z |= (child & 0x1u) >> 0;
	}

	HOST_DEVICE float3 as_position(uint32 extraShift = 0) const
	{
		return make_float3(
			float(path.x << extraShift),
			float(path.y << extraShift),
			float(path.z << extraShift)
		);
	}

	// level: level of the child!
	HOST_DEVICE uint8 child_index(uint32 level, uint32 totalLevels) const
	{
		check(level <= totalLevels);
		return uint8(
			(((path.x >> (totalLevels - level) & 0x1) == 0) ? 0 : 4) |
			(((path.y >> (totalLevels - level) & 0x1) == 0) ? 0 : 2) |
			(((path.z >> (totalLevels - level) & 0x1) == 0) ? 0 : 1));
	}

	HOST_DEVICE bool is_null() const
	{
		return path.x == 0 && path.y == 0 && path.z == 0;
	}

public:
	DEVICE static Path load(int32 x, int32 y, cudaSurfaceObject_t surface)
	{
#ifdef __CUDA_ARCH__
		Path path;
		path.path = make_uint3(surf2Dread<uint4>(surface, x * sizeof(uint4), y));
		return path;
#else
		check(false);
		return {};
#endif
	}
	DEVICE void store(int32 x, int32 y, cudaSurfaceObject_t surface)
	{
#ifdef __CUDA_ARCH__
		surf2Dwrite(make_uint4(path.x, path.y, path.z, 0), surface, x * sizeof(uint4), y);
#endif
	}
	// LOD: store path together with the hit level and (optionally) an axis-projected
	// coverage triplet into the 4th uint of the surface. Layout:
	//   [ cov_z : 8 | cov_y : 8 | cov_x : 8 | hitLevel : 8 ]
	// The default coverage 0xFFFFFF encodes (1,1,1) i.e. a fully opaque hit, which
	// disables alpha compositing in trace_shadows. trace_paths always writes with
	// the default; trace_colors overwrites the coverage bits for LOD-A pixels once
	// it has resolved the color-tree node.
	DEVICE void store_with_level(
		int32 x,
		int32 y,
		cudaSurfaceObject_t surface,
		uint32 hitLevel,
		uint32 coverage24 = 0xFFFFFFu)
	{
#ifdef __CUDA_ARCH__
		const uint32 w = (hitLevel & 0xFFu) | ((coverage24 & 0xFFFFFFu) << 8);
		surf2Dwrite(make_uint4(path.x, path.y, path.z, w), surface, x * sizeof(uint4), y);
#endif
	}

	// Read-modify-write: update only the coverage bits at the given pixel, keeping
	// the existing path and hitLevel intact. Used by trace_colors when it wants to
	// annotate a pixel with the LOD node's precomputed coverage without touching
	// the geometry data already produced by trace_paths.
	DEVICE static void store_coverage_only(
		int32 x,
		int32 y,
		cudaSurfaceObject_t surface,
		uint32 coverage24)
	{
#ifdef __CUDA_ARCH__
		const uint4 raw = surf2Dread<uint4>(surface, x * sizeof(uint4), y);
		const uint32 w = (raw.w & 0xFFu) | ((coverage24 & 0xFFFFFFu) << 8);
		surf2Dwrite(make_uint4(raw.x, raw.y, raw.z, w), surface, x * sizeof(uint4), y);
#else
		(void)x; (void)y; (void)surface; (void)coverage24;
#endif
	}

	// LOD: load path together with the hit level (4th uint in the surface).
	// If outCoverage24 is non-null it also receives the packed coverage triplet.
	DEVICE static Path load_with_level(
		int32 x,
		int32 y,
		cudaSurfaceObject_t surface,
		uint32& outHitLevel,
		uint32* outCoverage24 = nullptr)
	{
#ifdef __CUDA_ARCH__
		const uint4 raw = surf2Dread<uint4>(surface, x * sizeof(uint4), y);
		Path p;
		p.path = make_uint3(raw);
		outHitLevel = raw.w & 0xFFu;
		if (outCoverage24) *outCoverage24 = (raw.w >> 8) & 0xFFFFFFu;
		return p;
#else
		(void)x; (void)y; (void)surface;
		outHitLevel = 0;
		if (outCoverage24) *outCoverage24 = 0xFFFFFFu;
		check(false);
		return {};
#endif
	}

private:
	HOST_DEVICE Path() {}
};