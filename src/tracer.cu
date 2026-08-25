#include "tracer.h"
#include "dags/basic_dag/basic_dag.h"
#include "dags/hash_dag/hash_dag.h"
#include "dags/hash_dag/hash_dag_colors.h"

// order: (shouldFlipX, shouldFlipY, shouldFlipZ)
DEVICE uint8 next_child(uint8 order, uint8 mask)
{
	for (uint8 child = 0; child < 8; ++child)
	{
		uint8 childInOrder = child ^ order;
		if (mask & (1u << childInOrder))
			return childInOrder;
	}
	check(false);
	return 0;
}

// The part of the traversal test that only depends on the node's bounding box. Split out of
// compute_intersection_mask so that a caller which already needs `tmin` (the LOD test) can reuse
// it instead of recomputing the identical slab test one line earlier.
// 遍历测试中只依赖节点包围盒的那一部分。从 compute_intersection_mask 里拆出来，好让本来就
// 需要 tmin 的调用方（LOD 判据）直接复用，而不必在它前一行把完全相同的 slab test 再算一遍。
struct NodeSlab
{
	float3 centerRelativeToRay;
	float3 tmid;
	float tmin;
	float tmax;
};

template<typename TDAG>
DEVICE NodeSlab compute_node_slab(
	uint32 level,
	const Path& path,
	const TDAG& dag,
	const float3& rayOrigin,
	const float3& rayDirectionInverted)
{
	// Find node center = .5 * (boundsMin + boundsMax) + .5f
	const uint32 shift = dag.levels - level;

	const float radius = float(1u << (shift - 1));
	const float3 center = make_float3(radius) + path.as_position(shift);

	NodeSlab slab;
	slab.centerRelativeToRay = center - rayOrigin;

	// Ray intersection with axis-aligned planes centered on the node
	// => rayOrg + tmid * rayDir = center
	slab.tmid = slab.centerRelativeToRay * rayDirectionInverted;

	// t-values for where the ray intersects the slabs centered on the node
	// and extending to the side of the node
	{
		const float3 slabRadius = radius * abs(rayDirectionInverted);
		const float3 pmin = slab.tmid - slabRadius;
		slab.tmin = max(max(pmin), .0f);

		const float3 pmax = slab.tmid + slabRadius;
		slab.tmax = min(pmax);
	}

	return slab;
}

template<bool isRoot>
DEVICE uint8 compute_intersection_mask(
	const NodeSlab& slab,
	const float3& rayDirection)
{
	const float3 centerRelativeToRay = slab.centerRelativeToRay;
	const float3 tmid = slab.tmid;
	const float tmin = slab.tmin;
	const float tmax = slab.tmax;

	// Check if we actually hit the root node
	// This test may not be entirely safe due to float precision issues.
	// especially on lower levels. For the root node this seems OK, though.
	if (isRoot && (tmin >= tmax))
	{
		return 0;
	}

	// Identify first child that is intersected
	// NOTE: We assume that we WILL hit one child, since we assume that the
	//       parents bounding box is hit.
	// NOTE: To safely get the correct node, we cannot use o+ray_tmin*d as the
	//       intersection point, since this point might lie too close to an
	//       axis plane. Instead, we use the midpoint between max and min which
	//       will lie in the correct node IF the ray only intersects one node.
	//       Otherwise, it will still lie in an intersected node, so there are
	//       no false positives from this.
	uint8 intersectionMask = 0;
	{
		const float3 pointOnRay = (0.5f * (tmin + tmax)) * rayDirection;

		uint8 const firstChild =
			((pointOnRay.x >= centerRelativeToRay.x) ? 4 : 0) +
			((pointOnRay.y >= centerRelativeToRay.y) ? 2 : 0) +
			((pointOnRay.z >= centerRelativeToRay.z) ? 1 : 0);

		intersectionMask |= (1u << firstChild);
	}

	// We now check the points where the ray intersects the X, Y and Z plane.
	// If the intersection is within (ray_tmin, ray_tmax) then the intersection
	// point implies that two voxels will be touched by the ray. We find out
	// which voxels to mask for an intersection point at +X, +Y by setting
	// ALL voxels at +X and ALL voxels at +Y and ANDing these two masks.
	//
	// NOTE: When the intersection point is close enough to another axis plane,
	//       we must check both sides or we will get robustness issues.
	const float epsilon = 1e-4f;

	if (tmin <= tmid.x && tmid.x <= tmax)
	{
		const float3 pointOnRay = tmid.x * rayDirection;

		uint8 A = 0;
		if (pointOnRay.y >= centerRelativeToRay.y - epsilon) A |= 0xCC;
		if (pointOnRay.y <= centerRelativeToRay.y + epsilon) A |= 0x33;

		uint8 B = 0;
		if (pointOnRay.z >= centerRelativeToRay.z - epsilon) B |= 0xAA;
		if (pointOnRay.z <= centerRelativeToRay.z + epsilon) B |= 0x55;

		intersectionMask |= A & B;
	}
	if (tmin <= tmid.y && tmid.y <= tmax)
	{
		const float3 pointOnRay = tmid.y * rayDirection;

		uint8 C = 0;
		if (pointOnRay.x >= centerRelativeToRay.x - epsilon) C |= 0xF0;
		if (pointOnRay.x <= centerRelativeToRay.x + epsilon) C |= 0x0F;

		uint8 D = 0;
		if (pointOnRay.z >= centerRelativeToRay.z - epsilon) D |= 0xAA;
		if (pointOnRay.z <= centerRelativeToRay.z + epsilon) D |= 0x55;

		intersectionMask |= C & D;
	}
	if (tmin <= tmid.z && tmid.z <= tmax)
	{
		const float3 pointOnRay = tmid.z * rayDirection;

		uint8 E = 0;
		if (pointOnRay.x >= centerRelativeToRay.x - epsilon) E |= 0xF0;
		if (pointOnRay.x <= centerRelativeToRay.x + epsilon) E |= 0x0F;


		uint8 F = 0;
		if (pointOnRay.y >= centerRelativeToRay.y - epsilon) F |= 0xCC;
		if (pointOnRay.y <= centerRelativeToRay.y + epsilon) F |= 0x33;

		intersectionMask |= E & F;
	}

	return intersectionMask;
}

// Convenience overload for the call sites that have no use for the slab's t-values.
// 为那些用不到 slab t 值的调用点提供的便捷重载。
template<bool isRoot, typename TDAG>
DEVICE uint8 compute_intersection_mask(
	uint32 level,
	const Path& path,
	const TDAG& dag,
	const float3& rayOrigin,
	const float3& rayDirection,
	const float3& rayDirectionInverted)
{
	return compute_intersection_mask<isRoot>(
		compute_node_slab(level, path, dag, rayOrigin, rayDirectionInverted),
		rayDirection);
}

struct StackEntry
{
	uint32 index;
	uint8 childMask;
	uint8 visitMask;
};

template<typename TDAG>
__global__ void Tracer::trace_paths(const TracePathsParams traceParams, const TDAG dag)
{
	// Target pixel coordinate
	const uint2 pixel = make_uint2(
		blockIdx.x * blockDim.x + threadIdx.x,
		blockIdx.y * blockDim.y + threadIdx.y);

	if (pixel.x >= imageWidth || pixel.y >= imageHeight)
		return; // outside.

	// Pre-calculate per-pixel data
	const float3 rayOrigin = make_float3(traceParams.cameraPosition);
	const float3 rayDirection = make_float3(normalize(traceParams.rayMin + pixel.x * traceParams.rayDDx + pixel.y * traceParams.rayDDy - traceParams.cameraPosition));

	const float3 rayDirectionInverse = make_float3(make_double3(1. / rayDirection.x, 1. / rayDirection.y, 1. / rayDirection.z));
	const uint8 rayChildOrder =
		(rayDirection.x < 0.f ? 4 : 0) +
		(rayDirection.y < 0.f ? 2 : 0) +
		(rayDirection.z < 0.f ? 1 : 0);

	// State
	uint32 level = 0;
	Path path(0, 0, 0);

	StackEntry stack[MAX_LEVELS];
	StackEntry cache;
	Leaf cachedLeaf; // needed to iterate on the last few levels

	cache.index = dag.get_first_node_index();
	cache.childMask = Utils::child_mask(dag.get_node(0, cache.index));
	cache.visitMask = cache.childMask & compute_intersection_mask<true>(0, path, dag, rayOrigin, rayDirection, rayDirectionInverse);

	// LOD: level at which the traversal terminates. dag.levels means "full descent, hit a 1-voxel"
	// (the original behaviour). Set to 0 on background pixels, or a smaller value when LOD kicks in.
	uint32 hitLevel = dag.levels;

#if ENABLE_PREFILTERED_SHADING
	// Prefiltered appearance: packed 6-direction relief histogram of the node this ray stops
	// on. Stays 0 for background pixels, for full descents (where the geometry is exact) and
	// for LOD nodes with no internal relief — in all three cases the shading pass must use
	// the plain box normal, which is exactly what 0 tells it to do.
	// 预滤波外观：本条光线终止所在节点的打包 6 方向起伏直方图。背景像素、完全下降到底（几何
	// 精确）、以及没有内部起伏的 LOD 节点都保持为 0 —— 这三种情况着色阶段都必须使用普通盒
	// 法线，而 0 正是这个含义。
	uint32 prefilterPacked = LodPrefilter::C_emptyHistogram;
#endif

#if ENABLE_COVERAGE_AWARE_LOD
	uint32 coverageDescentBudget = uint32(PREFILTER_MAX_COVERAGE_DESCENT);
#endif

	// Traverse DAG
	for (;;)
	{
		// Ascend if there are no children left.
		{
			uint32 newLevel = level;
			while (newLevel > 0 && !cache.visitMask)
			{
				newLevel--;
				cache = stack[newLevel];
			}

			if (newLevel == 0 && !cache.visitMask)
			{
				path = Path(0, 0, 0);
				hitLevel = 0;
				break;
			}

			path.ascend(level - newLevel);
			level = newLevel;
		}

		// Find next child in order by the current ray's direction
		const uint8 nextChild = next_child(rayChildOrder, cache.visitMask);

		// Mark it as handled
		cache.visitMask &= ~(1u << nextChild);

		// Intersect that child with the ray
		{
			path.descend(nextChild);
			stack[level] = cache;
			level++;

			// If we're at the final level, we have intersected a single voxel.
			if (level == dag.levels)
			{
				break;
			}

			// Resolve the node we just descended into, once: the LOD test below and the
			// descent further down both need its index and its header word. cache.index is
			// only maintained above the leaf level, which bounds how deep an index can be
			// resolved at all; below that the geometry lives in cachedLeaf instead.
			// 一次性解析刚下降进入的这个节点：下面的 LOD 判据和再下面的下降都需要它的下标与
			// 头字。cache.index 仅在叶层以上被维护，这限定了下标最深能解析到哪一层；再往下
			// 几何存放在 cachedLeaf 中。
			const bool isInteriorNode = level < dag.leaf_level();
			const bool hasChildIndex = level <= dag.leaf_level();
			const uint32 childIndex = hasChildIndex
				? dag.get_child_index(level - 1, cache.index, cache.childMask, nextChild)
				: 0u;
			const uint32 childHeader = isInteriorNode ? dag.get_node(level, childIndex) : 0u;

			// Slab test for the node we just descended into. Computed once here and handed to
			// compute_intersection_mask below, so the LOD test costs no arithmetic of its own.
			// 刚下降进入的这个节点的 slab 测试。在这里只算一次，然后交给下面的
			// compute_intersection_mask 复用，因此 LOD 判据本身不产生任何额外算术开销。
			const NodeSlab slab = compute_node_slab(level, path, dag, rayOrigin, rayDirectionInverse);

			// LOD: stop here if this node already projects to fewer than the configured pixel
			// threshold. slab.tmin is the ray's entry t into the node; since rayDirection is
			// unit-length, that t equals the world-space camera distance.
			if (traceParams.lodScale > 0.f)
			{
				const float voxelSize = float(1u << (dag.levels - level));
				if (voxelSize < slab.tmin * traceParams.lodScale)
				{
					uint32 lodPrefilter = 0u;
#if ENABLE_PREFILTERED_SHADING
					if (dag.has_prefilter() && hasChildIndex)
						lodPrefilter = dag.get_prefilter(level, childIndex, childHeader);
#endif
#if ENABLE_COVERAGE_AWARE_LOD
					const uint32 viewAxis = LodPrefilter::dominant_axis(rayDirection);
					if (coverageDescentBudget > 0 && LodPrefilter::pass_axis(lodPrefilter, viewAxis))
					{
						--coverageDescentBudget;
					}
					else
#endif
					{
						hitLevel = level;
#if ENABLE_PREFILTERED_SHADING
						prefilterPacked = lodPrefilter;
#endif
						break;
					}
				}
			}

			// Are we in an internal node?
			if (isInteriorNode)
			{
				cache.index = childIndex;
				cache.childMask = Utils::child_mask(childHeader);
				cache.visitMask = cache.childMask &	compute_intersection_mask<false>(slab, rayDirection);
			}
			else
			{
				/* The second-to-last and last levels are different: the data
				 * of these two levels (2^3 voxels) are packed densely into a
				 * single 64-bit word.
				 */
				uint8 childMask;

				if (level == dag.leaf_level())
				{
					cachedLeaf = dag.get_leaf(childIndex);
					childMask = cachedLeaf.get_first_child_mask();
				}
				else
				{
					childMask = cachedLeaf.get_second_child_mask(nextChild);
				}

				// No need to set the index for bottom nodes
				cache.childMask = childMask;
				cache.visitMask = cache.childMask & compute_intersection_mask<false>(slab, rayDirection);
			}
		}
	}

	// LOD: align partial-descent paths so they share the same bit layout as full-descent paths.
	// After `hitLevel` calls to path.descend(), the descent bits live in positions [0, hitLevel-1]
	// (descend shifts existing bits left and ORs the new bit at the bottom). Full-descent paths,
	// however, have their first descent at bit (dag.levels - 1) and the last descent at bit 0,
	// which is the layout child_index(lv, dag.levels) and the DAG-coordinate interpretation
	// (path as world position in DAG units) both assume. Shift LOD paths left to match.
	if (hitLevel > 0 && hitLevel < dag.levels)
	{
		const uint32 alignShift = dag.levels - hitLevel;
		path.path.x <<= alignShift;
		path.path.y <<= alignShift;
		path.path.z <<= alignShift;
	}
	path.store_with_level(pixel.x, imageHeight - 1 - pixel.y, traceParams.pathsSurface, hitLevel);

#if ENABLE_PREFILTERED_SHADING
	// Same y-flip as store_with_level above: this pass writes flipped, the colour and shadow
	// passes read unflipped.
	// 与上面的 store_with_level 使用相同的 y 翻转：本阶段翻转写入，颜色与阴影阶段不翻转读取。
	if (traceParams.prefilterBuffer)
	{
		traceParams.prefilterBuffer[(imageHeight - 1 - pixel.y) * imageWidth + pixel.x] = prefilterPacked;
	}
#endif
}

template<typename TDAG, typename TDAGColors>
__global__ void Tracer::trace_colors(const TraceColorsParams traceParams, const TDAG dag, const TDAGColors colors)
{
	const uint2 pixel = make_uint2(
		blockIdx.x * blockDim.x + threadIdx.x,
		blockIdx.y * blockDim.y + threadIdx.y);

	if (pixel.x >= imageWidth || pixel.y >= imageHeight)
		return; // outside

	const auto setColorImpl = [&](uint32 color)
	{
		surf2Dwrite(color, traceParams.colorsSurface, (int)sizeof(uint32) * pixel.x, pixel.y, cudaBoundaryModeClamp);
	};

	// LOD: hitLevel is stored in the 4th uint of the pathsSurface. Equals dag.levels for
	// non-LOD pixels (full descent to a 1-voxel), or a smaller level when LOD stopped early.
	uint32 hitLevel = dag.levels;
	const Path path = Path::load_with_level(pixel.x, pixel.y, traceParams.pathsSurface, hitLevel);
	if (path.is_null())
    {
        setColorImpl(ColorUtils::float3_to_rgb888(make_float3(187, 242, 250) / 255.f));
        return;
	}

	const float toolStrength = traceParams.toolInfo.strength(path);
	const auto setColor = [&](uint32 color)
	{
#if TOOL_OVERLAY
		if (toolStrength > 0)
		{
			color = ColorUtils::float3_to_rgb888(lerp(ColorUtils::rgb888_to_float3(color), make_float3(1, 0, 0), clamp(100 * toolStrength, 0.f, .5f)));
		}
#endif
        setColorImpl(color);
    };

    const auto invalidColor = [&]()
    {
        uint32 b = (path.path.x ^ path.path.y ^ path.path.z) & 0x1;
        setColor(ColorUtils::float3_to_rgb888(make_float3(1, b, 1.f - b)));
    };

	const uint32 colorTreeLevels = colors.get_color_tree_levels();
	const bool lodMode = (hitLevel < dag.levels);

	// LOD-A: LOD stopped strictly above the color leaf level → fetch the precomputed average
	// from the corresponding color tree internal node and return immediately. Only meaningful
	// when the color tree actually exists (HashDAG); BasicDAG sets colorTreeLevels == 0 so
	// this branch is dead there.
	//
	// Note we use `<` (not `<=`) because at level == colorTreeLevels the color tree pointer
	// is a *leaf* index (into `leaves[]` / `offsets[]`), not an internal-node index, and we
	// don't store averages for leaves. The leaf-level case is handled by the main loop using
	// the first block's (min+max)/2 instead.
	if (lodMode && hitLevel < colorTreeLevels && colors.has_node_averages())
	{
		uint32 colorNodeIndexLod = 0;
		bool validPath = true;
		for (uint32 lv = 1; lv <= hitLevel; ++lv)
		{
			const uint8 c = path.child_index(lv, dag.levels);
			colorNodeIndexLod = colors.get_child_index(lv - 1, colorNodeIndexLod, c);
			// A zero child index past the root means the path doesn't actually exist in the
			// color tree (e.g. on freshly-edited regions that haven't allocated color nodes).
			if (lv < hitLevel && colorNodeIndexLod == 0)
			{
				validPath = false;
				break;
			}
		}
		if (!validPath)
		{
			invalidColor();
			return;
		}
		const uint32 avg = colors.get_node_average_color(colorNodeIndexLod);
		setColor(avg);
		return;
	}

    uint64 nof_leaves = 0;
#if LOD_COLOR_SAMPLES >= 2
	uint64 lodVoxelCount = 0;
#endif
	uint32 debugColorsIndex = 0;

	uint32 colorNodeIndex = 0;
	typename TDAGColors::ColorLeaf colorLeaf = colors.get_default_leaf();

	uint32 level = 0;
	uint32 nodeIndex = dag.get_first_node_index();
	while (level < dag.leaf_level())
	{
		level++;

		// Find the current childmask and which subnode we are in
		const uint32 node = dag.get_node(level - 1, nodeIndex);
		const uint8 childMask = Utils::child_mask(node);
		const uint8 child = path.child_index(level, dag.levels);

		// Make sure the node actually exists
		if (!(childMask & (1 << child)))
		{
			setColor(0xFF00FF);
			return;
		}

		ASSUME(level > 0);
		if (level - 1 < colors.get_color_tree_levels())
		{
			colorNodeIndex = colors.get_child_index(level - 1, colorNodeIndex, child);
			if (level == colors.get_color_tree_levels())
			{
				check(nof_leaves == 0);
				colorLeaf = colors.get_leaf(colorNodeIndex);
			}
			else
			{
				// TODO nicer interface
				if (!colorNodeIndex)
				{
					invalidColor();
					return;
				}
			}
		}

		// Debug
		if (traceParams.debugColors == EDebugColors::Index ||
			traceParams.debugColors == EDebugColors::Position ||
			traceParams.debugColors == EDebugColors::ColorTree)
		{
			if (traceParams.debugColors == EDebugColors::Index &&
				traceParams.debugColorsIndexLevel == level - 1)
			{
				debugColorsIndex = nodeIndex;
			}
			if (level == dag.leaf_level())
			{
				if (traceParams.debugColorsIndexLevel == dag.leaf_level())
				{
					check(debugColorsIndex == 0);
					const uint32 childIndex = dag.get_child_index(level - 1, nodeIndex, childMask, child);
					debugColorsIndex = childIndex;
				}

				if (traceParams.debugColors == EDebugColors::Index)
				{
					setColor(Utils::murmurhash32(debugColorsIndex));
				}
				else if (traceParams.debugColors == EDebugColors::Position)
				{
					constexpr uint32 checkerSize = 0x7FF;
					float color = ((path.path.x ^ path.path.y ^ path.path.z) & checkerSize) / float(checkerSize);
					color = (color + 0.5) / 2;
					setColor(ColorUtils::float3_to_rgb888(Utils::has_flag(nodeIndex) ? make_float3(color, 0, 0) : make_float3(color)));
				}
				else
				{
					check(traceParams.debugColors == EDebugColors::ColorTree);
					const uint32 offset = dag.levels - colors.get_color_tree_levels();
					const float color = ((path.path.x >> offset) ^ (path.path.y >> offset) ^ (path.path.z >> offset)) & 0x1;
					setColor(ColorUtils::float3_to_rgb888(make_float3(color)));
				}
				return;
			}
			else
			{
				nodeIndex = dag.get_child_index(level - 1, nodeIndex, childMask, child);
				continue;
			}
		}

		//////////////////////////////////////////////////////////////////////////
		// Find out how many leafs are in the children preceding this
		//////////////////////////////////////////////////////////////////////////
		// If at final level, just count nof children preceding and exit
		if (level == dag.leaf_level())
		{
			for (uint8 childBeforeChild = 0; childBeforeChild < child; ++childBeforeChild)
			{
				if (childMask & (1u << childBeforeChild))
				{
					const uint32 childIndex = dag.get_child_index(level - 1, nodeIndex, childMask, childBeforeChild);
					const Leaf leaf = dag.get_leaf(childIndex);
					nof_leaves += Utils::popcll(leaf.to_64());
				}
			}
			// LOD: if we're stopping at leaf_level(), nof_leaves now points to the start
			// of this leaf-level node and is the offset we use to pick a representative
			// block; skip the per-bit refinement (which requires the full path).
			if (lodMode && hitLevel <= level)
			{
#if LOD_COLOR_SAMPLES >= 2
				const uint32 lodChildIndex = dag.get_child_index(level - 1, nodeIndex, childMask, child);
				lodVoxelCount = Utils::popcll(dag.get_leaf(lodChildIndex).to_64());
#endif
				break;
			}
			const uint32 childIndex = dag.get_child_index(level - 1, nodeIndex, childMask, child);
			const Leaf leaf = dag.get_leaf(childIndex);
			const uint64 leafBits = leaf.to_64();
			const uint8 leafBitIndex =
				(((path.path.x & 0x1) == 0) ? 0 : 4) |
				(((path.path.y & 0x1) == 0) ? 0 : 2) |
				(((path.path.z & 0x1) == 0) ? 0 : 1) |
				(((path.path.x & 0x2) == 0) ? 0 : 32) |
				(((path.path.y & 0x2) == 0) ? 0 : 16) |
				(((path.path.z & 0x2) == 0) ? 0 : 8);
			nof_leaves += Utils::popcll(leafBits & ((uint64(1) << leafBitIndex) - 1));

#if LOD_COLOR_SAMPLES >= 2
			if (lodMode)
			{
				lodVoxelCount = Utils::popcll((leafBits >> leafBitIndex) & 0xFFull);
			}
#endif

			break;
		}
		else
		{
			ASSUME(level > 0);
			if (level > colors.get_color_tree_levels())
			{
				// Otherwise, fetch the next node (and accumulate leaves we pass by)
				for (uint8 childBeforeChild = 0; childBeforeChild < child; ++childBeforeChild)
				{
					if (childMask & (1u << childBeforeChild))
					{
						const uint32 childIndex = dag.get_child_index(level - 1, nodeIndex, childMask, childBeforeChild);
						const uint32 childNode = dag.get_node(level, childIndex);
						nof_leaves += colors.get_leaves_count(level, childNode);
					}
				}
			}
			nodeIndex = dag.get_child_index(level - 1, nodeIndex, childMask, child);

			// LOD: stop here when we've reached the LOD level. nof_leaves at this point
			// is the offset of this LOD node's first surface voxel within the color leaf.
			if (lodMode && level >= hitLevel)
			{
#if LOD_COLOR_SAMPLES >= 2
				if (level >= colors.get_color_tree_levels())
				{
					lodVoxelCount = colors.get_leaves_count(level, dag.get_node(level, nodeIndex));
				}
#endif
				break;
			}
		}
	}

	if (!colorLeaf.is_valid() || !colorLeaf.is_valid_index(nof_leaves))
	{
	    invalidColor();
		return;
	}

	if (lodMode)
	{
#if LOD_COLOR_SAMPLES == 0
		setColor(ColorUtils::float3_to_rgb888(colorLeaf.get_color(nof_leaves).get_lod_average()));
#elif LOD_COLOR_SAMPLES == 1
		setColor(ColorUtils::float3_to_rgb888(colorLeaf.get_color(nof_leaves).get_color()));
#else
		const int numSamples = (lodVoxelCount < uint64(LOD_COLOR_SAMPLES)) ? int(lodVoxelCount) : int(LOD_COLOR_SAMPLES);

		float3 acc = make_float3(0.f);
		int valid = 0;
		for (int s = 0; s < numSamples; ++s)
		{
			const uint64 idx = nof_leaves + (uint64(s) * lodVoxelCount + lodVoxelCount / 2) / uint64(numSamples);
			if (!colorLeaf.is_valid_index(idx)) continue;
			acc = acc + ColorUtils::to_average_space(colorLeaf.get_color(idx).get_color());
			++valid;
		}

		if (valid > 0)
		{
			setColor(ColorUtils::float3_to_rgb888(ColorUtils::from_average_space(acc / float(valid))));
		}
		else
		{
			setColor(ColorUtils::float3_to_rgb888(colorLeaf.get_color(nof_leaves).get_color()));
		}
#endif
		return;
	}

	auto compressedColor = colorLeaf.get_color(nof_leaves);

	uint32 color =
		traceParams.debugColors == EDebugColors::ColorBits
		? compressedColor.get_debug_hash()
		: ColorUtils::float3_to_rgb888(
			traceParams.debugColors == EDebugColors::MinColor
			? compressedColor.get_min_color()
			: traceParams.debugColors == EDebugColors::MaxColor
			? compressedColor.get_max_color()
			: traceParams.debugColors == EDebugColors::Weight
			? make_float3(compressedColor.get_weight())
			: compressedColor.get_color());
	setColor(color);
}

template<typename TDAG>
inline __device__ bool intersect_ray_node_out_of_order(const TDAG& dag, const float3 rayOrigin, const float3 rayDirection)
{
    const float3 rayDirectionInverse = make_float3(make_double3(1. / rayDirection.x, 1. / rayDirection.y, 1. / rayDirection.z));

	// State
	uint32 level = 0;
	Path path(0, 0, 0);

	StackEntry stack[MAX_LEVELS];
	StackEntry cache;
	Leaf cachedLeaf; // needed to iterate on the last few levels

	cache.index = dag.get_first_node_index();
	cache.childMask = Utils::child_mask(dag.get_node(0, cache.index));
	cache.visitMask = cache.childMask & compute_intersection_mask<true>(0, path, dag, rayOrigin, rayDirection, rayDirectionInverse);

	// Traverse DAG
	for (;;)
	{
		// Ascend if there are no children left.
		{
			uint32 newLevel = level;
			while (newLevel > 0 && !cache.visitMask)
			{
				newLevel--;
				cache = stack[newLevel];
			}

			if (newLevel == 0 && !cache.visitMask)
			{
				path = Path(0, 0, 0);
				break;
			}

			path.ascend(level - newLevel);
			level = newLevel;
		}

		// Find next child in order by the current ray's direction
		const uint8 nextChild = 31 - __clz(cache.visitMask);

		// Mark it as handled
		cache.visitMask &= ~(1u << nextChild);

		// Intersect that child with the ray
		{
			path.descend(nextChild);
			stack[level] = cache;
			level++;

			// If we're at the final level, we have intersected a single voxel.
			if (level == dag.levels)
			{
			    return true;
			}

			// Are we in an internal node?
			if (level < dag.leaf_level())
			{
				cache.index = dag.get_child_index(level - 1, cache.index, cache.childMask, nextChild);
				cache.childMask = Utils::child_mask(dag.get_node(level, cache.index));
				cache.visitMask = cache.childMask &	compute_intersection_mask<false>(level, path, dag, rayOrigin, rayDirection, rayDirectionInverse);
			}
			else
			{
				/* The second-to-last and last levels are different: the data
				 * of these two levels (2^3 voxels) are packed densely into a
				 * single 64-bit word.
				 */
				uint8 childMask;

				if (level == dag.leaf_level())
				{
					const uint32 addr = dag.get_child_index(level - 1, cache.index, cache.childMask, nextChild);
					cachedLeaf = dag.get_leaf(addr);
					childMask = cachedLeaf.get_first_child_mask();
				}
				else
				{
					childMask = cachedLeaf.get_second_child_mask(nextChild);
				}

				// No need to set the index for bottom nodes
				cache.childMask = childMask;
				cache.visitMask = cache.childMask & compute_intersection_mask<false>(level, path, dag, rayOrigin, rayDirection, rayDirectionInverse);
			}
		}
	}
	return false;
}

// Directed towards the sun
HOST_DEVICE float3 sun_direction()
{
    return normalize(make_float3(0.3f, 1.f, 0.5f));
}

constexpr float C_specularRoughness = 0.5f;
constexpr float C_specularAlpha = C_specularRoughness * C_specularRoughness;
constexpr float C_specularAlpha2 = C_specularAlpha * C_specularAlpha;

#if USE_GGX_BRDF
constexpr float C_invPi = 0.318309886183791f;
constexpr float C_dielectricF0 = 0.04f;
constexpr float C_sunIrradiance = 2.6f;
constexpr float C_skyIrradiance = 1.26f;
constexpr float C_normalVarianceScale = 1.f;

HOST_DEVICE float ggx_distribution(float nDotH, float alpha2)
{
	const float d = nDotH * nDotH * (alpha2 - 1.f) + 1.f;
	return alpha2 * C_invPi / max(1e-8f, d * d);
}

HOST_DEVICE float smith_visibility(float nDotL, float nDotV, float alpha2)
{
	const float lambdaV = nDotL * sqrt(nDotV * nDotV * (1.f - alpha2) + alpha2);
	const float lambdaL = nDotV * sqrt(nDotL * nDotL * (1.f - alpha2) + alpha2);
	return 0.5f / max(1e-8f, lambdaV + lambdaL);
}

HOST_DEVICE float schlick_fresnel(float vDotH)
{
	const float m = clamp(1.f - vDotH, 0.f, 1.f);
	const float m2 = m * m;
	return C_dielectricF0 + (1.f - C_dielectricF0) * (m2 * m2 * m);
}

HOST_DEVICE float specular_lobe(float3 N, float3 V, float3 L, float3 H, float alpha2)
{
	const float nDotL = dot(N, L);
	const float nDotV = dot(N, V);
	if (nDotL <= 0.f || nDotV <= 0.f) return 0.f;
	return ggx_distribution(max(0.f, dot(N, H)), alpha2) * smith_visibility(nDotL, nDotV, alpha2) * nDotL;
}

HOST_DEVICE float3 combine_shading(float3 albedo, float diffuse, float specular, float vDotH, bool isShadow)
{
	const float3 diffuseBRDF = albedo * C_invPi;

	float3 color = diffuseBRDF * C_skyIrradiance;
	if (!isShadow)
	{
		const float fresnel = schlick_fresnel(vDotH);
		color = color + diffuseBRDF * ((1.f - fresnel) * diffuse * C_sunIrradiance);
		color = color + make_float3(fresnel * specular * C_sunIrradiance);
	}
	return color;
}
#else
HOST_DEVICE float specular_lobe(float3 N, float3 /*V*/, float3 /*L*/, float3 H, float /*alpha2*/)
{
	return pow(max(0.f, dot(N, H)), 32.f);
}

HOST_DEVICE float3 combine_shading(float3 albedo, float diffuse, float specular, float /*vDotH*/, bool isShadow)
{
	float3 color = albedo * 0.4f;
	if (!isShadow)
	{
		color = color + albedo * diffuse * 0.8f;
		color = color + make_float3(1.f) * specular * 0.3f;
	}
	return color;
}
#endif

HOST_DEVICE float3 applyFog(float3 rgb,      // original color of the pixel
                            double distance, // camera to point distance
                            double3 rayDir,   // camera to point vector
                            double3 rayOri,
                            float fogDensity)  // camera position
{
#if 0
    constexpr float fogDensity = 0.0001f;
    constexpr float c = 1.f;
    constexpr float heightOffset = 20000.f;
    constexpr float heightScale = 1.f;
    double fogAmount = c * exp((heightOffset - rayOri.y * heightScale) * fogDensity) * (1.0 - exp(-distance * rayDir.y * fogDensity)) / rayDir.y;
#else
    fogDensity *= 0.00001f;
    double fogAmount = 1.0 - exp(-distance * fogDensity);
#endif
    double sunAmount = 1.01f * max(dot(rayDir, make_double3(sun_direction())), 0.0);
    float3 fogColor = lerp(ColorUtils::to_linear(make_float3(187, 242, 250) / 255.f), // blue
                              ColorUtils::to_linear(make_float3(1.0f)), // white
                              float(pow(sunAmount, 30.0)));
    return lerp(rgb, fogColor, clamp(float(fogAmount), 0.f, 1.f));
}

HOST_DEVICE double3 ray_box_intersection(double3 orig, double3 dir, double3 box_min, double3 box_max)
{
    double3 tmin = (box_min - orig) / dir;
    double3 tmax = (box_max - orig) / dir;

    double3 real_min = min(tmin, tmax);
    double3 real_max = max(tmin, tmax);

    // double minmax = min(min(real_max.x, real_max.y), real_max.z);
    double maxmin = max(max(real_min.x, real_min.y), real_min.z);

    // checkf(minmax >= maxmin, "%f > %f", minmax, maxmin);
    return orig + dir * maxmin;
}

template<typename TDAG>
__global__ void Tracer::trace_shadows(const TraceShadowsParams params, const TDAG dag)
{
    const uint2 pixel = make_uint2(
            blockIdx.x * blockDim.x + threadIdx.x,
            blockIdx.y * blockDim.y + threadIdx.y);

    if (pixel.x >= imageWidth || pixel.y >= imageHeight)
        return; // outside

    const auto setColorImpl = [&](float3 color)
    {
        const uint32 finalColor = ColorUtils::float3_to_rgb888(ColorUtils::from_linear(color));
        surf2Dwrite(finalColor, params.colorsSurface, (int)sizeof(uint32) * pixel.x, pixel.y, cudaBoundaryModeClamp);
    };
    const auto setColor = [&](float light, double distance, double3 direction)
    {
        const uint32 colorInt = surf2Dread<uint32>(params.colorsSurface, pixel.x * sizeof(uint32), pixel.y);
        float3 color = ColorUtils::to_linear(ColorUtils::rgb888_to_float3(colorInt));

        color = color * clamp(0.5f + light, 0.f, 1.f);
		//color = color * light;

        color = applyFog(
                color,
                distance,
                direction,
                params.cameraPosition,
                params.fogDensity);

        setColorImpl(color);
    };
	const auto setBRDFColor = [&](float light, double distance, double3 direction, double3 normal, bool isShadow)
		{
			const float3 L = sun_direction();
			const float3 V = make_float3( - normalize(direction));
			const float3 H = normalize(L + V);
			const float3 N = make_float3(normal);
			const uint32 colorInt = surf2Dread<uint32>(params.colorsSurface, pixel.x * sizeof(uint32), pixel.y);
			float3 albedo = ColorUtils::to_linear(ColorUtils::rgb888_to_float3(colorInt));
			// Lambert
			float diffuse = max(0.f, dot(N, L));

			float specular = specular_lobe(N, V, L, H, C_specularAlpha2);

			float3 color = combine_shading(albedo, diffuse, specular, max(0.f, dot(V, H)), isShadow);
			//color = color * clamp(0.5f + light, 0.f, 1.f);
			//color = color * light;

			color = applyFog(
				color,
				distance,
				direction,
				params.cameraPosition,
				params.fogDensity);

			setColorImpl(color);
		};
#if ENABLE_PREFILTERED_SHADING
	// Prefiltered appearance: same shading model as setBRDFColor, except that the single
	// geometric normal is replaced by a weighted set of normals.
	//
	// A LOD node's apparent surface is split in two parts:
	//   * the relief part — for every axis direction d the histogram stores how much
	//     *internal* face area points along n_d, as a fraction of the node's own projected
	//     face area. A direction can only be seen from the camera if dot(n_d, V) > 0, and
	//     what the pixel sees of it is foreshortened by that same factor, so the weight of
	//     direction d is relief[d] * max(0, dot(n_d, V)).
	//   * the flat part — whatever fraction of the projected face the relief does not claim
	//     is still a plain facet of the bounding box, shaded with the geometric box normal.
	// The two are then normalised together, which makes the whole thing degrade gracefully:
	// a node with an all-zero histogram gets weight 1 on the flat part and reproduces
	// setBRDFColor bit for bit.
	//
	// 预滤波外观：着色模型与 setBRDFColor 相同，只是把单一几何法线换成一组加权法线。
	//
	// 一个 LOD 节点的表观表面被拆成两部分：
	//   * 起伏部分 —— 直方图对每个坐标轴方向 d 记录了有多少"内部"面面积朝向 n_d，以节点自身
	//     投影面面积为单位。只有 dot(n_d, V) > 0 的方向才可能被相机看到，而像素看到的量又被
	//     同一个因子透视压缩，因此方向 d 的权重是 relief[d] * max(0, dot(n_d, V))。
	//   * 平坦部分 —— 投影面上没被起伏占据的那部分仍然是包围盒的平面，用几何盒法线着色。
	// 两部分一起归一化，使整套方案能平滑退化：直方图全零的节点会把权重 1 全给平坦部分，从而
	// 逐位复现 setBRDFColor 的结果。
	const auto setPrefilteredBRDFColor = [&](double distance, double3 direction, double3 boxNormal, uint32 packed, bool isShadow)
		{
			const float3 L = sun_direction();
			const float3 V = make_float3( - normalize(direction));
			const float3 H = normalize(L + V);
			const uint32 colorInt = surf2Dread<uint32>(params.colorsSurface, pixel.x * sizeof(uint32), pixel.y);
			float3 albedo = ColorUtils::to_linear(ColorUtils::rgb888_to_float3(colorInt));

			const float3 boxN = make_float3(boxNormal);

			float weightSum = 0.f;
			float diffuse = 0.f;
			float specular = 0.f;

#if USE_GGX_BRDF && PREFILTER_ROUGHNESS_FROM_VARIANCE
			float momentAlongBoxAxis = 0.f;
#endif

			// Relief part / 起伏部分
			for (uint32 d = 0; d < LodPrefilter::C_numDirections; ++d)
			{
				const float relief = LodPrefilter::bin_to_relief(LodPrefilter::unpack_bin(packed, d), d);
				if (relief <= 0.f) continue;

				const float3 N = LodPrefilter::direction_normal(d);
				const float nDotV = dot(N, V);
				if (nDotV <= 0.f) continue;

				const float weight = relief * nDotV;
				weightSum += weight;
				diffuse += weight * max(0.f, dot(N, L));
#if USE_GGX_BRDF && PREFILTER_ROUGHNESS_FROM_VARIANCE
				const float axisAlignment = dot(N, boxN);
				momentAlongBoxAxis += weight * axisAlignment * axisAlignment;
#else
				specular += weight * specular_lobe(N, V, L, H, C_specularAlpha2);
#endif
			}

			// Flat part / 平坦部分
			{
				const float weight = max(0.f, 1.f - weightSum);
				weightSum += weight;
				diffuse += weight * max(0.f, dot(boxN, L));
#if USE_GGX_BRDF && PREFILTER_ROUGHNESS_FROM_VARIANCE
				momentAlongBoxAxis += weight;
#else
				specular += weight * specular_lobe(boxN, V, L, H, C_specularAlpha2);
#endif
			}

			if (weightSum > 0.f)
			{
				diffuse /= weightSum;
#if !(USE_GGX_BRDF && PREFILTER_ROUGHNESS_FROM_VARIANCE)
				specular /= weightSum;
#endif
			}

#if USE_GGX_BRDF && PREFILTER_ROUGHNESS_FROM_VARIANCE
			float alpha2 = C_specularAlpha2;
			if (weightSum > 0.f)
			{
				const float sAxis = clamp(momentAlongBoxAxis / weightSum, 0.f, 1.f);
				alpha2 = clamp(
					C_specularAlpha2 + C_normalVarianceScale * (1.f - sAxis),
					C_specularAlpha2,
					1.f);
			}
			specular = specular_lobe(boxN, V, L, H, alpha2);
#endif

			float3 color = combine_shading(albedo, diffuse, specular, max(0.f, dot(V, H)), isShadow);

			color = applyFog(
				color,
				distance,
				direction,
				params.cameraPosition,
				params.fogDensity);

			setColorImpl(color);
		};
#endif
    // LOD: hitLevel is encoded in the 4th uint of pathsSurface. When LOD stopped early the
    // hit position represents the lower-left corner of a multi-voxel node rather than a 1-voxel.
    // dag.levels means full descent (1x1x1).
    uint32 hitLevel = dag.levels;
    const float3 rayOrigin = make_float3(
        Path::load_with_level(pixel.x, pixel.y, params.pathsSurface, hitLevel).path);
    const double3 cameraRayDirection = normalize(params.rayMin + pixel.x * params.rayDDx + (imageHeight - 1 - pixel.y) * params.rayDDy - params.cameraPosition);

#if ENABLE_PREFILTERED_SHADING
    // Prefiltered appearance: written by trace_paths for this pixel. 0 means "no relief".
    // 预滤波外观：由 trace_paths 为该像素写入。0 表示"无起伏"。
    const uint32 prefilterPacked = params.prefilterBuffer
        ? params.prefilterBuffer[pixel.y * imageWidth + pixel.x]
        : LodPrefilter::C_emptyHistogram;
#endif

#if EXACT_SHADOWS || PER_VOXEL_FACE_SHADING
    const double voxelSize = double(1u << (dag.levels - hitLevel));
    const double3 rayOriginDouble = make_double3(rayOrigin);
    const double3 hitPosition = ray_box_intersection(
            params.cameraPosition,
            cameraRayDirection,
            rayOriginDouble,
            rayOriginDouble + voxelSize);
#endif

#if EXACT_SHADOWS
    const float3 shadowStart = make_float3(hitPosition);
#else
    const float3 shadowStart = rayOrigin;
#endif

#if 0
    setColorImpl(make_float3(clamp_vector(normal, 0, 1)));
    return;
#endif

    if (length(rayOrigin) == 0.0f)
    {
        setColor(1, 1e9, cameraRayDirection);
        return; // Discard cleared or light-backfacing fragments
    }

    const float3 direction = sun_direction();
    const bool isShadowed = intersect_ray_node_out_of_order(dag, shadowStart + params.shadowBias * direction, direction);

    const double3 v = make_double3(rayOrigin) - params.cameraPosition;
    const double distance = length(v);
    const double3 nv = v / distance;

    if (isShadowed)
    {
#if PER_VOXEL_FACE_SHADING
		// LOD: voxel center is at rayOriginDouble + voxelSize*0.5, not (+0.5) which would
		// only be correct for 1x1x1 voxels.
		const double3 voxelOriginToHitPosition = normalize(hitPosition - (rayOriginDouble + voxelSize * 0.5));
		const auto truncate_signed = [](double3 d) { return make_double3(int32(d.x), int32(d.y), int32(d.z)); };
		const double3 normal = truncate_signed(voxelOriginToHitPosition / max(abs(voxelOriginToHitPosition)));
#if ENABLE_PREFILTERED_SHADING
		setPrefilteredBRDFColor(distance, nv, normal, prefilterPacked, true);
#else
		setBRDFColor(0, distance, nv, normal, true);
#endif
#else
		setColor(0, distance, nv);
#endif
		
    }
    else
    {
#if PER_VOXEL_FACE_SHADING
        const double3 voxelOriginToHitPosition = normalize(hitPosition - (rayOriginDouble + voxelSize * 0.5));
        const auto truncate_signed = [](double3 d) { return make_double3(int32(d.x), int32(d.y), int32(d.z)); };
        const double3 normal = truncate_signed(voxelOriginToHitPosition / max(abs(voxelOriginToHitPosition)));
        //setColor(max(0.f, dot(make_float3(normal), sun_direction())), distance, nv);
#if ENABLE_PREFILTERED_SHADING
		setPrefilteredBRDFColor(distance, nv, normal, prefilterPacked, false);
#else
		setBRDFColor(1, distance, nv, normal, false);
#endif
#else
        setColor(1, distance, nv);
#endif
    }

#if 0 // AO code copy-pasted from Erik's impl, doesn't compile at all
    constexpr int sqrtNofSamples = 8;

    float avgSum = 0;
    for (int y = 0; y < sqrtNofSamples; y++)
    {
        for (int x = 0; x < sqrtNofSamples; x++)
        {
            int2 coord = make_int2(blockIdx.x * blockDim.x + threadIdx.x, blockIdx.y * blockDim.y + threadIdx.y);
            float3 normal = make_float3(tex2D(normalTexture, float(coord.x), float(coord.y)));
            float3 tangent = normalize3(perp3(normal));
            float3 bitangent = cross(normal, tangent);
            //int2 randomCoord = make_int2((coord.x * sqrtNofSamples + x + randomSeed.x)%RAND_SIZE, (coord.y * sqrtNofSamples + y + randomSeed.y)%RAND_SIZE);
            int2 randomCoord = make_int2((coord.x * sqrtNofSamples + x + randomSeed.x) & RAND_BITMASK, (coord.y * sqrtNofSamples + y + randomSeed.y) & RAND_BITMASK);
            float2 randomSample = tex2D(randomTexture, randomCoord.x, randomCoord.y);
            float randomLength = tex2D(randomTexture, randomCoord.y, randomCoord.x).x;
            float2 dxdy = make_float2(1.0f / float(sqrtNofSamples), 1.0f / float(sqrtNofSamples));
            float3 sample = cosineSampleHemisphere(make_float2(x * dxdy.x, y * dxdy.y) + (1.0 / float(sqrtNofSamples)) * randomSample);
            float3 ray_d = normalize3(sample.x * tangent + sample.y * bitangent + sample.z * normal);
            avgSum += intersectRayNode_outOfOrder<maxLevels>(ray_o, ray_d, ray_tmax * randomLength, rootCenter, rootRadius, coneOpening) ? 0.0f : 1.0f;
        }
    }
    avgSum /= float(sqrtNofSamples * sqrtNofSamples);
#endif
}

template __global__ void Tracer::trace_paths<BasicDAG>(TracePathsParams, BasicDAG);
template __global__ void Tracer::trace_paths<HashDAG >(TracePathsParams, HashDAG);

template __global__ void Tracer::trace_shadows<BasicDAG>(TraceShadowsParams, BasicDAG);
template __global__ void Tracer::trace_shadows<HashDAG >(TraceShadowsParams, HashDAG);

#define COLORS_IMPL(Dag, Colors)\
template __global__ void Tracer::trace_colors<Dag, Colors>(TraceColorsParams, Dag, Colors);

COLORS_IMPL(BasicDAG, BasicDAGUncompressedColors)
COLORS_IMPL(BasicDAG, BasicDAGCompressedColors)
COLORS_IMPL(BasicDAG, BasicDAGColorErrors)
COLORS_IMPL(HashDAG, HashDAGColors)
