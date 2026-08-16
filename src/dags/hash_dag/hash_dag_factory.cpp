#include "hash_dag_factory.h"
#include "dag_tracer.h"
#include "serializer.h"

#include <unordered_map>
#include <algorithm>

uint32 create_hash_dag(
	const BasicDAG& sdag,
	HashDAG& hdag,
	std::vector<uint32>& map,
	const uint32 level,
	const uint32 index)
{
	const bool isLeaf = sdag.is_leaf(level);
	uint32 finalIndex;
	if (!isLeaf)
	{
		const uint32 node = sdag.get_node(level, index);
		const uint8 childMask = Utils::child_mask(node);

        uint32 nodeBuffer[9];
        uint32 nodeBufferSize = 0;
        nodeBuffer[nodeBufferSize++] = node;

		for (uint8 i = 0; i < 8; ++i)
		{
			if (childMask & (1u << i))
			{
				const uint32 childIndex = sdag.get_child_index(level, index, childMask, i);
				uint32 newChildIndex = map[childIndex];
				if (newChildIndex == 0)
				{
					newChildIndex = create_hash_dag(sdag, hdag, map, level + 1, childIndex);
					map[childIndex] = newChildIndex + 1;
				}
				else
				{
					newChildIndex--;
				}
				nodeBuffer[nodeBufferSize++] = newChildIndex;
			}
		}

		const uint32 hash = HashDagUtils::hash_interior(nodeBufferSize, nodeBuffer);

#if USE_BLOOM_FILTER
        BloomFilter filter;
        hdag.data.bloom_filter_init_interior(filter, nodeBufferSize, nodeBuffer);
#endif // ~ USE_BLOOM_FILTER

        finalIndex = hdag.data.add_interior_node(level, nodeBufferSize, nodeBuffer, hash BLOOM_FILTER_ARG(filter));
	}
	else
	{
		const uint64 leaf = sdag.get_leaf(index).to_64();
		const uint32 hash = HashDagUtils::hash_leaf(leaf);

#if USE_BLOOM_FILTER
        BloomFilter filter;
        hdag.data.bloom_filter_init_leaf(filter, leaf);
#endif // ~ USE_BLOOM_FILTER

        finalIndex = hdag.data.add_leaf_node(level, leaf, hash BLOOM_FILTER_ARG(filter));
    }

#if 0
    {
        for (int index = 0; index < newNodeData.size(); index++)
        {
            check(newNodeData[index] == hdag.get_node(level, finalIndex + index));
        }
        auto checkEqual = [&](int level, uint32 oldIndex, uint32 newIndex)
        {
            uint32 oldNodeTemp = sdag.get_node(level, oldIndex);
            uint32 newNodeTemp = hdag.get_node(level, newIndex);
            check(oldNodeTemp == newNodeTemp);
        };
        if (isLeaf)
        {
            checkEqual(level, index.index, finalIndex);
            checkEqual(level, index.index + 1, finalIndex + 1);
        }
        else
        {
            checkEqual(level, index.index, finalIndex);
            const uint32 node = sdag.get_node(level, index);
            const uint8 childMask = Utils::child_mask(node);
            for (uint8 i = 0; i < 8; ++i)
            {
                if (childMask & (1u << i))
                {
                    uint32 oldChildIndex = sdag.get_child_index(level, index, childMask, i);
                    uint32 newChildIndex = hdag.get_child_index(level, finalIndex, childMask, i);
                    checkEqual(level + 1, oldChildIndex.index, newChildIndex.index);
                }
            }
        }
    }
#endif
	return finalIndex;
}

// LOD: sample K positions in [offset, offset+size) and average their per-block representative
// colour. The average is taken in linear space; the result is packed RGB888 in storage space,
// or 0 if size == 0.
static uint32 sample_leaf_average_color(
	const CompressedColorLeaf& globalLeaf,
	uint64 offset,
	uint64 size)
{
	if (size == 0) return 0;

	constexpr int kMaxSamples = 8;
	const int numSamples = (size < uint64(kMaxSamples)) ? int(size) : kMaxSamples;

	float3 acc = make_float3(0.f);
	int valid = 0;
	for (int s = 0; s < numSamples; ++s)
	{
		// Centered samples: (s + 0.5) / numSamples maps to a position inside the range.
		const uint64 absIdx = offset + (uint64(s) * size + size / 2) / uint64(numSamples);
		if (!globalLeaf.is_valid_index(absIdx)) continue;


		acc = acc + ColorUtils::to_linear(globalLeaf.get_color(absIdx).get_lod_average());
		++valid;
	}
	if (valid == 0) return 0;
	return ColorUtils::float3_to_rgb888(ColorUtils::from_linear(acc / float(valid)));
}

// LOD: memoized recursive computation of the 3-axis projected coverage for a
// geometry DAG subtree. Return value is packed as
//   bits [ 0.. 7]: cov_x in 0..255
//   bits [ 8..15]: cov_y in 0..255
//   bits [16..23]: cov_z in 0..255
// cov_axis = "fraction of the axis-perpendicular projected face of the subtree
// that is covered by at least one surface voxel along that axis".
using CoverageCache = std::unordered_map<uint64, uint32>;

static uint32 pack_coverage(float cx, float cy, float cz)
{
	auto q = [](float f) -> uint32
	{
		const float c = std::max(0.f, std::min(1.f, f));
		return uint32(c * 255.f + 0.5f) & 0xFFu;
	};
	return q(cx) | (q(cy) << 8) | (q(cz) << 16);
}

static void unpack_coverage(uint32 packed, float& cx, float& cy, float& cz)
{
	cx = float((packed >> 0)  & 0xFFu) / 255.f;
	cy = float((packed >> 8)  & 0xFFu) / 255.f;
	cz = float((packed >> 16) & 0xFFu) / 255.f;
}

static uint32 compute_dag_coverage(
	const BasicDAG& sdag,
	uint32 level,
	uint32 index,
	CoverageCache& cache)
{
	// Key on (level, index) — index alone is likely unique but level makes it
	// unambiguous with essentially zero cost.
	const uint64 key = (uint64(level) << 32) | uint64(index);
	auto it = cache.find(key);
	if (it != cache.end()) return it->second;

	uint32 packed;

	if (level == sdag.leaf_level())
	{
		// Base case: a Leaf packs 4x4x4 = 64 voxels into a 64-bit mask.
		// Bit encoding for voxel (x,y,z) with x,y,z in [0,3]:
		//   bit = ((x&2)<<4) | ((y&2)<<3) | ((z&2)<<2) | ((x&1)<<2) | ((y&1)<<1) | (z&1)
		const Leaf leaf = sdag.get_leaf(index);
		const uint64 mask = leaf.to_64();
		auto bit_of = [](int x, int y, int z) -> int
		{
			return ((x & 2) << 4) | ((y & 2) << 3) | ((z & 2) << 2)
			     | ((x & 1) << 2) | ((y & 1) << 1) | (z & 1);
		};

		int nX = 0, nY = 0, nZ = 0;
		for (int y = 0; y < 4; ++y)
			for (int z = 0; z < 4; ++z)
			{
				for (int x = 0; x < 4; ++x)
					if (mask & (uint64(1) << bit_of(x, y, z))) { ++nX; break; }
			}
		for (int x = 0; x < 4; ++x)
			for (int z = 0; z < 4; ++z)
			{
				for (int y = 0; y < 4; ++y)
					if (mask & (uint64(1) << bit_of(x, y, z))) { ++nY; break; }
			}
		for (int x = 0; x < 4; ++x)
			for (int y = 0; y < 4; ++y)
			{
				for (int z = 0; z < 4; ++z)
					if (mask & (uint64(1) << bit_of(x, y, z))) { ++nZ; break; }
			}
		packed = pack_coverage(nX / 16.f, nY / 16.f, nZ / 16.f);
	}
	else
	{
		// Recursive case: combine 8 children's coverages. For each projection
		// axis, the parent's projected face is 2x2 quadrants; each quadrant is
		// the union of two children stacked along the projection axis. We use
		// the independence assumption (union = 1 - (1-a)(1-b)) as the combining
		// rule. The parent's coverage is the mean of the 4 quadrants.
		const uint32 node = sdag.get_node(level, index);
		const uint8 childMask = Utils::child_mask(node);

		float qX[4] = { 0.f, 0.f, 0.f, 0.f };
		float qY[4] = { 0.f, 0.f, 0.f, 0.f };
		float qZ[4] = { 0.f, 0.f, 0.f, 0.f };

		for (uint8 i = 0; i < 8; ++i)
		{
			if (!(childMask & (1u << i))) continue;
			const uint32 childIndex = sdag.get_child_index(level, index, childMask, i);
			const uint32 childCov = compute_dag_coverage(sdag, level + 1, childIndex, cache);
			float ccx, ccy, ccz;
			unpack_coverage(childCov, ccx, ccy, ccz);

			// Path::descend maps child bit 2 -> x, bit 1 -> y, bit 0 -> z.
			const uint32 xb = (i >> 2) & 1u;
			const uint32 yb = (i >> 1) & 1u;
			const uint32 zb = (i >> 0) & 1u;

			// For X projection the quadrant is indexed by (yb, zb), etc.
			const uint32 qxIdx = (yb << 1) | zb;
			const uint32 qyIdx = (xb << 1) | zb;
			const uint32 qzIdx = (xb << 1) | yb;

			qX[qxIdx] = 1.f - (1.f - qX[qxIdx]) * (1.f - ccx);
			qY[qyIdx] = 1.f - (1.f - qY[qyIdx]) * (1.f - ccy);
			qZ[qzIdx] = 1.f - (1.f - qZ[qzIdx]) * (1.f - ccz);
		}

		const float aX = 0.25f * (qX[0] + qX[1] + qX[2] + qX[3]);
		const float aY = 0.25f * (qY[0] + qY[1] + qY[2] + qY[3]);
		const float aZ = 0.25f * (qZ[0] + qZ[1] + qZ[2] + qZ[3]);
		packed = pack_coverage(aX, aY, aZ);
	}

	cache.emplace(key, packed);
	return packed;
}

#if ENABLE_PREFILTERED_SHADING

// Prefiltered appearance: build-time accumulator for one DAG node.
//   area[d]     = exposed face area along direction d, with "outside the node is empty"
//   boundary[d] = number of filled voxels in the node's own outermost slab along d
// Both are additive over a node's children (up to the interface correction below), so the
// whole thing is computed with one memoized bottom-up pass over the DAG.
//
// 预滤波外观：单个 DAG 节点的构建期累加器。
//   area[d]     = 沿 d 方向的暴露面面积，约定"节点之外为空"
//   boundary[d] = 节点自身沿 d 方向最外层薄片中的实心体素数
// 两者对子节点都是可加的（需要下面的界面修正），因此整体只需一次记忆化的自下而上遍历。
struct PrefilterNodeAreas
{
	float area[LodPrefilter::C_numDirections]     = { 0.f, 0.f, 0.f, 0.f, 0.f, 0.f };
	float boundary[LodPrefilter::C_numDirections] = { 0.f, 0.f, 0.f, 0.f, 0.f, 0.f };
};

// Memoized on the node index. HashDAG node indices are globally unique across levels, so
// the index alone is a complete key, and one entry per *unique* node means the cache size
// matches the DAG's own deduplicated node count.
// 以节点下标做记忆化。HashDAG 节点下标在所有层级上全局唯一，因此下标本身就是完整的键；
// 每个"唯一"节点一条记录，意味着缓存大小与 DAG 去重后的节点数一致。
using PrefilterCache = std::unordered_map<uint32, PrefilterNodeAreas>;

static PrefilterNodeAreas compute_node_prefilter(
	const HashDAG& dag,
	uint32 level,
	uint32 index,
	PrefilterCache& cache,
	std::vector<uint64>& outEntries)
{
	{
		const auto it = cache.find(index);
		if (it != cache.end()) return it->second;
	}

	PrefilterNodeAreas result;

	if (level == dag.leaf_level())
	{
		// Base case: exact counts straight out of the 64-bit occupancy mask.
		// 基础情形：直接从 64 bit 占用掩码精确计数。
		LodPrefilter::leaf_face_areas(dag.get_leaf(index).to_64(), result.area, result.boundary);
	}
	else
	{
		const uint32 node = dag.get_node(level, index);
		const uint8 childMask = Utils::child_mask(node);

		PrefilterNodeAreas children[8];
		bool present[8] = { false, false, false, false, false, false, false, false };

		for (uint8 i = 0; i < 8; ++i)
		{
			if (!(childMask & (1u << i))) continue;
			present[i] = true;
			const uint32 childIndex = dag.get_child_index(level, index, childMask, i);
			children[i] = compute_node_prefilter(dag, level + 1, childIndex, cache, outEntries);
		}

		// 1) Plain sum of the children's exposed areas.
		//    第一步：把子节点的暴露面积直接相加。
		for (uint8 i = 0; i < 8; ++i)
		{
			if (!present[i]) continue;
			for (uint32 d = 0; d < LodPrefilter::C_numDirections; ++d)
			{
				result.area[d] += children[i].area[d];
			}
		}

		// 2) Boundary slabs: only the four children that touch the parent's slab along a
		//    given direction contribute to it. Child bit layout matches Path::descend:
		//    bit 2 = x, bit 1 = y, bit 0 = z.
		//    第二步：边界薄片。沿某个方向，只有贴着父节点该薄片的四个子节点才有贡献。
		//    子节点位布局与 Path::descend 一致：bit 2 = x, bit 1 = y, bit 0 = z。
		for (uint8 i = 0; i < 8; ++i)
		{
			if (!present[i]) continue;
			for (uint32 axis = 0; axis < 3; ++axis)
			{
				const uint32 axisBit = 1u << (2 - axis);
				const uint32 dir = axis * 2 + ((i & axisBit) ? 1u : 0u);
				result.boundary[dir] += children[i].boundary[dir];
			}
		}

		// 3) Interface correction. Each child was measured as if everything outside it were
		//    empty, so the faces it has on a boundary it shares with a sibling were counted
		//    even though that sibling may fill them in. For a pair of children stacked along
		//    an axis we subtract the size of the intersection of the touching slabs, which we
		//    approximate as |A|*|B|/slabArea (the same independence assumption the existing
		//    coverage computation uses). This is exact whenever either slab is full or empty,
		//    which covers the important solid / empty cases, and without it the plain sum
		//    over-counts by roughly 2x per level.
		//    第三步：界面修正。每个子节点都是按"自身之外全为空"来统计的，因此它在与兄弟节点
		//    相邻的那个边界上的面也被计入了，而兄弟节点可能正好把它们填住。对沿某轴堆叠的一
		//    对子节点，我们减去两个相接薄片的交集大小，并用 |A|*|B|/薄片面积 来近似（与现有
		//    coverage 计算相同的独立性假设）。当任一薄片全满或全空时该近似是精确的，这覆盖了
		//    重要的实心 / 空心情形；若不做这一步，直接求和会在每一层上多算大约 2 倍。
		const float childEdge = float(1u << (dag.levels - level - 1));
		const float childSlabArea = childEdge * childEdge;
		for (uint32 axis = 0; axis < 3; ++axis)
		{
			const uint32 dirNeg = axis * 2 + 0;
			const uint32 dirPos = axis * 2 + 1;
			const uint32 axisBit = 1u << (2 - axis);

			float occluded = 0.f;
			for (uint8 i = 0; i < 8; ++i)
			{
				if (i & axisBit) continue;                  // only iterate the "low" children
				const uint8 j = uint8(i | axisBit);         // its neighbour on the "high" side
				if (!present[i] || !present[j]) continue;
				occluded += children[i].boundary[dirPos] * children[j].boundary[dirNeg] / childSlabArea;
			}
			// The same intersection hides the low child's +axis faces and the high child's
			// -axis faces. / 同一个交集同时挡住低侧子节点的 +axis 面和高侧子节点的 -axis 面。
			result.area[dirPos] = std::max(0.f, result.area[dirPos] - occluded);
			result.area[dirNeg] = std::max(0.f, result.area[dirNeg] - occluded);
		}
	}

	// Emit the quantised *internal* relief for this node. Subtracting boundary[] removes the
	// faces that lie on the node's own hull, which the bounding box normal already accounts
	// for; what remains is exactly the relief the box normal cannot express.
	// 输出该节点量化后的"内部"起伏。减去 boundary[] 去掉了位于节点自身外壳上的面 —— 那些面
	// 已被包围盒法线表达；剩下的正是包围盒法线无法表达的起伏。
	if (level >= uint32(PREFILTER_MIN_LEVEL) && level <= uint32(PREFILTER_MAX_LEVEL))
	{
		const float edge = float(1u << (dag.levels - level));
		const float faceArea = edge * edge;

		uint32 packed = 0;
		for (uint32 d = 0; d < LodPrefilter::C_numDirections; ++d)
		{
			const float relief = (result.area[d] - result.boundary[d]) / faceArea;
			packed |= LodPrefilter::pack_bin(LodPrefilter::quantise_bin(relief), d);
		}
		// Nodes with no relief are simply left out of the table: a miss and a zero histogram
		// mean the same thing, so skipping them shrinks the table for free.
		// 没有起伏的节点直接不入表：未命中与全零直方图含义相同，跳过它们可以白得一份瘦身。
		if (packed != LodPrefilter::C_emptyHistogram)
		{
			outEntries.push_back((uint64(index) << 32) | uint64(packed));
		}
	}

	cache.emplace(index, result);
	return result;
}

void HashDAGFactory::build_prefilter(HashDAG& dag)
{
	PROFILE_FUNCTION();
	SCOPED_STATS("Creating LOD prefilter table");

	Stats stats;
	stats.start_work("computing relief histograms");

	PrefilterCache cache;
	std::vector<uint64> entries;
	compute_node_prefilter(dag, 0, dag.firstNodeIndex, cache, entries);

	printf("\tLOD prefilter: visited %zu unique DAG nodes\n", cache.size());
	cache.clear();

	stats.start_work("building table");
	dag.prefilter.build(entries);
	dag.prefilter.print_stats(entries.size());
}

#endif // ~ ENABLE_PREFILTERED_SHADING

// Recursive build, also computes a per-internal-node average color for LOD shading.
// Returns the color tree index of this node; writes the subtree's voxel-count-weighted
// average color into `outAvgColor`.
uint32 create_hash_dag_colors(
	const BasicDAG& sdag,
	const BasicDAGCompressedColors& sdagcolors,
	HashColorsBuilder& colorBuilder,
	CoverageCache& covCache,
	const uint32 level,
	const uint32 index,
	uint64 leavesCount,
	uint32& outAvgColor)
{
	const uint32 node = sdag.get_node(level, index);
	const uint8 childMask = Utils::child_mask(node);
	const uint32 colorIndex = (uint32)colorBuilder.nodes.size();
	const uint32 nodeAvgSlot = (uint32)colorBuilder.nodeAverages.size();
	const uint32 nodeCovSlot = (uint32)colorBuilder.nodeCoverage.size();

	check(C_colorTreeLevels < sdag.leaf_level());

	// Reserve this node's average / coverage slots up-front so child recursions
	// don't reorder them.
	colorBuilder.nodeAverages.push_back(0);
	colorBuilder.nodeCoverage.push_back(0xFFFFFFu);

	// Use the global leaf in "absolute index" mode for direct sampling.
	CompressedColorLeaf globalLeafAbs = sdagcolors.leaf;
	globalLeafAbs.set_as_unique();

	float3 accColor = make_float3(0.f);
	uint64 accWeight = 0;

	if (level == C_colorTreeLevels - 1)
	{
		for (uint8 i = 0; i < 8; i++)
		{
			colorBuilder.nodes.push_back((uint32)colorBuilder.leaves.size());
			colorBuilder.leaves.emplace_back(HashColorsBuilder::BuildLeaf{ leavesCount });

			uint64 childSize = 0;
			if (childMask & (1u << i))
			{
				const uint32 childIndex = sdag.get_child_index(level, index, childMask, i);
				const uint32 childNode = sdag.get_node(level + 1, childIndex);
				childSize = sdagcolors.get_leaves_count(level + 1, childNode);
			}

			if (childSize > 0)
			{
				const uint32 childAvg = sample_leaf_average_color(globalLeafAbs, leavesCount, childSize);
				accColor = accColor + ColorUtils::to_linear(ColorUtils::rgb888_to_float3(childAvg)) * float(childSize);
				accWeight += childSize;
			}

			leavesCount += childSize;
		}
	}
	else
	{
		for (uint8 i = 0; i < 8; i++)
		{
			colorBuilder.nodes.push_back(0);
		}
		for (uint8 i = 0; i < 8; i++)
		{
			if (childMask & (1u << i))
			{
				const uint32 childIndex = sdag.get_child_index(level, index, childMask, i);
				uint32 childAvg = 0;
				const uint32 childColorIndex = create_hash_dag_colors(
					sdag, sdagcolors, colorBuilder, covCache, level + 1, childIndex, leavesCount, childAvg);
				check(colorBuilder.nodes[colorIndex + i] == 0);
				colorBuilder.nodes[colorIndex + i] = childColorIndex;

				const uint32 childNode = sdag.get_node(level + 1, childIndex);
				const uint64 childSize = sdagcolors.get_leaves_count(level + 1, childNode);
				if (childSize > 0)
				{
					accColor = accColor + ColorUtils::to_linear(ColorUtils::rgb888_to_float3(childAvg)) * float(childSize);
					accWeight += childSize;
				}
				leavesCount += childSize;
			}
		}
	}

	outAvgColor = (accWeight > 0)
		? ColorUtils::float3_to_rgb888(ColorUtils::from_linear(accColor / float(accWeight)))
		: 0;
	colorBuilder.nodeAverages[nodeAvgSlot] = outAvgColor;

	// LOD: 3-axis coverage for this color tree internal node. Uses the memoized
	// DAG-node coverage so shared subtrees are computed only once.
	colorBuilder.nodeCoverage[nodeCovSlot] = compute_dag_coverage(sdag, level, index, covCache);
	return colorIndex;
}

void HashDAGFactory::load_from_DAG(HashDAG& outDag, const BasicDAG& inDag, uint32 numPages)
{
	PROFILE_FUNCTION();
	SCOPED_STATS("Creating hash dag");

	Stats stats;

	stats.start_work("Allocating pool");
	outDag.data.create(numPages);

#if ADD_FULL_NODES_FIRST
	stats.start_work("Adding full nodes");
    outDag.data.cpuData.fullNodeIndices = new uint32[MAX_LEVELS];
    for (uint32 level = inDag.leaf_level(); level > 0; level--)
    {
        outDag.data.add_full_node(level);
    }
#endif
	
	stats.start_work("Hashing existing dag");
	std::vector<uint32> map(inDag.data.size(), 0);
	outDag.firstNodeIndex = create_hash_dag(inDag, outDag, map, 0, 0);

	stats.start_work("Checking");
	// outDag.check_nodes();

#if !MANUAL_VIRTUAL_MEMORY
	outDag.pool = outDag.data.gpuPool;
#endif

    stats.start_work("upload_to_gpu");
    outDag.data.upload_to_gpu();

#if ENABLE_PREFILTERED_SHADING
	// Prefiltered appearance: the histograms are a pure function of the geometry, so they
	// are built here, right after the geometry is final.
	// 预滤波外观：直方图是几何的纯函数，因此在几何定型之后立即在这里构建。
	build_prefilter(outDag);
#endif
}

void HashDAGFactory::load_colors_from_DAG(
	HashDAGColors& outDagColors,
	const BasicDAG& inDag, 
	const BasicDAGCompressedColors& inDagColors)
{
	PROFILE_FUNCTION();
	SCOPED_STATS("Creating hash dag colors");
	
	HashColorsBuilder colorBuilder;
	CoverageCache covCache;
	uint32 rootAvg = 0;
	const uint32 colorIndex = create_hash_dag_colors(
		inDag, inDagColors, colorBuilder, covCache, 0, 0, 0, rootAvg);
	checkAlways(colorIndex == 0);
	printf("\tLOD coverage cache: %zu unique DAG nodes\n", covCache.size());
	colorBuilder.build(outDagColors, inDagColors.leaf);
}

void HashDAGFactory::save_dag_to_file(const DAGInfo& info, const HashDAG& dag, const std::string& path)
{
	PROFILE_FUNCTION();
	checkAlways(dag.is_valid());
	
	FileWriter writer(path);
	
	writer.write(info);
	writer.write(dag.levels);

	writer.write(dag.firstNodeIndex);
	writer.write(dag.data.cpuData.poolMaxSize);
	writer.write(dag.data.pageTableSize);
	writer.write(dag.data.poolTop);

#if MANUAL_CPU_DATA
	writer.write(dag.data.cpuData.cpuPool, dag.data.poolTop * C_pageSize * sizeof(uint32));
	writer.write(dag.data.cpuData.cpuPageTable, dag.data.pageTableSize * sizeof(uint32));
#else
	checkAlways(false);
#endif
}

void HashDAGFactory::load_dag_from_file(DAGInfo& info, HashDAG& dag, const std::string& path)
{
	PROFILE_FUNCTION();
	checkAlways(!dag.is_valid());
	
	FileReader reader(path);
	
	reader.read(info);
	uint32 levels = 0;
	reader.read(levels);
    checkfAlways(levels == MAX_LEVELS, "MAX_LEVELS is %u, should be %u", MAX_LEVELS, levels);

	reader.read(dag.firstNodeIndex);
	reader.read(dag.data.cpuData.poolMaxSize);
	reader.read(dag.data.pageTableSize);
	reader.read(dag.data.poolTop);

#if MANUAL_CPU_DATA
	reader.read(dag.data.cpuData.cpuPool, dag.data.poolTop * C_pageSize * sizeof(uint32));
	reader.read(dag.data.cpuData.cpuPageTable, dag.data.pageTableSize * sizeof(uint32));
#else
	checkAlways(false);
#endif
}