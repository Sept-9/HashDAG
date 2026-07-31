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

// LOD: sample K positions in [offset, offset+size) and average their per-block (min+max)/2
// or single-color value. Returns RGB888 packed color, or 0 if size == 0.
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

		const CompressedColor cc = globalLeaf.get_color(absIdx);
		float3 sample;
		if (cc.bitsPerWeight == 0)
		{
			sample = ColorUtils::rgb101210_to_float3(cc.colorBits);
		}
		else
		{
			sample = 0.5f * (cc.get_min_color() + cc.get_max_color());
		}
		acc = acc + sample;
		++valid;
	}
	if (valid == 0) return 0;
	return ColorUtils::float3_to_rgb888(acc / float(valid));
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
				accColor = accColor + ColorUtils::rgb888_to_float3(childAvg) * float(childSize);
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
					accColor = accColor + ColorUtils::rgb888_to_float3(childAvg) * float(childSize);
					accWeight += childSize;
				}
				leavesCount += childSize;
			}
		}
	}

	outAvgColor = (accWeight > 0)
		? ColorUtils::float3_to_rgb888(accColor / float(accWeight))
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