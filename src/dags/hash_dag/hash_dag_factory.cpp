#include "hash_dag_factory.h"
#include "dag_tracer.h"
#include "serializer.h"

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

// Recursive build, also computes a per-internal-node average color for LOD shading.
// Returns the color tree index of this node; writes the subtree's voxel-count-weighted
// average color into `outAvgColor`.
uint32 create_hash_dag_colors(
	const BasicDAG& sdag,
	const BasicDAGCompressedColors& sdagcolors,
	HashColorsBuilder& colorBuilder,
	const uint32 level,
	const uint32 index,
	uint64 leavesCount,
	uint32& outAvgColor)
{
	const uint32 node = sdag.get_node(level, index);
	const uint8 childMask = Utils::child_mask(node);
	const uint32 colorIndex = (uint32)colorBuilder.nodes.size();
	const uint32 nodeAvgSlot = (uint32)colorBuilder.nodeAverages.size();

	check(C_colorTreeLevels < sdag.leaf_level());

	// Reserve this node's average slot up-front so child recursions don't reorder it.
	colorBuilder.nodeAverages.push_back(0);

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
					sdag, sdagcolors, colorBuilder, level + 1, childIndex, leavesCount, childAvg);
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
	uint32 rootAvg = 0;
	const uint32 colorIndex = create_hash_dag_colors(inDag, inDagColors, colorBuilder, 0, 0, 0, rootAvg);
	checkAlways(colorIndex == 0);
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