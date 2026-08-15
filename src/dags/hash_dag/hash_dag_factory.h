#pragma once

#include "typedefs.h"
#include "dags/basic_dag/basic_dag.h"
#include "dags/hash_dag/hash_dag.h"
#include <string>

struct DAGInfo;

struct HashDAGFactory
{
	static void load_from_DAG(
		HashDAG& outDag,
		const BasicDAG& inDag,
		uint32 numPages);

	static void load_colors_from_DAG(
		HashDAGColors& outDagColors,
		const BasicDAG& inDag,
		const BasicDAGCompressedColors& inDagColors);

#if ENABLE_PREFILTERED_SHADING
	/**
	 * Prefiltered appearance: walk the freshly built HashDAG bottom-up and fill its
	 * side table with one packed 6-direction relief histogram per node.
	 *
	 * Works directly on the HashDAG (not on the source BasicDAG) so that the table is
	 * keyed by the same node indices the tracer sees, and so that the same routine can
	 * rebuild the table after edits or after loading a serialised HashDAG.
	 *
	 * 预滤波外观：自下而上遍历刚构建好的 HashDAG，为每个节点在旁路表中填入一个打包的
	 * 6 方向起伏直方图。
	 *
	 * 直接作用于 HashDAG（而非源 BasicDAG），这样表的键与 tracer 看到的节点下标一致，
	 * 同一套代码也能在编辑之后、或加载序列化的 HashDAG 之后重建该表。
	 */
	static void build_prefilter(HashDAG& dag);
#endif

	static void save_dag_to_file(const DAGInfo& info, const HashDAG& dag, const std::string& path);
	static void load_dag_from_file(DAGInfo& info, HashDAG& dag, const std::string& path);
};