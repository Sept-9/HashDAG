#pragma once

#include "typedefs.h"
#include "utils.h"
#include "array.h"
#include "cuda_math.h"

#if ENABLE_PREFILTERED_SHADING

/**
 * ============================================================================
 * Prefiltered LOD appearance: 6-direction internal-relief area histogram
 * LOD 预滤波外观：6 方向"内部起伏"面积直方图
 * ============================================================================
 *
 * WHAT IS STORED / 存的是什么
 * ---------------------------
 * For a DAG node covering S x S x S voxels (S = 2^(levels - level)) we measure, for each
 * of the six axis directions d, the exposed surface area A[d]: the number of filled
 * voxels inside the node whose neighbour along d is empty. Everything outside the node is
 * treated as empty, which makes A[d] a pure function of the node's subtree and therefore
 * fully compatible with DAG deduplication and with bottom-up edits.
 *
 * We then subtract B[d], the number of filled voxels lying in the node's own outermost
 * slab along d. A[d] - B[d] is exactly the area of the faces that are *interior* to the
 * node, i.e. the relief that the node's bounding box does not already represent:
 *
 *   - a fully solid node          -> A[d] == B[d] == S^2  -> relief 0 in every direction
 *   - a flat slab (floor / wall)  -> relief 0             -> the box normal is exact
 *   - a staircase / bumpy surface -> relief along the step tops and risers
 *   - a pitted or porous node     -> relief in most directions
 *
 * So "all six bins are zero" cleanly means "the bounding box normal is already correct",
 * and the shading code can fall back to the original behaviour with no extra flag.
 *
 * 对于覆盖 S x S x S 个体素的 DAG 节点（S = 2^(levels - level)），我们对六个坐标轴方向 d
 * 统计暴露面积 A[d]：节点内部沿 d 方向邻居为空的实心体素数量。节点之外一律视为空，这样
 * A[d] 就是节点子树的纯函数，因此与 DAG 去重、以及自下而上的编辑完全兼容。
 *
 * 然后减去 B[d]：位于节点自身最外层（沿 d 方向）薄片中的实心体素数量。A[d] - B[d] 恰好是
 * 节点"内部"面的面积，也就是包围盒本身无法表达的起伏：
 *
 *   - 完全实心的节点        -> A[d] == B[d] == S^2 -> 六个方向起伏都是 0
 *   - 平板（地面 / 墙面）   -> 起伏为 0            -> 包围盒法线本来就是精确的
 *   - 阶梯 / 凹凸表面       -> 台阶顶面与立面产生起伏
 *   - 有坑洞 / 多孔的节点   -> 多个方向都有起伏
 *
 * 因此"六格全为零"干净地表示"包围盒法线已经正确"，着色代码可以无需额外标志位就退回原行为。
 *
 * PACKING / 打包格式
 * ------------------
 * One uint32 per node. relief[d] = clamp((A[d] - B[d]) / S^2, 0, 1) quantised to 5 bits.
 * Clamping at 1 is physically motivated: once the relief area equals the node's own
 * projected face area, adding more area cannot make more of it visible (self-occlusion).
 *
 *   bits [ 0.. 4] : -X    bits [ 5.. 9] : +X
 *   bits [10..14] : -Y    bits [15..19] : +Y
 *   bits [20..24] : -Z    bits [25..29] : +Z
 *   bits [30..31] : reserved (intended for a future ambient-occlusion term)
 *
 * 每个节点一个 uint32。relief[d] = clamp((A[d] - B[d]) / S^2, 0, 1)，量化为 5 bit。
 * 在 1 处截断有物理依据：一旦起伏面积达到节点自身的投影面面积，再多的面积也不会更多地被
 * 看见（自遮挡）。bits [30..31] 预留给后续的环境光遮蔽项。
 *
 * WHERE IT LIVES / 存在哪里
 * -------------------------
 * In a side hash table keyed by DAG node index (see PrefilterTable below), *not* inline in
 * the nodes. This keeps the node layout, the hashing and the edit code completely
 * untouched, at the cost of one hash probe per LOD-terminated ray.
 *
 * 放在以 DAG 节点下标为键的旁路哈希表中（见下方 PrefilterTable），而不是内联进节点。这样
 * 节点布局、哈希函数和编辑代码完全不用改动，代价是每条 LOD 终止的光线多一次哈希探测。
 */
namespace LodPrefilter
{
	// Direction slot order: 0 = -X, 1 = +X, 2 = -Y, 3 = +Y, 4 = -Z, 5 = +Z.
	// 方向槽位顺序：0 = -X, 1 = +X, 2 = -Y, 3 = +Y, 4 = -Z, 5 = +Z。
	constexpr uint32 C_numDirections = 6;
	constexpr uint32 C_binBits = 5;
	constexpr uint32 C_binMax = (1u << C_binBits) - 1u;

	// A packed value of 0 means "no internal relief", which is also what a table miss
	// returns. Both cases want the same thing: use the geometric box normal.
	// 打包值为 0 表示"没有内部起伏"，查表未命中也返回它。两种情况都希望使用几何盒法线。
	constexpr uint32 C_emptyHistogram = 0u;

	static_assert(C_numDirections * C_binBits <= 32, "histogram does not fit in a uint32");

	// --- Bit twiddling on the 4x4x4 leaf mask -------------------------------------
	// A Leaf packs 4x4x4 voxels into 64 bits. The bit index of voxel (x,y,z) is
	//   bit = 32*x1 + 16*y1 + 8*z1 + 4*x0 + 2*y0 + z0     (x = 2*x1 + x0, etc.)
	// 叶节点把 4x4x4 个体素打包成 64 bit，体素 (x,y,z) 的位下标如上。
	constexpr uint64 C_maskX0 = 0xF0F0F0F0F0F0F0F0ull; // x0 == 1
	constexpr uint64 C_maskX1 = 0xFFFFFFFF00000000ull; // x1 == 1
	constexpr uint64 C_maskY0 = 0xCCCCCCCCCCCCCCCCull; // y0 == 1
	constexpr uint64 C_maskY1 = 0xFFFF0000FFFF0000ull; // y1 == 1
	constexpr uint64 C_maskZ0 = 0xAAAAAAAAAAAAAAAAull; // z0 == 1
	constexpr uint64 C_maskZ1 = 0xFF00FF00FF00FF00ull; // z1 == 1

	// Translate the whole occupancy mask by +1 along one axis. Voxels that would leave the
	// 4x4x4 block are dropped, which encodes the "outside is empty" convention.
	// 把整个占用掩码沿某轴平移 +1。会移出 4x4x4 块的体素被丢弃，这正体现了"外部为空"的约定。
	//   lowShift : coordinate 0 -> 1 (bit0 of the coordinate flips up)
	//   highShift: coordinate 1 -> 2 (bit0 flips down, bit1 flips up)
	HOST_DEVICE uint64 shift_mask_plus(uint64 mask, uint64 m0, uint64 m1, uint32 lowShift, uint32 highShift)
	{
		return ((mask & ~m0) << lowShift) | ((mask & m0 & ~m1) << highShift);
	}
	// Translate the whole occupancy mask by -1 along one axis.
	// 把整个占用掩码沿某轴平移 -1。
	HOST_DEVICE uint64 shift_mask_minus(uint64 mask, uint64 m0, uint64 m1, uint32 lowShift, uint32 highShift)
	{
		return ((mask & m0) >> lowShift) | ((mask & ~m0 & m1) >> highShift);
	}

	/**
	 * Exact exposed-face areas and boundary-slab occupancies of one 4x4x4 leaf.
	 *   outArea[d]     = # filled voxels whose neighbour along d is empty (outside = empty)
	 *   outBoundary[d] = # filled voxels in the leaf's outermost slab along d
	 *
	 * 精确计算一个 4x4x4 叶节点的暴露面面积与边界薄片占用数。
	 *   outArea[d]     = 沿 d 方向邻居为空的实心体素数（外部视为空）
	 *   outBoundary[d] = 位于叶节点沿 d 方向最外层薄片中的实心体素数
	 *
	 * A voxel w has a filled neighbour at w - e_d exactly when w is in the +d translate of
	 * the mask, so the exposed area along -d is popcount(mask & ~shift_plus_d(mask)).
	 * 体素 w 在 w - e_d 处有实心邻居，当且仅当 w 属于掩码沿 +d 的平移，因此 -d 方向的暴露
	 * 面积就是 popcount(mask & ~shift_plus_d(mask))。
	 */
	HOST_DEVICE void leaf_face_areas(uint64 mask, float outArea[C_numDirections], float outBoundary[C_numDirections])
	{
		const uint64 plusX  = shift_mask_plus (mask, C_maskX0, C_maskX1, 4, 28);
		const uint64 minusX = shift_mask_minus(mask, C_maskX0, C_maskX1, 4, 28);
		const uint64 plusY  = shift_mask_plus (mask, C_maskY0, C_maskY1, 2, 14);
		const uint64 minusY = shift_mask_minus(mask, C_maskY0, C_maskY1, 2, 14);
		const uint64 plusZ  = shift_mask_plus (mask, C_maskZ0, C_maskZ1, 1, 7);
		const uint64 minusZ = shift_mask_minus(mask, C_maskZ0, C_maskZ1, 1, 7);

		outArea[0] = float(Utils::popcll(mask & ~plusX));  // -X
		outArea[1] = float(Utils::popcll(mask & ~minusX)); // +X
		outArea[2] = float(Utils::popcll(mask & ~plusY));  // -Y
		outArea[3] = float(Utils::popcll(mask & ~minusY)); // +Y
		outArea[4] = float(Utils::popcll(mask & ~plusZ));  // -Z
		outArea[5] = float(Utils::popcll(mask & ~minusZ)); // +Z

		outBoundary[0] = float(Utils::popcll(mask & ~C_maskX1 & ~C_maskX0)); // x == 0
		outBoundary[1] = float(Utils::popcll(mask &  C_maskX1 &  C_maskX0)); // x == 3
		outBoundary[2] = float(Utils::popcll(mask & ~C_maskY1 & ~C_maskY0)); // y == 0
		outBoundary[3] = float(Utils::popcll(mask &  C_maskY1 &  C_maskY0)); // y == 3
		outBoundary[4] = float(Utils::popcll(mask & ~C_maskZ1 & ~C_maskZ0)); // z == 0
		outBoundary[5] = float(Utils::popcll(mask &  C_maskZ1 &  C_maskZ0)); // z == 3
	}

	// --- Packing / unpacking ------------------------------------------------------

	// relief01 is the relief area divided by the node's own projected face area.
	// relief01 是起伏面积除以节点自身的投影面面积。
	HOST_DEVICE uint32 quantise_bin(float relief01)
	{
		const float clamped = (relief01 < 0.f) ? 0.f : ((relief01 > 1.f) ? 1.f : relief01);
		return uint32(clamped * float(C_binMax) + 0.5f) & C_binMax;
	}
	HOST_DEVICE uint32 pack_bin(uint32 bin, uint32 dir)
	{
		return (bin & C_binMax) << (dir * C_binBits);
	}
	HOST_DEVICE uint32 unpack_bin(uint32 packed, uint32 dir)
	{
		return (packed >> (dir * C_binBits)) & C_binMax;
	}
	HOST_DEVICE float bin_to_relief(uint32 bin)
	{
		return float(bin) * (1.f / float(C_binMax));
	}

	// Outward normal of a direction slot. / 方向槽位对应的外向法线。
	HOST_DEVICE float3 direction_normal(uint32 dir)
	{
		const float sign = (dir & 1u) ? 1.f : -1.f;
		const uint32 axis = dir >> 1;
		if (axis == 0) return make_float3(sign, 0.f, 0.f);
		if (axis == 1) return make_float3(0.f, sign, 0.f);
		return make_float3(0.f, 0.f, sign);
	}
}

/**
 * Side hash table: DAG node index -> packed 6-direction relief histogram.
 * 旁路哈希表：DAG 节点下标 -> 打包的 6 方向起伏直方图。
 *
 * Open addressing with linear probing, power-of-two capacity, load factor <= 0.5. One
 * slot is a single uint64 so a probe is one 8-byte load:
 *   slot = (nodeIndex << 32) | packedHistogram
 *
 * HashDAG node indices are *virtual addresses* produced by HashDagUtils::make_ptr. They
 * are globally unique across levels (each level owns a disjoint range of the address
 * space) and always strictly below C_totalVirtualAddresses, which a static_assert in
 * hash_dag_globals.h keeps below UINT32_MAX. 0xFFFFFFFF can therefore never be a real
 * key, so an all-ones slot is a safe "empty" marker.
 *
 * 开放寻址 + 线性探测，容量为 2 的幂，装载因子 <= 0.5。一个槽位就是一个 uint64，因此一次
 * 探测只需一次 8 字节读取：slot = (nodeIndex << 32) | packedHistogram
 *
 * HashDAG 的节点下标是 HashDagUtils::make_ptr 产生的"虚拟地址"，在所有层级上全局唯一（每
 * 层独占一段互不相交的地址空间），并且严格小于 C_totalVirtualAddresses —— hash_dag_globals.h
 * 中的 static_assert 保证它小于 UINT32_MAX。因此 0xFFFFFFFF 永远不会是真实的键，全 1 的
 * 槽位可以安全地作为"空"标记。
 */
struct PrefilterTable
{
	static constexpr uint64 C_emptySlot = ~uint64(0);
	// Hard probe cap so a malformed table can never spin a warp forever. The builder
	// asserts that no insertion needs more than this.
	// 硬性探测上限，保证损坏的表不会让 warp 死转。构建时会断言没有插入超过这个次数。
	static constexpr uint32 C_maxProbes = 64;

	StaticArray<uint64> slots_CPU;
	StaticArray<uint64> slots_GPU;
	// capacity - 1; also doubles as the "table exists" flag (a built table is never empty).
	// capacity - 1；同时兼作"表已存在"的标志（已构建的表容量不会为 0）。
	uint32 capacityMask = 0;

	HOST_DEVICE bool is_valid() const
	{
		return capacityMask != 0;
	}

	HOST_DEVICE uint32 find(uint32 nodeIndex) const
	{
#ifdef __CUDA_ARCH__
		const uint64* __restrict__ slots = slots_GPU.data();
#else
		const uint64* __restrict__ slots = slots_CPU.data();
#endif
		if (!slots) return LodPrefilter::C_emptyHistogram;

		uint32 slot = Utils::murmurhash32(nodeIndex) & capacityMask;
		for (uint32 probe = 0; probe < C_maxProbes; ++probe)
		{
			const uint64 entry = slots[slot];
			if (entry == C_emptySlot) break; // empty slot => the key is not in the table
			if (uint32(entry >> 32) == nodeIndex) return uint32(entry);
			slot = (slot + 1) & capacityMask;
		}
		return LodPrefilter::C_emptyHistogram;
	}

	// entries[i] = (nodeIndex << 32) | packedHistogram, in any order, no duplicate keys.
	// entries[i] = (节点下标 << 32) | 打包直方图，顺序任意，键不重复。
	HOST void build(const std::vector<uint64>& entries)
	{
		PROFILE_FUNCTION();
		free(); // rebuilding is allowed (e.g. after a GC pass reshuffles node indices)
		if (entries.empty()) return;

		uint64 capacity = 1024;
		while (capacity < entries.size() * 2) capacity <<= 1;
		checkAlways(capacity <= (uint64(1) << 32));

		slots_CPU = StaticArray<uint64>::allocate("lod prefilter table", capacity, EMemoryType::CPU);
		for (uint64 i = 0; i < capacity; ++i) slots_CPU[i] = C_emptySlot;

		capacityMask = uint32(capacity - 1);

		for (uint64 entry : entries)
		{
			const uint32 key = uint32(entry >> 32);
			uint32 slot = Utils::murmurhash32(key) & capacityMask;
			uint32 probe = 0;
			while (slots_CPU[slot] != C_emptySlot)
			{
				checkAlways(uint32(slots_CPU[slot] >> 32) != key); // duplicate key
				slot = (slot + 1) & capacityMask;
				++probe;
				checkfAlways(probe < C_maxProbes, "prefilter table probe run too long: %u", probe);
			}
			slots_CPU[slot] = entry;
		}

		slots_GPU = slots_CPU.create_gpu();
	}

	HOST void free()
	{
		if (slots_CPU.is_valid()) slots_CPU.free();
		if (slots_GPU.is_valid()) slots_GPU.free();
		capacityMask = 0;
	}

	HOST void print_stats(uint64 numEntries) const
	{
		if (!is_valid())
		{
			printf("\tLOD prefilter table: empty\n");
			return;
		}
		const uint64 capacity = uint64(capacityMask) + 1;
		printf(
			"\tLOD prefilter table: %" PRIu64 " entries in %" PRIu64 " slots (%.0f%% load)\n"
			"\t\tlevels [%u, %u], %fMB on the CPU + %fMB on the GPU\n",
			numEntries, capacity, 100.0 * double(numEntries) / double(capacity),
			uint32(PREFILTER_MIN_LEVEL), uint32(PREFILTER_MAX_LEVEL),
			slots_CPU.size_in_MB(), slots_GPU.size_in_MB());
	}
};

#endif // ~ ENABLE_PREFILTERED_SHADING
