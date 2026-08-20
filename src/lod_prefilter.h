#pragma once

#include "typedefs.h"
#include "utils.h"
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
 * One uint32 per node. X/Y use 5-bit bins, Z uses 4-bit bins, and bits 28..30 are pass-X/Y/Z.
 * Clamping at 1 is physically motivated: once the relief area equals the node's own
 * projected face area, adding more area cannot make more of it visible (self-occlusion).
 *
 *   bits [ 0.. 4] : -X    bits [ 5.. 9] : +X
 *   bits [10..14] : -Y    bits [15..19] : +Y
 *   bits [20..23] : -Z    bits [24..27] : +Z
 *   bits [28..30] : pass X/Y/Z    bit 31 : free
 *
 * 每节点一个 uint32：X/Y 各 5 bit，Z 各 4 bit，bits 28..30 为 X/Y/Z 通过位。
 *
 * The packed word is stored at the end of each HashDAG node.
 */
namespace LodPrefilter
{
	// Direction slot order: 0 = -X, 1 = +X, 2 = -Y, 3 = +Y, 4 = -Z, 5 = +Z.
	// 方向槽位顺序：0 = -X, 1 = +X, 2 = -Y, 3 = +Y, 4 = -Z, 5 = +Z。
	constexpr uint32 C_numDirections = 6;
	constexpr uint32 C_numAxes = 3;

	constexpr uint32 C_passShift = 28;

	// A packed value of 0 falls back to the geometric box normal.
	constexpr uint32 C_emptyHistogram = 0u;

	constexpr float C_coverageThreshold = PREFILTER_COVERAGE_THRESHOLD;

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
	HOST_DEVICE uint32 bin_bits(uint32 dir)
	{
		return dir < 4 ? 5u : 4u;
	}
	HOST_DEVICE uint32 bin_shift(uint32 dir)
	{
		return dir < 4 ? dir * 5u : 20u + (dir - 4u) * 4u;
	}
	HOST_DEVICE uint32 bin_max(uint32 dir)
	{
		return (1u << bin_bits(dir)) - 1u;
	}
	HOST_DEVICE uint32 quantise_bin(float relief01, uint32 dir)
	{
		const float clamped = (relief01 < 0.f) ? 0.f : ((relief01 > 1.f) ? 1.f : relief01);
		const uint32 maxBin = bin_max(dir);
		return uint32(clamped * float(maxBin) + 0.5f) & maxBin;
	}
	HOST_DEVICE uint32 pack_bin(uint32 bin, uint32 dir)
	{
		return (bin & bin_max(dir)) << bin_shift(dir);
	}
	HOST_DEVICE uint32 unpack_bin(uint32 packed, uint32 dir)
	{
		return (packed >> bin_shift(dir)) & bin_max(dir);
	}
	HOST_DEVICE float bin_to_relief(uint32 bin, uint32 dir)
	{
		return float(bin) / float(bin_max(dir));
	}

	// --- Coverage-aware LOD -------------------------------------------------------

	HOST_DEVICE uint32 pack_pass_axis(uint32 packed, uint32 axis, bool pass)
	{
		const uint32 bit = 1u << (C_passShift + axis);
		return pass ? packed | bit : packed & ~bit;
	}
	HOST_DEVICE bool pass_axis(uint32 packed, uint32 axis)
	{
		return (packed & (1u << (C_passShift + axis))) != 0u;
	}
	HOST_DEVICE uint32 dominant_axis(float3 direction)
	{
		const float3 a = abs(direction);
		if (a.x >= a.y && a.x >= a.z) return 0;
		return a.y >= a.z ? 1 : 2;
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

#endif // ~ ENABLE_PREFILTERED_SHADING
