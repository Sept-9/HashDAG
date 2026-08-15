#pragma once

#include "typedefs.h"
#include "tracer.h"
#include "camera_view.h"
#include "cuda_gl_buffer.h"
#include "dag_info.h"

class DAGTracer
{
public:
	const bool headLess;

	DAGTracer(bool headLess);
	~DAGTracer();

	inline GLuint get_colors_image() const
	{
		return colorsImage;
	}

	template<typename TDAG>
	float resolve_paths(const CameraView& camera, const DAGInfo& dagInfo, const TDAG& dag, float lodPixelThreshold = 0.f);
	template<typename TDAG, typename TDAGColors>
	float resolve_colors(const TDAG& dag, const TDAGColors& colors, EDebugColors debugColors, uint32 debugColorsIndexLevel, ToolInfo toolInfo);
	template<typename TDAG>
	float resolve_shadows(const CameraView& camera, const DAGInfo& dagInfo, const TDAG& dag, float shadowBias, float fogDensity);

	uint3 get_path(uint32 posX, uint32 posY);

private:
	GLuint pathsImage = 0;
	GLuint colorsImage = 0;

	CudaGLBuffer pathsBuffer;
	CudaGLBuffer colorsBuffer;

	cudaArray* pathArray = nullptr;
	cudaArray* colorsArray = nullptr;

	uint3* pathCache = nullptr;

#if ENABLE_PREFILTERED_SHADING
	// Prefiltered appearance: one packed 6-direction relief histogram per pixel, produced by
	// trace_paths and consumed by trace_shadows. It is never displayed, so it is plain device
	// memory rather than a GL-interop surface.
	// 预滤波外观：每像素一个打包的 6 方向起伏直方图，由 trace_paths 产生、trace_shadows 消费。
	// 它不会被显示，因此用普通显存而非 GL 互操作 surface。
	uint32* prefilterBuffer = nullptr;
#endif

	cudaEvent_t eventBeg, eventEnd;
};