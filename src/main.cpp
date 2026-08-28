#include "typedefs.h"

#include "dags/basic_dag/basic_dag.h"
#include "dags/hash_dag/hash_dag_factory.h"
#include "dags/dag_utils.h"
#include "engine.h"

#include <filesystem>
#include <fstream>
#include <iomanip>

namespace
{
void write_build_stats(
    const char* outputFile,
    double geometryBuildMs,
    double colorBuildMs,
    double totalBuildMs,
    const HashDAG& hashDag,
    const HashDAGColors& hashDagColors)
{
    const std::filesystem::path outputPath(outputFile);
    if (outputPath.has_parent_path())
    {
        std::error_code error;
        std::filesystem::create_directories(outputPath.parent_path(), error);
        checkfAlways(!error, "Could not create build stats directory: %s", error.message().c_str());
    }

    const double geometryLogicalMb = hashDag.data.get_virtual_used_size(false);
    const double geometryAllocatedMb = hashDag.data.get_allocated_pages_size();
    const double colorLogicalMb = hashDagColors.get_total_used_memory();

    std::ofstream output(outputPath);
    checkfAlways(output.is_open(), "Could not open build stats output: %s", outputFile);
    output << "configuration,prefilter_enabled,geometry_build_ms,color_build_ms,total_build_ms,"
              "geometry_logical_mb,geometry_allocated_mb,color_logical_mb,total_logical_mb\n";
    output << (ENABLE_PREFILTERED_SHADING ? "Inline prefilter" : "Without prefilter") << ','
           << ENABLE_PREFILTERED_SHADING << ','
           << std::fixed << std::setprecision(6)
           << geometryBuildMs << ','
           << colorBuildMs << ','
           << totalBuildMs << ','
           << geometryLogicalMb << ','
           << geometryAllocatedMb << ','
           << colorLogicalMb << ','
           << (geometryLogicalMb + colorLogicalMb) << '\n';
    checkfAlways(output.good(), "Could not write build stats output: %s", outputFile);

    printf("Build statistics written to %s\n", outputFile);
}

// EXIT_AFTER_BUILD_STATS stops before Engine::init(), so Engine::destroy()
// cannot be used: it also tears down graphics objects that were never created.
// Release only the data loaded or constructed above before returning.
void destroy_build_data(Engine& engine)
{
    engine.undoRedo.free();
    engine.hashDagColors.free();
    engine.hashDag.free();
    engine.basicDagColorErrors.free();
    engine.basicDagUncompressedColors.free();
    engine.basicDagCompressedColors.free();
    engine.basicDag.free();
}
}

int main(int argc, char** argv)
{
    PROFILE_FUNCTION();
	
	auto& engine = Engine::engine;

	printf("Using " SCENE "\n");
    printf("%d levels (resolution=%d^3)\n", MAX_LEVELS, 1 << MAX_LEVELS);
#if ENABLE_CHECKS
    std::fprintf(stderr, "CHECKS: ENABLED\n");
#else
    printf("CHECKS: DISABLED\n");
#endif
    printf("IMAGE RESOLUTION: %ux%u\n", imageWidth, imageHeight);

    const std::string fileName = std::string(SCENE) + std::to_string(1 << (SCENE_DEPTH - 10)) + "k";

    if (LOAD_UNCOMPRESSED_COLORS)
    {
        BasicDAGFactory::load_uncompressed_colors_from_file(engine.basicDagUncompressedColors, "data/" + fileName + ".basic_dag.uncompressed_colors.bin");
    }
    if (LOAD_COMPRESSED_COLORS)
    {
        BasicDAGFactory::load_compressed_colors_from_file(engine.basicDagCompressedColors, "data/" + fileName + ".basic_dag.compressed_colors.variable.bin");
    }
    BasicDAGFactory::load_dag_from_file(engine.dagInfo, engine.basicDag, "data/" + fileName + ".basic_dag.dag.bin");

#if 0
    DAGUtils::fix_enclosed_leaves(engine.basicDag, engine.basicDagCompressedColors.enclosedLeaves, engine.basicDagCompressedColors.topLevels);
#if 0
	BasicDAGFactory::save_compressed_colors_to_file(engine.basicDagCompressedColors, "data/" FILENAME ".basic_dag.compressed_colors.variable.bin");
    engine.basicDagCompressedColors.free();
    BasicDAGFactory::load_compressed_colors_from_file(engine.basicDagCompressedColors, "data/" FILENAME ".basic_dag.compressed_colors.variable.bin");
#endif
#endif

    double geometryBuildMs = 0.0;
    double colorBuildMs = 0.0;
    double totalBuildMs = 0.0;
	if (LOAD_COMPRESSED_COLORS)
    {
        const SimpleScopeStat totalBuildTimer;
        const SimpleScopeStat geometryBuildTimer;
        HashDAGFactory::load_from_DAG(engine.hashDag, engine.basicDag, 0x8FFFFFFF / C_pageSize / sizeof(uint32));
        geometryBuildMs = geometryBuildTimer.get_time();

        const SimpleScopeStat colorBuildTimer;
        HashDAGFactory::load_colors_from_DAG(engine.hashDagColors, engine.basicDag, engine.basicDagCompressedColors);
        colorBuildMs = colorBuildTimer.get_time();
        totalBuildMs = totalBuildTimer.get_time();
    }

#ifdef BUILD_STATS_OUTPUT
    write_build_stats(
        BUILD_STATS_OUTPUT,
        geometryBuildMs,
        colorBuildMs,
        totalBuildMs,
        engine.hashDag,
        engine.hashDagColors);
#if EXIT_AFTER_BUILD_STATS
    destroy_build_data(engine);
    return 0;
#endif
#endif

	engine.basicDagColorErrors.uncompressedColors = engine.basicDagUncompressedColors;
	engine.basicDagColorErrors.compressedColors = engine.basicDagCompressedColors;

#if FREE_BASIC_DAG_AFTER_BUILD
    {
        const size_t gpuBefore = Memory::get_gpu_allocated_memory();
        engine.basicDag.free();
        printf("Freed BasicDAG geometry: %fMB of GPU memory reclaimed\n",
               Utils::to_MB(gpuBefore - Memory::get_gpu_allocated_memory()));
    }
#endif

	engine.init(HEADLESS);
#if USE_NORMAL_DAG
	engine.set_dag(EDag::BasicDagCompressedColors);
#else
	engine.set_dag(EDag::HashDag);
#endif

#if USE_VIDEO
	engine.toggle_fullscreen();
    engine.videoManager.load_video("./videos/" SCENE "_" VIDEO_NAME ".txt");
	std::this_thread::sleep_for(std::chrono::seconds(5));
#elif ENABLE_REPLAY
    engine.replayReader.load_csv("./replays/" SCENE "_" REPLAY_NAME ".csv");
#endif

	printf("Starting...\n");

#ifdef PROFILING_PATH
    engine.hashDag.data.save_bucket_sizes(true);
#endif

	engine.loop();
	engine.destroy();

	return 0;
}
