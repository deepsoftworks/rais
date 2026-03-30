cmake_minimum_required(VERSION 3.20)
project(rais LANGUAGES CXX OBJCXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_OBJCXX_STANDARD 20)
set(CMAKE_OBJCXX_STANDARD_REQUIRED ON)
set(CMAKE_OBJCXX_EXTENSIONS OFF)

add_compile_options(-Wall -Wextra -Wpedantic)

# Library target
add_library(rais STATIC
    src/scheduler.cpp
    src/streaming.cpp
    src/layer_streamer.cpp
    src/memory_pressure.cpp
    src/model_manager.cpp
    src/metal_executor.mm
    src/metal_allocator.mm
    src/profiler.mm
)
target_include_directories(rais PUBLIC ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(rais PUBLIC "-framework Metal" "-framework Foundation" "-framework QuartzCore")
add_dependencies(rais metal_shaders)

# Metal shader compilation: .metal → .air → .metallib
set(METAL_SHADER_SOURCE ${CMAKE_SOURCE_DIR}/shaders/rais_kernels.metal)
set(METAL_AIR ${CMAKE_BINARY_DIR}/rais_kernels.air)
set(METAL_LIB ${CMAKE_BINARY_DIR}/rais_kernels.metallib)

add_custom_command(
    OUTPUT ${METAL_AIR}
    COMMAND xcrun -sdk macosx metal -c ${METAL_SHADER_SOURCE} -o ${METAL_AIR}
    DEPENDS ${METAL_SHADER_SOURCE}
    COMMENT "Compiling Metal shaders to AIR"
)

add_custom_command(
    OUTPUT ${METAL_LIB}
    COMMAND xcrun -sdk macosx metallib ${METAL_AIR} -o ${METAL_LIB}
    DEPENDS ${METAL_AIR}
    COMMENT "Linking AIR to Metal library"
)

add_custom_target(metal_shaders DEPENDS ${METAL_LIB})

# Catch2
find_package(Catch2 3 REQUIRED)

# Tests
enable_testing()

add_executable(test_queue tests/test_queue.cpp)
target_link_libraries(test_queue PRIVATE rais Catch2::Catch2WithMain)
add_test(NAME test_queue COMMAND test_queue)

add_executable(test_deque tests/test_deque.cpp)
target_link_libraries(test_deque PRIVATE rais Catch2::Catch2WithMain)
add_test(NAME test_deque COMMAND test_deque)

add_executable(test_scheduler tests/test_scheduler.cpp)
target_link_libraries(test_scheduler PRIVATE rais Catch2::Catch2WithMain)
add_test(NAME test_scheduler COMMAND test_scheduler)

add_executable(test_allocator tests/test_allocator.cpp)
target_link_libraries(test_allocator PRIVATE rais Catch2::Catch2WithMain)
add_test(NAME test_allocator COMMAND test_allocator)

add_executable(test_continuations tests/test_continuations.cpp)
target_link_libraries(test_continuations PRIVATE rais Catch2::Catch2WithMain)
add_test(NAME test_continuations COMMAND test_continuations)

add_executable(test_io_lane tests/test_io_lane.cpp)
target_link_libraries(test_io_lane PRIVATE rais Catch2::Catch2WithMain)
add_test(NAME test_io_lane COMMAND test_io_lane)

add_executable(test_layer_streamer tests/test_layer_streamer.mm)
target_link_libraries(test_layer_streamer PRIVATE rais Catch2::Catch2WithMain)
add_test(NAME test_layer_streamer COMMAND test_layer_streamer)

add_executable(test_memory_pressure tests/test_memory_pressure.mm)
target_link_libraries(test_memory_pressure PRIVATE rais Catch2::Catch2WithMain)
add_test(NAME test_memory_pressure COMMAND test_memory_pressure)

add_executable(test_model_swap tests/test_model_swap.cpp)
target_link_libraries(test_model_swap PRIVATE rais Catch2::Catch2WithMain)
add_test(NAME test_model_swap COMMAND test_model_swap)

add_executable(test_metal_executor tests/test_metal_executor.mm)
target_link_libraries(test_metal_executor PRIVATE rais Catch2::Catch2WithMain)
target_compile_definitions(test_metal_executor PRIVATE
    RAIS_METALLIB_PATH="${METAL_LIB}")
add_dependencies(test_metal_executor metal_shaders)
add_test(NAME test_metal_executor COMMAND test_metal_executor)

add_executable(test_metal_allocator tests/test_metal_allocator.mm)
target_link_libraries(test_metal_allocator PRIVATE rais Catch2::Catch2WithMain)
add_test(NAME test_metal_allocator COMMAND test_metal_allocator)

add_executable(test_profiler tests/test_profiler.mm)
target_link_libraries(test_profiler PRIVATE rais Catch2::Catch2WithMain)
add_test(NAME test_profiler COMMAND test_profiler)

# Benchmarks
add_executable(bench_queue benchmarks/bench_queue.cpp)
target_link_libraries(bench_queue PRIVATE rais)

add_executable(bench_deque benchmarks/bench_deque.cpp)
target_link_libraries(bench_deque PRIVATE rais)

add_executable(bench_scheduler benchmarks/bench_scheduler.cpp)
target_link_libraries(bench_scheduler PRIVATE rais)

add_executable(bench_allocator benchmarks/bench_allocator.cpp)
target_link_libraries(bench_allocator PRIVATE rais)

add_executable(bench_metal benchmarks/bench_metal.mm)
target_link_libraries(bench_metal PRIVATE rais)
target_compile_definitions(bench_metal PRIVATE
    RAIS_METALLIB_PATH="${METAL_LIB}")
add_dependencies(bench_metal metal_shaders)

# Examples
add_executable(priority_example examples/priority_scheduling.cpp)
target_link_libraries(priority_example PRIVATE rais)

# Experiments
add_executable(bench_inference_llm experiments/bench_inference_llm.cpp)
target_link_libraries(bench_inference_llm PRIVATE rais)
