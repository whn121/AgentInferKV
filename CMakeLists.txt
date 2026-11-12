cmake_minimum_required(VERSION 3.16)
project(AgentInferKV LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if (NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Debug CACHE STRING "Build type" FORCE)
endif()

# 核心库: 不依赖 llama.cpp，纯模拟可独立编译

add_library(agentinferkv_core
  src/Request.cpp
  src/FCFSScheduler.cpp
  src/NaivePreallocator.cpp
  src/TraceReplayer.cpp
  src/SimulatedExecutor.cpp
    src/SimulatedCellExecutor.cpp
  src/SimulationLoop.cpp
  src/PagedBlockManager.cpp
  src/PrefixCache.cpp
)
target_include_directories(agentinferkv_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_compile_options(agentinferkv_core PRIVATE -Wall -Wextra -Wpedantic)

add_executable(agentinferkv_demo src/main.cpp)
target_link_libraries(agentinferkv_demo PRIVATE agentinferkv_core)

# 实验1：同预算下 Naive vs Paged 的可复现对比（README 的表由它产出）
add_executable(agentinferkv_experiment1 src/experiment1.cpp)

# 块 → 物理 cell 映射演示（证明"分页不是假的"）
add_executable(agentinferkv_kvcell_demo tools/kv_cell_demo.cpp)

# L1：自管 cell 池的执行器（块→物理区间，统计 KV 写入量）
add_executable(agentinferkv_cell_demo src/experiment6.cpp)
target_link_libraries(agentinferkv_cell_demo PRIVATE agentinferkv_core)

# 最小 HTTP 服务（把外部请求接到控制面）
add_executable(agentinferkv_http tools/http_server.cpp)
target_link_libraries(agentinferkv_http PRIVATE agentinferkv_core)
target_link_libraries(agentinferkv_kvcell_demo PRIVATE agentinferkv_core)
target_link_libraries(agentinferkv_experiment1 PRIVATE agentinferkv_core)
target_compile_options(agentinferkv_experiment1 PRIVATE -Wall -Wextra -Wpedantic)

enable_testing()

foreach(test_name
    test_request
    test_model_config
    test_naive_preallocator
    test_fcfs_scheduler
    test_simulated_executor
    test_simulation_loop
     test_paged_block_manager
    test_prefix_cache
    test_prefill_once
    test_prefix_sharing
    test_lru_eviction)
  add_executable(${test_name} tests/${test_name}.cpp)
  target_link_libraries(${test_name} PRIVATE agentinferkv_core)
  add_test(NAME ${test_name} COMMAND ${test_name})
endforeach()

# 可选: 真实 llama.cpp 执行后端
# 配置时加 -DUSE_LLAMA_CPP=ON 才编译
# 如果 llama.cpp 不在 ~/src/llama.cpp，用 -DLLAMA_ROOT=/你的路径 覆盖

option(USE_LLAMA_CPP "Build with real llama.cpp executor" OFF)

if(USE_LLAMA_CPP)
  if(NOT DEFINED LLAMA_ROOT OR LLAMA_ROOT STREQUAL "")
    set(LLAMA_ROOT "$ENV{HOME}/src/llama.cpp")
  endif()

  message(STATUS "=== llama.cpp backend enabled ===")
  message(STATUS "LLAMA_ROOT = ${LLAMA_ROOT}")

  # 校验关键文件是否存在，早失败早报错
  if(NOT EXISTS "${LLAMA_ROOT}/include/llama.h")
    message(FATAL_ERROR "找不到 ${LLAMA_ROOT}/include/llama.h")
  endif()
  if(NOT EXISTS "${LLAMA_ROOT}/ggml/include/ggml.h")
    message(FATAL_ERROR "找不到 ${LLAMA_ROOT}/ggml/include/ggml.h")
  endif()
  if(NOT EXISTS "${LLAMA_ROOT}/build/bin/libllama.so")
    message(FATAL_ERROR "找不到 ${LLAMA_ROOT}/build/bin/libllama.so，先编译 llama.cpp")
  endif()

  add_library(agentinferkv_llama STATIC src/LlamaCppExecutor.cpp)

  target_include_directories(agentinferkv_llama PUBLIC
    ${LLAMA_ROOT}/include
    ${LLAMA_ROOT}/ggml/include
  )

  target_link_directories(agentinferkv_llama PUBLIC
    ${LLAMA_ROOT}/build/bin
  )

  target_link_libraries(agentinferkv_llama PUBLIC
    agentinferkv_core
    llama
    ggml
    ggml-base
    ggml-cpu
    ggml-cuda
  )

  add_executable(agentinferkv_demo_real src/main_real.cpp)
  target_link_libraries(agentinferkv_demo_real PRIVATE agentinferkv_llama)

  # 标定工具：真实 GPU 上测 Prefill/Decode 耗时，供 SimulatedExecutor 拟合
  add_executable(agentinferkv_calibrate tools/calibrate_executor.cpp)
  target_link_libraries(agentinferkv_calibrate PRIVATE agentinferkv_llama)

  # 让运行时自动找到 .so，不用手动 LD_LIBRARY_PATH
  set_target_properties(agentinferkv_demo_real PROPERTIES
    BUILD_RPATH   "${LLAMA_ROOT}/build/bin"
    INSTALL_RPATH "${LLAMA_ROOT}/build/bin"
  )
endif()