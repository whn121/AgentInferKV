cmake_minimum_required(VERSION 3.16)
project(AgentInferKV LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if (NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Debug CACHE STRING "Build type" FORCE)
endif()

add_library(agentinferkv_core
  src/Request.cpp
  src/FCFSScheduler.cpp
  src/NaivePreallocator.cpp
  src/TraceReplayer.cpp
  src/SimulatedExecutor.cpp
  src/SimulationLoop.cpp 
  src/PagedBlockManager.cpp
  src/PrefixCache.cpp 
)
target_include_directories(agentinferkv_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_compile_options(agentinferkv_core PRIVATE -Wall -Wextra -Wpedantic)

add_executable(agentinferkv_demo src/main.cpp)
target_link_libraries(agentinferkv_demo PRIVATE agentinferkv_core)

enable_testing()

foreach(test_name
    test_request
    test_model_config
    test_naive_preallocator
    test_fcfs_scheduler
    test_simulated_executor
    test_simulation_loop
    test_paged_block_manager
    test_prefix_cache) 
  add_executable(${test_name} tests/${test_name}.cpp)
  target_link_libraries(${test_name} PRIVATE agentinferkv_core)
  add_test(NAME ${test_name} COMMAND ${test_name})
endforeach()