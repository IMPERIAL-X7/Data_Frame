cmake_minimum_required(VERSION 3.15)
project(dataframelib CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Point CMake to conda Arrow/Parquet installation if present
list(APPEND CMAKE_PREFIX_PATH "$ENV{HOME}/anaconda3")

find_package(Arrow REQUIRED)
find_package(Parquet REQUIRED)

# Public headers live under include/. The public umbrella header is
# include/dataframelib/dataframelib.h, accessed as <dataframelib/dataframelib.h>.
# Internal headers live under src/.

add_library(dataframelib
    src/ExpressionSystem.cpp
    src/EagerDataFrame_impl/read-write.cpp
    src/EagerDataFrame_impl/head-select.cpp
    src/EagerDataFrame_impl/table_view.cpp
    src/EagerDataFrame_impl/with-column-sort.cpp
    src/EagerDataFrame_impl/with-column-join.cpp
    src/EagerDataFrame_impl/group-by-aggregate.cpp
    src/LazyDataFrame_impl/LazyDataFrame.cpp
    src/LazyDataFrame_impl/PlanNodes.cpp
    src/LazyDataFrame_impl/GraphVizUtils.cpp
    src/LazyDataFrame_impl/QueryOptimizerFacade.cpp        # pipeline orchestration
    src/LazyDataFrame_impl/QueryOptimizerExpression.cpp    # constant folding + algebraic simplification
    src/LazyDataFrame_impl/QueryOptimizerPredicate.cpp     # predicate pushdown
    src/LazyDataFrame_impl/QueryOptimizerProjection.cpp    # projection pushdown
    src/LazyDataFrame_impl/QueryOptimizerStructural.cpp    # Sort+Head → TopN fusion (limit pushdown)
    src/LazyDataFrame_impl/QueryOptimizerSchema.cpp        # column-reference analysis (helper)
)

target_include_directories(dataframelib PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)
target_include_directories(dataframelib PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)

target_link_libraries(dataframelib PUBLIC
    Arrow::arrow_shared
    Parquet::parquet_shared
)

# Force the compiled binary to prioritize the system's C++ standard library 
# over Anaconda's outdated libstdc++.so.6 at runtime.
target_link_options(dataframelib PUBLIC "-Wl,-rpath=/usr/lib/x86_64-linux-gnu")

# Driver binary for local development. Skip when this CMakeLists is consumed
# as a sub-project (e.g. by the autograder), so the autograder's own build
# produces only the library.
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    add_executable(main src/main.cpp)
    target_link_libraries(main PRIVATE dataframelib)
    target_link_options(main PRIVATE "-Wl,-rpath=/usr/lib/x86_64-linux-gnu")

    add_executable(correctness_check src/correctness_check.cpp)
    target_link_libraries(correctness_check PRIVATE dataframelib)
endif()
