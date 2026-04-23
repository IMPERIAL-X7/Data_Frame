cmake_minimum_required(VERSION 3.15)
project(DataFrameLib CXX)

set(CMAKE_CXX_STANDARD 17) # or 20, depending on your preference
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Point CMake to conda Arrow/Parquet installation
list(APPEND CMAKE_PREFIX_PATH "$ENV{HOME}/anaconda3")

# Find Apache Arrow
find_package(Arrow REQUIRED)
find_package(Parquet REQUIRED)
find_package(ArrowDataset REQUIRED)

# Include directories
include_directories(include)

# Find Graphviz/Boost for DAG rendering
# Not required for compilation as we use the 'dot' CLI tool directly
# find_package(Graphviz)
# find_package(Boost)

# Your core library
add_library(DataFrameLib 
    src/Eager.cpp 
    src/Lazy.cpp 
    src/Expr.cpp 
    src/Optimizer.cpp
    src/IOUtils.cpp
)

# Link Arrow to your library
target_link_libraries(DataFrameLib PUBLIC
    Arrow::arrow_shared
    ArrowDataset::arrow_dataset_shared
    Parquet::parquet_shared
)

# Add main executable
add_executable(main main.cpp)
target_link_libraries(main PRIVATE DataFrameLib)

# Force the compiled binary to prioritize the system's C++ standard library 
# over Anaconda's outdated libstdc++.so.6 at runtime.
target_link_options(main PRIVATE "-Wl,-rpath=/usr/lib/x86_64-linux-gnu")
