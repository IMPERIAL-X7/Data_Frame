cmake_minimum_required(VERSION 3.15)
project(DataFrameLib CXX)

set(CMAKE_CXX_STANDARD 17) # or 20, depending on your preference
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Point CMake to conda Arrow/Parquet installation
list(APPEND CMAKE_PREFIX_PATH "$ENV{HOME}/anaconda3")

# Find Apache Arrow
find_package(Arrow REQUIRED)
find_package(Parquet REQUIRED)
# Find Graphviz/Boost for DAG rendering
# find_package(Graphviz REQUIRED) 

# Include directories
include_directories(include)

# Your core library
add_library(DataFrameLib 
    src/Eager.cpp 
    src/Lazy.cpp 
    src/Expr.cpp 
    src/Optimizer.cpp
    src/IOUtils.cpp
)

# Link Arrow to your library
target_link_libraries(DataFrameLib PUBLIC Arrow::arrow_shared Parquet::parquet_shared)

# Add main executable
add_executable(main main.cpp)
target_link_libraries(main PRIVATE DataFrameLib)
