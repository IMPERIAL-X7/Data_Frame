cmake_minimum_required(VERSION 3.15)
project(DataFrameLib CXX)

set(CMAKE_CXX_STANDARD 17) # or 20, depending on your preference
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find Apache Arrow
find_package(Arrow REQUIRED)
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
target_link_libraries(DataFrameLib PRIVATE Arrow::arrow_shared)