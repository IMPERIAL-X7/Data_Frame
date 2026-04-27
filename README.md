# Data_Frame

A high-performance C++ DataFrame library built on top of Apache Arrow and Parquet. It supports both Eager execution (like Pandas) and Lazy execution with query optimization.

## Prerequisites

- **C++17** compatible compiler
- **CMake** (version 3.15 or higher)
- **Apache Arrow** and **Parquet** C++ libraries installed. 

*Note for Conda users:* The `CMakeLists.txt` is configured to look in `$HOME/anaconda3` for Arrow and Parquet automatically, but you can also install them system-wide.

## Build Instructions

To build the library and the local development executables (`main` and `correctness_check`):

```bash
# Clone the repository
git clone https://github.com/IMPERIAL-X7/Data_Frame.git
cd Data_Frame

# Create a build directory
mkdir build
cd build

# Configure with CMake
cmake -DCMAKE_BUILD_TYPE=Release ..

# Compile the project
make -j4
```

After a successful compilation, you will find the generated binaries (e.g., `main`, `correctness_check`) and the `libdataframelib.a` or `libdataframelib.so` library in the `build` directory.

## Installation Instructions

If you want to use `dataframelib` in another CMake project, you can add it as a subdirectory.

```cmake
add_subdirectory(path/to/Data_Frame)
target_link_libraries(your_target PRIVATE dataframelib)
```

Alternatively, you can install the library system-wide:

```bash
cd build
sudo make install
```

<!-- ## Running Tests

If you are using the provided Python autograder (A4-Tester), you can run the test suite as follows:

```bash
cd A4-Tester
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python3 autograder.py --student-dir ..
``` -->
