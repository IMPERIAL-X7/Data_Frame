#include <dataframelib/dataframelib.h>
#include <iostream>

// Lightweight smoke check for the public API. Used during development to
// confirm that read/select/filter compile and link against the public header.
int main() {
    using namespace dataframelib;
    try {
        std::cout << "DataFrameLib correctness_check: API is reachable." << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "correctness_check failed: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}
