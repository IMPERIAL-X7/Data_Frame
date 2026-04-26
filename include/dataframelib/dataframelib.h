#pragma once

// Public umbrella header for DataFrameLib.
// All public types and free functions are declared in `namespace dataframelib`.

#include "../../src/ExpressionSystem.h"
#include "../../src/EagerDataFrame.h"
#include "../../src/LazyDataFrame.h"

namespace dataframelib {

// ---------- Free function I/O API ----------
// These mirror the spec API: read_csv(path), read_parquet(path), scan_csv(path),
// scan_parquet(path), from_columns(map). They forward to the static class methods.

inline EagerDataFrame read_csv(const std::string& path) {
    return EagerDataFrame::read_csv(path);
}

inline EagerDataFrame read_parquet(const std::string& path) {
    return EagerDataFrame::read_parquet(path);
}

inline LazyDataFrame scan_csv(const std::string& path) {
    return LazyDataFrame::scan_csv(path);
}

inline LazyDataFrame scan_parquet(const std::string& path) {
    return LazyDataFrame::scan_parquet(path);
}

inline EagerDataFrame from_columns(
    const std::unordered_map<std::string, std::shared_ptr<arrow::Array>>& col_map) {
    return EagerDataFrame::from_columns(col_map);
}

}
