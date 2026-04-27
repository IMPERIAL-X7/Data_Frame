#pragma once

#include <arrow/api.h>
#include <string>
#include <vector>
#include <memory>
#include <unordered_map>
#include <utility>
#include "ExpressionSystem.h"

namespace dataframelib {

class EagerDataFrame;

// Aggregation spec: ordered list of (column_name, function_name) pairs,
// e.g. {{"salary", "sum"}, {"salary", "count"}}. The same column may appear
// multiple times with different functions; the result columns are named
// "<col>_<func>".
using AggSpec = std::vector<std::pair<std::string, std::string>>;

// Intermediate object returned by EagerDataFrame::group_by(). It holds the
// input table plus the grouping keys until aggregate() is called. We don't
// actually do the grouping until aggregate() so that group_by alone has
// no cost.
class GroupedDataFrame {
private:
    std::shared_ptr<arrow::Table> table_;
    std::vector<std::string> keys_;

public:
    GroupedDataFrame(std::shared_ptr<arrow::Table> table, std::vector<std::string> keys)
        : table_(std::move(table)), keys_(std::move(keys)) {}

    EagerDataFrame aggregate(const AggSpec& aggs) const;
};

// Materialised dataframe — every operation runs immediately and returns a
// new EagerDataFrame backed by a fresh Arrow table. The underlying storage
// is `std::shared_ptr<arrow::Table>` so copies are cheap (just a refcount
// bump) and the data layout is always single-chunk after read/from_columns
// (we call CombineChunks so per-row indexing stays O(1)).
class EagerDataFrame {
    friend class GroupedDataFrame;
private:
    std::shared_ptr<arrow::Table> table_;

public:
    explicit EagerDataFrame(std::shared_ptr<arrow::Table> table) : table_(std::move(table)) {}

    static EagerDataFrame read_csv(const std::string& path);
    static EagerDataFrame read_parquet(const std::string& path);
    // Projection-pushdown variants: read only the listed columns from the
    // source. Used by the lazy optimizer when downstream needs a subset.
    static EagerDataFrame read_csv(const std::string& path, const std::vector<std::string>& columns);
    static EagerDataFrame read_parquet(const std::string& path, const std::vector<std::string>& columns);

    void write_csv(const std::string& path) const;
    void write_parquet(const std::string& path) const;

    EagerDataFrame sort(const std::vector<std::string>& columns, bool asc) const;
    EagerDataFrame head(int64_t n) const;
    // Internal fast-path used by the lazy optimizer to fuse Sort+Head: pick
    // the top n rows by sort key without fully ordering the rest.
    EagerDataFrame sort_top_n(const std::vector<std::string>& columns, bool asc, int64_t n) const;

    EagerDataFrame select(const std::vector<std::string>& columns) const;
    EagerDataFrame filter(const Expr& predicate) const;
    EagerDataFrame with_column(const std::string& name, const Expr& expr) const;
    static EagerDataFrame from_columns(const std::vector<std::pair<std::string, std::shared_ptr<arrow::Array>>>& cols);

    EagerDataFrame join(const EagerDataFrame& other, const std::vector<std::string>& on, const std::string& how) const;
    GroupedDataFrame group_by(const std::vector<std::string>& keys) const {
        return GroupedDataFrame(table_, keys);
    }

    int64_t num_rows() const { return table_ ? table_->num_rows() : 0; }
    int64_t num_columns() const { return table_ ? table_->num_columns() : 0; }

    std::shared_ptr<arrow::Table> get_table() const { return table_; }

    void print() const;
};

}
