#pragma once

#include "ExpressionSystem.h"
#include "LazyDataFrame_impl/GraphNodes.h"
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace dataframelib {

class EagerDataFrame;
class LazyGroupedDataFrame;

// Deferred-execution dataframe. Every operation is a constant-time builder
// call that wraps the previous LogicalNode in a new node — no data is read
// or processed until collect()/sink_*() runs the plan through the optimizer
// and physical compiler.
//
// The trade-off versus EagerDataFrame: more bookkeeping per call, but the
// optimizer can rewrite the whole plan (predicate/projection pushdown,
// Sort+Head fusion, ...) before any I/O happens.
class LazyDataFrame {
private:
    std::shared_ptr<LogicalNode> logical_plan_;

public:
    explicit LazyDataFrame(std::shared_ptr<LogicalNode> plan)
        : logical_plan_(std::move(plan)) {}

    static LazyDataFrame scan_csv(const std::string& path);
    static LazyDataFrame scan_parquet(const std::string& path);

    LazyDataFrame filter(const Expr& predicate) const;
    LazyDataFrame select(const std::vector<std::string>& columns) const;
    LazyDataFrame with_column(const std::string& name, const Expr& expr) const;
    LazyDataFrame sort(const std::vector<std::string>& columns, bool asc) const;
    LazyDataFrame head(int64_t n) const;
    LazyDataFrame join(const LazyDataFrame& other,
                       const std::vector<std::string>& on,
                       const std::string& how) const;

    LazyGroupedDataFrame group_by(const std::vector<std::string>& keys) const;

    void explain(const std::string& plan_path) const;

    EagerDataFrame collect() const;

    void sink_csv(const std::string& path) const;
    void sink_parquet(const std::string& path) const;

    std::shared_ptr<LogicalNode> plan() const { return logical_plan_; }
};

// Lazy counterpart of GroupedDataFrame: holds the in-flight plan plus the
// grouping keys until aggregate() finishes the pair into a GroupAggNode.
class LazyGroupedDataFrame {
    std::shared_ptr<LogicalNode> input_;
    std::vector<std::string> keys_;
public:
    LazyGroupedDataFrame(std::shared_ptr<LogicalNode> input, std::vector<std::string> keys)
        : input_(std::move(input)), keys_(std::move(keys)) {}

    LazyDataFrame aggregate(const std::vector<std::pair<std::string, std::string>>& aggs) const;
};

}
