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

class LazyGroupedDataFrame {
    std::shared_ptr<LogicalNode> input_;
    std::vector<std::string> keys_;
public:
    LazyGroupedDataFrame(std::shared_ptr<LogicalNode> input, std::vector<std::string> keys)
        : input_(std::move(input)), keys_(std::move(keys)) {}

    LazyDataFrame aggregate(const std::vector<std::pair<std::string, std::string>>& aggs) const;
};

}
