// LazyDataFrame builder methods — every operation just wraps the current
// logical plan in a new LogicalNode. No data is read or processed here.
// The real work happens in collect()/sink_*(), which call the optimizer
// and then PhysicalPlanCompiler::execute.

#include "../LazyDataFrame.h"
#include "../EagerDataFrame.h"
#include "QueryOptimizer.h"

namespace dataframelib {

LazyDataFrame LazyDataFrame::scan_csv(const std::string& path) {
    return LazyDataFrame(std::make_shared<ScanNode>(path, "csv"));
}

LazyDataFrame LazyDataFrame::scan_parquet(const std::string& path) {
    return LazyDataFrame(std::make_shared<ScanNode>(path, "parquet"));
}

LazyDataFrame LazyDataFrame::filter(const Expr& predicate) const {
    return LazyDataFrame(std::make_shared<FilterNode>(logical_plan_, predicate));
}

LazyDataFrame LazyDataFrame::select(const std::vector<std::string>& columns) const {
    return LazyDataFrame(std::make_shared<ProjectNode>(logical_plan_, columns));
}

LazyDataFrame LazyDataFrame::with_column(const std::string& name, const Expr& expr) const {
    return LazyDataFrame(std::make_shared<WithColumnNode>(logical_plan_, name, expr));
}

LazyDataFrame LazyDataFrame::sort(const std::vector<std::string>& columns, bool asc) const {
    return LazyDataFrame(std::make_shared<SortNode>(logical_plan_, columns, asc));
}

LazyDataFrame LazyDataFrame::head(int64_t n) const {
    return LazyDataFrame(std::make_shared<HeadNode>(logical_plan_, n));
}

LazyDataFrame LazyDataFrame::join(const LazyDataFrame& other,
                                  const std::vector<std::string>& on,
                                  const std::string& how) const {
    return LazyDataFrame(std::make_shared<JoinNode>(logical_plan_, other.logical_plan_, on, how));
}

LazyGroupedDataFrame LazyDataFrame::group_by(const std::vector<std::string>& keys) const {
    return LazyGroupedDataFrame(logical_plan_, keys);
}

EagerDataFrame LazyDataFrame::collect() const {
    auto optimized_plan = QueryOptimizer::optimize(logical_plan_);
    return PhysicalPlanCompiler::execute(optimized_plan);
}

void LazyDataFrame::sink_csv(const std::string& path) const {
    collect().write_csv(path);
}
void LazyDataFrame::sink_parquet(const std::string& path) const {
    collect().write_parquet(path);
}

LazyDataFrame LazyGroupedDataFrame::aggregate(
    const std::vector<std::pair<std::string, std::string>>& aggs) const {
    return LazyDataFrame(std::make_shared<GroupAggNode>(input_, keys_, aggs));
}

}
