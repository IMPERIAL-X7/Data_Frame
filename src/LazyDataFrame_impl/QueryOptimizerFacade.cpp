#include "QueryOptimizer.h"
#include "../EagerDataFrame.h"
#include <stdexcept>

namespace dataframelib {

std::shared_ptr<LogicalNode> QueryOptimizer::optimize(std::shared_ptr<LogicalNode> plan) {
    return pushdown_predicates(plan);
}

EagerDataFrame PhysicalPlanCompiler::execute(const std::shared_ptr<LogicalNode>& node) {
    if (!node) throw std::runtime_error("Cannot execute null plan");

    if (auto scan = std::dynamic_pointer_cast<ScanNode>(node)) {
        if (scan->format() == "csv") return EagerDataFrame::read_csv(scan->path());
        if (scan->format() == "parquet") return EagerDataFrame::read_parquet(scan->path());
        throw std::runtime_error("Unsupported scan format");
    }
    if (auto filter = std::dynamic_pointer_cast<FilterNode>(node)) {
        return execute(filter->input()).filter(filter->predicate());
    }
    if (auto project = std::dynamic_pointer_cast<ProjectNode>(node)) {
        return execute(project->input()).select(project->columns());
    }
    if (auto wc = std::dynamic_pointer_cast<WithColumnNode>(node)) {
        return execute(wc->input()).with_column(wc->name(), wc->expr());
    }
    if (auto sort = std::dynamic_pointer_cast<SortNode>(node)) {
        return execute(sort->input()).sort(sort->columns(), sort->asc());
    }
    if (auto head = std::dynamic_pointer_cast<HeadNode>(node)) {
        return execute(head->input()).head(head->n());
    }
    if (auto join = std::dynamic_pointer_cast<JoinNode>(node)) {
        return execute(join->left()).join(execute(join->right()), join->on(), join->how());
    }
    if (auto ga = std::dynamic_pointer_cast<GroupAggNode>(node)) {
        return execute(ga->input()).group_by(ga->keys()).aggregate(ga->aggs());
    }

    throw std::runtime_error("Unknown LogicalNode encountered during execution");
}

}
