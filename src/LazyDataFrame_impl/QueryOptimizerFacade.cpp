// Optimizer pipeline driver + physical-plan compiler.
//
// optimize() composes the rule passes in a fixed order (see comment inside).
// PhysicalPlanCompiler::execute() walks the optimized plan post-order and
// dispatches each LogicalNode to the matching EagerDataFrame method, so
// the heavy lifting is reused — lazy execution = "build a tree, optimize,
// then run the eager engine".

#include "QueryOptimizer.h"
#include "../EagerDataFrame.h"
#include <stdexcept>

namespace dataframelib {

std::shared_ptr<LogicalNode> QueryOptimizer::optimize(std::shared_ptr<LogicalNode> plan) {
    // Pass order:
    //   1. Constant folding — simplifies expressions before later passes look
    //      at them (e.g. so a folded predicate may now be a trivial bool).
    //   2. Predicate pushdown — moves filters below projections so they run
    //      on the smaller intermediate.
    //   3. Projection pushdown — restricts the scan to only columns the plan
    //      actually consumes; biggest IO win on wide tables.
    //   4. TopN fusion — Head(Sort(...)) → TopN, partial-sort fast path.
    plan = fold_constants(plan);
    plan = pushdown_predicates(plan);
    plan = pushdown_projection(plan);
    plan = fuse_top_n(plan);
    return plan;
}

EagerDataFrame PhysicalPlanCompiler::execute(const std::shared_ptr<LogicalNode>& node) {
    if (!node) throw std::runtime_error("Cannot execute null plan");

    if (auto scan = std::dynamic_pointer_cast<ScanNode>(node)) {
        const auto& cols = scan->restrict_columns();
        if (scan->format() == "csv") {
            return cols.empty() ? EagerDataFrame::read_csv(scan->path())
                                : EagerDataFrame::read_csv(scan->path(), cols);
        }
        if (scan->format() == "parquet") {
            return cols.empty() ? EagerDataFrame::read_parquet(scan->path())
                                : EagerDataFrame::read_parquet(scan->path(), cols);
        }
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
    if (auto topn = std::dynamic_pointer_cast<TopNNode>(node)) {
        return execute(topn->input()).sort_top_n(topn->columns(), topn->asc(), topn->n());
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
