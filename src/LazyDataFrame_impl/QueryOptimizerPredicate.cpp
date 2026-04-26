#include "QueryOptimizer.h"

namespace dataframelib {

// Recursively rewrite the plan, swapping `Filter(Project(...))` →
// `Project(Filter(...))` so filters run before projection narrows the schema.
// All other node types are passed through with their inputs recursed.
std::shared_ptr<LogicalNode> QueryOptimizer::pushdown_predicates(std::shared_ptr<LogicalNode> node) {
    if (!node) return nullptr;

    if (auto filter = std::dynamic_pointer_cast<FilterNode>(node)) {
        auto child = filter->input();
        if (auto project = std::dynamic_pointer_cast<ProjectNode>(child)) {
            auto inner = pushdown_predicates(project->input());
            auto new_filter = std::make_shared<FilterNode>(inner, filter->predicate());
            return std::make_shared<ProjectNode>(new_filter, project->columns());
        }
        return std::make_shared<FilterNode>(pushdown_predicates(child), filter->predicate());
    }
    if (auto project = std::dynamic_pointer_cast<ProjectNode>(node)) {
        return std::make_shared<ProjectNode>(pushdown_predicates(project->input()), project->columns());
    }
    if (auto wc = std::dynamic_pointer_cast<WithColumnNode>(node)) {
        return std::make_shared<WithColumnNode>(pushdown_predicates(wc->input()), wc->name(), wc->expr());
    }
    if (auto sort = std::dynamic_pointer_cast<SortNode>(node)) {
        return std::make_shared<SortNode>(pushdown_predicates(sort->input()), sort->columns(), sort->asc());
    }
    if (auto head = std::dynamic_pointer_cast<HeadNode>(node)) {
        return std::make_shared<HeadNode>(pushdown_predicates(head->input()), head->n());
    }
    if (auto join = std::dynamic_pointer_cast<JoinNode>(node)) {
        return std::make_shared<JoinNode>(pushdown_predicates(join->left()),
                                          pushdown_predicates(join->right()),
                                          join->on(), join->how());
    }
    if (auto ga = std::dynamic_pointer_cast<GroupAggNode>(node)) {
        return std::make_shared<GroupAggNode>(pushdown_predicates(ga->input()), ga->keys(), ga->aggs());
    }
    return node;
}

}
