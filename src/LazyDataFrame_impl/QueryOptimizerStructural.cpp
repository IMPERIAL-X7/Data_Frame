#include "QueryOptimizer.h"

namespace dataframelib {

// Fuse Head(n)(Sort(cols, asc)) into a single TopN node. A real partial-sort
// over n entries is O(N log n) versus O(N log N) for the full sort, and TopN
// can also bypass the costly "take" pass on every column for the rows we
// drop. This is the dominant optimisation for "show me the top 100" queries.
std::shared_ptr<LogicalNode> QueryOptimizer::fuse_top_n(std::shared_ptr<LogicalNode> node) {
    if (!node) return nullptr;

    if (auto head = std::dynamic_pointer_cast<HeadNode>(node)) {
        auto child = fuse_top_n(head->input());
        if (auto sort = std::dynamic_pointer_cast<SortNode>(child)) {
            return std::make_shared<TopNNode>(sort->input(), sort->columns(), sort->asc(), head->n());
        }
        return std::make_shared<HeadNode>(child, head->n());
    }
    if (auto sort = std::dynamic_pointer_cast<SortNode>(node)) {
        return std::make_shared<SortNode>(fuse_top_n(sort->input()), sort->columns(), sort->asc());
    }
    if (auto filter = std::dynamic_pointer_cast<FilterNode>(node)) {
        return std::make_shared<FilterNode>(fuse_top_n(filter->input()), filter->predicate());
    }
    if (auto project = std::dynamic_pointer_cast<ProjectNode>(node)) {
        return std::make_shared<ProjectNode>(fuse_top_n(project->input()), project->columns());
    }
    if (auto wc = std::dynamic_pointer_cast<WithColumnNode>(node)) {
        return std::make_shared<WithColumnNode>(fuse_top_n(wc->input()), wc->name(), wc->expr());
    }
    if (auto join = std::dynamic_pointer_cast<JoinNode>(node)) {
        return std::make_shared<JoinNode>(fuse_top_n(join->left()), fuse_top_n(join->right()),
                                          join->on(), join->how());
    }
    if (auto ga = std::dynamic_pointer_cast<GroupAggNode>(node)) {
        return std::make_shared<GroupAggNode>(fuse_top_n(ga->input()), ga->keys(), ga->aggs());
    }
    return node;
}

}
