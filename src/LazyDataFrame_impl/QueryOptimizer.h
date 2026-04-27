#pragma once

#include "GraphNodes.h"
#include "../EagerDataFrame.h"
#include "../ExpressionSystem.h"
#include <memory>
#include <string>
#include <unordered_set>

namespace dataframelib {

// Returns the set of column names referenced by an expression tree
// (i.e., every distinct ColExpr name reachable from the root).
std::unordered_set<std::string> collect_expr_cols(const ExprNodePtr& n);


// Rule-based logical-plan optimizer. Each pass is a pure tree-to-tree
// rewrite: take a LogicalNode root, return a new root with the rule applied
// recursively. Passes are composed in optimize() — see the docstring there
// for ordering rationale. Each rule lives in its own .cpp file.
class QueryOptimizer {
public:
    // Apply the full pipeline.
    static std::shared_ptr<LogicalNode> optimize(std::shared_ptr<LogicalNode> plan);

    // Individual passes (exposed for testing / composing).
    static std::shared_ptr<LogicalNode> fold_constants(std::shared_ptr<LogicalNode> node);
    static std::shared_ptr<LogicalNode> pushdown_predicates(std::shared_ptr<LogicalNode> node);
    static std::shared_ptr<LogicalNode> pushdown_projection(std::shared_ptr<LogicalNode> node);
    static std::shared_ptr<LogicalNode> fuse_top_n(std::shared_ptr<LogicalNode> node);
};

// Walks an (optimized) logical plan and produces a materialized
// EagerDataFrame. This is what LazyDataFrame::collect() calls after the
// optimizer runs. Each LogicalNode subclass has a corresponding case here.
class PhysicalPlanCompiler {
public:
    static EagerDataFrame execute(const std::shared_ptr<LogicalNode>& node);
};

}
