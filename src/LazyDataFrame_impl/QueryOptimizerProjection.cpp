// Projection pushdown.
//
// Walks the plan top-down, tracking the set of columns the parent needs from
// the current subtree's output. When we hit a ScanNode, we know exactly
// which columns the rest of the plan touches, so we tell the scan to only
// read those — saving CSV parse time and memory on wide tables.
//
// The "needed" set is `optional`:
//   * nullopt — caller doesn't know what's needed (we're at the root and
//     the user might call .num_columns() / .write_csv() expecting every
//     column). Don't restrict; just recurse.
//   * a set    — only these columns flow upward through this subtree.
//
// A Project node *introduces* a known needed-set: from there down we know
// exactly what's required. Filter, WithColumn, Sort, GroupAgg adjust the
// set as they propagate it.
//
// Joins are skipped (left/right column origins matter and the safe transform
// is more involved than a one-pass walk).

#include "QueryOptimizer.h"
#include <optional>
#include <unordered_set>

namespace dataframelib {

namespace {

using ColSet = std::unordered_set<std::string>;

void union_into(ColSet& dst, const ColSet& src) {
    for (const auto& s : src) dst.insert(s);
}
void union_into(ColSet& dst, const std::vector<std::string>& src) {
    for (const auto& s : src) dst.insert(s);
}

std::shared_ptr<LogicalNode> visit(const std::shared_ptr<LogicalNode>& node,
                                   std::optional<ColSet> needed);

std::shared_ptr<LogicalNode> recurse_unrestricted(const std::shared_ptr<LogicalNode>& node) {
    return visit(node, std::nullopt);
}

std::shared_ptr<LogicalNode> visit(const std::shared_ptr<LogicalNode>& node,
                                   std::optional<ColSet> needed) {
    if (!node) return nullptr;

    if (auto scan = std::dynamic_pointer_cast<ScanNode>(node)) {
        if (!needed.has_value()) return scan;
        // Build a fresh ScanNode with the restriction (don't mutate shared
        // input — the same plan might be optimized twice).
        std::vector<std::string> cols(needed->begin(), needed->end());
        // Stable order makes plan output deterministic for explain().
        std::sort(cols.begin(), cols.end());
        auto fresh = std::make_shared<ScanNode>(scan->path(), scan->format());
        fresh->set_restrict_columns(std::move(cols));
        return fresh;
    }

    if (auto project = std::dynamic_pointer_cast<ProjectNode>(node)) {
        // Project pins the column set. Replace whatever was needed above with
        // exactly the projected list — those are all that flow up.
        ColSet new_needed;
        union_into(new_needed, project->columns());
        auto child = visit(project->input(), new_needed);
        return std::make_shared<ProjectNode>(child, project->columns());
    }

    if (auto filter = std::dynamic_pointer_cast<FilterNode>(node)) {
        std::optional<ColSet> child_needed = needed;
        if (child_needed.has_value()) {
            union_into(*child_needed, collect_expr_cols(filter->predicate().node()));
        }
        return std::make_shared<FilterNode>(visit(filter->input(), child_needed),
                                            filter->predicate());
    }

    if (auto wc = std::dynamic_pointer_cast<WithColumnNode>(node)) {
        // The output of WithColumn includes its `name`. Strip that from
        // needed before passing down (the input doesn't have it yet) and
        // add the columns referenced by the expression.
        std::optional<ColSet> child_needed = needed;
        if (child_needed.has_value()) {
            child_needed->erase(wc->name());
            union_into(*child_needed, collect_expr_cols(wc->expr().node()));
        }
        return std::make_shared<WithColumnNode>(visit(wc->input(), child_needed),
                                                wc->name(), wc->expr());
    }

    if (auto sort = std::dynamic_pointer_cast<SortNode>(node)) {
        std::optional<ColSet> child_needed = needed;
        if (child_needed.has_value()) union_into(*child_needed, sort->columns());
        return std::make_shared<SortNode>(visit(sort->input(), child_needed),
                                          sort->columns(), sort->asc());
    }

    if (auto head = std::dynamic_pointer_cast<HeadNode>(node)) {
        return std::make_shared<HeadNode>(visit(head->input(), needed), head->n());
    }

    if (auto topn = std::dynamic_pointer_cast<TopNNode>(node)) {
        std::optional<ColSet> child_needed = needed;
        if (child_needed.has_value()) union_into(*child_needed, topn->columns());
        return std::make_shared<TopNNode>(visit(topn->input(), child_needed),
                                          topn->columns(), topn->asc(), topn->n());
    }

    if (auto ga = std::dynamic_pointer_cast<GroupAggNode>(node)) {
        // GroupAgg's output is exactly keys ∪ {agg_col} — its own contract.
        // Propagate that as the input requirement regardless of `needed`.
        ColSet ga_needed;
        union_into(ga_needed, ga->keys());
        for (const auto& [col, _func] : ga->aggs()) ga_needed.insert(col);
        return std::make_shared<GroupAggNode>(visit(ga->input(), ga_needed),
                                              ga->keys(), ga->aggs());
    }

    if (auto join = std::dynamic_pointer_cast<JoinNode>(node)) {
        // Conservative: don't push past joins (would need per-side column
        // origin tracking). Just recurse without restriction.
        return std::make_shared<JoinNode>(recurse_unrestricted(join->left()),
                                          recurse_unrestricted(join->right()),
                                          join->on(), join->how());
    }

    return node;
}

}

std::shared_ptr<LogicalNode> QueryOptimizer::pushdown_projection(std::shared_ptr<LogicalNode> node) {
    return visit(node, std::nullopt);
}

}
