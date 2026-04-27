// Column-reference analysis used by projection pushdown.
//
// Given an Expr tree, collect_expr_cols() returns the set of column names it
// reads from its input table. Used to compute "needed columns" for the
// projection-pushdown pass: the input of Filter(predicate) needs every
// column in its parent's needed-set plus every column referenced by the
// predicate.

#include "QueryOptimizer.h"
#include "../ExpressionSystem.h"

namespace dataframelib {

namespace {

void collect_into(const ExprNodePtr& n, std::unordered_set<std::string>& out) {
    if (!n) return;
    if (auto c = std::dynamic_pointer_cast<ColExpr>(n)) {
        out.insert(c->name());
        return;
    }
    if (std::dynamic_pointer_cast<LitExpr>(n)) return;
    if (auto bin = std::dynamic_pointer_cast<BinaryExpr>(n)) {
        collect_into(bin->left(), out);
        collect_into(bin->right(), out);
        return;
    }
    if (auto un = std::dynamic_pointer_cast<UnaryExpr>(n)) {
        collect_into(un->child(), out);
        return;
    }
    if (auto sp = std::dynamic_pointer_cast<StringPredicateExpr>(n)) {
        collect_into(sp->child(), out);
        return;
    }
    if (auto ag = std::dynamic_pointer_cast<AggExpr>(n)) {
        collect_into(ag->child(), out);
        return;
    }
    if (auto al = std::dynamic_pointer_cast<AliasExpr>(n)) {
        collect_into(al->child(), out);
        return;
    }
}

}

std::unordered_set<std::string> collect_expr_cols(const ExprNodePtr& n) {
    std::unordered_set<std::string> out;
    collect_into(n, out);
    return out;
}

}
