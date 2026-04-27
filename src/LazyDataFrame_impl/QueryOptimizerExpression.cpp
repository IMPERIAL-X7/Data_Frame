// Constant folding + algebraic simplification on Expr trees.
//
// Two transformations:
//   (1) Constant folding — BinaryExpr(LitExpr, LitExpr) is evaluated at plan
//       construction time and replaced with a single LitExpr. Same for
//       UnaryExpr(LitExpr).
//   (2) Algebraic simplification — strip identity ops:
//         x + 0 → x       0 + x → x
//         x - 0 → x
//         x * 1 → x       1 * x → x
//         x / 1 → x
//
// Correctness: arithmetic on literal numbers preserves the spec's int+float
// promotion rule. We avoid simplifying expressions whose operand might be a
// null column (e.g. col("x") * 0 ≠ lit(0) when col has nulls), so the x*0
// rule is omitted.

#include "QueryOptimizer.h"
#include "../ExpressionSystem.h"
#include <cmath>

namespace dataframelib {

namespace {

bool is_lit(const ExprNodePtr& n, LitValue& out) {
    auto l = std::dynamic_pointer_cast<LitExpr>(n);
    if (!l) return false;
    out = l->value();
    return true;
}

bool lit_as_double(const LitValue& v, double& out) {
    return std::visit([&](auto&& x) -> bool {
        using T = std::decay_t<decltype(x)>;
        if constexpr (std::is_same_v<T, int32_t> || std::is_same_v<T, int64_t> ||
                      std::is_same_v<T, float>   || std::is_same_v<T, double>) {
            out = static_cast<double>(x);
            return true;
        } else {
            return false;
        }
    }, v);
}

ExprNodePtr fold_numeric(const LitValue& a, const LitValue& b, const std::string& op_name) {
    double av = 0, bv = 0;
    if (!lit_as_double(a, av) || !lit_as_double(b, bv)) return nullptr;

    int ai = static_cast<int>(a.index());
    int bi = static_cast<int>(b.index());
    int ri = std::max(ai, bi);

    auto make_lit = [&](double r) -> ExprNodePtr {
        switch (ri) {
            case 0: return std::make_shared<LitExpr>(LitValue(static_cast<int32_t>(r)));
            case 1: return std::make_shared<LitExpr>(LitValue(static_cast<int64_t>(r)));
            case 2: return std::make_shared<LitExpr>(LitValue(static_cast<float>(r)));
            case 3: return std::make_shared<LitExpr>(LitValue(static_cast<double>(r)));
        }
        return nullptr;
    };
    auto make_bool = [&](bool r) -> ExprNodePtr {
        return std::make_shared<LitExpr>(LitValue(r));
    };

    if (op_name == "add") return make_lit(av + bv);
    if (op_name == "sub") return make_lit(av - bv);
    if (op_name == "mul") return make_lit(av * bv);
    if (op_name == "div") {
        if (bv == 0) return nullptr;  // leave runtime to handle div-by-zero
        return make_lit(av / bv);
    }
    if (op_name == "mod") {
        if (ai > 1 || bi > 1) return nullptr;  // mod requires ints
        if (static_cast<int64_t>(bv) == 0) return nullptr;
        return make_lit(static_cast<double>(static_cast<int64_t>(av) % static_cast<int64_t>(bv)));
    }
    if (op_name == "eq") return make_bool(av == bv);
    if (op_name == "ne") return make_bool(av != bv);
    if (op_name == "lt") return make_bool(av < bv);
    if (op_name == "le") return make_bool(av <= bv);
    if (op_name == "gt") return make_bool(av > bv);
    if (op_name == "ge") return make_bool(av >= bv);
    return nullptr;
}

bool is_numeric_value(const LitValue& v, double target) {
    double x = 0;
    if (!lit_as_double(v, x)) return false;
    return x == target;
}

ExprNodePtr fold_expr(const ExprNodePtr& node);

ExprNodePtr fold_binary(const std::shared_ptr<BinaryExpr>& bin) {
    auto l = fold_expr(bin->left());
    auto r = fold_expr(bin->right());
    const std::string& op = bin->op_name();

    LitValue lv, rv;
    bool l_lit = is_lit(l, lv);
    bool r_lit = is_lit(r, rv);

    if (l_lit && r_lit) {
        if (auto folded = fold_numeric(lv, rv, op)) return folded;
    }

    if (op == "add" || op == "sub") {
        if (r_lit && is_numeric_value(rv, 0.0)) return l;
        if (op == "add" && l_lit && is_numeric_value(lv, 0.0)) return r;
    }
    if (op == "mul") {
        if (r_lit && is_numeric_value(rv, 1.0)) return l;
        if (l_lit && is_numeric_value(lv, 1.0)) return r;
    }
    if (op == "div") {
        if (r_lit && is_numeric_value(rv, 1.0)) return l;
    }

    if (l == bin->left() && r == bin->right()) return bin;
    return std::make_shared<BinaryExpr>(l, r, op, op);
}

ExprNodePtr fold_unary(const std::shared_ptr<UnaryExpr>& un) {
    auto c = fold_expr(un->child());
    LitValue lv;
    if (is_lit(c, lv)) {
        const std::string& op = un->op_name();
        double x = 0;
        if (op == "neg" && lit_as_double(lv, x)) {
            int idx = static_cast<int>(lv.index());
            switch (idx) {
                case 0: return std::make_shared<LitExpr>(LitValue(static_cast<int32_t>(-x)));
                case 1: return std::make_shared<LitExpr>(LitValue(static_cast<int64_t>(-x)));
                case 2: return std::make_shared<LitExpr>(LitValue(static_cast<float>(-x)));
                case 3: return std::make_shared<LitExpr>(LitValue(static_cast<double>(-x)));
            }
        }
        if (op == "abs" && lit_as_double(lv, x)) {
            int idx = static_cast<int>(lv.index());
            double r = std::fabs(x);
            switch (idx) {
                case 0: return std::make_shared<LitExpr>(LitValue(static_cast<int32_t>(r)));
                case 1: return std::make_shared<LitExpr>(LitValue(static_cast<int64_t>(r)));
                case 2: return std::make_shared<LitExpr>(LitValue(static_cast<float>(r)));
                case 3: return std::make_shared<LitExpr>(LitValue(static_cast<double>(r)));
            }
        }
        if (op == "not") {
            if (auto* b = std::get_if<bool>(&lv)) {
                return std::make_shared<LitExpr>(LitValue(!*b));
            }
        }
    }
    if (c == un->child()) return un;
    return std::make_shared<UnaryExpr>(c, un->op_name());
}

ExprNodePtr fold_expr(const ExprNodePtr& node) {
    if (!node) return node;
    if (auto bin = std::dynamic_pointer_cast<BinaryExpr>(node)) return fold_binary(bin);
    if (auto un = std::dynamic_pointer_cast<UnaryExpr>(node)) return fold_unary(un);
    if (auto al = std::dynamic_pointer_cast<AliasExpr>(node)) {
        auto c = fold_expr(al->child());
        if (c == al->child()) return al;
        return std::make_shared<AliasExpr>(c, al->name());
    }
    if (auto ag = std::dynamic_pointer_cast<AggExpr>(node)) {
        auto c = fold_expr(ag->child());
        if (c == ag->child()) return ag;
        return std::make_shared<AggExpr>(c, ag->func());
    }
    return node;
}

}

std::shared_ptr<LogicalNode> QueryOptimizer::fold_constants(std::shared_ptr<LogicalNode> node) {
    if (!node) return nullptr;

    if (auto filter = std::dynamic_pointer_cast<FilterNode>(node)) {
        return std::make_shared<FilterNode>(fold_constants(filter->input()),
                                            Expr(fold_expr(filter->predicate().node())));
    }
    if (auto wc = std::dynamic_pointer_cast<WithColumnNode>(node)) {
        return std::make_shared<WithColumnNode>(fold_constants(wc->input()), wc->name(),
                                                Expr(fold_expr(wc->expr().node())));
    }
    if (auto project = std::dynamic_pointer_cast<ProjectNode>(node)) {
        return std::make_shared<ProjectNode>(fold_constants(project->input()), project->columns());
    }
    if (auto sort = std::dynamic_pointer_cast<SortNode>(node)) {
        return std::make_shared<SortNode>(fold_constants(sort->input()), sort->columns(), sort->asc());
    }
    if (auto head = std::dynamic_pointer_cast<HeadNode>(node)) {
        return std::make_shared<HeadNode>(fold_constants(head->input()), head->n());
    }
    if (auto topn = std::dynamic_pointer_cast<TopNNode>(node)) {
        return std::make_shared<TopNNode>(fold_constants(topn->input()), topn->columns(), topn->asc(), topn->n());
    }
    if (auto join = std::dynamic_pointer_cast<JoinNode>(node)) {
        return std::make_shared<JoinNode>(fold_constants(join->left()),
                                          fold_constants(join->right()),
                                          join->on(), join->how());
    }
    if (auto ga = std::dynamic_pointer_cast<GroupAggNode>(node)) {
        return std::make_shared<GroupAggNode>(fold_constants(ga->input()), ga->keys(), ga->aggs());
    }
    return node;
}

}
