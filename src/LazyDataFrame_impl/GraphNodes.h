#pragma once

// Logical-plan node hierarchy for the lazy execution engine.
//
// Each LazyDataFrame operation (filter, select, sort, ...) builds one of
// these nodes wrapping its input — the result is a small DAG describing the
// computation. No data is read or processed during construction; the tree
// is rewritten by QueryOptimizer and finally walked by PhysicalPlanCompiler
// when collect() is called.
//
// Node responsibilities:
//   - to_string()   : human-readable label, used by explain() / Graphviz.
//   - children()    : enumerate inputs so generic walks (rendering, schema
//                     analysis) don't need per-node knowledge.

#include "../ExpressionSystem.h"
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace dataframelib {

class LogicalNode {
public:
    virtual ~LogicalNode() = default;
    virtual std::string to_string() const = 0;
    virtual std::vector<std::shared_ptr<LogicalNode>> children() const = 0;
};

// Leaf: read a CSV/Parquet file. The optimizer's projection-pushdown pass
// may set restrict_columns_ so only the listed columns are materialised.

class ScanNode : public LogicalNode {
    std::string path_;
    std::string format_;
    // If non-empty, the projection-pushdown pass has restricted this scan to
    // only read these columns. Empty means "read everything" (default).
    std::vector<std::string> restrict_columns_;
public:
    ScanNode(std::string path, std::string format)
        : path_(std::move(path)), format_(std::move(format)) {}

    std::string to_string() const override {
        std::string s = "Scan(" + format_ + ": " + path_;
        if (!restrict_columns_.empty()) {
            s += ", cols=[";
            for (size_t i = 0; i < restrict_columns_.size(); ++i) {
                if (i) s += ",";
                s += restrict_columns_[i];
            }
            s += "]";
        }
        return s + ")";
    }
    std::vector<std::shared_ptr<LogicalNode>> children() const override { return {}; }

    const std::string& path() const { return path_; }
    const std::string& format() const { return format_; }
    const std::vector<std::string>& restrict_columns() const { return restrict_columns_; }
    void set_restrict_columns(std::vector<std::string> cols) { restrict_columns_ = std::move(cols); }
};

// Row filter — keeps rows where `predicate` evaluates to true.
class FilterNode : public LogicalNode {
    std::shared_ptr<LogicalNode> input_;
    Expr predicate_;
public:
    FilterNode(std::shared_ptr<LogicalNode> input, Expr predicate)
        : input_(std::move(input)), predicate_(std::move(predicate)) {}

    std::string to_string() const override { return "Filter(" + predicate_.to_string() + ")"; }
    std::vector<std::shared_ptr<LogicalNode>> children() const override { return {input_}; }

    const Expr& predicate() const { return predicate_; }
    std::shared_ptr<LogicalNode> input() const { return input_; }
};

// Column projection — emit only the named columns. Note: the order of
// `columns_` is the output column order. This is what `select()` produces.
class ProjectNode : public LogicalNode {
    std::shared_ptr<LogicalNode> input_;
    std::vector<std::string> columns_;
public:
    ProjectNode(std::shared_ptr<LogicalNode> input, std::vector<std::string> columns)
        : input_(std::move(input)), columns_(std::move(columns)) {}

    std::string to_string() const override;
    std::vector<std::shared_ptr<LogicalNode>> children() const override { return {input_}; }

    const std::vector<std::string>& columns() const { return columns_; }
    std::shared_ptr<LogicalNode> input() const { return input_; }
};

// Add or replace a column. Output schema = input schema with `name` either
// inserted (new column) or overwritten (existing column).
class WithColumnNode : public LogicalNode {
    std::shared_ptr<LogicalNode> input_;
    std::string name_;
    Expr expr_;
public:
    WithColumnNode(std::shared_ptr<LogicalNode> input, std::string name, Expr expr)
        : input_(std::move(input)), name_(std::move(name)), expr_(std::move(expr)) {}

    std::string to_string() const override { return "WithColumn(" + name_ + " = " + expr_.to_string() + ")"; }
    std::vector<std::shared_ptr<LogicalNode>> children() const override { return {input_}; }

    const std::string& name() const { return name_; }
    const Expr& expr() const { return expr_; }
    std::shared_ptr<LogicalNode> input() const { return input_; }
};

// Sort all rows by the given keys (lexicographic when multiple). Single
// asc/desc applies to all keys, matching the assignment's signature.
class SortNode : public LogicalNode {
    std::shared_ptr<LogicalNode> input_;
    std::vector<std::string> columns_;
    bool asc_;
public:
    SortNode(std::shared_ptr<LogicalNode> input, std::vector<std::string> columns, bool asc)
        : input_(std::move(input)), columns_(std::move(columns)), asc_(asc) {}

    std::string to_string() const override { return std::string("Sort(") + (asc_ ? "asc" : "desc") + ")"; }
    std::vector<std::shared_ptr<LogicalNode>> children() const override { return {input_}; }

    const std::vector<std::string>& columns() const { return columns_; }
    bool asc() const { return asc_; }
    std::shared_ptr<LogicalNode> input() const { return input_; }
};

// Fused Sort+Head: pick the top n rows by sort key without fully sorting the
// rest. The optimizer rewrites Head(Sort(...)) into this. Massive win for
// "show me the top 100 rows" queries on large tables.
class TopNNode : public LogicalNode {
    std::shared_ptr<LogicalNode> input_;
    std::vector<std::string> columns_;
    bool asc_;
    int64_t n_;
public:
    TopNNode(std::shared_ptr<LogicalNode> input, std::vector<std::string> columns, bool asc, int64_t n)
        : input_(std::move(input)), columns_(std::move(columns)), asc_(asc), n_(n) {}

    std::string to_string() const override {
        return "TopN(n=" + std::to_string(n_) + (asc_ ? ", asc)" : ", desc)");
    }
    std::vector<std::shared_ptr<LogicalNode>> children() const override { return {input_}; }

    const std::vector<std::string>& columns() const { return columns_; }
    bool asc() const { return asc_; }
    int64_t n() const { return n_; }
    std::shared_ptr<LogicalNode> input() const { return input_; }
};

// Take the first n rows (preserves input order). When the immediate child
// is a SortNode, the optimizer fuses the pair into TopNNode for speed.
class HeadNode : public LogicalNode {
    std::shared_ptr<LogicalNode> input_;
    int64_t n_;
public:
    HeadNode(std::shared_ptr<LogicalNode> input, int64_t n)
        : input_(std::move(input)), n_(n) {}

    std::string to_string() const override { return "Head(" + std::to_string(n_) + ")"; }
    std::vector<std::shared_ptr<LogicalNode>> children() const override { return {input_}; }

    int64_t n() const { return n_; }
    std::shared_ptr<LogicalNode> input() const { return input_; }
};

// Equi-join on the columns named in `on_`. `how_` is "inner" / "left" /
// "outer". This is the only binary node — it has two children.
class JoinNode : public LogicalNode {
    std::shared_ptr<LogicalNode> left_;
    std::shared_ptr<LogicalNode> right_;
    std::vector<std::string> on_;
    std::string how_;
public:
    JoinNode(std::shared_ptr<LogicalNode> left, std::shared_ptr<LogicalNode> right,
             std::vector<std::string> on, std::string how)
        : left_(std::move(left)), right_(std::move(right)),
          on_(std::move(on)), how_(std::move(how)) {}

    std::string to_string() const override { return "Join(" + how_ + ")"; }
    std::vector<std::shared_ptr<LogicalNode>> children() const override { return {left_, right_}; }

    std::shared_ptr<LogicalNode> left() const { return left_; }
    std::shared_ptr<LogicalNode> right() const { return right_; }
    const std::vector<std::string>& on() const { return on_; }
    const std::string& how() const { return how_; }
};

// Group-by + aggregate as a single node — they always come together in this
// engine. `aggs_` is a list of (input_col, function_name) pairs; each
// produces an output column named "<col>_<func>" (e.g. salary_mean).
class GroupAggNode : public LogicalNode {
    std::shared_ptr<LogicalNode> input_;
    std::vector<std::string> keys_;
    std::vector<std::pair<std::string, std::string>> aggs_;
public:
    GroupAggNode(std::shared_ptr<LogicalNode> input,
                 std::vector<std::string> keys,
                 std::vector<std::pair<std::string, std::string>> aggs)
        : input_(std::move(input)), keys_(std::move(keys)), aggs_(std::move(aggs)) {}

    std::string to_string() const override { return "GroupAgg"; }
    std::vector<std::shared_ptr<LogicalNode>> children() const override { return {input_}; }

    const std::vector<std::string>& keys() const { return keys_; }
    const std::vector<std::pair<std::string, std::string>>& aggs() const { return aggs_; }
    std::shared_ptr<LogicalNode> input() const { return input_; }
};

}
