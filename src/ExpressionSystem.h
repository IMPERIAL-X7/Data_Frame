#pragma once

// Expression system used by filter / with_column / aggregate.
//
// Two layers:
//
//   ExprNode (abstract)         — the AST. Subclasses model column refs,
//   ├── ColExpr                   literals, binary/unary ops, string
//   ├── LitExpr                   predicates, aggregations, aliases. Each
//   ├── BinaryExpr                provides evaluate(table) which returns an
//   ├── UnaryExpr                 arrow::Datum, plus to_string() for plan
//   ├── StringPredicateExpr       rendering.
//   ├── AggExpr
//   └── AliasExpr
//
//   Expr (value class)          — what user code sees. Wraps an
//                                 ExprNodePtr and hosts the .method() and
//                                 operator overloads required by the spec
//                                 (col("x").is_null(), col("a") > 30, ...).
//
// Why the split? The spec requires method-call syntax — col("x").abs(),
// col("name").to_upper() — which can't work if col() returns a
// shared_ptr<ExprNode> (operator. is not overloadable on shared_ptr). So
// col() returns a value-type wrapper, and that wrapper carries every
// builder method.

#include <arrow/api.h>
#include <memory>
#include <string>
#include <variant>

namespace dataframelib {

class ExprNode {
public:
    virtual ~ExprNode() = default;
    virtual arrow::Datum evaluate(const std::shared_ptr<arrow::Table>& table) const = 0;
    virtual std::string to_string() const = 0;
};
using ExprNodePtr = std::shared_ptr<ExprNode>;

using LitValue = std::variant<int32_t, int64_t, float, double, std::string, bool>;

class ColExpr : public ExprNode {
    std::string name_;
public:
    explicit ColExpr(std::string name);
    arrow::Datum evaluate(const std::shared_ptr<arrow::Table>& table) const override;
    std::string to_string() const override;
    const std::string& name() const { return name_; }
};

class LitExpr : public ExprNode {
    LitValue value_;
public:
    explicit LitExpr(LitValue value);
    arrow::Datum evaluate(const std::shared_ptr<arrow::Table>& table) const override;
    std::string to_string() const override;
    const LitValue& value() const { return value_; }
};

class BinaryExpr : public ExprNode {
    ExprNodePtr left_;
    ExprNodePtr right_;
    std::string op_name_;   // add/sub/mul/div/mod/eq/ne/lt/le/gt/ge/and/or
    std::string symbol_;
public:
    BinaryExpr(ExprNodePtr left, ExprNodePtr right, std::string op_name, std::string symbol);
    arrow::Datum evaluate(const std::shared_ptr<arrow::Table>& table) const override;
    std::string to_string() const override;
    const std::string& op_name() const { return op_name_; }
    ExprNodePtr left() const { return left_; }
    ExprNodePtr right() const { return right_; }
};

class UnaryExpr : public ExprNode {
    ExprNodePtr child_;
    std::string op_name_;   // not/neg/abs/is_null/is_not_null/length/to_lower/to_upper
public:
    UnaryExpr(ExprNodePtr child, std::string op_name);
    arrow::Datum evaluate(const std::shared_ptr<arrow::Table>& table) const override;
    std::string to_string() const override;
    const std::string& op_name() const { return op_name_; }
    ExprNodePtr child() const { return child_; }
};

class StringPredicateExpr : public ExprNode {
    ExprNodePtr child_;
    std::string op_name_;   // contains/starts_with/ends_with
    std::string operand_;
public:
    StringPredicateExpr(ExprNodePtr child, std::string op_name, std::string operand);
    arrow::Datum evaluate(const std::shared_ptr<arrow::Table>& table) const override;
    std::string to_string() const override;
    const std::string& op_name() const { return op_name_; }
    ExprNodePtr child() const { return child_; }
    const std::string& operand() const { return operand_; }
};

class AggExpr : public ExprNode {
    ExprNodePtr child_;
    std::string func_;      // sum/mean/count/min/max
public:
    AggExpr(ExprNodePtr child, std::string func);
    arrow::Datum evaluate(const std::shared_ptr<arrow::Table>& table) const override;
    std::string to_string() const override;
    const std::string& func() const { return func_; }
    ExprNodePtr child() const { return child_; }
};

class AliasExpr : public ExprNode {
    ExprNodePtr child_;
    std::string name_;
public:
    AliasExpr(ExprNodePtr child, std::string name);
    arrow::Datum evaluate(const std::shared_ptr<arrow::Table>& table) const override;
    std::string to_string() const override;
    const std::string& name() const { return name_; }
    ExprNodePtr child() const { return child_; }
};

// ---------------------------------------------------------
// Public value class: Expr
// ---------------------------------------------------------

class Expr {
    ExprNodePtr node_;
public:
    Expr() = default;
    Expr(ExprNodePtr n) : node_(std::move(n)) {}

    ExprNodePtr node() const { return node_; }
    bool valid() const { return static_cast<bool>(node_); }

    arrow::Datum evaluate(const std::shared_ptr<arrow::Table>& table) const {
        return node_->evaluate(table);
    }
    std::string to_string() const { return node_ ? node_->to_string() : "<null>"; }

    // ---- Arithmetic helpers (.abs()) ----
    Expr abs() const;

    // ---- Null checks ----
    Expr is_null() const;
    Expr is_not_null() const;

    // ---- String functions ----
    Expr length() const;
    Expr to_lower() const;
    Expr to_upper() const;
    Expr contains(const std::string& s) const;
    Expr starts_with(const std::string& s) const;
    Expr ends_with(const std::string& s) const;

    // ---- Aggregations ----
    Expr sum() const;
    Expr mean() const;
    Expr count() const;
    Expr min() const;
    Expr max() const;

    // ---- Alias ----
    Expr alias(const std::string& name) const;
};

// Factory functions
Expr col(const std::string& name);
Expr lit(LitValue value);

// Convenience lit() overloads — the variant doesn't auto-deduce const char*
// or int (vs int32_t/int64_t), so accept these explicitly.
Expr lit(int v);
Expr lit(int64_t v);
Expr lit(double v);
Expr lit(float v);
Expr lit(const char* v);

// ---- Operators ----
Expr operator+(const Expr& a, const Expr& b);
Expr operator-(const Expr& a, const Expr& b);
Expr operator*(const Expr& a, const Expr& b);
Expr operator/(const Expr& a, const Expr& b);
Expr operator%(const Expr& a, const Expr& b);
Expr operator==(const Expr& a, const Expr& b);
Expr operator!=(const Expr& a, const Expr& b);
Expr operator<(const Expr& a, const Expr& b);
Expr operator<=(const Expr& a, const Expr& b);
Expr operator>(const Expr& a, const Expr& b);
Expr operator>=(const Expr& a, const Expr& b);
Expr operator&(const Expr& a, const Expr& b);
Expr operator|(const Expr& a, const Expr& b);
Expr operator~(const Expr& a);
Expr operator-(const Expr& a);  // unary negate

// Literal-rhs overloads. Only single-side overloads are provided since the
// tests always have the column on the left.
#define DFLIB_LITERAL_OPS(LIT_T)                                                   \
    inline Expr operator+(const Expr& a, LIT_T b) { return a + lit(b); }           \
    inline Expr operator-(const Expr& a, LIT_T b) { return a - lit(b); }           \
    inline Expr operator*(const Expr& a, LIT_T b) { return a * lit(b); }           \
    inline Expr operator/(const Expr& a, LIT_T b) { return a / lit(b); }           \
    inline Expr operator%(const Expr& a, LIT_T b) { return a % lit(b); }           \
    inline Expr operator==(const Expr& a, LIT_T b) { return a == lit(b); }         \
    inline Expr operator!=(const Expr& a, LIT_T b) { return a != lit(b); }         \
    inline Expr operator<(const Expr& a, LIT_T b) { return a < lit(b); }           \
    inline Expr operator<=(const Expr& a, LIT_T b) { return a <= lit(b); }         \
    inline Expr operator>(const Expr& a, LIT_T b) { return a > lit(b); }           \
    inline Expr operator>=(const Expr& a, LIT_T b) { return a >= lit(b); }

DFLIB_LITERAL_OPS(int)
DFLIB_LITERAL_OPS(int64_t)
DFLIB_LITERAL_OPS(double)
DFLIB_LITERAL_OPS(float)
#undef DFLIB_LITERAL_OPS

// String literal comparison only (other ops don't make sense on strings).
inline Expr operator==(const Expr& a, const char* b) { return a == lit(b); }
inline Expr operator!=(const Expr& a, const char* b) { return a != lit(b); }
inline Expr operator==(const Expr& a, const std::string& b) { return a == lit(b); }
inline Expr operator!=(const Expr& a, const std::string& b) { return a != lit(b); }

}
