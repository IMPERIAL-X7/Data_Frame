#pragma once

#include <arrow/api.h>
#include <memory>
#include <string>
#include <variant>

// Forward declaration for the smart pointers
class Expr;
using ExprPtr = std::shared_ptr<Expr>;

// The allowed literal types as per assignment strict typing rules
using LitValue = std::variant<int32_t, int64_t, float, double, std::string, bool>;

// ---------------------------------------------------------
// Abstract Base Class
// ---------------------------------------------------------
class Expr {
public:
    virtual ~Expr() = default;
    
    // Evaluates the expression against a given table, returning an Arrow Datum (array/scalar)
    virtual arrow::Datum evaluate(const std::shared_ptr<arrow::Table>& table) const = 0;
    
    // Returns a string representation (useful later for Lazy DAG explain)
    virtual std::string to_string() const = 0;
};

// ---------------------------------------------------------
// Leaf Nodes
// ---------------------------------------------------------
class ColExpr : public Expr {
    std::string name_;
public:
    explicit ColExpr(std::string name);
    arrow::Datum evaluate(const std::shared_ptr<arrow::Table>& table) const override;
    std::string to_string() const override;
};

class LitExpr : public Expr {
    LitValue value_;
public:
    explicit LitExpr(LitValue value);
    arrow::Datum evaluate(const std::shared_ptr<arrow::Table>& table) const override;
    std::string to_string() const override;
};

// ---------------------------------------------------------
// Operator Nodes
// ---------------------------------------------------------
class BinaryExpr : public Expr {
    ExprPtr left_;
    ExprPtr right_;
    std::string op_name_; // Arrow compute function name, e.g., "add", "greater"
    std::string symbol_;  // String representation, e.g., "+", ">"

public:
    BinaryExpr(ExprPtr left, ExprPtr right, std::string op_name, std::string symbol);
    arrow::Datum evaluate(const std::shared_ptr<arrow::Table>& table) const override;
    std::string to_string() const override;
};

// ---------------------------------------------------------
// Public API Functions
// ---------------------------------------------------------
ExprPtr col(std::string name);
ExprPtr lit(LitValue value);

// Example Operator Overloads (+ and >)
ExprPtr operator+(const ExprPtr& left, const ExprPtr& right);
ExprPtr operator>(const ExprPtr& left, const ExprPtr& right);

// Convenience overloads so you can write `col("age") > 30` directly
ExprPtr operator+(const ExprPtr& left, LitValue right);
ExprPtr operator>(const ExprPtr& left, LitValue right);