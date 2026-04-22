#include "DataFrameLib/Expr.hpp"
#include <arrow/compute/api.h>
#include <stdexcept>




// ---------------------------------------------------------
// ColExpr Implementation
// ---------------------------------------------------------
ColExpr::ColExpr(std::string name) : name_(std::move(name)) {}

arrow::Datum ColExpr::evaluate(const std::shared_ptr<arrow::Table>& table) const {
    auto chunked_array = table->GetColumnByName(name_);
    if (!chunked_array) {
        throw std::runtime_error("Column not found: " + name_);
    }
    return arrow::Datum(chunked_array);
}

std::string ColExpr::to_string() const {
    return "col(\"" + name_ + "\")";
}

// ---------------------------------------------------------
// LitExpr Implementation
// ---------------------------------------------------------
LitExpr::LitExpr(LitValue value) : value_(std::move(value)) {}

// Visitor to convert our strict C++ types into Arrow Scalar objects
struct MakeArrowScalarVisitor {
    std::shared_ptr<arrow::Scalar> operator()(int32_t v) const { return arrow::MakeScalar(v); }
    std::shared_ptr<arrow::Scalar> operator()(int64_t v) const { return arrow::MakeScalar(v); }
    std::shared_ptr<arrow::Scalar> operator()(float v) const { return arrow::MakeScalar(v); }
    std::shared_ptr<arrow::Scalar> operator()(double v) const { return arrow::MakeScalar(v); }
    std::shared_ptr<arrow::Scalar> operator()(const std::string& v) const { return arrow::MakeScalar(v); }
    std::shared_ptr<arrow::Scalar> operator()(bool v) const { return arrow::MakeScalar(v); }
};

arrow::Datum LitExpr::evaluate(const std::shared_ptr<arrow::Table>& table) const {
    // Literal evaluation ignores the table and just builds an Arrow scalar
    auto scalar = std::visit(MakeArrowScalarVisitor{}, value_);
    return arrow::Datum(scalar);
}

// Visitor to convert variants to strings for the to_string() method
struct ToStringVisitor {
    std::string operator()(int32_t v) const { return std::to_string(v); }
    std::string operator()(int64_t v) const { return std::to_string(v) + "L"; }
    std::string operator()(float v) const { return std::to_string(v) + "f"; }
    std::string operator()(double v) const { return std::to_string(v); }
    std::string operator()(const std::string& v) const { return "\"" + v + "\""; }
    std::string operator()(bool v) const { return v ? "true" : "false"; }
};

std::string LitExpr::to_string() const {
    return "lit(" + std::visit(ToStringVisitor{}, value_) + ")";
}

// ---------------------------------------------------------
// BinaryExpr Implementation
// ---------------------------------------------------------
BinaryExpr::BinaryExpr(ExprPtr left, ExprPtr right, std::string op_name, std::string symbol)
    : left_(std::move(left)), right_(std::move(right)), op_name_(std::move(op_name)), symbol_(std::move(symbol)) {}

arrow::Datum BinaryExpr::evaluate(const std::shared_ptr<arrow::Table>& table) const {
    auto left_datum = left_->evaluate(table);
    auto right_datum = right_->evaluate(table);

    // Call the Arrow Compute API to execute the operation in C++
    auto result = arrow::compute::CallFunction(op_name_, {left_datum, right_datum});
    if (!result.ok()) {
        throw std::runtime_error("Compute error in '" + symbol_ + "': " + result.status().ToString());
    }
    return *result;
}

std::string BinaryExpr::to_string() const {
    return "(" + left_->to_string() + " " + symbol_ + " " + right_->to_string() + ")";
}

// ---------------------------------------------------------
// API Factory Functions & Operator Overloads
// ---------------------------------------------------------
ExprPtr col(std::string name) {
    return std::make_shared<ColExpr>(std::move(name));
}

ExprPtr lit(LitValue value) {
    return std::make_shared<LitExpr>(std::move(value));
}

ExprPtr operator+(const ExprPtr& left, const ExprPtr& right) {
    // "add" is the internal registered name in Arrow's compute registry
    return std::make_shared<BinaryExpr>(left, right, "add", "+");
}

ExprPtr operator>(const ExprPtr& left, const ExprPtr& right) {
    // "greater" is the internal registered name in Arrow
    return std::make_shared<BinaryExpr>(left, right, "greater", ">");
}

ExprPtr operator+(const ExprPtr& left, LitValue right) {
    return left + lit(std::move(right));
}

ExprPtr operator>(const ExprPtr& left, LitValue right) {
    return left > lit(std::move(right));
}