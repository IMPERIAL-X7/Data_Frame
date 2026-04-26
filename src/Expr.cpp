#include "DataFrameLib/Expr.hpp"
#include <arrow/builder.h>
#include <arrow/type_traits.h>
#include <arrow/array/util.h>
#include <stdexcept>
#include <iostream>

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


// Engine 1: Math Operations (Returns the same type as the input)
template <typename ArrowType, typename Op>
std::shared_ptr<arrow::Array> manual_math_op(
    const std::shared_ptr<arrow::Array>& left,
    const std::shared_ptr<arrow::Array>& right,
    Op op) 
{
    using ArrayType = typename arrow::TypeTraits<ArrowType>::ArrayType;
    using BuilderType = typename arrow::TypeTraits<ArrowType>::BuilderType;

    auto left_typed = std::static_pointer_cast<ArrayType>(left);
    auto right_typed = std::static_pointer_cast<ArrayType>(right);
    
    BuilderType builder;
    ARROW_UNUSED(builder.Reserve(left->length()));

    for (int64_t i = 0; i < left->length(); ++i) {
        // Only compute if both rows are NOT null
        if (left_typed->IsValid(i) && right_typed->IsValid(i)) {
            builder.Append(op(left_typed->GetView(i), right_typed->GetView(i)));
        } else {
            builder.AppendNull();
        }
    }
    
    std::shared_ptr<arrow::Array> result;
    ARROW_UNUSED(builder.Finish(&result));
    return result;
}

// Engine 2: Comparison Operations (Always returns a BooleanArray)
template <typename ArrowType, typename Op>
std::shared_ptr<arrow::Array> manual_cmp_op(
    const std::shared_ptr<arrow::Array>& left,
    const std::shared_ptr<arrow::Array>& right,
    Op op) 
{
    using ArrayType = typename arrow::TypeTraits<ArrowType>::ArrayType;
    
    auto left_typed = std::static_pointer_cast<ArrayType>(left);
    auto right_typed = std::static_pointer_cast<ArrayType>(right);
    
    arrow::BooleanBuilder builder;
    ARROW_UNUSED(builder.Reserve(left->length()));

    for (int64_t i = 0; i < left->length(); ++i) {
        if (left_typed->IsValid(i) && right_typed->IsValid(i)) {
            builder.Append(op(left_typed->GetView(i), right_typed->GetView(i)));
        } else {
            builder.AppendNull();
        }
    }
    
    std::shared_ptr<arrow::Array> result;
    ARROW_UNUSED(builder.Finish(&result));
    return result;
}



// ---------------------------------------------------------
// BinaryExpr Implementation
// ---------------------------------------------------------
BinaryExpr::BinaryExpr(ExprPtr left, ExprPtr right, std::string op_name, std::string symbol)
    : left_(std::move(left)), right_(std::move(right)), op_name_(std::move(op_name)), symbol_(std::move(symbol)) {}

// #include <arrow/array/util.h>

// Add this helper function right above BinaryExpr::evaluate
std::shared_ptr<arrow::Scalar> manual_cast_scalar(const std::shared_ptr<arrow::Scalar>& scalar, arrow::Type::type target_type) {
    if (scalar->type->id() == target_type) return scalar;

    // Extract the raw number (using double as a safe intermediate container)
    double val = 0;
    if (scalar->type->id() == arrow::Type::INT32) val = std::static_pointer_cast<arrow::Int32Scalar>(scalar)->value;
    else if (scalar->type->id() == arrow::Type::INT64) val = std::static_pointer_cast<arrow::Int64Scalar>(scalar)->value;
    else if (scalar->type->id() == arrow::Type::FLOAT) val = std::static_pointer_cast<arrow::FloatScalar>(scalar)->value;
    else if (scalar->type->id() == arrow::Type::DOUBLE) val = std::static_pointer_cast<arrow::DoubleScalar>(scalar)->value;
    else return scalar; // String/Bool fallback

    // Repackage as the target type
    if (target_type == arrow::Type::INT32) return std::make_shared<arrow::Int32Scalar>(static_cast<int32_t>(val));
    if (target_type == arrow::Type::INT64) return std::make_shared<arrow::Int64Scalar>(static_cast<int64_t>(val));
    if (target_type == arrow::Type::FLOAT) return std::make_shared<arrow::FloatScalar>(static_cast<float>(val));
    if (target_type == arrow::Type::DOUBLE) return std::make_shared<arrow::DoubleScalar>(static_cast<double>(val));
    
    return scalar;
}

// Replace your existing BinaryExpr::evaluate with this:
arrow::Datum BinaryExpr::evaluate(const std::shared_ptr<arrow::Table>& table) const {
    auto left_datum = left_->evaluate(table);
    auto right_datum = right_->evaluate(table);

    // 1. Handle Type Promotion (e.g. col("age") [INT64] > lit(30) [INT32])
    if ((left_datum.is_chunked_array() || left_datum.is_array()) && right_datum.is_scalar()) {
        right_datum = manual_cast_scalar(right_datum.scalar(), left_datum.type()->id());
    } else if ((right_datum.is_chunked_array() || right_datum.is_array()) && left_datum.is_scalar()) {
        left_datum = manual_cast_scalar(left_datum.scalar(), right_datum.type()->id());
    }

    // 2. Safely Extract Arrays (Broadcasts Scalars to Arrays of length N so the variant doesn't crash)
    int64_t num_rows = table->num_rows();
    std::shared_ptr<arrow::Array> left_arr;
    std::shared_ptr<arrow::Array> right_arr;

    if (left_datum.is_chunked_array()) left_arr = left_datum.chunked_array()->chunk(0);
    else if (left_datum.is_array()) left_arr = left_datum.make_array();
    else left_arr = *arrow::MakeArrayFromScalar(*left_datum.scalar(), num_rows);

    if (right_datum.is_chunked_array()) right_arr = right_datum.chunked_array()->chunk(0);
    else if (right_datum.is_array()) right_arr = right_datum.make_array();
    else right_arr = *arrow::MakeArrayFromScalar(*right_datum.scalar(), num_rows);

    if (left_arr->type_id() != right_arr->type_id()) {
        throw std::runtime_error("Type mismatch in '" + symbol_ + "'. Both sides must be the same type.");
    }

    auto type_id = left_arr->type_id();

    // --- MATH OPERATIONS ---
    if (op_name_ == "add") {
        if (type_id == arrow::Type::INT32) return manual_math_op<arrow::Int32Type>(left_arr, right_arr, [](auto a, auto b) { return a + b; });
        if (type_id == arrow::Type::INT64) return manual_math_op<arrow::Int64Type>(left_arr, right_arr, [](auto a, auto b) { return a + b; });
        if (type_id == arrow::Type::FLOAT) return manual_math_op<arrow::FloatType>(left_arr, right_arr, [](auto a, auto b) { return a + b; });
        if (type_id == arrow::Type::DOUBLE) return manual_math_op<arrow::DoubleType>(left_arr, right_arr, [](auto a, auto b) { return a + b; });
    }
    
    // --- COMPARISON OPERATIONS ---
    else if (op_name_ == "greater") {
        if (type_id == arrow::Type::INT32) return manual_cmp_op<arrow::Int32Type>(left_arr, right_arr, [](auto a, auto b) { return a > b; });
        if (type_id == arrow::Type::INT64) return manual_cmp_op<arrow::Int64Type>(left_arr, right_arr, [](auto a, auto b) { return a > b; });
        if (type_id == arrow::Type::FLOAT) return manual_cmp_op<arrow::FloatType>(left_arr, right_arr, [](auto a, auto b) { return a > b; });
        if (type_id == arrow::Type::DOUBLE) return manual_cmp_op<arrow::DoubleType>(left_arr, right_arr, [](auto a, auto b) { return a > b; });
    }
    else if (op_name_ == "equal") {
        if (type_id == arrow::Type::INT32) return manual_cmp_op<arrow::Int32Type>(left_arr, right_arr, [](auto a, auto b) { return a == b; });
        if (type_id == arrow::Type::INT64) return manual_cmp_op<arrow::Int64Type>(left_arr, right_arr, [](auto a, auto b) { return a == b; });
        if (type_id == arrow::Type::FLOAT) return manual_cmp_op<arrow::FloatType>(left_arr, right_arr, [](auto a, auto b) { return a == b; });
        if (type_id == arrow::Type::DOUBLE) return manual_cmp_op<arrow::DoubleType>(left_arr, right_arr, [](auto a, auto b) { return a == b; });
        if (type_id == arrow::Type::STRING) return manual_cmp_op<arrow::StringType>(left_arr, right_arr, [](auto a, auto b) { return a == b; });
    }

    throw std::runtime_error("Unsupported manual operation: " + op_name_ + " on type ID " + std::to_string(type_id));
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