#include "ExpressionSystem.h"
#include <arrow/builder.h>
#include <arrow/type_traits.h>
#include <arrow/array/util.h>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>

namespace dataframelib {

// =====================================================================
// Helpers — array extraction and type promotion (no Arrow compute).
// =====================================================================
namespace {

// Pull a single Array out of a Datum. Scalars get broadcast to length n.
std::shared_ptr<arrow::Array> datum_to_array(const arrow::Datum& d, int64_t n) {
    if (d.is_chunked_array()) return d.chunked_array()->chunk(0);
    if (d.is_array()) return d.make_array();
    auto res = arrow::MakeArrayFromScalar(*d.scalar(), n);
    return *res;
}

bool is_int_type(arrow::Type::type t) {
    return t == arrow::Type::INT32 || t == arrow::Type::INT64;
}
bool is_float_type(arrow::Type::type t) {
    return t == arrow::Type::FLOAT || t == arrow::Type::DOUBLE;
}
bool is_numeric_type(arrow::Type::type t) {
    return is_int_type(t) || is_float_type(t);
}

// Type promotion: int + float = float. Returns the promoted type id.
arrow::Type::type promote_numeric(arrow::Type::type a, arrow::Type::type b) {
    if (a == b) return a;
    if (a == arrow::Type::DOUBLE || b == arrow::Type::DOUBLE) return arrow::Type::DOUBLE;
    if (a == arrow::Type::FLOAT || b == arrow::Type::FLOAT) return arrow::Type::FLOAT;
    if (a == arrow::Type::INT64 || b == arrow::Type::INT64) return arrow::Type::INT64;
    return arrow::Type::INT32;
}

// Cast a column-style array to a target numeric type. Element-wise loop.
std::shared_ptr<arrow::Array> cast_numeric(const std::shared_ptr<arrow::Array>& arr,
                                           arrow::Type::type target) {
    if (arr->type_id() == target) return arr;

    auto build = [&](auto builder, auto reader) {
        ARROW_UNUSED(builder.Reserve(arr->length()));
        for (int64_t i = 0; i < arr->length(); ++i) {
            if (arr->IsValid(i)) {
                ARROW_UNUSED(builder.Append(static_cast<decltype(builder.Append(0))>(0), 0));
            } else {
                ARROW_UNUSED(builder.AppendNull());
            }
            (void)reader;
        }
        std::shared_ptr<arrow::Array> out;
        ARROW_UNUSED(builder.Finish(&out));
        return out;
    };
    (void)build;

    // Read source as double (widest), append into builder for target.
    auto read_as_double = [&](int64_t i) -> double {
        switch (arr->type_id()) {
            case arrow::Type::INT32:  return std::static_pointer_cast<arrow::Int32Array>(arr)->Value(i);
            case arrow::Type::INT64:  return static_cast<double>(std::static_pointer_cast<arrow::Int64Array>(arr)->Value(i));
            case arrow::Type::FLOAT:  return std::static_pointer_cast<arrow::FloatArray>(arr)->Value(i);
            case arrow::Type::DOUBLE: return std::static_pointer_cast<arrow::DoubleArray>(arr)->Value(i);
            default: throw std::runtime_error("cast_numeric: non-numeric source");
        }
    };

    std::shared_ptr<arrow::Array> out;
    switch (target) {
        case arrow::Type::INT32: {
            arrow::Int32Builder b;
            ARROW_UNUSED(b.Reserve(arr->length()));
            for (int64_t i = 0; i < arr->length(); ++i) {
                if (arr->IsValid(i)) ARROW_UNUSED(b.Append(static_cast<int32_t>(read_as_double(i))));
                else ARROW_UNUSED(b.AppendNull());
            }
            ARROW_UNUSED(b.Finish(&out));
            return out;
        }
        case arrow::Type::INT64: {
            arrow::Int64Builder b;
            ARROW_UNUSED(b.Reserve(arr->length()));
            for (int64_t i = 0; i < arr->length(); ++i) {
                if (arr->IsValid(i)) ARROW_UNUSED(b.Append(static_cast<int64_t>(read_as_double(i))));
                else ARROW_UNUSED(b.AppendNull());
            }
            ARROW_UNUSED(b.Finish(&out));
            return out;
        }
        case arrow::Type::FLOAT: {
            arrow::FloatBuilder b;
            ARROW_UNUSED(b.Reserve(arr->length()));
            for (int64_t i = 0; i < arr->length(); ++i) {
                if (arr->IsValid(i)) ARROW_UNUSED(b.Append(static_cast<float>(read_as_double(i))));
                else ARROW_UNUSED(b.AppendNull());
            }
            ARROW_UNUSED(b.Finish(&out));
            return out;
        }
        case arrow::Type::DOUBLE: {
            arrow::DoubleBuilder b;
            ARROW_UNUSED(b.Reserve(arr->length()));
            for (int64_t i = 0; i < arr->length(); ++i) {
                if (arr->IsValid(i)) ARROW_UNUSED(b.Append(read_as_double(i)));
                else ARROW_UNUSED(b.AppendNull());
            }
            ARROW_UNUSED(b.Finish(&out));
            return out;
        }
        default:
            throw std::runtime_error("cast_numeric: unsupported target type");
    }
}

template <typename ArrowType, typename Op>
std::shared_ptr<arrow::Array> manual_math_op(const std::shared_ptr<arrow::Array>& left,
                                             const std::shared_ptr<arrow::Array>& right,
                                             Op op) {
    using ArrayType = typename arrow::TypeTraits<ArrowType>::ArrayType;
    using BuilderType = typename arrow::TypeTraits<ArrowType>::BuilderType;

    auto l = std::static_pointer_cast<ArrayType>(left);
    auto r = std::static_pointer_cast<ArrayType>(right);

    BuilderType b;
    ARROW_UNUSED(b.Reserve(left->length()));
    for (int64_t i = 0; i < left->length(); ++i) {
        if (l->IsValid(i) && r->IsValid(i)) {
            ARROW_UNUSED(b.Append(op(l->Value(i), r->Value(i))));
        } else {
            ARROW_UNUSED(b.AppendNull());
        }
    }
    std::shared_ptr<arrow::Array> out;
    ARROW_UNUSED(b.Finish(&out));
    return out;
}

template <typename ArrowType, typename Op>
std::shared_ptr<arrow::Array> manual_cmp_op(const std::shared_ptr<arrow::Array>& left,
                                            const std::shared_ptr<arrow::Array>& right,
                                            Op op) {
    using ArrayType = typename arrow::TypeTraits<ArrowType>::ArrayType;
    auto l = std::static_pointer_cast<ArrayType>(left);
    auto r = std::static_pointer_cast<ArrayType>(right);

    arrow::BooleanBuilder b;
    ARROW_UNUSED(b.Reserve(left->length()));
    for (int64_t i = 0; i < left->length(); ++i) {
        if (l->IsValid(i) && r->IsValid(i)) {
            ARROW_UNUSED(b.Append(op(l->GetView(i), r->GetView(i))));
        } else {
            ARROW_UNUSED(b.AppendNull());
        }
    }
    std::shared_ptr<arrow::Array> out;
    ARROW_UNUSED(b.Finish(&out));
    return out;
}

// Dispatch a numeric-vs-numeric arithmetic op given the (already promoted) type.
template <typename Op>
std::shared_ptr<arrow::Array> dispatch_math(arrow::Type::type t,
                                            const std::shared_ptr<arrow::Array>& l,
                                            const std::shared_ptr<arrow::Array>& r,
                                            Op op) {
    switch (t) {
        case arrow::Type::INT32:  return manual_math_op<arrow::Int32Type>(l, r, op);
        case arrow::Type::INT64:  return manual_math_op<arrow::Int64Type>(l, r, op);
        case arrow::Type::FLOAT:  return manual_math_op<arrow::FloatType>(l, r, op);
        case arrow::Type::DOUBLE: return manual_math_op<arrow::DoubleType>(l, r, op);
        default: throw std::runtime_error("dispatch_math: non-numeric type");
    }
}

template <typename Op>
std::shared_ptr<arrow::Array> dispatch_cmp_numeric(arrow::Type::type t,
                                                   const std::shared_ptr<arrow::Array>& l,
                                                   const std::shared_ptr<arrow::Array>& r,
                                                   Op op) {
    switch (t) {
        case arrow::Type::INT32:  return manual_cmp_op<arrow::Int32Type>(l, r, op);
        case arrow::Type::INT64:  return manual_cmp_op<arrow::Int64Type>(l, r, op);
        case arrow::Type::FLOAT:  return manual_cmp_op<arrow::FloatType>(l, r, op);
        case arrow::Type::DOUBLE: return manual_cmp_op<arrow::DoubleType>(l, r, op);
        default: throw std::runtime_error("dispatch_cmp_numeric: non-numeric type");
    }
}

}

// =====================================================================
// ColExpr / LitExpr / AliasExpr
// =====================================================================
ColExpr::ColExpr(std::string name) : name_(std::move(name)) {}

arrow::Datum ColExpr::evaluate(const std::shared_ptr<arrow::Table>& table) const {
    auto chunked = table->GetColumnByName(name_);
    if (!chunked) throw std::runtime_error("Column not found: " + name_);
    return arrow::Datum(chunked);
}
std::string ColExpr::to_string() const { return "col(\"" + name_ + "\")"; }

LitExpr::LitExpr(LitValue value) : value_(std::move(value)) {}

namespace {
struct MakeArrowScalarVisitor {
    std::shared_ptr<arrow::Scalar> operator()(int32_t v) const { return arrow::MakeScalar(v); }
    std::shared_ptr<arrow::Scalar> operator()(int64_t v) const { return arrow::MakeScalar(v); }
    std::shared_ptr<arrow::Scalar> operator()(float v) const { return arrow::MakeScalar(v); }
    std::shared_ptr<arrow::Scalar> operator()(double v) const { return arrow::MakeScalar(v); }
    std::shared_ptr<arrow::Scalar> operator()(const std::string& v) const { return arrow::MakeScalar(v); }
    std::shared_ptr<arrow::Scalar> operator()(bool v) const { return arrow::MakeScalar(v); }
};
struct ToStringVisitor {
    std::string operator()(int32_t v) const { return std::to_string(v); }
    std::string operator()(int64_t v) const { return std::to_string(v) + "L"; }
    std::string operator()(float v) const { return std::to_string(v) + "f"; }
    std::string operator()(double v) const { return std::to_string(v); }
    std::string operator()(const std::string& v) const { return "\"" + v + "\""; }
    std::string operator()(bool v) const { return v ? "true" : "false"; }
};
}

arrow::Datum LitExpr::evaluate(const std::shared_ptr<arrow::Table>& /*table*/) const {
    return arrow::Datum(std::visit(MakeArrowScalarVisitor{}, value_));
}
std::string LitExpr::to_string() const { return "lit(" + std::visit(ToStringVisitor{}, value_) + ")"; }

AliasExpr::AliasExpr(ExprNodePtr child, std::string name) : child_(std::move(child)), name_(std::move(name)) {}
arrow::Datum AliasExpr::evaluate(const std::shared_ptr<arrow::Table>& table) const {
    return child_->evaluate(table);
}
std::string AliasExpr::to_string() const { return child_->to_string() + ".alias(\"" + name_ + "\")"; }

// =====================================================================
// BinaryExpr
// =====================================================================
BinaryExpr::BinaryExpr(ExprNodePtr left, ExprNodePtr right, std::string op_name, std::string symbol)
    : left_(std::move(left)), right_(std::move(right)), op_name_(std::move(op_name)), symbol_(std::move(symbol)) {}

std::string BinaryExpr::to_string() const {
    return "(" + left_->to_string() + " " + symbol_ + " " + right_->to_string() + ")";
}

namespace {

bool op_is_arith(const std::string& op) {
    return op == "add" || op == "sub" || op == "mul" || op == "div" || op == "mod";
}
bool op_is_cmp(const std::string& op) {
    return op == "eq" || op == "ne" || op == "lt" || op == "le" || op == "gt" || op == "ge";
}
bool op_is_bool(const std::string& op) {
    return op == "and" || op == "or";
}

}

arrow::Datum BinaryExpr::evaluate(const std::shared_ptr<arrow::Table>& table) const {
    auto ld = left_->evaluate(table);
    auto rd = right_->evaluate(table);

    int64_t n = table->num_rows();
    auto la = datum_to_array(ld, n);
    auto ra = datum_to_array(rd, n);

    // ---- Boolean ops ----
    if (op_is_bool(op_name_)) {
        if (la->type_id() != arrow::Type::BOOL || ra->type_id() != arrow::Type::BOOL) {
            throw std::runtime_error("Boolean op '" + symbol_ + "' requires boolean operands");
        }
        auto l = std::static_pointer_cast<arrow::BooleanArray>(la);
        auto r = std::static_pointer_cast<arrow::BooleanArray>(ra);
        arrow::BooleanBuilder b;
        ARROW_UNUSED(b.Reserve(la->length()));
        for (int64_t i = 0; i < la->length(); ++i) {
            if (!l->IsValid(i) || !r->IsValid(i)) {
                ARROW_UNUSED(b.AppendNull());
            } else {
                bool lv = l->Value(i), rv = r->Value(i);
                ARROW_UNUSED(b.Append(op_name_ == "and" ? (lv && rv) : (lv || rv)));
            }
        }
        std::shared_ptr<arrow::Array> out;
        ARROW_UNUSED(b.Finish(&out));
        return out;
    }

    // ---- String == / != ----
    if (op_is_cmp(op_name_) && la->type_id() == arrow::Type::STRING && ra->type_id() == arrow::Type::STRING) {
        if (op_name_ != "eq" && op_name_ != "ne") {
            throw std::runtime_error("Strings only support == and !=");
        }
        auto l = std::static_pointer_cast<arrow::StringArray>(la);
        auto r = std::static_pointer_cast<arrow::StringArray>(ra);
        arrow::BooleanBuilder b;
        ARROW_UNUSED(b.Reserve(la->length()));
        for (int64_t i = 0; i < la->length(); ++i) {
            if (!l->IsValid(i) || !r->IsValid(i)) {
                ARROW_UNUSED(b.AppendNull());
            } else {
                bool eq = l->GetView(i) == r->GetView(i);
                ARROW_UNUSED(b.Append(op_name_ == "eq" ? eq : !eq));
            }
        }
        std::shared_ptr<arrow::Array> out;
        ARROW_UNUSED(b.Finish(&out));
        return out;
    }

    // ---- Numeric arith / cmp with type promotion ----
    if (!is_numeric_type(la->type_id()) || !is_numeric_type(ra->type_id())) {
        throw std::runtime_error("Operator '" + symbol_ + "' requires numeric operands");
    }
    auto promoted = promote_numeric(la->type_id(), ra->type_id());
    auto lp = cast_numeric(la, promoted);
    auto rp = cast_numeric(ra, promoted);

    if (op_is_arith(op_name_)) {
        if (op_name_ == "add") return dispatch_math(promoted, lp, rp, [](auto a, auto b) { return a + b; });
        if (op_name_ == "sub") return dispatch_math(promoted, lp, rp, [](auto a, auto b) { return a - b; });
        if (op_name_ == "mul") return dispatch_math(promoted, lp, rp, [](auto a, auto b) { return a * b; });
        if (op_name_ == "div") {
            // Integer division throws on zero; float division yields inf/nan as native.
            if (is_int_type(promoted)) {
                if (promoted == arrow::Type::INT32) {
                    return manual_math_op<arrow::Int32Type>(lp, rp, [](int32_t a, int32_t b) {
                        return b == 0 ? 0 : a / b;
                    });
                }
                return manual_math_op<arrow::Int64Type>(lp, rp, [](int64_t a, int64_t b) {
                    return b == 0 ? 0 : a / b;
                });
            }
            return dispatch_math(promoted, lp, rp, [](auto a, auto b) { return a / b; });
        }
        if (op_name_ == "mod") {
            if (!is_int_type(promoted)) throw std::runtime_error("Modulo requires integer operands");
            if (promoted == arrow::Type::INT32) {
                return manual_math_op<arrow::Int32Type>(lp, rp, [](int32_t a, int32_t b) {
                    return b == 0 ? 0 : a % b;
                });
            }
            return manual_math_op<arrow::Int64Type>(lp, rp, [](int64_t a, int64_t b) {
                return b == 0 ? 0 : a % b;
            });
        }
    }

    if (op_is_cmp(op_name_)) {
        if (op_name_ == "eq") return dispatch_cmp_numeric(promoted, lp, rp, [](auto a, auto b) { return a == b; });
        if (op_name_ == "ne") return dispatch_cmp_numeric(promoted, lp, rp, [](auto a, auto b) { return a != b; });
        if (op_name_ == "lt") return dispatch_cmp_numeric(promoted, lp, rp, [](auto a, auto b) { return a < b; });
        if (op_name_ == "le") return dispatch_cmp_numeric(promoted, lp, rp, [](auto a, auto b) { return a <= b; });
        if (op_name_ == "gt") return dispatch_cmp_numeric(promoted, lp, rp, [](auto a, auto b) { return a > b; });
        if (op_name_ == "ge") return dispatch_cmp_numeric(promoted, lp, rp, [](auto a, auto b) { return a >= b; });
    }

    throw std::runtime_error("Unsupported binary operation: " + op_name_);
}

// =====================================================================
// UnaryExpr
// =====================================================================
UnaryExpr::UnaryExpr(ExprNodePtr child, std::string op_name)
    : child_(std::move(child)), op_name_(std::move(op_name)) {}
std::string UnaryExpr::to_string() const { return op_name_ + "(" + child_->to_string() + ")"; }

namespace {

template <typename ArrowType, typename Op>
std::shared_ptr<arrow::Array> manual_unary_numeric(const std::shared_ptr<arrow::Array>& a, Op op) {
    using ArrayType = typename arrow::TypeTraits<ArrowType>::ArrayType;
    using BuilderType = typename arrow::TypeTraits<ArrowType>::BuilderType;
    auto t = std::static_pointer_cast<ArrayType>(a);
    BuilderType b;
    ARROW_UNUSED(b.Reserve(a->length()));
    for (int64_t i = 0; i < a->length(); ++i) {
        if (t->IsValid(i)) ARROW_UNUSED(b.Append(op(t->Value(i))));
        else ARROW_UNUSED(b.AppendNull());
    }
    std::shared_ptr<arrow::Array> out;
    ARROW_UNUSED(b.Finish(&out));
    return out;
}

std::shared_ptr<arrow::Array> manual_is_null(const std::shared_ptr<arrow::Array>& a, bool inverted) {
    arrow::BooleanBuilder b;
    ARROW_UNUSED(b.Reserve(a->length()));
    for (int64_t i = 0; i < a->length(); ++i) {
        bool isn = !a->IsValid(i);
        ARROW_UNUSED(b.Append(inverted ? !isn : isn));
    }
    std::shared_ptr<arrow::Array> out;
    ARROW_UNUSED(b.Finish(&out));
    return out;
}

std::shared_ptr<arrow::Array> manual_str_length(const std::shared_ptr<arrow::Array>& a) {
    auto s = std::static_pointer_cast<arrow::StringArray>(a);
    arrow::Int32Builder b;
    ARROW_UNUSED(b.Reserve(a->length()));
    for (int64_t i = 0; i < a->length(); ++i) {
        if (s->IsValid(i)) ARROW_UNUSED(b.Append(static_cast<int32_t>(s->GetView(i).size())));
        else ARROW_UNUSED(b.AppendNull());
    }
    std::shared_ptr<arrow::Array> out;
    ARROW_UNUSED(b.Finish(&out));
    return out;
}

std::shared_ptr<arrow::Array> manual_str_case(const std::shared_ptr<arrow::Array>& a, bool upper) {
    auto s = std::static_pointer_cast<arrow::StringArray>(a);
    arrow::StringBuilder b;
    ARROW_UNUSED(b.Reserve(a->length()));
    for (int64_t i = 0; i < a->length(); ++i) {
        if (s->IsValid(i)) {
            std::string v(s->GetView(i));
            std::transform(v.begin(), v.end(), v.begin(), [upper](unsigned char c) {
                return upper ? std::toupper(c) : std::tolower(c);
            });
            ARROW_UNUSED(b.Append(v));
        } else {
            ARROW_UNUSED(b.AppendNull());
        }
    }
    std::shared_ptr<arrow::Array> out;
    ARROW_UNUSED(b.Finish(&out));
    return out;
}

std::shared_ptr<arrow::Array> manual_bool_not(const std::shared_ptr<arrow::Array>& a) {
    auto t = std::static_pointer_cast<arrow::BooleanArray>(a);
    arrow::BooleanBuilder b;
    ARROW_UNUSED(b.Reserve(a->length()));
    for (int64_t i = 0; i < a->length(); ++i) {
        if (t->IsValid(i)) ARROW_UNUSED(b.Append(!t->Value(i)));
        else ARROW_UNUSED(b.AppendNull());
    }
    std::shared_ptr<arrow::Array> out;
    ARROW_UNUSED(b.Finish(&out));
    return out;
}

}

arrow::Datum UnaryExpr::evaluate(const std::shared_ptr<arrow::Table>& table) const {
    auto cd = child_->evaluate(table);
    int64_t n = table->num_rows();
    auto a = datum_to_array(cd, n);

    if (op_name_ == "is_null")     return manual_is_null(a, false);
    if (op_name_ == "is_not_null") return manual_is_null(a, true);
    if (op_name_ == "not") {
        if (a->type_id() != arrow::Type::BOOL) throw std::runtime_error("~ requires boolean operand");
        return manual_bool_not(a);
    }
    if (op_name_ == "length") {
        if (a->type_id() != arrow::Type::STRING) throw std::runtime_error(".length() requires string");
        return manual_str_length(a);
    }
    if (op_name_ == "to_lower") {
        if (a->type_id() != arrow::Type::STRING) throw std::runtime_error(".to_lower() requires string");
        return manual_str_case(a, false);
    }
    if (op_name_ == "to_upper") {
        if (a->type_id() != arrow::Type::STRING) throw std::runtime_error(".to_upper() requires string");
        return manual_str_case(a, true);
    }
    if (op_name_ == "abs") {
        if (!is_numeric_type(a->type_id())) throw std::runtime_error(".abs() requires numeric");
        switch (a->type_id()) {
            case arrow::Type::INT32:  return manual_unary_numeric<arrow::Int32Type>(a, [](int32_t x) { return std::abs(x); });
            case arrow::Type::INT64:  return manual_unary_numeric<arrow::Int64Type>(a, [](int64_t x) { return std::abs(x); });
            case arrow::Type::FLOAT:  return manual_unary_numeric<arrow::FloatType>(a, [](float x) { return std::fabs(x); });
            case arrow::Type::DOUBLE: return manual_unary_numeric<arrow::DoubleType>(a, [](double x) { return std::fabs(x); });
            default: break;
        }
    }
    if (op_name_ == "neg") {
        if (!is_numeric_type(a->type_id())) throw std::runtime_error("unary - requires numeric");
        switch (a->type_id()) {
            case arrow::Type::INT32:  return manual_unary_numeric<arrow::Int32Type>(a, [](int32_t x) { return -x; });
            case arrow::Type::INT64:  return manual_unary_numeric<arrow::Int64Type>(a, [](int64_t x) { return -x; });
            case arrow::Type::FLOAT:  return manual_unary_numeric<arrow::FloatType>(a, [](float x) { return -x; });
            case arrow::Type::DOUBLE: return manual_unary_numeric<arrow::DoubleType>(a, [](double x) { return -x; });
            default: break;
        }
    }
    throw std::runtime_error("Unsupported unary op: " + op_name_);
}

// =====================================================================
// StringPredicateExpr
// =====================================================================
StringPredicateExpr::StringPredicateExpr(ExprNodePtr child, std::string op_name, std::string operand)
    : child_(std::move(child)), op_name_(std::move(op_name)), operand_(std::move(operand)) {}
std::string StringPredicateExpr::to_string() const {
    return child_->to_string() + "." + op_name_ + "(\"" + operand_ + "\")";
}

arrow::Datum StringPredicateExpr::evaluate(const std::shared_ptr<arrow::Table>& table) const {
    auto cd = child_->evaluate(table);
    int64_t n = table->num_rows();
    auto a = datum_to_array(cd, n);
    if (a->type_id() != arrow::Type::STRING) {
        throw std::runtime_error("." + op_name_ + " requires string column");
    }
    auto s = std::static_pointer_cast<arrow::StringArray>(a);

    arrow::BooleanBuilder b;
    ARROW_UNUSED(b.Reserve(a->length()));
    const std::string& needle = operand_;
    for (int64_t i = 0; i < a->length(); ++i) {
        if (!s->IsValid(i)) {
            ARROW_UNUSED(b.AppendNull());
            continue;
        }
        std::string_view v = s->GetView(i);
        bool result = false;
        if (op_name_ == "contains") {
            result = v.find(needle) != std::string_view::npos;
        } else if (op_name_ == "starts_with") {
            result = v.size() >= needle.size() && v.compare(0, needle.size(), needle) == 0;
        } else if (op_name_ == "ends_with") {
            result = v.size() >= needle.size() &&
                     v.compare(v.size() - needle.size(), needle.size(), needle) == 0;
        }
        ARROW_UNUSED(b.Append(result));
    }
    std::shared_ptr<arrow::Array> out;
    ARROW_UNUSED(b.Finish(&out));
    return out;
}

// =====================================================================
// AggExpr — collapses a column to a scalar
// =====================================================================
AggExpr::AggExpr(ExprNodePtr child, std::string func)
    : child_(std::move(child)), func_(std::move(func)) {}
std::string AggExpr::to_string() const { return child_->to_string() + "." + func_ + "()"; }

arrow::Datum AggExpr::evaluate(const std::shared_ptr<arrow::Table>& table) const {
    auto cd = child_->evaluate(table);
    auto a = datum_to_array(cd, table->num_rows());

    auto for_each_valid_double = [&](auto&& f) {
        for (int64_t i = 0; i < a->length(); ++i) {
            if (!a->IsValid(i)) continue;
            double v = 0;
            switch (a->type_id()) {
                case arrow::Type::INT32:  v = std::static_pointer_cast<arrow::Int32Array>(a)->Value(i); break;
                case arrow::Type::INT64:  v = static_cast<double>(std::static_pointer_cast<arrow::Int64Array>(a)->Value(i)); break;
                case arrow::Type::FLOAT:  v = std::static_pointer_cast<arrow::FloatArray>(a)->Value(i); break;
                case arrow::Type::DOUBLE: v = std::static_pointer_cast<arrow::DoubleArray>(a)->Value(i); break;
                default: throw std::runtime_error("Aggregation requires numeric column");
            }
            f(v);
        }
    };

    if (func_ == "count") {
        int64_t c = 0;
        for (int64_t i = 0; i < a->length(); ++i) if (a->IsValid(i)) c++;
        return arrow::Datum(arrow::MakeScalar(c));
    }

    if (a->length() == 0) {
        return arrow::Datum(arrow::MakeNullScalar(a->type()));
    }

    if (func_ == "sum") {
        double s = 0; bool any = false;
        for_each_valid_double([&](double v) { s += v; any = true; });
        if (!any) return arrow::Datum(arrow::MakeNullScalar(a->type()));
        switch (a->type_id()) {
            case arrow::Type::INT32:  return arrow::Datum(arrow::MakeScalar(static_cast<int32_t>(s)));
            case arrow::Type::INT64:  return arrow::Datum(arrow::MakeScalar(static_cast<int64_t>(s)));
            case arrow::Type::FLOAT:  return arrow::Datum(arrow::MakeScalar(static_cast<float>(s)));
            case arrow::Type::DOUBLE: return arrow::Datum(arrow::MakeScalar(s));
            default: break;
        }
    }
    if (func_ == "mean") {
        double s = 0; int64_t c = 0;
        for_each_valid_double([&](double v) { s += v; c++; });
        if (c == 0) return arrow::Datum(arrow::MakeNullScalar(arrow::float64()));
        return arrow::Datum(arrow::MakeScalar(s / c));
    }
    if (func_ == "min" || func_ == "max") {
        bool any = false; double best = 0;
        for_each_valid_double([&](double v) {
            if (!any) { best = v; any = true; }
            else if (func_ == "min") best = std::min(best, v);
            else best = std::max(best, v);
        });
        if (!any) return arrow::Datum(arrow::MakeNullScalar(a->type()));
        switch (a->type_id()) {
            case arrow::Type::INT32:  return arrow::Datum(arrow::MakeScalar(static_cast<int32_t>(best)));
            case arrow::Type::INT64:  return arrow::Datum(arrow::MakeScalar(static_cast<int64_t>(best)));
            case arrow::Type::FLOAT:  return arrow::Datum(arrow::MakeScalar(static_cast<float>(best)));
            case arrow::Type::DOUBLE: return arrow::Datum(arrow::MakeScalar(best));
            default: break;
        }
    }
    throw std::runtime_error("Unknown aggregation: " + func_);
}

// =====================================================================
// Expr value class — methods + factories + operators
// =====================================================================
Expr Expr::abs()         const { return Expr(std::make_shared<UnaryExpr>(node_, "abs")); }
Expr Expr::is_null()     const { return Expr(std::make_shared<UnaryExpr>(node_, "is_null")); }
Expr Expr::is_not_null() const { return Expr(std::make_shared<UnaryExpr>(node_, "is_not_null")); }
Expr Expr::length()      const { return Expr(std::make_shared<UnaryExpr>(node_, "length")); }
Expr Expr::to_lower()    const { return Expr(std::make_shared<UnaryExpr>(node_, "to_lower")); }
Expr Expr::to_upper()    const { return Expr(std::make_shared<UnaryExpr>(node_, "to_upper")); }

Expr Expr::contains(const std::string& s)    const { return Expr(std::make_shared<StringPredicateExpr>(node_, "contains", s)); }
Expr Expr::starts_with(const std::string& s) const { return Expr(std::make_shared<StringPredicateExpr>(node_, "starts_with", s)); }
Expr Expr::ends_with(const std::string& s)   const { return Expr(std::make_shared<StringPredicateExpr>(node_, "ends_with", s)); }

Expr Expr::sum()   const { return Expr(std::make_shared<AggExpr>(node_, "sum")); }
Expr Expr::mean()  const { return Expr(std::make_shared<AggExpr>(node_, "mean")); }
Expr Expr::count() const { return Expr(std::make_shared<AggExpr>(node_, "count")); }
Expr Expr::min()   const { return Expr(std::make_shared<AggExpr>(node_, "min")); }
Expr Expr::max()   const { return Expr(std::make_shared<AggExpr>(node_, "max")); }

Expr Expr::alias(const std::string& name) const { return Expr(std::make_shared<AliasExpr>(node_, name)); }

Expr col(const std::string& name) { return Expr(std::make_shared<ColExpr>(name)); }
Expr lit(LitValue value)          { return Expr(std::make_shared<LitExpr>(std::move(value))); }
Expr lit(int v)                   { return lit(LitValue(static_cast<int32_t>(v))); }
Expr lit(const char* v)           { return lit(LitValue(std::string(v))); }

namespace {
inline Expr mk_bin(const Expr& a, const Expr& b, const char* op, const char* sym) {
    return Expr(std::make_shared<BinaryExpr>(a.node(), b.node(), op, sym));
}
}

Expr operator+(const Expr& a, const Expr& b) { return mk_bin(a, b, "add", "+"); }
Expr operator-(const Expr& a, const Expr& b) { return mk_bin(a, b, "sub", "-"); }
Expr operator*(const Expr& a, const Expr& b) { return mk_bin(a, b, "mul", "*"); }
Expr operator/(const Expr& a, const Expr& b) { return mk_bin(a, b, "div", "/"); }
Expr operator%(const Expr& a, const Expr& b) { return mk_bin(a, b, "mod", "%"); }
Expr operator==(const Expr& a, const Expr& b) { return mk_bin(a, b, "eq", "=="); }
Expr operator!=(const Expr& a, const Expr& b) { return mk_bin(a, b, "ne", "!="); }
Expr operator<(const Expr& a, const Expr& b) { return mk_bin(a, b, "lt", "<"); }
Expr operator<=(const Expr& a, const Expr& b) { return mk_bin(a, b, "le", "<="); }
Expr operator>(const Expr& a, const Expr& b) { return mk_bin(a, b, "gt", ">"); }
Expr operator>=(const Expr& a, const Expr& b) { return mk_bin(a, b, "ge", ">="); }
Expr operator&(const Expr& a, const Expr& b) { return mk_bin(a, b, "and", "&"); }
Expr operator|(const Expr& a, const Expr& b) { return mk_bin(a, b, "or", "|"); }

Expr operator~(const Expr& a) { return Expr(std::make_shared<UnaryExpr>(a.node(), "not")); }
Expr operator-(const Expr& a) { return Expr(std::make_shared<UnaryExpr>(a.node(), "neg")); }

}
