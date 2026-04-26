#include "../EagerDataFrame.h"
#include <arrow/builder.h>
#include <arrow/type_traits.h>
#include <map>
#include <sstream>
#include <stdexcept>

namespace dataframelib {

namespace {

// Compose a string key from the values at row `i` across all key columns.
// Used as the bucket key in the group map. We embed the validity marker so
// nulls in different key columns don't collide.
std::string compose_key(const std::vector<std::shared_ptr<arrow::Array>>& key_arrays, int64_t i) {
    std::ostringstream oss;
    for (size_t k = 0; k < key_arrays.size(); ++k) {
        if (k) oss << "\x1f";
        const auto& a = key_arrays[k];
        if (!a->IsValid(i)) {
            oss << "\x00null";
            continue;
        }
        switch (a->type_id()) {
            case arrow::Type::INT32:  oss << std::static_pointer_cast<arrow::Int32Array>(a)->Value(i); break;
            case arrow::Type::INT64:  oss << std::static_pointer_cast<arrow::Int64Array>(a)->Value(i); break;
            case arrow::Type::FLOAT:  oss << std::static_pointer_cast<arrow::FloatArray>(a)->Value(i); break;
            case arrow::Type::DOUBLE: oss << std::static_pointer_cast<arrow::DoubleArray>(a)->Value(i); break;
            case arrow::Type::STRING: oss << std::static_pointer_cast<arrow::StringArray>(a)->GetString(i); break;
            case arrow::Type::BOOL:   oss << (std::static_pointer_cast<arrow::BooleanArray>(a)->Value(i) ? 1 : 0); break;
            default: throw std::runtime_error("group_by: unsupported key type");
        }
    }
    return oss.str();
}

// Append a representative value from `src` (taking the first row of the group)
// to a builder of matching type. Used to materialise the output key columns.
void append_repr_value(arrow::ArrayBuilder& builder,
                       const std::shared_ptr<arrow::Array>& src,
                       int64_t row) {
    if (!src->IsValid(row)) {
        ARROW_UNUSED(builder.AppendNull());
        return;
    }
    switch (src->type_id()) {
        case arrow::Type::INT32:
            ARROW_UNUSED(static_cast<arrow::Int32Builder&>(builder).Append(
                std::static_pointer_cast<arrow::Int32Array>(src)->Value(row))); break;
        case arrow::Type::INT64:
            ARROW_UNUSED(static_cast<arrow::Int64Builder&>(builder).Append(
                std::static_pointer_cast<arrow::Int64Array>(src)->Value(row))); break;
        case arrow::Type::FLOAT:
            ARROW_UNUSED(static_cast<arrow::FloatBuilder&>(builder).Append(
                std::static_pointer_cast<arrow::FloatArray>(src)->Value(row))); break;
        case arrow::Type::DOUBLE:
            ARROW_UNUSED(static_cast<arrow::DoubleBuilder&>(builder).Append(
                std::static_pointer_cast<arrow::DoubleArray>(src)->Value(row))); break;
        case arrow::Type::STRING:
            ARROW_UNUSED(static_cast<arrow::StringBuilder&>(builder).Append(
                std::static_pointer_cast<arrow::StringArray>(src)->GetString(row))); break;
        case arrow::Type::BOOL:
            ARROW_UNUSED(static_cast<arrow::BooleanBuilder&>(builder).Append(
                std::static_pointer_cast<arrow::BooleanArray>(src)->Value(row))); break;
        default: throw std::runtime_error("group_by: unsupported key type");
    }
}

std::unique_ptr<arrow::ArrayBuilder> make_builder(arrow::Type::type t) {
    switch (t) {
        case arrow::Type::INT32:  return std::make_unique<arrow::Int32Builder>();
        case arrow::Type::INT64:  return std::make_unique<arrow::Int64Builder>();
        case arrow::Type::FLOAT:  return std::make_unique<arrow::FloatBuilder>();
        case arrow::Type::DOUBLE: return std::make_unique<arrow::DoubleBuilder>();
        case arrow::Type::STRING: return std::make_unique<arrow::StringBuilder>();
        case arrow::Type::BOOL:   return std::make_unique<arrow::BooleanBuilder>();
        default: throw std::runtime_error("group_by: unsupported builder type");
    }
}

template <typename ArrowType>
std::shared_ptr<arrow::Array> manual_agg_column(
    const std::shared_ptr<arrow::Array>& array,
    const std::vector<std::vector<int64_t>>& groups,
    const std::string& func_name) {
    using ArrayType = typename arrow::TypeTraits<ArrowType>::ArrayType;
    using BuilderType = typename arrow::TypeTraits<ArrowType>::BuilderType;
    using CType = typename arrow::TypeTraits<ArrowType>::CType;

    auto t = std::static_pointer_cast<ArrayType>(array);
    BuilderType b;
    ARROW_UNUSED(b.Reserve(groups.size()));

    for (const auto& indices : groups) {
        if (func_name == "count") {
            int64_t c = 0;
            for (int64_t idx : indices) if (t->IsValid(idx)) c++;
            ARROW_UNUSED(b.Append(static_cast<CType>(c)));
            continue;
        }
        if (func_name == "sum" || func_name == "mean") {
            double s = 0; int64_t c = 0;
            for (int64_t idx : indices) {
                if (t->IsValid(idx)) { s += t->Value(idx); c++; }
            }
            if (c == 0) ARROW_UNUSED(b.AppendNull());
            else if (func_name == "mean") ARROW_UNUSED(b.Append(static_cast<CType>(s / c)));
            else ARROW_UNUSED(b.Append(static_cast<CType>(s)));
            continue;
        }
        if (func_name == "min" || func_name == "max") {
            bool any = false; CType best{};
            for (int64_t idx : indices) {
                if (!t->IsValid(idx)) continue;
                CType v = t->Value(idx);
                if (!any) { best = v; any = true; }
                else if (func_name == "min") { if (v < best) best = v; }
                else { if (v > best) best = v; }
            }
            if (!any) ARROW_UNUSED(b.AppendNull());
            else ARROW_UNUSED(b.Append(best));
            continue;
        }
        ARROW_UNUSED(b.AppendNull());
    }

    std::shared_ptr<arrow::Array> out;
    ARROW_UNUSED(b.Finish(&out));
    return out;
}

}

EagerDataFrame GroupedDataFrame::aggregate(const AggSpec& aggs) const {
    if (keys_.empty()) throw std::runtime_error("No group_by keys specified");

    std::vector<std::shared_ptr<arrow::Array>> key_arrays;
    key_arrays.reserve(keys_.size());
    for (const auto& k : keys_) {
        auto chunked = table_->GetColumnByName(k);
        if (!chunked) throw std::runtime_error("group_by: column '" + k + "' not found");
        key_arrays.push_back(chunked->chunk(0));
    }

    int64_t n = table_->num_rows();
    std::map<std::string, std::vector<int64_t>> groups;
    std::vector<std::string> group_order;
    std::vector<int64_t> group_repr_row;

    for (int64_t i = 0; i < n; ++i) {
        std::string key = compose_key(key_arrays, i);
        auto it = groups.find(key);
        if (it == groups.end()) {
            groups[key] = {i};
            group_order.push_back(key);
            group_repr_row.push_back(i);
        } else {
            it->second.push_back(i);
        }
    }

    std::vector<std::vector<int64_t>> ordered_groups;
    ordered_groups.reserve(group_order.size());
    for (const auto& k : group_order) ordered_groups.push_back(groups[k]);

    // Build output key columns (one per group key, in input order).
    std::vector<std::shared_ptr<arrow::Field>> out_fields;
    std::vector<std::shared_ptr<arrow::ChunkedArray>> out_columns;

    for (size_t k = 0; k < keys_.size(); ++k) {
        auto src = key_arrays[k];
        auto builder = make_builder(src->type_id());
        for (int64_t r : group_repr_row) append_repr_value(*builder, src, r);
        std::shared_ptr<arrow::Array> out;
        ARROW_UNUSED(builder->Finish(&out));
        out_fields.push_back(arrow::field(keys_[k], src->type()));
        out_columns.push_back(std::make_shared<arrow::ChunkedArray>(out));
    }

    // Per-aggregation column.
    for (const auto& [col_name, func] : aggs) {
        auto chunked = table_->GetColumnByName(col_name);
        if (!chunked) throw std::runtime_error("aggregate: column '" + col_name + "' not found");
        auto val = chunked->chunk(0);
        std::shared_ptr<arrow::Array> agg;

        switch (val->type_id()) {
            case arrow::Type::INT32:  agg = manual_agg_column<arrow::Int32Type>(val, ordered_groups, func); break;
            case arrow::Type::INT64:  agg = manual_agg_column<arrow::Int64Type>(val, ordered_groups, func); break;
            case arrow::Type::FLOAT:  agg = manual_agg_column<arrow::FloatType>(val, ordered_groups, func); break;
            case arrow::Type::DOUBLE: agg = manual_agg_column<arrow::DoubleType>(val, ordered_groups, func); break;
            default: throw std::runtime_error("aggregate: unsupported value type for '" + col_name + "'");
        }

        out_fields.push_back(arrow::field(col_name + "_" + func, val->type()));
        out_columns.push_back(std::make_shared<arrow::ChunkedArray>(agg));
    }

    return EagerDataFrame(arrow::Table::Make(std::make_shared<arrow::Schema>(out_fields), out_columns));
}

}
