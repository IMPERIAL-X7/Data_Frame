#include "../EagerDataFrame.h"
#include <arrow/builder.h>
#include <arrow/type_traits.h>
#include <arrow/array/util.h>
#include <numeric>
#include <algorithm>
#include <stdexcept>

namespace dataframelib {

namespace {

template <typename ArrowType>
std::vector<int64_t> manual_get_sorted_indices(const std::shared_ptr<arrow::Array>& array, bool asc) {
    using ArrayType = typename arrow::TypeTraits<ArrowType>::ArrayType;
    auto typed_array = std::static_pointer_cast<ArrayType>(array);

    std::vector<int64_t> indices(array->length());
    std::iota(indices.begin(), indices.end(), 0);

    std::sort(indices.begin(), indices.end(), [&](int64_t a, int64_t b) {
        bool a_valid = typed_array->IsValid(a);
        bool b_valid = typed_array->IsValid(b);
        if (!a_valid && !b_valid) return false;
        if (!a_valid) return false;
        if (!b_valid) return true;

        if (asc) return typed_array->GetView(a) < typed_array->GetView(b);
        return typed_array->GetView(a) > typed_array->GetView(b);
    });
    return indices;
}

template <typename ArrowType>
std::shared_ptr<arrow::Array> manual_take(const std::shared_ptr<arrow::Array>& array,
                                          const std::vector<int64_t>& indices) {
    using ArrayType = typename arrow::TypeTraits<ArrowType>::ArrayType;
    using BuilderType = typename arrow::TypeTraits<ArrowType>::BuilderType;

    auto typed_array = std::static_pointer_cast<ArrayType>(array);
    BuilderType builder;
    ARROW_UNUSED(builder.Reserve(indices.size()));

    for (int64_t idx : indices) {
        if (idx >= 0 && typed_array->IsValid(idx)) {
            ARROW_UNUSED(builder.Append(typed_array->GetView(idx)));
        } else {
            ARROW_UNUSED(builder.AppendNull());
        }
    }

    std::shared_ptr<arrow::Array> result;
    ARROW_UNUSED(builder.Finish(&result));
    return result;
}

}

EagerDataFrame EagerDataFrame::sort(const std::vector<std::string>& columns, bool asc) const {
    if (columns.empty() || table_->num_rows() == 0) return *this;

    auto sort_col_chunked = table_->GetColumnByName(columns[0]);
    if (!sort_col_chunked) throw std::runtime_error("Sort column not found");
    auto sort_array = sort_col_chunked->chunk(0);

    std::vector<int64_t> sorted_indices;

    switch (sort_array->type_id()) {
        case arrow::Type::INT32: sorted_indices = manual_get_sorted_indices<arrow::Int32Type>(sort_array, asc); break;
        case arrow::Type::INT64: sorted_indices = manual_get_sorted_indices<arrow::Int64Type>(sort_array, asc); break;
        case arrow::Type::FLOAT: sorted_indices = manual_get_sorted_indices<arrow::FloatType>(sort_array, asc); break;
        case arrow::Type::DOUBLE: sorted_indices = manual_get_sorted_indices<arrow::DoubleType>(sort_array, asc); break;
        case arrow::Type::STRING: sorted_indices = manual_get_sorted_indices<arrow::StringType>(sort_array, asc); break;
        case arrow::Type::BOOL: sorted_indices = manual_get_sorted_indices<arrow::BooleanType>(sort_array, asc); break;
        default: throw std::runtime_error("Unsupported type for manual sort");
    }

    std::vector<std::shared_ptr<arrow::ChunkedArray>> sorted_columns;
    for (int i = 0; i < table_->num_columns(); ++i) {
        auto col_array = table_->column(i)->chunk(0);
        std::shared_ptr<arrow::Array> taken_arr;

        switch (col_array->type_id()) {
            case arrow::Type::INT32: taken_arr = manual_take<arrow::Int32Type>(col_array, sorted_indices); break;
            case arrow::Type::INT64: taken_arr = manual_take<arrow::Int64Type>(col_array, sorted_indices); break;
            case arrow::Type::FLOAT: taken_arr = manual_take<arrow::FloatType>(col_array, sorted_indices); break;
            case arrow::Type::DOUBLE: taken_arr = manual_take<arrow::DoubleType>(col_array, sorted_indices); break;
            case arrow::Type::STRING: taken_arr = manual_take<arrow::StringType>(col_array, sorted_indices); break;
            case arrow::Type::BOOL: taken_arr = manual_take<arrow::BooleanType>(col_array, sorted_indices); break;
            default: throw std::runtime_error("Unsupported type for manual take");
        }

        sorted_columns.push_back(std::make_shared<arrow::ChunkedArray>(taken_arr));
    }

    auto sorted_table = arrow::Table::Make(table_->schema(), sorted_columns);
    return EagerDataFrame(sorted_table);
}

}
