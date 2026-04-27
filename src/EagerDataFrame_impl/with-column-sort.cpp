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
void manual_sort_indices(std::vector<int64_t>& indices, const std::shared_ptr<arrow::Array>& array, bool asc) {
    using ArrayType = typename arrow::TypeTraits<ArrowType>::ArrayType;
    auto typed_array = std::static_pointer_cast<ArrayType>(array);

    std::stable_sort(indices.begin(), indices.end(), [&](int64_t a, int64_t b) {
        bool a_valid = typed_array->IsValid(a);
        bool b_valid = typed_array->IsValid(b);
        if (!a_valid && !b_valid) return false;
        if (!a_valid) return false;
        if (!b_valid) return true;

        if (asc) return typed_array->GetView(a) < typed_array->GetView(b);
        return typed_array->GetView(a) > typed_array->GetView(b);
    });
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

    std::vector<int64_t> sorted_indices(table_->num_rows());
    std::iota(sorted_indices.begin(), sorted_indices.end(), 0);

    for (auto it = columns.rbegin(); it != columns.rend(); ++it) {
        auto sort_col_chunked = table_->GetColumnByName(*it);
        if (!sort_col_chunked) throw std::runtime_error("Sort column not found");
        auto sort_array = sort_col_chunked->chunk(0);

        switch (sort_array->type_id()) {
            case arrow::Type::INT32: manual_sort_indices<arrow::Int32Type>(sorted_indices, sort_array, asc); break;
            case arrow::Type::INT64: manual_sort_indices<arrow::Int64Type>(sorted_indices, sort_array, asc); break;
            case arrow::Type::FLOAT: manual_sort_indices<arrow::FloatType>(sorted_indices, sort_array, asc); break;
            case arrow::Type::DOUBLE: manual_sort_indices<arrow::DoubleType>(sorted_indices, sort_array, asc); break;
            case arrow::Type::STRING: manual_sort_indices<arrow::StringType>(sorted_indices, sort_array, asc); break;
            case arrow::Type::BOOL: manual_sort_indices<arrow::BooleanType>(sorted_indices, sort_array, asc); break;
            default: throw std::runtime_error("Unsupported type for manual sort");
        }
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

// Partial-sort fast path: keep only the top n rows under the sort order.
// Used by the lazy optimizer's TopN fusion rule (Head(n) on Sort → TopN(n)).
// Complexity: O(N log n) versus O(N log N) for the full sort, plus we only
// "take" n rows per column instead of N, which is the bigger constant win.
EagerDataFrame EagerDataFrame::sort_top_n(const std::vector<std::string>& columns,
                                          bool asc, int64_t n) const {
    int64_t total = table_->num_rows();
    if (columns.empty() || total == 0) return head(n);
    if (n <= 0) return head(0);
    if (n >= total) return sort(columns, asc);

    std::vector<int64_t> indices(total);
    std::iota(indices.begin(), indices.end(), 0);

    // Apply partial_sort once per key, in reverse order so the primary key
    // wins (mirrors the stable_sort layering in sort()).
    for (auto it = columns.rbegin(); it != columns.rend(); ++it) {
        auto col_chunked = table_->GetColumnByName(*it);
        if (!col_chunked) throw std::runtime_error("sort_top_n: column not found: " + *it);
        auto arr = col_chunked->chunk(0);

        auto run = [&](auto type_tag) {
            using ArrowType = decltype(type_tag);
            using ArrayType = typename arrow::TypeTraits<ArrowType>::ArrayType;
            auto typed = std::static_pointer_cast<ArrayType>(arr);
            std::partial_sort(indices.begin(), indices.begin() + n, indices.end(),
                [&](int64_t a, int64_t b) {
                    bool av = typed->IsValid(a), bv = typed->IsValid(b);
                    if (!av && !bv) return false;
                    if (!av) return false;
                    if (!bv) return true;
                    if (asc) return typed->GetView(a) < typed->GetView(b);
                    return typed->GetView(a) > typed->GetView(b);
                });
        };

        switch (arr->type_id()) {
            case arrow::Type::INT32:  run(arrow::Int32Type{});  break;
            case arrow::Type::INT64:  run(arrow::Int64Type{});  break;
            case arrow::Type::FLOAT:  run(arrow::FloatType{});  break;
            case arrow::Type::DOUBLE: run(arrow::DoubleType{}); break;
            case arrow::Type::STRING: run(arrow::StringType{}); break;
            case arrow::Type::BOOL:   run(arrow::BooleanType{}); break;
            default: throw std::runtime_error("sort_top_n: unsupported type");
        }
    }

    indices.resize(n);

    std::vector<std::shared_ptr<arrow::ChunkedArray>> taken_columns;
    for (int i = 0; i < table_->num_columns(); ++i) {
        auto col_array = table_->column(i)->chunk(0);
        std::shared_ptr<arrow::Array> taken_arr;

        switch (col_array->type_id()) {
            case arrow::Type::INT32:  taken_arr = manual_take<arrow::Int32Type>(col_array, indices);  break;
            case arrow::Type::INT64:  taken_arr = manual_take<arrow::Int64Type>(col_array, indices);  break;
            case arrow::Type::FLOAT:  taken_arr = manual_take<arrow::FloatType>(col_array, indices);  break;
            case arrow::Type::DOUBLE: taken_arr = manual_take<arrow::DoubleType>(col_array, indices); break;
            case arrow::Type::STRING: taken_arr = manual_take<arrow::StringType>(col_array, indices); break;
            case arrow::Type::BOOL:   taken_arr = manual_take<arrow::BooleanType>(col_array, indices); break;
            default: throw std::runtime_error("sort_top_n: unsupported take type");
        }
        taken_columns.push_back(std::make_shared<arrow::ChunkedArray>(taken_arr));
    }
    return EagerDataFrame(arrow::Table::Make(table_->schema(), taken_columns));
}

}
