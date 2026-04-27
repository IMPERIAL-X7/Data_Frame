// filter() + print() — operations that produce a "view" of the table.
//
// filter is mask-driven: evaluate the predicate to a BooleanArray, then
// iterate every column and copy rows where the mask is true. We dispatch
// per-column on Arrow type so each pass is a tight typed loop.

#include "../EagerDataFrame.h"
#include <arrow/builder.h>
#include <arrow/type_traits.h>
#include <arrow/pretty_print.h>
#include <iostream>
#include <stdexcept>

namespace dataframelib {

namespace {

// Manual mask-driven filter. Iterates the array and keeps rows where the
// boolean mask is valid and true.
template <typename ArrowType>
std::shared_ptr<arrow::Array> manual_filter_array(
    const std::shared_ptr<arrow::Array>& array,
    const std::shared_ptr<arrow::BooleanArray>& mask) {
    using ArrayType = typename arrow::TypeTraits<ArrowType>::ArrayType;
    using BuilderType = typename arrow::TypeTraits<ArrowType>::BuilderType;

    auto typed_array = std::static_pointer_cast<ArrayType>(array);
    BuilderType builder;

    int64_t num_true = 0;
    for (int64_t i = 0; i < mask->length(); ++i) {
        if (mask->IsValid(i) && mask->Value(i)) num_true++;
    }
    ARROW_UNUSED(builder.Reserve(num_true));

    for (int64_t i = 0; i < array->length(); ++i) {
        if (mask->IsValid(i) && mask->Value(i)) {
            if (typed_array->IsValid(i)) {
                ARROW_UNUSED(builder.Append(typed_array->GetView(i)));
            } else {
                ARROW_UNUSED(builder.AppendNull());
            }
        }
    }

    std::shared_ptr<arrow::Array> result;
    ARROW_UNUSED(builder.Finish(&result));
    return result;
}

}

EagerDataFrame EagerDataFrame::filter(const Expr& predicate) const {
    auto mask_datum = predicate.evaluate(table_);
    auto mask_array = std::static_pointer_cast<arrow::BooleanArray>(mask_datum.make_array());

    std::vector<std::shared_ptr<arrow::ChunkedArray>> filtered_columns;

    for (int col_idx = 0; col_idx < table_->num_columns(); ++col_idx) {
        auto array = table_->column(col_idx)->chunk(0);
        std::shared_ptr<arrow::Array> filtered_arr;

        switch (array->type_id()) {
            case arrow::Type::INT32: filtered_arr = manual_filter_array<arrow::Int32Type>(array, mask_array); break;
            case arrow::Type::INT64: filtered_arr = manual_filter_array<arrow::Int64Type>(array, mask_array); break;
            case arrow::Type::FLOAT: filtered_arr = manual_filter_array<arrow::FloatType>(array, mask_array); break;
            case arrow::Type::DOUBLE: filtered_arr = manual_filter_array<arrow::DoubleType>(array, mask_array); break;
            case arrow::Type::STRING: filtered_arr = manual_filter_array<arrow::StringType>(array, mask_array); break;
            case arrow::Type::BOOL: filtered_arr = manual_filter_array<arrow::BooleanType>(array, mask_array); break;
            default: throw std::runtime_error("Unsupported type for manual filter");
        }

        filtered_columns.push_back(std::make_shared<arrow::ChunkedArray>(filtered_arr));
    }

    auto filtered_table = arrow::Table::Make(table_->schema(), filtered_columns);
    return EagerDataFrame(filtered_table);
}

void EagerDataFrame::print() const {
    arrow::PrettyPrintOptions options;
    ARROW_UNUSED(arrow::PrettyPrint(*table_, options, &std::cout));
    std::cout << std::endl;
}

}
