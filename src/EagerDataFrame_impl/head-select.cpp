#include "../EagerDataFrame.h"
#include <stdexcept>

namespace dataframelib {

EagerDataFrame EagerDataFrame::select(const std::vector<std::string>& columns) const {
    std::vector<std::shared_ptr<arrow::Field>> new_fields;
    std::vector<std::shared_ptr<arrow::ChunkedArray>> new_columns;

    for (const auto& col_name : columns) {
        auto field = table_->schema()->GetFieldByName(col_name);
        auto chunked_array = table_->GetColumnByName(col_name);

        if (!field || !chunked_array) {
            throw std::runtime_error("Select Error: Column '" + col_name + "' not found.");
        }

        new_fields.push_back(field);
        new_columns.push_back(chunked_array);
    }

    auto new_schema = std::make_shared<arrow::Schema>(new_fields);
    auto new_table = arrow::Table::Make(new_schema, new_columns);
    return EagerDataFrame(new_table);
}

EagerDataFrame EagerDataFrame::head(int64_t n) const {
    if (n >= table_->num_rows()) return *this;
    return EagerDataFrame(table_->Slice(0, n));
}

}
