#include "DataFrameLib/Eager.hpp"
#include <arrow/compute/api.h>
#include <arrow/array/util.h>
#include <arrow/pretty_print.h>
#include <iostream>

// ---------------------------------------------------------
// Core Operations
// ---------------------------------------------------------

EagerDataFrame EagerDataFrame::select(const std::vector<std::string>& columns) const {
    std::vector<int> indices;
    for (const auto& col : columns) {
        int field_idx = table_->schema()->GetFieldIndex(col);
        if (field_idx == -1) {
            throw std::runtime_error("Select Error: Column '" + col + "' not found.");
        }
        indices.push_back(field_idx);
    }
    
    // SelectColumns is zero-copy.
    auto result = table_->SelectColumns(indices);
    if (!result.ok()) throw std::runtime_error(result.status().ToString());
    
    return EagerDataFrame(*result);
}

EagerDataFrame EagerDataFrame::filter(const ExprPtr& predicate) const {
    // Evaluate the AST to get a boolean mask
    auto mask_datum = predicate->evaluate(table_);
    
    auto filter_options = arrow::compute::FilterOptions::Defaults();
    auto result = arrow::compute::Filter(table_, mask_datum, filter_options);
    
    if (!result.ok()) throw std::runtime_error("Filter Error: " + result.status().ToString());
    
    return EagerDataFrame(result->table());
}

EagerDataFrame EagerDataFrame::with_column(const std::string& name, const ExprPtr& expr) const {
    auto datum = expr->evaluate(table_);
    std::shared_ptr<arrow::ChunkedArray> new_col;
    
    if (datum.is_chunked_array()) {
        new_col = datum.chunked_array();
    } else if (datum.is_array()) {
        new_col = std::make_shared<arrow::ChunkedArray>(datum.make_array());
    } else if (datum.is_scalar()) {
        auto array_res = arrow::MakeArrayFromScalar(*datum.scalar(), table_->num_rows());
        if (!array_res.ok()) throw std::runtime_error("Scalar broadcast failed: " + array_res.status().ToString());
        new_col = std::make_shared<arrow::ChunkedArray>(*array_res);
    } else {
        throw std::runtime_error("Expression did not return a valid column format");
    }

    auto field = arrow::field(name, new_col->type());
    int num_cols = table_->num_columns();
    auto result = table_->AddColumn(num_cols, field, new_col);

    if (!result.ok()) throw std::runtime_error("AddColumn Error: " + result.status().ToString());

    return EagerDataFrame(*result);
}

EagerDataFrame EagerDataFrame::sort(const std::vector<std::string>& columns, bool asc) const {
    std::vector<arrow::compute::SortKey> sort_keys;
    auto order = asc ? arrow::compute::SortOrder::Ascending : arrow::compute::SortOrder::Descending;
    
    for (const auto& col : columns) {
        sort_keys.emplace_back(col, order);
    }
    
    arrow::compute::SortOptions options(sort_keys);
    
    // Step 1: Get the sorted indices
    auto indices = arrow::compute::SortIndices(table_, options);
    if (!indices.ok()) throw std::runtime_error("Sort Error: " + indices.status().ToString());

    // Step 2: Reorder the table using those indices
    auto sorted_table = arrow::compute::Take(table_, *indices);
    if (!sorted_table.ok()) throw std::runtime_error("Take Error: " + sorted_table.status().ToString());

    return EagerDataFrame(sorted_table->table());
}

EagerDataFrame EagerDataFrame::head(int64_t n) const {
    // Arrow's Slice is an O(1) zero-copy operation
    if (n >= table_->num_rows()) return *this; 
    return EagerDataFrame(table_->Slice(0, n));
}

// ---------------------------------------------------------
// Debugging Utility
// ---------------------------------------------------------
void EagerDataFrame::print() const {
    arrow::PrettyPrintOptions options;
    arrow::PrettyPrint(*table_, options, &std::cout);
    std::cout << std::endl;
}