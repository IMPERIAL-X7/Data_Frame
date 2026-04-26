#include "DataFrameLib/Eager.hpp"
#include <arrow/builder.h>
#include <arrow/type_traits.h>
// #include <arrow/array/util.h>
#include <arrow/pretty_print.h>
#include <iostream>
#include <stdexcept>
#include <numeric>
#include <algorithm>
#include <unordered_map>
#include <map>
#include <string_view>
#include <set>

// This template manually iterates over an array and filters it based on a boolean mask
template <typename ArrowType>
std::shared_ptr<arrow::Array> manual_filter_array(
    const std::shared_ptr<arrow::Array>& array, 
    const std::shared_ptr<arrow::BooleanArray>& mask) 
{
    // Fetch the correct C++ array and builder types for the given ArrowType
    using ArrayType = typename arrow::TypeTraits<ArrowType>::ArrayType;
    using BuilderType = typename arrow::TypeTraits<ArrowType>::BuilderType;

    // Cast the raw memory buffer to our specific type
    auto typed_array = std::static_pointer_cast<ArrayType>(array);
    BuilderType builder;
    
    // We only allocate memory for the rows we know we are keeping
    int64_t num_true = 0;
    for (int64_t i = 0; i < mask->length(); ++i) {
        if (mask->IsValid(i) && mask->Value(i)) num_true++;
    }
    ARROW_UNUSED(builder.Reserve(num_true)); // Pre-allocate for speed

    // The O(N) manual scan
    for (int64_t i = 0; i < array->length(); ++i) {
        // If the mask is true at this row
        if (mask->IsValid(i) && mask->Value(i)) {
            // Append the value, or append null if it was missing data
            if (typed_array->IsValid(i)) {
                builder.Append(typed_array->GetView(i));
            } else {
                builder.AppendNull();
            }
        }
    }
    
    std::shared_ptr<arrow::Array> result;
    ARROW_UNUSED(builder.Finish(&result));
    return result;
}

//Manual Sorters
// 1. Sorts an array and returns the rearranged row indices
template <typename ArrowType>
std::vector<int64_t> manual_get_sorted_indices(const std::shared_ptr<arrow::Array>& array, bool asc) {
    using ArrayType = typename arrow::TypeTraits<ArrowType>::ArrayType;
    auto typed_array = std::static_pointer_cast<ArrayType>(array);
    
    std::vector<int64_t> indices(array->length());
    std::iota(indices.begin(), indices.end(), 0); // Fill with 0, 1, 2...
    
    std::sort(indices.begin(), indices.end(), [&](int64_t a, int64_t b) {
        bool a_valid = typed_array->IsValid(a);
        bool b_valid = typed_array->IsValid(b);
        if (!a_valid && !b_valid) return false;
        if (!a_valid) return false; // Nulls go to the bottom
        if (!b_valid) return true;
        
        if (asc) return typed_array->GetView(a) < typed_array->GetView(b);
        return typed_array->GetView(a) > typed_array->GetView(b);
    });
    return indices;
}

// 2. Rebuilds a column using the sorted indices
template <typename ArrowType>
std::shared_ptr<arrow::Array> manual_take(const std::shared_ptr<arrow::Array>& array, const std::vector<int64_t>& indices) {
    using ArrayType = typename arrow::TypeTraits<ArrowType>::ArrayType;
    using BuilderType = typename arrow::TypeTraits<ArrowType>::BuilderType;
    
    auto typed_array = std::static_pointer_cast<ArrayType>(array);
    BuilderType builder;
    ARROW_UNUSED(builder.Reserve(indices.size()));
    
    for (int64_t idx : indices) {
        if (idx >= 0 && typed_array->IsValid(idx)) builder.Append(typed_array->GetView(idx));
        else builder.AppendNull(); // Handle the -1 nulls from outer/left joins
    }
    
    std::shared_ptr<arrow::Array> result;
    ARROW_UNUSED(builder.Finish(&result));
    return result;
}


// Manual Hash Join Engine
// Helper to handle string_view hashing in C++17 unordered_maps
struct StringViewHash {
    std::size_t operator()(std::string_view sv) const { return std::hash<std::string_view>{}(sv); }
};

template <typename ArrowType>
void manual_compute_join_indices(
    const std::shared_ptr<arrow::Array>& left_key,
    const std::shared_ptr<arrow::Array>& right_key,
    const std::string& how,
    std::vector<int64_t>& left_indices,
    std::vector<int64_t>& right_indices)
{
    using ArrayType = typename arrow::TypeTraits<ArrowType>::ArrayType;
    auto left_typed = std::static_pointer_cast<ArrayType>(left_key);
    auto right_typed = std::static_pointer_cast<ArrayType>(right_key);

    // We use a generalized map. (Using conditional types to handle string_view safely)
    using KeyType = decltype(left_typed->GetView(0));
    
    // Fallback to std::map for safety across all types and C++ versions
    std::map<KeyType, std::vector<int64_t>> hash_table;

    // 1. BUILD PHASE (Hash the Right Table)
    for (int64_t i = 0; i < right_typed->length(); ++i) {
        if (right_typed->IsValid(i)) {
            hash_table[right_typed->GetView(i)].push_back(i);
        }
    }

    std::set<int64_t> matched_right_indices;

    // 2. PROBE PHASE (Scan the Left Table)
    for (int64_t i = 0; i < left_typed->length(); ++i) {
        if (!left_typed->IsValid(i)) continue; // Skip nulls for standard joins

        KeyType key = left_typed->GetView(i);
        auto it = hash_table.find(key);

        if (it != hash_table.end()) {
            // Match found! Append all matching right rows
            for (int64_t right_idx : it->second) {
                left_indices.push_back(i);
                right_indices.push_back(right_idx);
                matched_right_indices.insert(right_idx);
            }
        } else if (how == "left" || how == "outer") {
            // No match, but we must keep the left row
            left_indices.push_back(i);
            right_indices.push_back(-1); // -1 represents a null row
        }
    }

    // 3. OUTER JOIN COMPLETION (Add unmatched Right rows)
    if (how == "outer") {
        for (int64_t i = 0; i < right_typed->length(); ++i) {
            if (right_typed->IsValid(i) && matched_right_indices.find(i) == matched_right_indices.end()) {
                left_indices.push_back(-1); // -1 represents a null row
                right_indices.push_back(i);
            }
        }
    }
}

// Visitor to dynamically dispatch the join based on data type
struct JoinIndicesVisitor {
    std::shared_ptr<arrow::Array> left_array;
    std::shared_ptr<arrow::Array> right_array;
    std::string how;
    std::vector<int64_t> left_indices;
    std::vector<int64_t> right_indices;

    template <typename T> arrow::Status Visit(const T&) {
        manual_compute_join_indices<T>(left_array, right_array, how, left_indices, right_indices);
        return arrow::Status::OK();
    }
};

// Core Operations
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

EagerDataFrame EagerDataFrame::filter(const ExprPtr& predicate) const {
    auto mask_datum = predicate->evaluate(table_);
    auto mask_array = std::static_pointer_cast<arrow::BooleanArray>(mask_datum.make_array());

    std::vector<std::shared_ptr<arrow::ChunkedArray>> filtered_columns;

    for (int col_idx = 0; col_idx < table_->num_columns(); ++col_idx) {
        auto array = table_->column(col_idx)->chunk(0); 
        std::shared_ptr<arrow::Array> filtered_arr;

        // SAFE DISPATCH: Only instantiate templates for assignment-supported types
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

EagerDataFrame EagerDataFrame::sort(const std::vector<std::string>& columns, bool asc) const {
    if (columns.empty() || table_->num_rows() == 0) return *this;

    auto sort_col_chunked = table_->GetColumnByName(columns[0]);
    if (!sort_col_chunked) throw std::runtime_error("Sort column not found");
    auto sort_array = sort_col_chunked->chunk(0);

    std::vector<int64_t> sorted_indices;

    // SAFE DISPATCH: Get Indices
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

        // SAFE DISPATCH: Take Data
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

EagerDataFrame EagerDataFrame::head(int64_t n) const {
    // Arrow's Slice is an O(1) zero-copy operation
    if (n >= table_->num_rows()) return *this; 
    return EagerDataFrame(table_->Slice(0, n));
}

EagerDataFrame EagerDataFrame::join(const EagerDataFrame& other, const std::string& on, const std::string& how) const {
    // 1. Extract the join keys
    auto left_key_chunk = table_->GetColumnByName(on);
    auto right_key_chunk = other.get_table()->GetColumnByName(on);
    
    if (!left_key_chunk || !right_key_chunk) {
        throw std::runtime_error("Join column '" + on + "' not found in one or both tables");
    }

    auto left_array = left_key_chunk->chunk(0);
    auto right_array = right_key_chunk->chunk(0);

    if (left_array->type_id() != right_array->type_id()) {
        throw std::runtime_error("Join column types do not match");
    }

    // 2. Dispatch the Visitor to get the matching indices
    JoinIndicesVisitor visitor{left_array, right_array, how, {}, {}};
    
    switch (left_array->type_id()) {
        case arrow::Type::INT32: visitor.Visit(arrow::Int32Type()); break;
        case arrow::Type::INT64: visitor.Visit(arrow::Int64Type()); break;
        case arrow::Type::FLOAT: visitor.Visit(arrow::FloatType()); break;
        case arrow::Type::DOUBLE: visitor.Visit(arrow::DoubleType()); break;
        case arrow::Type::STRING: visitor.Visit(arrow::StringType()); break;
        case arrow::Type::BOOL: visitor.Visit(arrow::BooleanType()); break;
        default: throw std::runtime_error("Unsupported type for manual join");
    }

    // 3. Rebuild columns from the Left Table using left_indices
    std::vector<std::shared_ptr<arrow::Field>> joined_fields;
    std::vector<std::shared_ptr<arrow::ChunkedArray>> joined_columns;

    for (int i = 0; i < table_->num_columns(); ++i) {
        auto col_array = table_->column(i)->chunk(0);
        std::shared_ptr<arrow::Array> taken_arr;
        
        switch (col_array->type_id()) {
            case arrow::Type::INT32: taken_arr = manual_take<arrow::Int32Type>(col_array, visitor.left_indices); break;
            case arrow::Type::INT64: taken_arr = manual_take<arrow::Int64Type>(col_array, visitor.left_indices); break;
            case arrow::Type::FLOAT: taken_arr = manual_take<arrow::FloatType>(col_array, visitor.left_indices); break;
            case arrow::Type::DOUBLE: taken_arr = manual_take<arrow::DoubleType>(col_array, visitor.left_indices); break;
            case arrow::Type::STRING: taken_arr = manual_take<arrow::StringType>(col_array, visitor.left_indices); break;
            case arrow::Type::BOOL: taken_arr = manual_take<arrow::BooleanType>(col_array, visitor.left_indices); break;
            default: throw std::runtime_error("Unsupported type");
        }
        joined_fields.push_back(table_->schema()->field(i));
        joined_columns.push_back(std::make_shared<arrow::ChunkedArray>(taken_arr));
    }

    // 4. Rebuild columns from the Right Table using right_indices (excluding the join key)
    auto right_table = other.get_table();
    for (int i = 0; i < right_table->num_columns(); ++i) {
        if (right_table->schema()->field(i)->name() == on) continue; // Don't duplicate the join key

        auto col_array = right_table->column(i)->chunk(0);
        std::shared_ptr<arrow::Array> taken_arr;
        
        switch (col_array->type_id()) {
            case arrow::Type::INT32: taken_arr = manual_take<arrow::Int32Type>(col_array, visitor.right_indices); break;
            case arrow::Type::INT64: taken_arr = manual_take<arrow::Int64Type>(col_array, visitor.right_indices); break;
            case arrow::Type::FLOAT: taken_arr = manual_take<arrow::FloatType>(col_array, visitor.right_indices); break;
            case arrow::Type::DOUBLE: taken_arr = manual_take<arrow::DoubleType>(col_array, visitor.right_indices); break;
            case arrow::Type::STRING: taken_arr = manual_take<arrow::StringType>(col_array, visitor.right_indices); break;
            case arrow::Type::BOOL: taken_arr = manual_take<arrow::BooleanType>(col_array, visitor.right_indices); break;
            default: throw std::runtime_error("Unsupported type");
        }
        // Rename right columns if there's a collision (optional but good practice)
        std::string col_name = right_table->schema()->field(i)->name();
        if (table_->GetColumnByName(col_name)) col_name += "_right";

        joined_fields.push_back(arrow::field(col_name, col_array->type()));
        joined_columns.push_back(std::make_shared<arrow::ChunkedArray>(taken_arr));
    }

    auto joined_table = arrow::Table::Make(std::make_shared<arrow::Schema>(joined_fields), joined_columns);
    return EagerDataFrame(joined_table);
}

// =========================================================
// 2. Manual Aggregate & GroupBy Implementation
// =========================================================
template <typename ArrowType>
std::shared_ptr<arrow::Array> manual_agg_column(
    const std::shared_ptr<arrow::Array>& array, 
    const std::map<std::string, std::vector<int64_t>>& groups, 
    const std::string& func_name) 
{
    using ArrayType = typename arrow::TypeTraits<ArrowType>::ArrayType;
    using BuilderType = typename arrow::TypeTraits<ArrowType>::BuilderType;
    using CType = typename arrow::TypeTraits<ArrowType>::CType;

    auto typed_array = std::static_pointer_cast<ArrayType>(array);
    BuilderType builder;
    
    for (const auto& [grp_name, indices] : groups) {
        if (func_name == "count") {
            int64_t count = 0;
            for (int64_t idx : indices) if (typed_array->IsValid(idx)) count++;
            builder.Append(static_cast<CType>(count));
        } 
        else if (func_name == "sum" || func_name == "mean") {
            double sum = 0; int64_t count = 0;
            for (int64_t idx : indices) {
                if (typed_array->IsValid(idx)) { sum += typed_array->GetView(idx); count++; }
            }
            if (count == 0) builder.AppendNull();
            else if (func_name == "mean") builder.Append(static_cast<CType>(sum / count)); // TA Clarification: return original type
            else builder.Append(static_cast<CType>(sum));
        }
        else builder.AppendNull();
    }
    
    std::shared_ptr<arrow::Array> result;
    ARROW_UNUSED(builder.Finish(&result));
    return result;
}

EagerDataFrame GroupedDataFrame::aggregate(const std::unordered_map<std::string, std::string>& agg_map) const {
    if (keys_.empty()) throw std::runtime_error("No group by keys specified");
    std::string key_name = keys_[0]; // Handling 1 key for simplicity
    
    auto key_array = table_->GetColumnByName(key_name)->chunk(0);
    std::map<std::string, std::vector<int64_t>> groups;

    // 1. Grouping Phase (Map keys to row indices)
    if (key_array->type_id() == arrow::Type::STRING) {
        auto typed_key = std::static_pointer_cast<arrow::StringArray>(key_array);
        for (int64_t i = 0; i < typed_key->length(); ++i) {
            if (typed_key->IsValid(i)) groups[typed_key->GetString(i)].push_back(i);
        }
    } else {
        throw std::runtime_error("Manual GroupBy currently only supports String keys");
    }

    // 2. Build Output Key Column
    arrow::StringBuilder key_builder;
    for (const auto& [grp_name, indices] : groups) ARROW_UNUSED(key_builder.Append(grp_name));
    std::shared_ptr<arrow::Array> out_key_array;
    ARROW_UNUSED(key_builder.Finish(&out_key_array));

    std::vector<std::shared_ptr<arrow::Field>> out_fields = {arrow::field(key_name, arrow::utf8())};
    std::vector<std::shared_ptr<arrow::ChunkedArray>> out_columns = {std::make_shared<arrow::ChunkedArray>(out_key_array)};

    // 3. Aggregation Phase
    for (const auto& [col_name, func] : agg_map) {
        auto val_array = table_->GetColumnByName(col_name)->chunk(0);
        std::shared_ptr<arrow::Array> agg_arr;
        
        switch (val_array->type_id()) {
            case arrow::Type::INT32: agg_arr = manual_agg_column<arrow::Int32Type>(val_array, groups, func); break;
            case arrow::Type::INT64: agg_arr = manual_agg_column<arrow::Int64Type>(val_array, groups, func); break;
            case arrow::Type::FLOAT: agg_arr = manual_agg_column<arrow::FloatType>(val_array, groups, func); break;
            case arrow::Type::DOUBLE: agg_arr = manual_agg_column<arrow::DoubleType>(val_array, groups, func); break;
            default: throw std::runtime_error("Unsupported type for manual aggregation");
        }

        std::string out_col_name = col_name + "_" + func;
        out_fields.push_back(arrow::field(out_col_name, val_array->type()));
        out_columns.push_back(std::make_shared<arrow::ChunkedArray>(agg_arr));
    }

    return EagerDataFrame(arrow::Table::Make(std::make_shared<arrow::Schema>(out_fields), out_columns));
}
// // ---------------------------------------------------------
// // Debugging Utility
// // ---------------------------------------------------------
void EagerDataFrame::print() const {
    arrow::PrettyPrintOptions options;
    arrow::PrettyPrint(*table_, options, &std::cout);
    std::cout << std::endl;
}