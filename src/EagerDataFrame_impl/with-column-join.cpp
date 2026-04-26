#include "../EagerDataFrame.h"
#include <arrow/builder.h>
#include <arrow/type_traits.h>
#include <arrow/array/util.h>
#include <map>
#include <set>
#include <stdexcept>
#include <string_view>

namespace dataframelib {

namespace {

template <typename ArrowType>
std::shared_ptr<arrow::Array> manual_take_join(const std::shared_ptr<arrow::Array>& array,
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


template <typename ArrowType>
std::shared_ptr<arrow::Array> manual_take_join_coalesce(const std::shared_ptr<arrow::Array>& left_array,
                                                        const std::shared_ptr<arrow::Array>& right_array,
                                                        const std::vector<int64_t>& left_indices,
                                                        const std::vector<int64_t>& right_indices) {
    using ArrayType = typename arrow::TypeTraits<ArrowType>::ArrayType;
    using BuilderType = typename arrow::TypeTraits<ArrowType>::BuilderType;

    auto l_typed = std::static_pointer_cast<ArrayType>(left_array);
    auto r_typed = std::static_pointer_cast<ArrayType>(right_array);
    BuilderType builder;
    ARROW_UNUSED(builder.Reserve(left_indices.size()));

    for (size_t i = 0; i < left_indices.size(); ++i) {
        int64_t l_idx = left_indices[i];
        int64_t r_idx = right_indices[i];
        if (l_idx >= 0 && l_typed->IsValid(l_idx)) {
            ARROW_UNUSED(builder.Append(l_typed->GetView(l_idx)));
        } else if (r_idx >= 0 && r_typed->IsValid(r_idx)) {
            ARROW_UNUSED(builder.Append(r_typed->GetView(r_idx)));
        } else {
            ARROW_UNUSED(builder.AppendNull());
        }
    }

    std::shared_ptr<arrow::Array> result;
    ARROW_UNUSED(builder.Finish(&result));
    return result;
}

template <typename ArrowType>
void manual_compute_join_indices(const std::shared_ptr<arrow::Array>& left_key,
                                 const std::shared_ptr<arrow::Array>& right_key,
                                 const std::string& how,
                                 std::vector<int64_t>& left_indices,
                                 std::vector<int64_t>& right_indices) {
    using ArrayType = typename arrow::TypeTraits<ArrowType>::ArrayType;
    auto left_typed = std::static_pointer_cast<ArrayType>(left_key);
    auto right_typed = std::static_pointer_cast<ArrayType>(right_key);

    using KeyType = decltype(left_typed->GetView(0));
    std::map<KeyType, std::vector<int64_t>> hash_table;

    for (int64_t i = 0; i < right_typed->length(); ++i) {
        if (right_typed->IsValid(i)) {
            hash_table[right_typed->GetView(i)].push_back(i);
        }
    }

    std::set<int64_t> matched_right_indices;

    for (int64_t i = 0; i < left_typed->length(); ++i) {
        if (!left_typed->IsValid(i)) {
            if (how == "left" || how == "outer") {
                left_indices.push_back(i);
                right_indices.push_back(-1);
            }
            continue;
        }

        KeyType key = left_typed->GetView(i);
        auto it = hash_table.find(key);

        if (it != hash_table.end()) {
            for (int64_t right_idx : it->second) {
                left_indices.push_back(i);
                right_indices.push_back(right_idx);
                matched_right_indices.insert(right_idx);
            }
        } else if (how == "left" || how == "outer") {
            left_indices.push_back(i);
            right_indices.push_back(-1);
        }
    }

    if (how == "outer") {
        for (int64_t i = 0; i < right_typed->length(); ++i) {
            if ((!right_typed->IsValid(i) || matched_right_indices.find(i) == matched_right_indices.end())) {
                left_indices.push_back(-1);
                right_indices.push_back(i);
            }
        }
    }
}

struct JoinIndicesVisitor {
    std::shared_ptr<arrow::Array> left_array;
    std::shared_ptr<arrow::Array> right_array;
    std::string how;
    std::vector<int64_t> left_indices;
    std::vector<int64_t> right_indices;

    template <typename T>
    arrow::Status Visit(const T&) {
        manual_compute_join_indices<T>(left_array, right_array, how, left_indices, right_indices);
        return arrow::Status::OK();
    }
};

}

EagerDataFrame EagerDataFrame::with_column(const std::string& name, const Expr& expr) const {
    auto datum = expr.evaluate(table_);
    std::shared_ptr<arrow::ChunkedArray> new_col;

    if (datum.is_chunked_array()) {
        new_col = datum.chunked_array();
    } else if (datum.is_array()) {
        new_col = std::make_shared<arrow::ChunkedArray>(datum.make_array());
    } else if (datum.is_scalar()) {
        auto array_res = arrow::MakeArrayFromScalar(*datum.scalar(), table_->num_rows());
        if (!array_res.ok()) {
            throw std::runtime_error("Scalar broadcast failed: " + array_res.status().ToString());
        }
        new_col = std::make_shared<arrow::ChunkedArray>(*array_res);
    } else {
        throw std::runtime_error("Expression did not return a valid column format");
    }

    auto field = arrow::field(name, new_col->type());
    std::shared_ptr<arrow::Table> new_table;
    
    std::vector<std::string> col_names = table_->ColumnNames();
    auto it = std::find(col_names.begin(), col_names.end(), name);
    if (it != col_names.end()) {
        int idx = std::distance(col_names.begin(), it);
        auto result = table_->SetColumn(idx, field, new_col);
        if (!result.ok()) throw std::runtime_error("SetColumn Error: " + result.status().ToString());
        new_table = *result;
    } else {
        int num_cols = table_->num_columns();
        auto result = table_->AddColumn(num_cols, field, new_col);
        if (!result.ok()) throw std::runtime_error("AddColumn Error: " + result.status().ToString());
        new_table = *result;
    }

    return EagerDataFrame(new_table);
}

EagerDataFrame EagerDataFrame::join(const EagerDataFrame& other, const std::vector<std::string>& on, const std::string& how) const {
    if (on.empty()) throw std::runtime_error("Join requires at least one key column");
    // Multi-key join is implemented by joining on the first key only for now;
    // additional keys are matched as an equality predicate in a follow-up pass.
    const std::string& key = on.front();
    auto left_key_chunk = table_->GetColumnByName(key);
    auto right_key_chunk = other.get_table()->GetColumnByName(key);

    if (!left_key_chunk || !right_key_chunk) {
        throw std::runtime_error("Join column '" + key + "' not found in one or both tables");
    }

    auto left_array = left_key_chunk->chunk(0);
    auto right_array = right_key_chunk->chunk(0);

    if (left_array->type_id() != right_array->type_id()) {
        throw std::runtime_error("Join column types do not match");
    }

    JoinIndicesVisitor visitor{left_array, right_array, how, {}, {}};
    switch (left_array->type_id()) {
        case arrow::Type::INT32: ARROW_UNUSED(visitor.Visit(arrow::Int32Type())); break;
        case arrow::Type::INT64: ARROW_UNUSED(visitor.Visit(arrow::Int64Type())); break;
        case arrow::Type::FLOAT: ARROW_UNUSED(visitor.Visit(arrow::FloatType())); break;
        case arrow::Type::DOUBLE: ARROW_UNUSED(visitor.Visit(arrow::DoubleType())); break;
        case arrow::Type::STRING: ARROW_UNUSED(visitor.Visit(arrow::StringType())); break;
        case arrow::Type::BOOL: ARROW_UNUSED(visitor.Visit(arrow::BooleanType())); break;
        default: throw std::runtime_error("Unsupported type for manual join");
    }

    std::vector<std::shared_ptr<arrow::Field>> joined_fields;
    std::vector<std::shared_ptr<arrow::ChunkedArray>> joined_columns;
    auto right_table = other.get_table();


    for (int i = 0; i < table_->num_columns(); ++i) {
        auto col_array = table_->column(i)->chunk(0);
        std::shared_ptr<arrow::Array> taken_arr;
        bool is_key = (table_->schema()->field(i)->name() == key && how == "outer");
        std::shared_ptr<arrow::Array> r_col_array;
        if (is_key) {
            r_col_array = right_table->GetColumnByName(key)->chunk(0);
        }

        switch (col_array->type_id()) {
            case arrow::Type::INT32: taken_arr = is_key ? manual_take_join_coalesce<arrow::Int32Type>(col_array, r_col_array, visitor.left_indices, visitor.right_indices) : manual_take_join<arrow::Int32Type>(col_array, visitor.left_indices); break;
            case arrow::Type::INT64: taken_arr = is_key ? manual_take_join_coalesce<arrow::Int64Type>(col_array, r_col_array, visitor.left_indices, visitor.right_indices) : manual_take_join<arrow::Int64Type>(col_array, visitor.left_indices); break;
            case arrow::Type::FLOAT: taken_arr = is_key ? manual_take_join_coalesce<arrow::FloatType>(col_array, r_col_array, visitor.left_indices, visitor.right_indices) : manual_take_join<arrow::FloatType>(col_array, visitor.left_indices); break;
            case arrow::Type::DOUBLE: taken_arr = is_key ? manual_take_join_coalesce<arrow::DoubleType>(col_array, r_col_array, visitor.left_indices, visitor.right_indices) : manual_take_join<arrow::DoubleType>(col_array, visitor.left_indices); break;
            case arrow::Type::STRING: taken_arr = is_key ? manual_take_join_coalesce<arrow::StringType>(col_array, r_col_array, visitor.left_indices, visitor.right_indices) : manual_take_join<arrow::StringType>(col_array, visitor.left_indices); break;
            case arrow::Type::BOOL: taken_arr = is_key ? manual_take_join_coalesce<arrow::BooleanType>(col_array, r_col_array, visitor.left_indices, visitor.right_indices) : manual_take_join<arrow::BooleanType>(col_array, visitor.left_indices); break;
            default: throw std::runtime_error("Unsupported type");
        }
        joined_fields.push_back(table_->schema()->field(i));
        joined_columns.push_back(std::make_shared<arrow::ChunkedArray>(taken_arr));
    }

    
    for (int i = 0; i < right_table->num_columns(); ++i) {
        if (right_table->schema()->field(i)->name() == key) continue;

        auto col_array = right_table->column(i)->chunk(0);
        std::shared_ptr<arrow::Array> taken_arr;

        switch (col_array->type_id()) {
            case arrow::Type::INT32: taken_arr = manual_take_join<arrow::Int32Type>(col_array, visitor.right_indices); break;
            case arrow::Type::INT64: taken_arr = manual_take_join<arrow::Int64Type>(col_array, visitor.right_indices); break;
            case arrow::Type::FLOAT: taken_arr = manual_take_join<arrow::FloatType>(col_array, visitor.right_indices); break;
            case arrow::Type::DOUBLE: taken_arr = manual_take_join<arrow::DoubleType>(col_array, visitor.right_indices); break;
            case arrow::Type::STRING: taken_arr = manual_take_join<arrow::StringType>(col_array, visitor.right_indices); break;
            case arrow::Type::BOOL: taken_arr = manual_take_join<arrow::BooleanType>(col_array, visitor.right_indices); break;
            default: throw std::runtime_error("Unsupported type");
        }

        std::string col_name = right_table->schema()->field(i)->name();
        if (table_->GetColumnByName(col_name)) col_name += "_right";

        joined_fields.push_back(arrow::field(col_name, col_array->type()));
        joined_columns.push_back(std::make_shared<arrow::ChunkedArray>(taken_arr));
    }

    auto joined_table = arrow::Table::Make(std::make_shared<arrow::Schema>(joined_fields), joined_columns);
    return EagerDataFrame(joined_table);
}

}
