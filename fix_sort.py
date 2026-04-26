import re

with open("/home/imperial-x/Documents/GitHub/Data_Frame/src/EagerDataFrame_impl/with-column-sort.cpp", "r") as f:
    text = f.read()

new_code = """
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
"""

text = re.sub(r'template <typename ArrowType>\nstd::vector<int64_t> manual_get_sorted_indices.*?return indices;\n}', new_code, text, flags=re.DOTALL)

sort_func = """
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
"""

text = re.sub(r'EagerDataFrame EagerDataFrame::sort\(.*?for \(int i = 0; i < table_->num_columns\(\); \+\+i\) \{', sort_func, text, flags=re.DOTALL)

with open("/home/imperial-x/Documents/GitHub/Data_Frame/src/EagerDataFrame_impl/with-column-sort.cpp", "w") as f:
    f.write(text)
