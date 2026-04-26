#pragma once

#include <arrow/api.h>
#include <string>
#include <vector>
#include <memory>
#include <unordered_map>
#include <utility>
#include "ExpressionSystem.h"

namespace dataframelib {

class EagerDataFrame;

// (column_name, function_name) e.g. {"salary", "mean"}.
using AggSpec = std::vector<std::pair<std::string, std::string>>;

class GroupedDataFrame {
private:
    std::shared_ptr<arrow::Table> table_;
    std::vector<std::string> keys_;

public:
    GroupedDataFrame(std::shared_ptr<arrow::Table> table, std::vector<std::string> keys)
        : table_(std::move(table)), keys_(std::move(keys)) {}

    EagerDataFrame aggregate(const AggSpec& aggs) const;
};

class EagerDataFrame {
    friend class GroupedDataFrame;
private:
    std::shared_ptr<arrow::Table> table_;

public:
    explicit EagerDataFrame(std::shared_ptr<arrow::Table> table) : table_(std::move(table)) {}

    static EagerDataFrame read_csv(const std::string& path);
    static EagerDataFrame read_parquet(const std::string& path);

    void write_csv(const std::string& path) const;
    void write_parquet(const std::string& path) const;

    EagerDataFrame sort(const std::vector<std::string>& columns, bool asc) const;
    EagerDataFrame head(int64_t n) const;

    EagerDataFrame select(const std::vector<std::string>& columns) const;
    EagerDataFrame filter(const Expr& predicate) const;
    EagerDataFrame with_column(const std::string& name, const Expr& expr) const;
    static EagerDataFrame from_columns(const std::unordered_map<std::string, std::shared_ptr<arrow::Array>>& col_map);

    EagerDataFrame join(const EagerDataFrame& other, const std::vector<std::string>& on, const std::string& how) const;
    GroupedDataFrame group_by(const std::vector<std::string>& keys) const {
        return GroupedDataFrame(table_, keys);
    }

    int64_t num_rows() const { return table_ ? table_->num_rows() : 0; }
    int64_t num_columns() const { return table_ ? table_->num_columns() : 0; }

    std::shared_ptr<arrow::Table> get_table() const { return table_; }

    void print() const;
};

}
