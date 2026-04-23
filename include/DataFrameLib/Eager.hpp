#pragma once

#include <arrow/api.h>
#include <string>
#include <vector>
#include <memory>
#include <unordered_map>
#include "DataFrameLib/Expr.hpp"

class EagerDataFrame; // Forward declaration for the GroupedDataFrame return type

class GroupedDataFrame {
private:
    std::shared_ptr<arrow::Table> table_;
    std::vector<std::string> keys_;

public:
    GroupedDataFrame(std::shared_ptr<arrow::Table> table, std::vector<std::string> keys)
        : table_(std::move(table)), keys_(std::move(keys)) {}

    // The map is { "column_name" : "function_name" } e.g., {"salary": "mean"}
    EagerDataFrame aggregate(const std::unordered_map<std::string, std::string>& agg_map) const;
};

class EagerDataFrame {
    friend class GroupedDataFrame;
private:
    std::shared_ptr<arrow::Table> table_;

    // Private constructor: enforce building through I/O methods
    explicit EagerDataFrame(std::shared_ptr<arrow::Table> table) : table_(std::move(table)) {}

public:
    // --- I/O Functions ---
    static EagerDataFrame read_csv(const std::string& path);
    static EagerDataFrame read_parquet(const std::string& path);
    
    // (You will implement these two later)
    void write_csv(const std::string& path) const;
    void write_parquet(const std::string& path) const;

    // --- Missing Single-Table Ops ---
    EagerDataFrame sort(const std::vector<std::string>& columns, bool asc) const;
    EagerDataFrame head(int64_t n) const;

    // --- Core Operations ---
    EagerDataFrame select(const std::vector<std::string>& columns) const;
    EagerDataFrame filter(const ExprPtr& predicate) const;
    EagerDataFrame with_column(const std::string& name, const ExprPtr& expr) const;
    static EagerDataFrame from_columns(const std::unordered_map<std::string, std::shared_ptr<arrow::Array>>& col_map);

    // --- Complex Relational Ops ---
    EagerDataFrame join(const EagerDataFrame& other, const std::string& on, const std::string& how) const;
    GroupedDataFrame group_by(const std::vector<std::string>& keys) const {
        return GroupedDataFrame(table_, keys);
    }

    // --- Utilities & Getters ---
    int64_t num_rows() const { return table_ ? table_->num_rows() : 0; }
    int64_t num_columns() const { return table_ ? table_->num_columns() : 0; }
    
    // Crucial for the Lazy Engine to extract data later
    std::shared_ptr<arrow::Table> get_table() const { return table_; }

    // Quick helper to print the table for debugging
    void print() const; 
};