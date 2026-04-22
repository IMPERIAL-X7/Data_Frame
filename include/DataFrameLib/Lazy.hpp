#pragma once

#include "DataFrameLib/Expr.hpp"
#include <memory>
#include <string>
#include <vector>

// ---------------------------------------------------------
// Logical Plan Nodes (The DAG)
// ---------------------------------------------------------
class LogicalNode {
public:
    virtual ~LogicalNode() = default;
    
    // Every node must be able to describe itself for Graphviz
    virtual std::string to_string() const = 0;
    
    // Returns child dependencies (empty for leaf nodes like Scan)
    virtual std::vector<std::shared_ptr<LogicalNode>> children() const = 0;
};

// Leaf Node: Represents reading a file
class ScanNode : public LogicalNode {
    std::string path_;
    std::string format_; // "csv" or "parquet"
public:
    ScanNode(std::string path, std::string format) 
        : path_(std::move(path)), format_(std::move(format)) {}
        
    std::string to_string() const override { return "Scan(" + format_ + ": " + path_ + ")"; }
    std::vector<std::shared_ptr<LogicalNode>> children() const override { return {}; }
    
    const std::string& path() const { return path_; }
    const std::string& format() const { return format_; }
};

// Internal Node: Represents a filter operation
class FilterNode : public LogicalNode {
    std::shared_ptr<LogicalNode> input_;
    ExprPtr predicate_;
public:
    FilterNode(std::shared_ptr<LogicalNode> input, ExprPtr predicate)
        : input_(std::move(input)), predicate_(std::move(predicate)) {}

    std::string to_string() const override { return "Filter(" + predicate_->to_string() + ")"; }
    std::vector<std::shared_ptr<LogicalNode>> children() const override { return {input_}; }
    
    ExprPtr predicate() const { return predicate_; }
    std::shared_ptr<LogicalNode> input() const { return input_; }
};

// Internal Node: Represents a select operation
class ProjectNode : public LogicalNode {
    std::shared_ptr<LogicalNode> input_;
    std::vector<std::string> columns_;
public:
    ProjectNode(std::shared_ptr<LogicalNode> input, std::vector<std::string> columns)
        : input_(std::move(input)), columns_(std::move(columns)) {}

    std::string to_string() const override; // You'll implement this string joining in the .cpp
    std::vector<std::shared_ptr<LogicalNode>> children() const override { return {input_}; }
};

// ---------------------------------------------------------
// The LazyDataFrame API
// ---------------------------------------------------------
class LazyDataFrame {
private:
    std::shared_ptr<LogicalNode> logical_plan_;

    explicit LazyDataFrame(std::shared_ptr<LogicalNode> plan) 
        : logical_plan_(std::move(plan)) {}

public:
    // I/O: These create ScanNodes
    static LazyDataFrame scan_csv(const std::string& path);
    static LazyDataFrame scan_parquet(const std::string& path);

    // Operations: These wrap the existing plan in a new Node
    LazyDataFrame filter(const ExprPtr& predicate) const;
    LazyDataFrame select(const std::vector<std::string>& columns) const;

    // Execution & Debugging
    void explain(const std::string& plan_path) const;
    
    // EagerDataFrame collect() const; // We will build this in the Optimizer phase!
};