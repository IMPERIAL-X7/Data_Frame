#pragma once

#include "../ExpressionSystem.h"
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace dataframelib {

class LogicalNode {
public:
    virtual ~LogicalNode() = default;
    virtual std::string to_string() const = 0;
    virtual std::vector<std::shared_ptr<LogicalNode>> children() const = 0;
};

class ScanNode : public LogicalNode {
    std::string path_;
    std::string format_;
public:
    ScanNode(std::string path, std::string format)
        : path_(std::move(path)), format_(std::move(format)) {}

    std::string to_string() const override { return "Scan(" + format_ + ": " + path_ + ")"; }
    std::vector<std::shared_ptr<LogicalNode>> children() const override { return {}; }

    const std::string& path() const { return path_; }
    const std::string& format() const { return format_; }
};

class FilterNode : public LogicalNode {
    std::shared_ptr<LogicalNode> input_;
    Expr predicate_;
public:
    FilterNode(std::shared_ptr<LogicalNode> input, Expr predicate)
        : input_(std::move(input)), predicate_(std::move(predicate)) {}

    std::string to_string() const override { return "Filter(" + predicate_.to_string() + ")"; }
    std::vector<std::shared_ptr<LogicalNode>> children() const override { return {input_}; }

    const Expr& predicate() const { return predicate_; }
    std::shared_ptr<LogicalNode> input() const { return input_; }
};

class ProjectNode : public LogicalNode {
    std::shared_ptr<LogicalNode> input_;
    std::vector<std::string> columns_;
public:
    ProjectNode(std::shared_ptr<LogicalNode> input, std::vector<std::string> columns)
        : input_(std::move(input)), columns_(std::move(columns)) {}

    std::string to_string() const override;
    std::vector<std::shared_ptr<LogicalNode>> children() const override { return {input_}; }

    const std::vector<std::string>& columns() const { return columns_; }
    std::shared_ptr<LogicalNode> input() const { return input_; }
};

class WithColumnNode : public LogicalNode {
    std::shared_ptr<LogicalNode> input_;
    std::string name_;
    Expr expr_;
public:
    WithColumnNode(std::shared_ptr<LogicalNode> input, std::string name, Expr expr)
        : input_(std::move(input)), name_(std::move(name)), expr_(std::move(expr)) {}

    std::string to_string() const override { return "WithColumn(" + name_ + " = " + expr_.to_string() + ")"; }
    std::vector<std::shared_ptr<LogicalNode>> children() const override { return {input_}; }

    const std::string& name() const { return name_; }
    const Expr& expr() const { return expr_; }
    std::shared_ptr<LogicalNode> input() const { return input_; }
};

class SortNode : public LogicalNode {
    std::shared_ptr<LogicalNode> input_;
    std::vector<std::string> columns_;
    bool asc_;
public:
    SortNode(std::shared_ptr<LogicalNode> input, std::vector<std::string> columns, bool asc)
        : input_(std::move(input)), columns_(std::move(columns)), asc_(asc) {}

    std::string to_string() const override { return std::string("Sort(") + (asc_ ? "asc" : "desc") + ")"; }
    std::vector<std::shared_ptr<LogicalNode>> children() const override { return {input_}; }

    const std::vector<std::string>& columns() const { return columns_; }
    bool asc() const { return asc_; }
    std::shared_ptr<LogicalNode> input() const { return input_; }
};

class HeadNode : public LogicalNode {
    std::shared_ptr<LogicalNode> input_;
    int64_t n_;
public:
    HeadNode(std::shared_ptr<LogicalNode> input, int64_t n)
        : input_(std::move(input)), n_(n) {}

    std::string to_string() const override { return "Head(" + std::to_string(n_) + ")"; }
    std::vector<std::shared_ptr<LogicalNode>> children() const override { return {input_}; }

    int64_t n() const { return n_; }
    std::shared_ptr<LogicalNode> input() const { return input_; }
};

class JoinNode : public LogicalNode {
    std::shared_ptr<LogicalNode> left_;
    std::shared_ptr<LogicalNode> right_;
    std::vector<std::string> on_;
    std::string how_;
public:
    JoinNode(std::shared_ptr<LogicalNode> left, std::shared_ptr<LogicalNode> right,
             std::vector<std::string> on, std::string how)
        : left_(std::move(left)), right_(std::move(right)),
          on_(std::move(on)), how_(std::move(how)) {}

    std::string to_string() const override { return "Join(" + how_ + ")"; }
    std::vector<std::shared_ptr<LogicalNode>> children() const override { return {left_, right_}; }

    std::shared_ptr<LogicalNode> left() const { return left_; }
    std::shared_ptr<LogicalNode> right() const { return right_; }
    const std::vector<std::string>& on() const { return on_; }
    const std::string& how() const { return how_; }
};

class GroupAggNode : public LogicalNode {
    std::shared_ptr<LogicalNode> input_;
    std::vector<std::string> keys_;
    std::vector<std::pair<std::string, std::string>> aggs_;
public:
    GroupAggNode(std::shared_ptr<LogicalNode> input,
                 std::vector<std::string> keys,
                 std::vector<std::pair<std::string, std::string>> aggs)
        : input_(std::move(input)), keys_(std::move(keys)), aggs_(std::move(aggs)) {}

    std::string to_string() const override { return "GroupAgg"; }
    std::vector<std::shared_ptr<LogicalNode>> children() const override { return {input_}; }

    const std::vector<std::string>& keys() const { return keys_; }
    const std::vector<std::pair<std::string, std::string>>& aggs() const { return aggs_; }
    std::shared_ptr<LogicalNode> input() const { return input_; }
};

}
