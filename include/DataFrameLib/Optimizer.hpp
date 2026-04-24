#pragma once

#include "DataFrameLib/Lazy.hpp"
#include "DataFrameLib/Eager.hpp"
#include <memory>

class QueryOptimizer {
public:
    // The main entry point that applies all rules
    static std::shared_ptr<LogicalNode> optimize(std::shared_ptr<LogicalNode> plan);

private:
    // Rule 1: Predicate Pushdown 
    static std::shared_ptr<LogicalNode> pushdown_predicates(std::shared_ptr<LogicalNode> node);
};

class PhysicalPlanCompiler {
public:
    // Takes a logical node and physically executes it using EagerDataFrame
    static EagerDataFrame execute(const std::shared_ptr<LogicalNode>& node);
};