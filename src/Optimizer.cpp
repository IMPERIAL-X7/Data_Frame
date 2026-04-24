#include "DataFrameLib/Optimizer.hpp"
#include <stdexcept>

// ---------------------------------------------------------
// Query Optimizer Rules
// ---------------------------------------------------------
std::shared_ptr<LogicalNode> QueryOptimizer::optimize(std::shared_ptr<LogicalNode> plan) {
    return pushdown_predicates(plan);
}

std::shared_ptr<LogicalNode> QueryOptimizer::pushdown_predicates(std::shared_ptr<LogicalNode> node) {
    if (!node) return nullptr;

    // 1. If it's a Filter node, check if we can push it lower
    if (auto filter = std::dynamic_pointer_cast<FilterNode>(node)) {
        auto child = filter->input();
        
        // PUSHDOWN RULE: If a Filter sits on top of a Project, swap them!
        if (auto project = std::dynamic_pointer_cast<ProjectNode>(child)) {
            // Recurse down the tree first
            auto optimized_project_input = pushdown_predicates(project->input());
            
            // Create the swapped tree: Project -> Filter -> Input
            auto new_filter = std::make_shared<FilterNode>(optimized_project_input, filter->predicate());
            return std::make_shared<ProjectNode>(new_filter, project->columns());
        }
        // If we can't swap, just recurse down
        return std::make_shared<FilterNode>(pushdown_predicates(child), filter->predicate());
    }

    // 2. For Project nodes, just recurse down
    if (auto project = std::dynamic_pointer_cast<ProjectNode>(node)) {
        return std::make_shared<ProjectNode>(pushdown_predicates(project->input()), project->columns());
    }

    // 3. Leaf nodes (Scan) return themselves
    return node;
}

// ---------------------------------------------------------
// Physical Execution (The Compiler)
// ---------------------------------------------------------
EagerDataFrame PhysicalPlanCompiler::execute(const std::shared_ptr<LogicalNode>& node) {
    if (!node) throw std::runtime_error("Cannot execute null plan");

    if (auto scan = std::dynamic_pointer_cast<ScanNode>(node)) {
        if (scan->format() == "csv") return EagerDataFrame::read_csv(scan->path());
        if (scan->format() == "parquet") return EagerDataFrame::read_parquet(scan->path());
        throw std::runtime_error("Unsupported scan format");
    }
    
    if (auto filter = std::dynamic_pointer_cast<FilterNode>(node)) {
        // Execute the child first, then apply the filter eager operation
        EagerDataFrame child_df = execute(filter->input());
        return child_df.filter(filter->predicate());
    }
    
    if (auto project = std::dynamic_pointer_cast<ProjectNode>(node)) {
        // Execute the child first, then apply the select eager operation
        EagerDataFrame child_df = execute(project->input());
        return child_df.select(project->columns());
    }

    throw std::runtime_error("Unknown LogicalNode encountered during execution");
}