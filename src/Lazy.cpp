#include "DataFrameLib/Lazy.hpp"
#include "DataFrameLib/Optimizer.hpp"
#include "DataFrameLib/Eager.hpp"
#include <fstream>
#include <iostream>
#include <cstdlib>
#include <numeric>

std::string ProjectNode::to_string() const {
    std::string cols = "";
    for(size_t i=0; i<columns_.size(); ++i) {
        cols += columns_[i] + (i < columns_.size()-1 ? ", " : "");
    }
    return "Project(" + cols + ")";
}

LazyDataFrame LazyDataFrame::scan_csv(const std::string& path) {
    return LazyDataFrame(std::make_shared<ScanNode>(path, "csv"));
}

LazyDataFrame LazyDataFrame::scan_parquet(const std::string& path) {
    return LazyDataFrame(std::make_shared<ScanNode>(path, "parquet"));
}

LazyDataFrame LazyDataFrame::filter(const ExprPtr& predicate) const {
    auto node = std::make_shared<FilterNode>(logical_plan_, predicate);
    return LazyDataFrame(node);
}

LazyDataFrame LazyDataFrame::select(const std::vector<std::string>& columns) const {
    auto node = std::make_shared<ProjectNode>(logical_plan_, columns);
    return LazyDataFrame(node);
}

// --- Helper to escape quotes for Graphviz ---
std::string escape_quotes(const std::string& str) {
    std::string escaped;
    for (char c : str) {
        if (c == '"') escaped += "\\\"";
        else escaped += c;
    }
    return escaped;
}

// --- Helper function for recursive Graphviz DOT generation ---
void generate_dot(const std::shared_ptr<LogicalNode>& node, std::ofstream& out) {
    if (!node) return;

    // Create a unique ID for this node using its memory address
    std::string node_id = "node_" + std::to_string(reinterpret_cast<uintptr_t>(node.get()));

    // Define the node visually (e.g., shape, label)
    out << "  " << node_id << " [label=\"" << escape_quotes(node->to_string()) 
        << "\", shape=box, style=filled, fillcolor=lightblue, fontname=\"Arial\"];\n";

    // Recursively draw edges to children
    for (const auto& child : node->children()) {
        if (child) {
            std::string child_id = "node_" + std::to_string(reinterpret_cast<uintptr_t>(child.get()));
            // In a query plan, data flows FROM children TO parents. 
            // So the arrow points from the child (input) to the parent (operation).
            out << "  " << child_id << " -> " << node_id << ";\n";
            generate_dot(child, out);
        }
    }
}

// --- The Explain Method ---
void LazyDataFrame::explain(const std::string& plan_path) const {
    if (!logical_plan_) {
        std::cerr << "Warning: Cannot explain an empty LazyDataFrame." << std::endl;
        return;
    }

    std::string dot_file_path = plan_path + ".dot";
    std::ofstream out(dot_file_path);
    
    if (!out.is_open()) {
        throw std::runtime_error("Failed to create temporary .dot file for Graphviz");
    }

    // Write the Graphviz digraph wrapper
    out << "digraph LogicalPlan {\n";
    out << "  rankdir=BT; // Bottom-to-Top layout (Data flows up)\n";
    
    // Start the recursive walk from the root
    generate_dot(logical_plan_, out);
    
    out << "}\n";
    out.close();

    // Call the system Graphviz CLI to render the PNG
    std::string command = "dot -Tpng " + dot_file_path + " -o " + plan_path;
    int result = std::system(command.c_str());

    if (result == 0) {
        std::cout << "Successfully generated query plan: " << plan_path << std::endl;
        // Optionally delete the temporary .dot file
        std::remove(dot_file_path.c_str());
    } else {
        throw std::runtime_error("Graphviz failed. Make sure 'dot' is installed via 'sudo apt install graphviz'");
    }
}

EagerDataFrame LazyDataFrame::collect() const {
    // 1. Optimize the Logical Plan
    auto optimized_plan = QueryOptimizer::optimize(logical_plan_);
    
    // 2. Compile and execute it into a physical EagerDataFrame
    return PhysicalPlanCompiler::execute(optimized_plan);
}