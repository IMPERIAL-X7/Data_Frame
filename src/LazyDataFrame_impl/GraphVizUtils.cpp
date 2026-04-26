#include "../LazyDataFrame.h"
#include <fstream>
#include <iostream>
#include <cstdlib>
#include <stdexcept>

namespace dataframelib {

namespace {

std::string escape_quotes(const std::string& str) {
    std::string escaped;
    for (char c : str) {
        if (c == '"') escaped += "\\\"";
        else escaped += c;
    }
    return escaped;
}

void generate_dot(const std::shared_ptr<LogicalNode>& node, std::ofstream& out) {
    if (!node) return;

    std::string node_id = "node_" + std::to_string(reinterpret_cast<uintptr_t>(node.get()));

    out << "  " << node_id << " [label=\"" << escape_quotes(node->to_string())
        << "\", shape=box, style=filled, fillcolor=lightblue, fontname=\"Arial\"];\n";

    for (const auto& child : node->children()) {
        if (child) {
            std::string child_id = "node_" + std::to_string(reinterpret_cast<uintptr_t>(child.get()));
            out << "  " << child_id << " -> " << node_id << ";\n";
            generate_dot(child, out);
        }
    }
}

}

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

    out << "digraph LogicalPlan {\n";
    out << "  rankdir=BT;\n";
    generate_dot(logical_plan_, out);
    out << "}\n";
    out.close();

    std::string command = "dot -Tpng " + dot_file_path + " -o " + plan_path;
    int result = std::system(command.c_str());

    if (result == 0) {
        std::cout << "Successfully generated query plan: " << plan_path << std::endl;
        std::remove(dot_file_path.c_str());
    } else {
        throw std::runtime_error("Graphviz failed. Make sure 'dot' is installed via 'sudo apt install graphviz'");
    }
}

}
