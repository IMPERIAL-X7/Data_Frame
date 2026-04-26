#include "GraphNodes.h"

namespace dataframelib {

std::string ProjectNode::to_string() const {
    std::string cols;
    for (size_t i = 0; i < columns_.size(); ++i) {
        cols += columns_[i] + (i < columns_.size() - 1 ? ", " : "");
    }
    return "Project(" + cols + ")";
}

}
