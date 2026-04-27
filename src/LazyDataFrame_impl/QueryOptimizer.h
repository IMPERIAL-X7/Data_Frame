#pragma once

#include "GraphNodes.h"
#include "../EagerDataFrame.h"
#include <memory>

namespace dataframelib {

class QueryOptimizer {
public:
    static std::shared_ptr<LogicalNode> optimize(std::shared_ptr<LogicalNode> plan);
    static std::shared_ptr<LogicalNode> pushdown_predicates(std::shared_ptr<LogicalNode> node);
    static std::shared_ptr<LogicalNode> fuse_top_n(std::shared_ptr<LogicalNode> node);
};

class PhysicalPlanCompiler {
public:
    static EagerDataFrame execute(const std::shared_ptr<LogicalNode>& node);
};

}
