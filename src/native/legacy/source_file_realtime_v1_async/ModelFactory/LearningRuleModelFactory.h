/*
* 文件名：LearningRuleModelFactory.h
* 定义了学习规则工厂类，用于创建学习规则模型
*/

#ifndef LEARNINGRULEMODELFACTORY_H
#define LEARNINGRULEMODELFACTORY_H

#include "source_file_realtime_v1_async/LearningRule/inc/LearningRule.h"
#include "source_file_realtime_v1_async/Network/inc/NetworkConstructStructure.h"
class LearningRuleModelFactory {
public:
    static LearningRule* createLearningRuleModel(LearningRuleDescription description);
};
#endif
