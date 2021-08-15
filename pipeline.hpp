//
// Created by Vincent on 12/08/2021.
//

#ifndef VOXELS_PIPELINE_HPP
#define VOXELS_PIPELINE_HPP

#include "init.hpp"
#include <vector>

class Pipeline {
public:
    std::vector<VkPipelineShaderStageCreateInfo> shaderStages;
    VkPipelineVertexInputStateCreateInfo vertexInputInfo;
    VkPipelineInputAssemblyStateCreateInfo inputAssembly;
    VkViewport viewport;
    VkRect2D scissor;
    VkPipelineRasterizationStateCreateInfo rasterizer;
    VkPipelineColorBlendAttachmentState colorBlendAttachment;
    VkPipelineMultisampleStateCreateInfo multisampling;
    VkPipelineLayout pipelineLayout;
    VkPipelineDepthStencilStateCreateInfo depthStencil;

    VkPipeline buildPipeline(VkDevice device, VkRenderPass pass);
};


#endif //VOXELS_PIPELINE_HPP
