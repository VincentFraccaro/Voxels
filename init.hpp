//
// Created by Vincent on 11/08/2021.
//

#ifndef VOXELS_INIT_HPP
#define VOXELS_INIT_HPP


#include <vulkan/vulkan.h>

class Init {
public:
    VkCommandPoolCreateInfo command_pool_create_info(uint32_t queueFamilyIndex, VkCommandPoolCreateFlags flags);
    VkCommandBufferAllocateInfo command_buffer_allocate_info(VkCommandPool pool, uint32_t count, VkCommandBufferLevel level);

    VkPipelineShaderStageCreateInfo pipelineShaderStageCreateInfo(VkShaderStageFlagBits stage, VkShaderModule shaderModule);
    VkPipelineVertexInputStateCreateInfo vertexInputStateCreateInfo();
    VkPipelineInputAssemblyStateCreateInfo inputAssemblyStateCreateInfo(VkPrimitiveTopology topology);
    VkPipelineRasterizationStateCreateInfo rasterizationStateCreateInfo(VkPolygonMode polygonMode);
    VkPipelineMultisampleStateCreateInfo multisampleStateCreateInfo();
    VkPipelineColorBlendAttachmentState colorBlendAttachmentState();
    VkPipelineLayoutCreateInfo pipelineLayoutCreateInfo();
    VkPipelineDepthStencilStateCreateInfo depthStencilStateCreateInfo(bool depthTest, bool depthWrite, VkCompareOp compareOp);

    VkImageCreateInfo imageCreateInfo(VkFormat format, VkImageUsageFlags usageFlags, VkExtent3D extent);
    VkImageViewCreateInfo imageViewCreateInfo(VkFormat format, VkImage image, VkImageAspectFlags aspectFlags);


};


#endif //VOXELS_INIT_HPP
