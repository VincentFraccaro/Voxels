//
// Created by Vincent on 11/08/2021.
//

#ifndef VOXELS_TYPES_HPP
#define VOXELS_TYPES_HPP

#include <vulkan/vulkan.h>
#include <vk_mem_alloc.h>

struct AllocatedBuffer {
    VkBuffer buffer;
    VmaAllocation allocation;
};

struct AllocatedImage {
    VkImage image;
    VmaAllocation allocation;
};


#endif //VOXELS_TYPES_HPP
