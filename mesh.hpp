//
// Created by Vincent on 12/08/2021.
//

#ifndef VOXELS_MESH_HPP
#define VOXELS_MESH_HPP

#include "types.hpp"
#include <vector>
#include <glm/vec3.hpp>

struct VertexInputDescription {
    std::vector<VkVertexInputBindingDescription> bindings;
    std::vector<VkVertexInputAttributeDescription> attributes;

    VkPipelineVertexInputStateCreateFlags flags = 0;
};

struct Vertex {
    glm::vec3 position;
    glm::vec3 normal;
    glm::vec3 color;

    static VertexInputDescription getVertexDescription();
};

struct Mesh {
    std::vector<Vertex> vertices;
    AllocatedBuffer vertexBuffer;

    bool loadFromObj(const char *filename);
};


#endif //VOXELS_MESH_HPP
