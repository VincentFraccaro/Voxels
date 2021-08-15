//
// Created by Vincent on 12/08/2021.
//
#include <iostream>
#define TINYOBJLOADER_IMPLEMENTATION
#include <tiny_obj_loader.h>

#include "mesh.hpp"

VertexInputDescription Vertex::getVertexDescription() {
    VertexInputDescription description;

    //we will have just 1 vertex buffer binding, with a per-vertex rate
    VkVertexInputBindingDescription mainBinding = {};
    mainBinding.binding = 0;
    mainBinding.stride = sizeof(Vertex);
    mainBinding.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

    description.bindings.push_back(mainBinding);

    //Position will be stored at Location 0
    VkVertexInputAttributeDescription positionAttribute = {};
    positionAttribute.binding = 0;
    positionAttribute.location = 0;
    positionAttribute.format = VK_FORMAT_R32G32B32_SFLOAT;
    positionAttribute.offset = offsetof(Vertex, position);

    //Normal will be stored at Location 1
    VkVertexInputAttributeDescription normalAttribute = {};
    normalAttribute.binding = 0;
    normalAttribute.location = 1;
    normalAttribute.format = VK_FORMAT_R32G32B32_SFLOAT;
    normalAttribute.offset = offsetof(Vertex, normal);

    //Color will be stored at Location 2
    VkVertexInputAttributeDescription colorAttribute = {};
    colorAttribute.binding = 0;
    colorAttribute.location = 2;
    colorAttribute.format = VK_FORMAT_R32G32B32_SFLOAT;
    colorAttribute.offset = offsetof(Vertex, color);

    description.attributes.push_back(positionAttribute);
    description.attributes.push_back(normalAttribute);
    description.attributes.push_back(colorAttribute);
    return description;
}

bool Mesh::loadFromObj(const char *filename) {
    tinyobj::attrib_t attrib;
    std::vector<tinyobj::shape_t> shapes;
    std::vector<tinyobj::material_t> materials;

    std::string warn;
    std::string err;

    tinyobj::LoadObj(&attrib, &shapes, &materials, &warn, &err, filename, nullptr);

    if(!warn.empty()){
        std::cout << "Warn: " << warn << std::endl;
    }

    if(!err.empty()){
        std::cerr << err << std::endl;
        return false;
    }

    for(size_t s = 0; s < shapes.size(); s++){
        size_t indexOffset = 0;
        for(size_t f = 0; f < shapes[s].mesh.num_face_vertices.size(); f++){
            int fv = 3;
            for(size_t v = 0; v < fv; v++){
                tinyobj::index_t idx = shapes[s].mesh.indices[indexOffset + v];

                tinyobj::real_t vx = attrib.vertices[3 * idx.vertex_index + 0];
                tinyobj::real_t vy = attrib.vertices[3 * idx.vertex_index + 1];
                tinyobj::real_t vz = attrib.vertices[3 * idx.vertex_index + 2];

                tinyobj::real_t nx = attrib.vertices[3 * idx.normal_index + 0];
                tinyobj::real_t ny = attrib.vertices[3 * idx.normal_index + 1];
                tinyobj::real_t nz = attrib.vertices[3 * idx.normal_index + 2];

                Vertex newVert;
                newVert.position = {vx, vy, vz};
                newVert.normal = {nx, ny, nz};

                newVert.color = newVert.normal;
                vertices.push_back(newVert);

            }
            indexOffset += fv;
        }
    }



    return true;
}
