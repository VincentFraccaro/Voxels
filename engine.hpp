//
// Created by Vincent on 11/08/2021.
//

#ifndef VOXELS_ENGINE_HPP
#define VOXELS_ENGINE_HPP

#include "types.hpp"
#include "pipeline.hpp"
#include "mesh.hpp"
#include <vector>
#include <functional>
#include <deque>
#include <glm/glm.hpp>
#include <glm/gtx/transform.hpp>
#include <unordered_map>

struct FrameData {
    VkSemaphore presentSemaphore, renderSemaphore;
    VkFence renderFence;

    VkCommandPool commandPool;
    VkCommandBuffer mainCommandBuffer;

    static constexpr unsigned int FRAME_OVERLAP = 2;
};

struct Material {
    VkPipeline pipeline;
    VkPipelineLayout  pipelineLayout;
};

struct RenderObject {
    Mesh *mesh;
    Material *material;
    glm::mat4 transformMatrix;
};

struct MeshPushConstants {
    glm::vec4 data;
    glm::mat4 renderMatrix;
};

class Engine {

public:

    struct DeletionQueue{
        std::deque<std::function<void()>> deletors;

        void push_function(std::function<void()>&& function) {
            deletors.push_back(function);
        }

        void flush() {
            // reverse iterate the deletion queue to execute all the functions
            for (auto it = deletors.rbegin(); it != deletors.rend(); it++) {
                (*it)(); //call the function
            }

            deletors.clear();
        }
    };

    bool isInitialized{false};
    int frameNumber{0};

    VkExtent2D windowExtent{800,600};

    struct SDL_Window* window {nullptr};

    void init();
    void cleanup();
    void draw();
    void run();
    void initVulkan();
    void initSwapchain();
    void initCommands();
    void initDefaultRenderpass();
    void initFramebuffers();
    void initSync();
    bool loadShaderModule(const char* filepath, VkShaderModule *outShaderModule);
    void initPipeline();
    void loadMeshes();
    void uploadMesh(Mesh &mesh);
    Material *createMaterial(VkPipeline pipeline, VkPipelineLayout layout, const std::string &name);
    Material *getMaterial(const std::string name);
    Mesh *getMesh(const std::string &name);
    void drawObjects(VkCommandBuffer cmd, RenderObject *first, int count);
    void initScene();

    VkInstance instance;
    VkDebugUtilsMessengerEXT debugMessenger;

    VkSurfaceKHR surface;
    VkDevice device;
    VkPhysicalDevice gpu;
    VkSwapchainKHR swapchain;
    VkFormat swapchainImageFormat;
    std::vector<VkImage> swapchainImages;

    std::vector<VkImageView> swapchainImageViews;
    VkQueue graphicsQueue;

    uint32_t  graphicsQueueFamily;
    VkRenderPass renderPass;

    std::vector<VkFramebuffer> frameBuffers;

    VkPipelineLayout pipelineLayout;
    VkPipelineLayout meshPipelineLayout;
    VkPipeline pipeline;
    Mesh triangleMesh;

    Mesh monkeyMesh;
    std::vector<RenderObject> renderables;
    std::unordered_map<std::string, Material> materials;

    std::unordered_map<std::string, Mesh> meshes;

    VkShaderModule vertShader;
    VkShaderModule fragShader;

    Pipeline pipelineBuilder;
    Init init1;
    DeletionQueue deletionQueue;

    VmaAllocator allocator;

    VkImageView depthImageView;
    AllocatedImage depthImage;

    VkFormat depthFormat;

    FrameData frames[FrameData::FRAME_OVERLAP];
    FrameData& getCurrentFrame();

};


#endif //VOXELS_ENGINE_HPP
