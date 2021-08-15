//
// Created by Vincent on 11/08/2021.
//

#include "engine.hpp"

#include <SDL.h>
#include <SDL_vulkan.h>
#include "types.hpp"
#include "init.hpp"

#include "VkBootstrap.h"
#include <algorithm>
#include <iostream>
#include <fstream>

#define VMA_IMPLEMENTATION
#include "vk_mem_alloc.h"

void Engine::init() {
    SDL_Init(SDL_INIT_VIDEO);
    SDL_WindowFlags windowFlags = (SDL_WindowFlags)(SDL_WINDOW_VULKAN);
    window = SDL_CreateWindow("Voxel Engine", SDL_WINDOWPOS_UNDEFINED, SDL_WINDOWPOS_UNDEFINED, windowExtent.width, windowExtent.height, windowFlags);

    initVulkan();
    initSwapchain();
    initDefaultRenderpass();
    initFramebuffers();
    initCommands();
    initSync();
    initDescriptors();
    initPipeline();
    loadMeshes();
    initScene();
    isInitialized = true;
}

void Engine::cleanup() {
    if(isInitialized){
        vkDeviceWaitIdle(device);
        vkDestroyRenderPass(device, renderPass, nullptr);
        vkDestroySwapchainKHR(device, swapchain, nullptr);

        //destroy swapchain resources
        for (int i = 0; i < swapchainImageViews.size(); i++) {
            vkDestroyFramebuffer(device, frameBuffers[i], nullptr);
            vkDestroyImageView(device, swapchainImageViews[i], nullptr);
        }
        vkDestroyDevice(device, nullptr);
        vkDestroySurfaceKHR(instance, surface, nullptr);
        vkb::destroy_debug_utils_messenger(instance, debugMessenger);
        vkDestroyInstance(instance, nullptr);
        SDL_DestroyWindow(window);
    }
}

void Engine::draw() {
    if(vkWaitForFences(device, 1, &getCurrentFrame().renderFence, true, 1000000000) != VK_SUCCESS){
        std::cout << "Why can't it be patient?\n";
    }
    if(vkResetFences(device, 1, &getCurrentFrame().renderFence) != VK_SUCCESS){
        std::cout << "Why can't I reset fence\n";
    }

    uint32_t swapchainImageIndex;
    if(vkAcquireNextImageKHR(device, swapchain, 1000000000, getCurrentFrame().presentSemaphore, nullptr, &swapchainImageIndex) != VK_SUCCESS){
        std::cout << "couldn't get next thing mate\n";
    }

    if(vkResetCommandBuffer(getCurrentFrame().mainCommandBuffer, 0) != VK_SUCCESS){
        std::cout << "Couldn't reset commandbuffer xD \n";
    }

    VkCommandBuffer cmd = getCurrentFrame().mainCommandBuffer;
    VkCommandBufferBeginInfo cmdBeginInfo = {};
    cmdBeginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    cmdBeginInfo.pNext = nullptr;

    cmdBeginInfo.pInheritanceInfo = nullptr;
    cmdBeginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

    if(vkBeginCommandBuffer(cmd, &cmdBeginInfo) != VK_SUCCESS){
        std::cout << "Couldn't make new command buffer\n";
    }

    VkClearValue clearValue;
    float flash = std::clamp( std::abs(sin(frameNumber / 120.f)), 0.005f, 0.995f);
    clearValue.color = { { flash, 0.0f, flash, 1.0f } };

    VkClearValue depthClear;
    depthClear.depthStencil.depth = 1.f;

    VkRenderPassBeginInfo rpInfo = {};
    rpInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
    rpInfo.pNext = nullptr;

    rpInfo.renderPass = renderPass;
    rpInfo.renderArea.offset.x = 0;
    rpInfo.renderArea.offset.y = 0;
    rpInfo.renderArea.extent = windowExtent;
    rpInfo.framebuffer = frameBuffers[swapchainImageIndex];

    //connect clear values
    rpInfo.clearValueCount = 2;
    VkClearValue clearValues[] = {clearValue, depthClear};
    rpInfo.pClearValues = &clearValues[0];

    vkCmdBeginRenderPass(cmd, &rpInfo, VK_SUBPASS_CONTENTS_INLINE);

    drawObjects(cmd, renderables.data(), renderables.size());

    vkCmdEndRenderPass(cmd);
    if(vkEndCommandBuffer(cmd) != VK_SUCCESS){
        std::cout << "Couldn't end command buffer lol what\n";
    }

    VkSubmitInfo submit = {};
    submit.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submit.pNext = nullptr;

    VkPipelineStageFlags waitStage = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    submit.pWaitDstStageMask = &waitStage;
    submit.waitSemaphoreCount = 1;
    submit.pWaitSemaphores = &getCurrentFrame().presentSemaphore;
    submit.signalSemaphoreCount = 1;
    submit.pSignalSemaphores = &getCurrentFrame().renderSemaphore;
    submit.commandBufferCount = 1;
    submit.pCommandBuffers = &cmd;

    if(vkQueueSubmit(graphicsQueue, 1, &submit, getCurrentFrame().renderFence) != VK_SUCCESS){
        std::cout << "Couldn't submit the render stuff to the queue thing \n";
    }

    VkPresentInfoKHR presentInfo = {};
    presentInfo.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
    presentInfo.pNext = nullptr;

    presentInfo.pSwapchains = &swapchain;
    presentInfo.swapchainCount = 1;

    presentInfo.pWaitSemaphores = &getCurrentFrame().renderSemaphore;
    presentInfo.waitSemaphoreCount = 1;

    presentInfo.pImageIndices = &swapchainImageIndex;

    if(vkQueuePresentKHR(graphicsQueue, &presentInfo) != VK_SUCCESS){
        std::cout << "I kill myself if it fails here cunt\n";
    }
    //increase the number of frames drawn
    frameNumber++;
}

void Engine::run() {
    SDL_Event e;
    bool quit = false;

    while(!quit) {
        while(SDL_PollEvent(&e) != 0) {
            if(e.type == SDL_QUIT) quit = true;
        }
        draw();
    }
}

void Engine::initVulkan() {

    vkb::InstanceBuilder builder;
    auto instReturn = builder.set_app_name("Vulkan Application")
            .request_validation_layers(true)
            .require_api_version(1,2, 0)
            .use_default_debug_messenger()
            .build();

    vkb::Instance vkbInst = instReturn.value();
    instance = vkbInst.instance;
    debugMessenger = vkbInst.debug_messenger;


    SDL_Vulkan_CreateSurface(window, instance, &surface);

    vkb::PhysicalDeviceSelector selector{ vkbInst };
    vkb::PhysicalDevice physicalDevice = selector
            .set_minimum_version(1, 2)
            .set_surface(surface)
            .select()
            .value();

    vkb::DeviceBuilder deviceBuilder{physicalDevice};
    vkb::Device vkbDevice = deviceBuilder.build().value();
    device = vkbDevice.device;
    gpu = physicalDevice.physical_device;

    graphicsQueue = vkbDevice.get_queue(vkb::QueueType::graphics).value();
    graphicsQueueFamily = vkbDevice.get_queue_index(vkb::QueueType::graphics).value();

    VmaAllocatorCreateInfo allocatorInfo = {};
    allocatorInfo.physicalDevice = gpu;
    allocatorInfo.device = device;
    allocatorInfo.instance = instance;
    vmaCreateAllocator(&allocatorInfo, &allocator);

    deletionQueue.push_function([&](){
        vmaDestroyAllocator(allocator);
    });
}

void Engine::initSwapchain() {
    vkb::SwapchainBuilder swapchainBuilder{gpu, device, surface};
    vkb::Swapchain vkbSwapchain = swapchainBuilder.use_default_format_selection()
            .set_desired_present_mode(VK_PRESENT_MODE_FIFO_KHR)
            .set_desired_extent(windowExtent.width, windowExtent.height)
            .build()
            .value();

    swapchain = vkbSwapchain.swapchain;
    swapchainImages = vkbSwapchain.get_images().value();
    swapchainImageViews = vkbSwapchain.get_image_views().value();
    swapchainImageFormat = vkbSwapchain.image_format;

    VkExtent3D depthImageExtent = {windowExtent.width, windowExtent.height, 1};
    depthFormat = VK_FORMAT_D32_SFLOAT;

    VkImageCreateInfo depthImageInfo = init1.imageCreateInfo(depthFormat, VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT, depthImageExtent);

    VmaAllocationCreateInfo depthImageAllocationInfo = {};
    depthImageAllocationInfo.usage = VMA_MEMORY_USAGE_GPU_ONLY;
    depthImageAllocationInfo.requiredFlags = VkMemoryPropertyFlags(VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

    vmaCreateImage(allocator, &depthImageInfo, &depthImageAllocationInfo, &depthImage.image, &depthImage.allocation, nullptr);

    VkImageViewCreateInfo depthViewInfo = init1.imageViewCreateInfo(depthFormat, depthImage.image, VK_IMAGE_ASPECT_DEPTH_BIT);

    if(!vkCreateImageView(device, &depthViewInfo, nullptr, &depthImageView) != VK_SUCCESS){
        std::cout << "Failed making image view \n";
    }

    deletionQueue.push_function([=](){
        vkDestroyImageView(device, depthImageView, nullptr);
        vmaDestroyImage(allocator, depthImage.image, depthImage.allocation);
    });


}

void Engine::initCommands() {
    VkCommandPoolCreateInfo commandPoolInfo = init1.command_pool_create_info(graphicsQueueFamily, VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT);

    for(int i = 0; i < frames->FRAME_OVERLAP; i++){
        if(vkCreateCommandPool(device, &commandPoolInfo, nullptr, &frames[i].commandPool) != VK_SUCCESS){
            std::cerr << "Error in making command pool\n";
        }

        VkCommandBufferAllocateInfo cmdAllocInfo = init1.command_buffer_allocate_info(frames[i].commandPool, 1, VK_COMMAND_BUFFER_LEVEL_PRIMARY);
        if(vkAllocateCommandBuffers(device, &cmdAllocInfo, &frames[i].mainCommandBuffer) != VK_SUCCESS){
            std::cerr << "Couldn't make main command buffer\n";
        }

        deletionQueue.push_function([=](){
            vkDestroyCommandPool(device, frames[i].commandPool, nullptr);
        });
    }
}

void Engine::initDefaultRenderpass() {
    VkAttachmentDescription colorAttachment = {};
    colorAttachment.format = swapchainImageFormat;
    colorAttachment.samples = VK_SAMPLE_COUNT_1_BIT;
    colorAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    colorAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    colorAttachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    colorAttachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    colorAttachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    colorAttachment.finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;

    VkAttachmentReference colorAttachmentRef = {};
    colorAttachmentRef.attachment = 0;
    colorAttachmentRef.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

    VkAttachmentDescription depthAttachment = {};
    // Depth attachment
    depthAttachment.flags = 0;
    depthAttachment.format = depthFormat;
    depthAttachment.samples = VK_SAMPLE_COUNT_1_BIT;
    depthAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    depthAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    depthAttachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    depthAttachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    depthAttachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    depthAttachment.finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

    VkAttachmentReference depthAttachmentRef = {};
    depthAttachmentRef.attachment = 1;
    depthAttachmentRef.layout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

    VkSubpassDescription subpass = {};
    subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
    subpass.colorAttachmentCount = 1;
    subpass.pColorAttachments = &colorAttachmentRef;
    subpass.pDepthStencilAttachment = &depthAttachmentRef;

    VkAttachmentDescription attachments[2] = { colorAttachment, depthAttachment };

    VkRenderPassCreateInfo renderPassInfo = {};
    renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
    //2 attachments from said array
    renderPassInfo.attachmentCount = 2;
    renderPassInfo.pAttachments = &attachments[0];
    renderPassInfo.subpassCount = 1;
    renderPassInfo.pSubpasses = &subpass;

    if(vkCreateRenderPass(device, &renderPassInfo, nullptr, &renderPass) != VK_SUCCESS){
        throw std::runtime_error("Shits fucked in the render pass making of");
    }


}

void Engine::initFramebuffers() {
    VkFramebufferCreateInfo frameBufferInfo = {};
    frameBufferInfo.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
    frameBufferInfo.pNext = nullptr;

    frameBufferInfo.renderPass = renderPass;
    frameBufferInfo.attachmentCount = 1;
    frameBufferInfo.width = windowExtent.width;
    frameBufferInfo.height = windowExtent.height;
    frameBufferInfo.layers = 1;

    //grab how many images we have in the swapchain
    const uint32_t swapchainImageCount = swapchainImages.size();
    frameBuffers = std::vector<VkFramebuffer>(swapchainImageCount);

    //create framebuffers for each of the swapchain image views
    for (int i = 0; i < swapchainImageCount; i++) {

        VkImageView attachments[2];
        attachments[0] = swapchainImageViews[i];
        attachments[1] = depthImageView;

        frameBufferInfo.pAttachments = attachments;
        frameBufferInfo.attachmentCount = 2;
        if(vkCreateFramebuffer(device, &frameBufferInfo, nullptr, &frameBuffers[i]) != VK_SUCCESS){
            throw std::runtime_error("Shits fucked making the framebuffer");
        }
    }
}

void Engine::initSync() {
    VkFenceCreateInfo fenceCreateInfo = {};
    fenceCreateInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    fenceCreateInfo.pNext = nullptr;
    //we want to create the fence with the Create Signaled flag, so we can wait on it before using it on a GPU command (for the first frame)
    fenceCreateInfo.flags = VK_FENCE_CREATE_SIGNALED_BIT;
    //for the semaphores we don't need any flags
    VkSemaphoreCreateInfo semaphoreCreateInfo = {};
    semaphoreCreateInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
    semaphoreCreateInfo.pNext = nullptr;
    semaphoreCreateInfo.flags = 0;

    for(int i = 0; i < frames->FRAME_OVERLAP; i++){
        if(vkCreateFence(device, &fenceCreateInfo, nullptr, &frames[i].renderFence) != VK_SUCCESS){
            std::cerr << "Failed at creating the fence\n";
        }

        deletionQueue.push_function([=](){
            vkDestroyFence(device, frames[i].renderFence, nullptr);
        });

        if(vkCreateSemaphore(device, &semaphoreCreateInfo, nullptr, &frames[i].presentSemaphore) != VK_SUCCESS){
            std::cerr << "Failed at creating the present sempahore\n";
        }
        if(vkCreateSemaphore(device, &semaphoreCreateInfo, nullptr, &frames[i].renderSemaphore) != VK_SUCCESS){
            std::cerr << "Failed at creating the render Sempahore\n";
        }

        deletionQueue.push_function([=](){
            vkDestroySemaphore(device, frames[i].presentSemaphore, nullptr);
            vkDestroySemaphore(device, frames[i].renderSemaphore, nullptr);
        });


    }

}

bool Engine::loadShaderModule(const char *filepath, VkShaderModule *outShaderFile) {
    std::ifstream file(filepath, std::ios::ate | std::ios::binary);
    if(!file.is_open()){
        return false;
    }

    size_t fileSize = size_t(file.tellg());
    std::vector<uint32_t> buffer(fileSize/sizeof(uint32_t));
    file.seekg(0);
    file.read((char*)buffer.data(), fileSize);

    file.close();

    VkShaderModuleCreateInfo createInfo = {};
    createInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    createInfo.pNext = nullptr;

    //codeSize has to be in bytes, so multiply the ints in the buffer by size of int to know the real size of the buffer
    createInfo.codeSize = buffer.size() * sizeof(uint32_t);
    createInfo.pCode = buffer.data();

    //check that the creation goes well.
    VkShaderModule shaderModule;
    if (vkCreateShaderModule(device, &createInfo, nullptr, &shaderModule) != VK_SUCCESS) {
        return false;
    }
    *outShaderFile = shaderModule;
    return true;

}

void Engine::initPipeline() {
    if(!loadShaderModule("shaders/frag.spv", &fragShader)) {
        std::cout << "Error building frag shader module\n";
    }
    else{
        std::cout << "It worked building frag\n" << fragShader << std::endl;
    }

    if(!loadShaderModule("shaders/vert.spv", &vertShader)) {
        std::cout << "Error building frag shader module\n";
    }
    else{
        std::cout << "It worked building vert\n";
    }

    VkPipelineLayoutCreateInfo pipelineLayoutCreateInfo = init1.pipelineLayoutCreateInfo();
    if(vkCreatePipelineLayout(device, &pipelineLayoutCreateInfo, nullptr, &pipelineLayout) != VK_SUCCESS){
        std::cout << "Failed creating pipeline\n";
    }
    else{
        std::cout << "creating pipeline success\n";
    }

    VertexInputDescription vertexDescription = Vertex::getVertexDescription();

    VkPipelineLayoutCreateInfo meshPipelineLayoutInfo = init1.pipelineLayoutCreateInfo();
    VkPushConstantRange pushConstant;
    pushConstant.offset = 0;
    pushConstant.size = sizeof(MeshPushConstants);
    pushConstant.stageFlags = VK_SHADER_STAGE_VERTEX_BIT;

    meshPipelineLayoutInfo.pPushConstantRanges = &pushConstant;
    meshPipelineLayoutInfo.pushConstantRangeCount = 1;

    meshPipelineLayoutInfo.setLayoutCount = 1;
    meshPipelineLayoutInfo.pSetLayouts = &globalSetLayout;

    vkCreatePipelineLayout(device, &meshPipelineLayoutInfo, nullptr, &meshPipelineLayout);

    pipelineBuilder.shaderStages.push_back(init1.pipelineShaderStageCreateInfo(VK_SHADER_STAGE_VERTEX_BIT, vertShader));
    pipelineBuilder.shaderStages.push_back(init1.pipelineShaderStageCreateInfo(VK_SHADER_STAGE_FRAGMENT_BIT, fragShader));
    //vertex input controls how to read vertices from vertex buffers. We aren't using it yet
    pipelineBuilder.vertexInputInfo = init1.vertexInputStateCreateInfo();
    pipelineBuilder.vertexInputInfo.pVertexAttributeDescriptions = vertexDescription.attributes.data();
    pipelineBuilder.vertexInputInfo.vertexAttributeDescriptionCount = vertexDescription.attributes.size();

    pipelineBuilder.vertexInputInfo.pVertexBindingDescriptions = vertexDescription.bindings.data();
    pipelineBuilder.vertexInputInfo.vertexBindingDescriptionCount = vertexDescription.bindings.size();
    //input assembly is the configuration for drawing triangle lists, strips, or individual points.
    //we are just going to draw triangle list
    pipelineBuilder.inputAssembly = init1.inputAssemblyStateCreateInfo(VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST);
    //build viewport and scissor from the swapchain extents
    pipelineBuilder.viewport.x = 0.0f;
    pipelineBuilder.viewport.y = 0.0f;
    pipelineBuilder.viewport.width = (float)windowExtent.width;
    pipelineBuilder.viewport.height = (float)windowExtent.height;
    pipelineBuilder.viewport.minDepth = 0.0f;
    pipelineBuilder.viewport.maxDepth = 1.0f;

    pipelineBuilder.scissor.offset = { 0, 0 };
    pipelineBuilder.scissor.extent = windowExtent;
    //configure the rasterizer to draw filled triangles
    pipelineBuilder.rasterizer = init1.rasterizationStateCreateInfo(VK_POLYGON_MODE_FILL);
    //we don't use multisampling, so just run the default one
    pipelineBuilder.multisampling = init1.multisampleStateCreateInfo();
    //a single blend attachment with no blending and writing to RGBA
    pipelineBuilder.colorBlendAttachment = init1.colorBlendAttachmentState();
    pipelineBuilder.depthStencil = init1.depthStencilStateCreateInfo(true, true, VK_COMPARE_OP_LESS_OR_EQUAL);
    //use the triangle layout we created
    pipelineBuilder.pipelineLayout = meshPipelineLayout;

    pipeline = pipelineBuilder.buildPipeline(device, renderPass);

    createMaterial(pipeline, meshPipelineLayout, "defaultmesh");

    pipelineBuilder.shaderStages.clear();


    deletionQueue.push_function([=]() {
        vkDestroyPipeline(device, pipeline, nullptr);
        vkDestroyPipelineLayout(device, pipelineLayout, nullptr);
        vkDestroyPipelineLayout(device, meshPipelineLayout, nullptr);
    });


}

void Engine::loadMeshes() {
    triangleMesh.vertices.resize(3);

    triangleMesh.vertices[0].position = {1.f, 1.f, 0.0f};
    triangleMesh.vertices[1].position = {-1.f, 1.f, 0.0f};
    triangleMesh.vertices[2].position = {0.0f, -1.f, 0.0f};

    triangleMesh.vertices[0].normal = {0.f, 0.f, 1.0f};
    triangleMesh.vertices[1].normal = {0.f, 0.f, 1.0f};
    triangleMesh.vertices[2].normal = {0.0f, 0.f, 1.0f};

    triangleMesh.vertices[0].color = {0.0f, 1.0f, 0.0f};
    triangleMesh.vertices[1].color = {1.0f, 0.0f, 0.0f};
    triangleMesh.vertices[2].color = {0.0f, 0.0f, 1.0f};

    monkeyMesh.loadFromObj("monkey_smooth.obj");

    uploadMesh(triangleMesh);
    uploadMesh(monkeyMesh);

    meshes["monkey"] = monkeyMesh;
    meshes["triangle"] = triangleMesh;
}

void Engine::uploadMesh(Mesh &mesh) {
    VkBufferCreateInfo bufferInfo = {};
    bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bufferInfo.pNext = nullptr;
    //this is the total size, in bytes, of the buffer we are allocating
    bufferInfo.size = mesh.vertices.size() * sizeof(Vertex);
    //this buffer is going to be used as a Vertex Buffer
    bufferInfo.usage = VK_BUFFER_USAGE_VERTEX_BUFFER_BIT;


    //let the VMA library know that this data should be writeable by CPU, but also readable by GPU
    VmaAllocationCreateInfo vmaallocInfo = {};
    vmaallocInfo.usage = VMA_MEMORY_USAGE_CPU_TO_GPU;

    //allocate the buffer
    vmaCreateBuffer(allocator, &bufferInfo, &vmaallocInfo, &mesh.vertexBuffer.buffer, &mesh.vertexBuffer.allocation,nullptr);

    //add the destruction of triangle mesh buffer to the deletion queue
    deletionQueue.push_function([=]() {

        vmaDestroyBuffer(allocator, mesh.vertexBuffer.buffer, mesh.vertexBuffer.allocation);
    });

    //copy vertex data
    void* data;
    vmaMapMemory(allocator, mesh.vertexBuffer.allocation, &data);

    memcpy(data, mesh.vertices.data(), mesh.vertices.size() * sizeof(Vertex));

    vmaUnmapMemory(allocator, mesh.vertexBuffer.allocation);


}

Material *Engine::createMaterial(VkPipeline pipeline, VkPipelineLayout layout, const std::string &name) {
    Material mat;
    mat.pipeline = pipeline;
    mat.pipelineLayout = layout;
    materials[name] = mat;
    return &materials[name];
}

Material *Engine::getMaterial(const std::string name) {
    auto it = materials.find(name);
    if(it == materials.end()){
        return nullptr;
    }
    else return &(*it).second;
}

Mesh *Engine::getMesh(const std::string &name) {
    auto it = meshes.find(name);
    if(it == meshes.end()) {
        return nullptr;
    }
    else return &(*it).second;
}

void Engine::drawObjects(VkCommandBuffer cmd, RenderObject *first, int count) {
    glm::vec3 camPos = {0.f, -5.f, -10.f};
    glm::mat4 view = glm::translate(glm::mat4(1.f), camPos);
    glm::mat4 projection = glm::perspective(glm::radians(70.0f), 1700.0f/900.0f, 0.1f, 200.0f);
    projection[1][1] *= -1;

    GPUCameraData camData;
    camData.projection = projection;
    camData.view = view;
    camData.viewproj = projection * view;

    void* data;
    vmaMapMemory(allocator, getCurrentFrame().cameraBuffer.allocation, &data);
    memcpy(data, &camData, sizeof(GPUCameraData));
    vmaUnmapMemory(allocator, getCurrentFrame().cameraBuffer.allocation);

    Mesh *lastMesh = nullptr;
    Material* lastMaterial = nullptr;

    for(int i = 0; i < count; i++){
        RenderObject& object = first[i];
        if(object.material != lastMaterial) {
            vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, object.material->pipeline);
            lastMaterial = object.material;

            vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, object.material->pipelineLayout, 0, 1, &getCurrentFrame().globalDescriptor, 0, nullptr);
        }

        glm::mat4 model = object.transformMatrix;
        glm::mat4 meshMatrix = projection * view * model;

        MeshPushConstants constants;
        constants.renderMatrix = object.transformMatrix;

        vkCmdPushConstants(cmd, object.material->pipelineLayout, VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(MeshPushConstants), &constants);
        if(object.mesh != lastMesh) {
            VkDeviceSize offset = 0;
            vkCmdBindVertexBuffers(cmd, 0, 1, &object.mesh->vertexBuffer.buffer, &offset);
            lastMesh = object.mesh;
        }

        vkCmdDraw(cmd, object.mesh->vertices.size(), 1, 0, 0);

    }



}

void Engine::initScene() {
    RenderObject monkey;
    monkey.mesh = getMesh("monkey");
    monkey.material = getMaterial("defaultmesh");
    monkey.transformMatrix = glm::mat4{1.0f};
    renderables.push_back(monkey);

    for(int x = -20; x <= 20; x++){
        for(int y = -20; y <= 20; y++){
            RenderObject tri;
            tri.mesh = getMesh("triangle");
            tri.material = getMaterial("defaultmesh");
            glm::mat4 translation = glm::translate(glm::mat4{1.0f}, glm::vec3(x, 0, y));
            glm::mat4 scale = glm::scale(glm::mat4{1.0f}, glm::vec3{0.2f, 0.2f, 0.2f});
            tri.transformMatrix = translation * scale;

            renderables.push_back(tri);
        }
    }
}

FrameData &Engine::getCurrentFrame() {
    return frames[frameNumber % frames->FRAME_OVERLAP];
}

AllocatedBuffer Engine::createBuffer(size_t allocSize, VkBufferUsageFlags usage, VmaMemoryUsage memoryUsage) {
    VkBufferCreateInfo bufferInfo = {};
    bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bufferInfo.pNext = nullptr;
    bufferInfo.size = allocSize;
    bufferInfo.usage = usage;

    VmaAllocationCreateInfo  vmaAllocationCreateInfo = {};
    vmaAllocationCreateInfo.usage = memoryUsage;

    AllocatedBuffer newBuffer;

    if(vmaCreateBuffer(allocator, &bufferInfo, &vmaAllocationCreateInfo, &newBuffer.buffer, &newBuffer.allocation, nullptr) != VK_SUCCESS){
        std::cerr << "FAILED TO CREATE THHE BUFFER FOR VMA!!!!!!!!!!!\n";
    }

    return newBuffer;
}

void Engine::initDescriptors() {

    //create a descriptor pool that will hold 10 uniform buffers
    std::vector<VkDescriptorPoolSize> sizes = {{ VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 10 }};

    VkDescriptorPoolCreateInfo poolInfo ={};
    poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    poolInfo.flags = 0;
    poolInfo.maxSets = 10;
    poolInfo.poolSizeCount = (uint32_t)sizes.size();
    poolInfo.pPoolSizes = sizes.data();

    vkCreateDescriptorPool(device, &poolInfo, nullptr, &descriptorPool);


    VkDescriptorSetLayoutBinding camBufferBinding = {};
    camBufferBinding.binding = 0;
    camBufferBinding.descriptorCount = 1;
    camBufferBinding.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    camBufferBinding.stageFlags = VK_SHADER_STAGE_VERTEX_BIT;

    VkDescriptorSetLayoutCreateInfo setInfo = {};
    setInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    setInfo.pNext = nullptr;
    setInfo.bindingCount = 1;
    setInfo.flags = 0;
    setInfo.pBindings = &camBufferBinding;

    vkCreateDescriptorSetLayout(device, &setInfo, nullptr, &globalSetLayout);

    for(int i = 0; i < frames->FRAME_OVERLAP; i++){
        frames[i].cameraBuffer = createBuffer(sizeof (GPUCameraData), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, VMA_MEMORY_USAGE_CPU_TO_GPU);

        VkDescriptorSetAllocateInfo allocInfo = {};
        allocInfo.pNext = nullptr;
        allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        allocInfo.descriptorPool = descriptorPool;
        allocInfo.descriptorSetCount = 1;
        allocInfo.pSetLayouts = &globalSetLayout;

        vkAllocateDescriptorSets(device, &allocInfo, &frames[i].globalDescriptor);

        VkDescriptorBufferInfo bufferInfo = {};
        bufferInfo.buffer = frames[i].cameraBuffer.buffer;
        bufferInfo.offset = 0;
        bufferInfo.range = sizeof(GPUCameraData);

        VkWriteDescriptorSet setWrite = {};
        setWrite.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        setWrite.pNext = nullptr;
        setWrite.dstBinding = 0;
        setWrite.dstSet = frames[i].globalDescriptor;
        setWrite.descriptorCount = 1;
        setWrite.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        setWrite.pBufferInfo = &bufferInfo;

        vkUpdateDescriptorSets(device, 1, &setWrite, 0, nullptr);
    }
}
