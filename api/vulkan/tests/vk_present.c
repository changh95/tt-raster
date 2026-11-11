/* SPDX-License-Identifier: Apache-2.0
 *
 * Vulkan through an X11 swapchain: draws the grid of depth-tested, per-vertex-coloured triangles of vk_offscreen into the images of a
 * swapchain, presents them, and at the end compares what the X server holds for the window with a copy of the last swapchain image
 * taken on the GPU side. It checks what a headless test cannot: render passes that clear and draw into presentable images (the
 * Tensix offload of llvmpipe/lavapipe writes the whole frame into the surface when the queue is flushed), and the present path that
 * reads the same surface afterwards.
 *
 *   vk_present [--size WxH] [--tris N] [--frames N] [--out FILE.ppm]
 *
 * Run under an X server (scripts/run_vk.sh starts xvfb). Exit status 1 if the window and the copy differ.
 */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <xcb/xcb.h>
#define VK_USE_PLATFORM_XCB_KHR
#include <vulkan/vulkan.h>
#include <vulkan/vulkan_xcb.h>

#include "tri_frag.h"
#include "tri_vert.h"

#define CHECK(x)                                                                     \
   do {                                                                              \
      VkResult r_ = (x);                                                             \
      if (r_ != VK_SUCCESS) {                                                        \
         fprintf(stderr, "%s:%d: %s failed (%d)\n", __FILE__, __LINE__, #x, (int)r_); \
         exit(2);                                                                    \
      }                                                                              \
   } while (0)

static VkPhysicalDevice g_pd;
static VkDevice g_dev;

static uint32_t find_memory(uint32_t bits, VkMemoryPropertyFlags want)
{
   VkPhysicalDeviceMemoryProperties mp;
   vkGetPhysicalDeviceMemoryProperties(g_pd, &mp);
   for (uint32_t i = 0; i < mp.memoryTypeCount; i++)
      if ((bits & (1u << i)) && (mp.memoryTypes[i].propertyFlags & want) == want)
         return i;
   fprintf(stderr, "no memory type\n");
   exit(2);
}

static void make_buffer(VkDeviceSize size, VkBufferUsageFlags usage, VkBuffer *buf, VkDeviceMemory *mem, void **map)
{
   VkBufferCreateInfo bi = {VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, .size = size, .usage = usage, .sharingMode = VK_SHARING_MODE_EXCLUSIVE};
   CHECK(vkCreateBuffer(g_dev, &bi, NULL, buf));
   VkMemoryRequirements mr;
   vkGetBufferMemoryRequirements(g_dev, *buf, &mr);
   VkMemoryAllocateInfo ai = {VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, .allocationSize = mr.size,
                              .memoryTypeIndex = find_memory(mr.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)};
   CHECK(vkAllocateMemory(g_dev, &ai, NULL, mem));
   CHECK(vkBindBufferMemory(g_dev, *buf, *mem, 0));
   CHECK(vkMapMemory(g_dev, *mem, 0, size, 0, map));
}

static VkShaderModule make_shader(const uint32_t *code, size_t bytes)
{
   VkShaderModuleCreateInfo si = {VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO, .codeSize = bytes, .pCode = code};
   VkShaderModule m;
   CHECK(vkCreateShaderModule(g_dev, &si, NULL, &m));
   return m;
}

int main(int argc, char **argv)
{
   uint32_t w = 640, h = 480;
   int tris = 2000, frames = 3;
   const char *out = NULL;
   for (int i = 1; i < argc; i++) {
      if (!strcmp(argv[i], "--size") && i + 1 < argc) sscanf(argv[++i], "%ux%u", &w, &h);
      else if (!strcmp(argv[i], "--tris") && i + 1 < argc) tris = atoi(argv[++i]);
      else if (!strcmp(argv[i], "--frames") && i + 1 < argc) frames = atoi(argv[++i]);
      else if (!strcmp(argv[i], "--out") && i + 1 < argc) out = argv[++i];
   }

   /* The window. */
   int screen_no = 0;
   xcb_connection_t *conn = xcb_connect(NULL, &screen_no);
   if (xcb_connection_has_error(conn)) {
      fprintf(stderr, "cannot connect to the X server (run under xvfb)\n");
      return 2;
   }
   const xcb_setup_t *setup = xcb_get_setup(conn);
   xcb_screen_iterator_t it = xcb_setup_roots_iterator(setup);
   for (int i = 0; i < screen_no; i++) xcb_screen_next(&it);
   xcb_screen_t *screen = it.data;
   xcb_window_t win = xcb_generate_id(conn);
   uint32_t mask = XCB_CW_BACK_PIXEL | XCB_CW_EVENT_MASK, vals[2] = {screen->black_pixel, XCB_EVENT_MASK_EXPOSURE};
   xcb_create_window(conn, XCB_COPY_FROM_PARENT, win, screen->root, 0, 0, (uint16_t)w, (uint16_t)h, 0, XCB_WINDOW_CLASS_INPUT_OUTPUT,
                     screen->root_visual, mask, vals);
   xcb_map_window(conn, win);
   xcb_flush(conn);

   const char *iext[] = {"VK_KHR_surface", "VK_KHR_xcb_surface"};
   VkApplicationInfo app = {VK_STRUCTURE_TYPE_APPLICATION_INFO, .pApplicationName = "vk_present", .apiVersion = VK_API_VERSION_1_1};
   VkInstanceCreateInfo ici = {VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO, .pApplicationInfo = &app, .enabledExtensionCount = 2, .ppEnabledExtensionNames = iext};
   VkInstance inst;
   CHECK(vkCreateInstance(&ici, NULL, &inst));
   uint32_t n = 1;
   vkEnumeratePhysicalDevices(inst, &n, &g_pd);
   VkPhysicalDeviceProperties props;
   vkGetPhysicalDeviceProperties(g_pd, &props);
   fprintf(stderr, "device: %s\n", props.deviceName);

   VkXcbSurfaceCreateInfoKHR sci = {VK_STRUCTURE_TYPE_XCB_SURFACE_CREATE_INFO_KHR, .connection = conn, .window = win};
   VkSurfaceKHR surface;
   CHECK(vkCreateXcbSurfaceKHR(inst, &sci, NULL, &surface));

   uint32_t qf = 0, nq = 0;
   vkGetPhysicalDeviceQueueFamilyProperties(g_pd, &nq, NULL);
   VkQueueFamilyProperties *qp = calloc(nq, sizeof *qp);
   vkGetPhysicalDeviceQueueFamilyProperties(g_pd, &nq, qp);
   for (uint32_t i = 0; i < nq; i++) {
      VkBool32 ok = VK_FALSE;
      vkGetPhysicalDeviceSurfaceSupportKHR(g_pd, i, surface, &ok);
      if ((qp[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) && ok) { qf = i; break; }
   }
   float prio = 1.0f;
   const char *dext[] = {"VK_KHR_swapchain"};
   VkDeviceQueueCreateInfo qi = {VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO, .queueFamilyIndex = qf, .queueCount = 1, .pQueuePriorities = &prio};
   VkDeviceCreateInfo di = {VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO, .queueCreateInfoCount = 1, .pQueueCreateInfos = &qi, .enabledExtensionCount = 1,
                            .ppEnabledExtensionNames = dext};
   CHECK(vkCreateDevice(g_pd, &di, NULL, &g_dev));
   VkQueue queue;
   vkGetDeviceQueue(g_dev, qf, 0, &queue);

   VkSurfaceCapabilitiesKHR caps;
   CHECK(vkGetPhysicalDeviceSurfaceCapabilitiesKHR(g_pd, surface, &caps));
   VkExtent2D ext = caps.currentExtent.width != 0xFFFFFFFFu ? caps.currentExtent : (VkExtent2D){w, h};
   w = ext.width, h = ext.height;
   const VkFormat fmt = VK_FORMAT_B8G8R8A8_UNORM;
   uint32_t want = caps.minImageCount + 1;
   if (caps.maxImageCount && want > caps.maxImageCount) want = caps.maxImageCount;
   VkSwapchainCreateInfoKHR sw = {VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR, .surface = surface, .minImageCount = want, .imageFormat = fmt,
                                  .imageColorSpace = VK_COLOR_SPACE_SRGB_NONLINEAR_KHR, .imageExtent = ext, .imageArrayLayers = 1,
                                  .imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT,
                                  .imageSharingMode = VK_SHARING_MODE_EXCLUSIVE, .preTransform = VK_SURFACE_TRANSFORM_IDENTITY_BIT_KHR,
                                  .compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR, .presentMode = VK_PRESENT_MODE_FIFO_KHR, .clipped = VK_TRUE};
   VkSwapchainKHR chain;
   CHECK(vkCreateSwapchainKHR(g_dev, &sw, NULL, &chain));
   uint32_t nimg = 0;
   vkGetSwapchainImagesKHR(g_dev, chain, &nimg, NULL);
   VkImage *imgs = calloc(nimg, sizeof *imgs);
   vkGetSwapchainImagesKHR(g_dev, chain, &nimg, imgs);

   /* One depth image for all frames (they are serialized by a fence). */
   VkImage depth;
   VkDeviceMemory dmem;
   VkImageView dview;
   {
      VkImageCreateInfo ii = {VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO, .imageType = VK_IMAGE_TYPE_2D, .format = VK_FORMAT_D32_SFLOAT, .extent = {w, h, 1},
                              .mipLevels = 1, .arrayLayers = 1, .samples = VK_SAMPLE_COUNT_1_BIT, .tiling = VK_IMAGE_TILING_OPTIMAL,
                              .usage = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT, .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
                              .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED};
      CHECK(vkCreateImage(g_dev, &ii, NULL, &depth));
      VkMemoryRequirements mr;
      vkGetImageMemoryRequirements(g_dev, depth, &mr);
      VkMemoryAllocateInfo ai = {VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, .allocationSize = mr.size,
                                 .memoryTypeIndex = find_memory(mr.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)};
      CHECK(vkAllocateMemory(g_dev, &ai, NULL, &dmem));
      CHECK(vkBindImageMemory(g_dev, depth, dmem, 0));
      VkImageViewCreateInfo vi = {VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO, .image = depth, .viewType = VK_IMAGE_VIEW_TYPE_2D,
                                  .format = VK_FORMAT_D32_SFLOAT, .subresourceRange = {VK_IMAGE_ASPECT_DEPTH_BIT, 0, 1, 0, 1}};
      CHECK(vkCreateImageView(g_dev, &vi, NULL, &dview));
   }

   VkAttachmentDescription at[2] = {
      {.format = fmt, .samples = VK_SAMPLE_COUNT_1_BIT, .loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR, .storeOp = VK_ATTACHMENT_STORE_OP_STORE,
       .stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE, .stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE,
       .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED, .finalLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL},
      {.format = VK_FORMAT_D32_SFLOAT, .samples = VK_SAMPLE_COUNT_1_BIT, .loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR,
       .storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE, .stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE, .stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE,
       .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED, .finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL}};
   VkAttachmentReference cref = {0, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL}, dref = {1, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL};
   VkSubpassDescription sub = {.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS, .colorAttachmentCount = 1, .pColorAttachments = &cref,
                               .pDepthStencilAttachment = &dref};
   VkRenderPassCreateInfo rpi = {VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO, .attachmentCount = 2, .pAttachments = at, .subpassCount = 1, .pSubpasses = &sub};
   VkRenderPass rp;
   CHECK(vkCreateRenderPass(g_dev, &rpi, NULL, &rp));

   VkFramebuffer *fbs = calloc(nimg, sizeof *fbs);
   for (uint32_t i = 0; i < nimg; i++) {
      VkImageView cv;
      VkImageViewCreateInfo vi = {VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO, .image = imgs[i], .viewType = VK_IMAGE_VIEW_TYPE_2D, .format = fmt,
                                  .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1}};
      CHECK(vkCreateImageView(g_dev, &vi, NULL, &cv));
      VkImageView views[2] = {cv, dview};
      VkFramebufferCreateInfo fbi = {VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO, .renderPass = rp, .attachmentCount = 2, .pAttachments = views, .width = w,
                                     .height = h, .layers = 1};
      CHECK(vkCreateFramebuffer(g_dev, &fbi, NULL, &fbs[i]));
   }

   VkPipelineLayoutCreateInfo pli = {VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
   VkPipelineLayout layout;
   CHECK(vkCreatePipelineLayout(g_dev, &pli, NULL, &layout));
   VkShaderModule vs = make_shader(tri_vert, sizeof tri_vert), fs = make_shader(tri_frag, sizeof tri_frag);
   VkPipelineShaderStageCreateInfo stages[2] = {
      {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, .stage = VK_SHADER_STAGE_VERTEX_BIT, .module = vs, .pName = "main"},
      {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, .stage = VK_SHADER_STAGE_FRAGMENT_BIT, .module = fs, .pName = "main"}};
   VkVertexInputBindingDescription bind = {0, 6 * sizeof(float), VK_VERTEX_INPUT_RATE_VERTEX};
   VkVertexInputAttributeDescription attrs[2] = {{0, 0, VK_FORMAT_R32G32B32_SFLOAT, 0}, {1, 0, VK_FORMAT_R32G32B32_SFLOAT, 3 * sizeof(float)}};
   VkPipelineVertexInputStateCreateInfo vin = {VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO, .vertexBindingDescriptionCount = 1,
                                               .pVertexBindingDescriptions = &bind, .vertexAttributeDescriptionCount = 2, .pVertexAttributeDescriptions = attrs};
   VkPipelineInputAssemblyStateCreateInfo ia = {VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO, .topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST};
   VkViewport vp = {0, 0, (float)w, (float)h, 0, 1};
   VkRect2D sc = {{0, 0}, {w, h}};
   VkPipelineViewportStateCreateInfo vps = {VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO, .viewportCount = 1, .pViewports = &vp, .scissorCount = 1,
                                            .pScissors = &sc};
   VkPipelineRasterizationStateCreateInfo rs = {VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO, .polygonMode = VK_POLYGON_MODE_FILL,
                                                .cullMode = VK_CULL_MODE_NONE, .frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE, .lineWidth = 1.0f};
   VkPipelineMultisampleStateCreateInfo msaa = {VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO, .rasterizationSamples = VK_SAMPLE_COUNT_1_BIT};
   VkPipelineDepthStencilStateCreateInfo ds = {VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO, .depthTestEnable = VK_TRUE,
                                               .depthWriteEnable = VK_TRUE, .depthCompareOp = VK_COMPARE_OP_LESS};
   VkPipelineColorBlendAttachmentState cba = {.colorWriteMask = 0xF};
   VkPipelineColorBlendStateCreateInfo cb = {VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO, .attachmentCount = 1, .pAttachments = &cba};
   VkGraphicsPipelineCreateInfo gpi = {VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO, .stageCount = 2, .pStages = stages, .pVertexInputState = &vin,
                                       .pInputAssemblyState = &ia, .pViewportState = &vps, .pRasterizationState = &rs, .pMultisampleState = &msaa,
                                       .pDepthStencilState = &ds, .pColorBlendState = &cb, .layout = layout, .renderPass = rp, .subpass = 0};
   VkPipeline pipe;
   CHECK(vkCreateGraphicsPipelines(g_dev, VK_NULL_HANDLE, 1, &gpi, NULL, &pipe));

   int cells = (tris + 1) / 2 > 0 ? (tris + 1) / 2 : 1, g = 1;
   while (g * g < cells) g++;
   int ntri = 2 * g * g;
   VkBuffer vb;
   VkDeviceMemory vmem;
   float *v;
   make_buffer((VkDeviceSize)ntri * 3 * 6 * sizeof(float), VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, &vb, &vmem, (void **)&v);
   int t = 0;
   for (int cy = 0; cy < g; cy++)
      for (int cx = 0; cx < g; cx++) {
         float x0 = -1 + 2.0f * cx / g, x1 = -1 + 2.0f * (cx + 1) / g, y0 = -1 + 2.0f * cy / g, y1 = -1 + 2.0f * (cy + 1) / g;
         float z = 0.2f + 0.6f * ((cx * 7 + cy * 13) % 16) / 16.0f;
         float q[2][3][2] = {{{x0, y0}, {x1, y0}, {x1, y1}}, {{x0, y0}, {x1, y1}, {x0, y1}}};
         for (int k = 0; k < 2; k++, t++)
            for (int i = 0; i < 3; i++) {
               float *p = v + (3 * t + i) * 6;
               p[0] = q[k][i][0], p[1] = q[k][i][1], p[2] = z;
               p[3] = (q[k][i][0] + 1) * 0.5f, p[4] = (q[k][i][1] + 1) * 0.5f, p[5] = z;
            }
      }

   VkBuffer rb;
   VkDeviceMemory rmem;
   uint8_t *rpx;
   make_buffer((VkDeviceSize)w * h * 4, VK_BUFFER_USAGE_TRANSFER_DST_BIT, &rb, &rmem, (void **)&rpx);

   VkCommandPoolCreateInfo cpi = {VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO, .flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT, .queueFamilyIndex = qf};
   VkCommandPool pool;
   CHECK(vkCreateCommandPool(g_dev, &cpi, NULL, &pool));
   VkCommandBufferAllocateInfo cai = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO, .commandPool = pool, .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
                                      .commandBufferCount = 1};
   VkCommandBuffer cmd;
   CHECK(vkAllocateCommandBuffers(g_dev, &cai, &cmd));
   VkSemaphoreCreateInfo semi = {VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
   VkSemaphore acquired, rendered;
   CHECK(vkCreateSemaphore(g_dev, &semi, NULL, &acquired));
   CHECK(vkCreateSemaphore(g_dev, &semi, NULL, &rendered));
   VkFenceCreateInfo fci = {VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
   VkFence fence;
   CHECK(vkCreateFence(g_dev, &fci, NULL, &fence));

   for (int f = 0; f < frames; f++) {
      uint32_t idx;
      CHECK(vkAcquireNextImageKHR(g_dev, chain, UINT64_MAX, acquired, VK_NULL_HANDLE, &idx));
      CHECK(vkResetCommandBuffer(cmd, 0));
      VkCommandBufferBeginInfo bi = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
      CHECK(vkBeginCommandBuffer(cmd, &bi));
      /* A different clear colour every frame, so that a frame the offload failed to write is not mistaken for the previous one. */
      const float shade = 0.05f + 0.05f * (float)(f % 4);
      VkClearValue clears[2] = {{.color = {{shade, shade, 0.1f, 1.0f}}}, {.depthStencil = {1.0f, 0}}};
      VkRenderPassBeginInfo rbi = {VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO, .renderPass = rp, .framebuffer = fbs[idx], .renderArea = {{0, 0}, {w, h}},
                                   .clearValueCount = 2, .pClearValues = clears};
      vkCmdBeginRenderPass(cmd, &rbi, VK_SUBPASS_CONTENTS_INLINE);
      vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipe);
      VkDeviceSize off = 0;
      vkCmdBindVertexBuffers(cmd, 0, 1, &vb, &off);
      vkCmdDraw(cmd, 3 * ntri, 1, 0, 0);
      vkCmdEndRenderPass(cmd);
      VkBufferImageCopy bic = {.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1}, .imageExtent = {w, h, 1}};
      vkCmdCopyImageToBuffer(cmd, imgs[idx], VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, rb, 1, &bic);
      VkImageMemoryBarrier bar = {VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER, .srcAccessMask = VK_ACCESS_TRANSFER_READ_BIT, .dstAccessMask = 0,
                                  .oldLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, .newLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
                                  .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED, .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED, .image = imgs[idx],
                                  .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1}};
      vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, 0, 0, NULL, 0, NULL, 1, &bar);
      CHECK(vkEndCommandBuffer(cmd));
      VkPipelineStageFlags wait = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
      VkSubmitInfo si = {VK_STRUCTURE_TYPE_SUBMIT_INFO, .waitSemaphoreCount = 1, .pWaitSemaphores = &acquired, .pWaitDstStageMask = &wait,
                         .commandBufferCount = 1, .pCommandBuffers = &cmd, .signalSemaphoreCount = 1, .pSignalSemaphores = &rendered};
      CHECK(vkQueueSubmit(queue, 1, &si, fence));
      VkPresentInfoKHR pi = {VK_STRUCTURE_TYPE_PRESENT_INFO_KHR, .waitSemaphoreCount = 1, .pWaitSemaphores = &rendered, .swapchainCount = 1,
                             .pSwapchains = &chain, .pImageIndices = &idx};
      CHECK(vkQueuePresentKHR(queue, &pi));
      CHECK(vkWaitForFences(g_dev, 1, &fence, VK_TRUE, UINT64_MAX));
      CHECK(vkResetFences(g_dev, 1, &fence));
   }
   vkDeviceWaitIdle(g_dev);
   xcb_flush(conn);

   /* What the X server holds for the window against the copy of the last image. */
   xcb_get_image_reply_t *im = xcb_get_image_reply(conn, xcb_get_image(conn, XCB_IMAGE_FORMAT_Z_PIXMAP, win, 0, 0, (uint16_t)w, (uint16_t)h, ~0u), NULL);
   if (!im || xcb_get_image_data_length(im) < (int)(w * h * 4)) {
      fprintf(stderr, "cannot read the window\n");
      return 2;
   }
   const uint8_t *xp = xcb_get_image_data(im);
   uint64_t differing = 0, nonbackground = 0;
   for (uint32_t i = 0; i < w * h; i++) {
      /* both are B, G, R, (A or unused) in memory order */
      if (xp[4 * i] != rpx[4 * i] || xp[4 * i + 1] != rpx[4 * i + 1] || xp[4 * i + 2] != rpx[4 * i + 2])
         differing++;
      if (rpx[4 * i + 2] > 60 || rpx[4 * i + 1] > 60)
         nonbackground++;
   }
   printf("%ux%u, %d frames, %d triangles: window and copy of the last image differ in %llu of %u pixels; %llu pixels show triangles\n", w, h, frames,
          ntri, (unsigned long long)differing, w * h, (unsigned long long)nonbackground);
   if (out) {
      FILE *fp = fopen(out, "wb");
      fprintf(fp, "P6\n%u %u\n255\n", w, h);
      for (uint32_t i = 0; i < w * h; i++) {
         const uint8_t rgb[3] = {xp[4 * i + 2], xp[4 * i + 1], xp[4 * i]};
         fwrite(rgb, 1, 3, fp);
      }
      fclose(fp);
   }
   free(im);
   return differing == 0 && nonbackground > 100 ? 0 : 1;
}
