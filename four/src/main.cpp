#include "webgpu.h"

#include <string.h>
#include <vector>

WGPUDevice device;
WGPUQueue queue;
WGPUSwapChain swapchain;


WGPUBuffer uniformBuffer;
WGPUBuffer vertexBuffer;
WGPUBuffer indexBuffer;
WGPUBindGroupLayout bindGroupLayout;
WGPUBindGroup bindGroup;
WGPURenderPipeline renderPipeline;

/**
 * Current rotation angle (in degrees, updated per frame).
 */
float rotDeg = 0.0f;

/**
 * WGSL equivalent of \c triangle_vert_spirv.
 */
static char const triangle_vert_wgsl[] = R"(
	struct VertexIn {
		@location(0) aPos : vec2<f32>,
		@location(1) aCol : vec3<f32>
	}
	struct VertexOut {
		@location(0) vCol : vec3<f32>,
		@builtin(position) Position : vec4<f32>
	}
	struct Rotation {
		@location(0) degs : f32
	}
	@group(0) @binding(0) var<uniform> uRot : Rotation;
	@vertex
	fn main(input : VertexIn) -> VertexOut {
		var rads : f32 = radians(uRot.degs);
		var cosA : f32 = cos(rads);
		var sinA : f32 = sin(rads);
		var rot : mat3x3<f32> = mat3x3<f32>(
			vec3<f32>( cosA, sinA, 0.0),
			vec3<f32>(-sinA, cosA, 0.0),
			vec3<f32>( 0.0,  0.0,  1.0));
		var output : VertexOut;
		output.Position = vec4<f32>(rot * vec3<f32>(input.aPos, 1.0), 1.0);
		output.Position.x = min(1, max(output.Position.x, -1));
		output.Position.y = min(1, max(output.Position.y, -1));
		output.vCol = input.aCol;
		return output;
	}
)";

/**
 * WGSL equivalent of \c triangle_frag_spirv.
 */
static char const triangle_frag_wgsl[] = R"(
	@fragment
	fn main(@location(0) vCol : vec3<f32>) -> @location(0) vec4<f32> {
		return vec4<f32>(vCol, 1.0);
	}
)";

/**
 * Helper to create a shader from WGSL source.
 *
 * \param[in] code WGSL shader source
 * \param[in] label optional shader name
 */
static WGPUShaderModule createShader(const char* const code, const char* label = nullptr) {
	WGPUShaderModuleWGSLDescriptor wgsl = {};
	wgsl.chain.sType = WGPUSType_ShaderModuleWGSLDescriptor;
	wgsl.code = code;
	WGPUShaderModuleDescriptor desc = {};
	desc.nextInChain = reinterpret_cast<WGPUChainedStruct*>(&wgsl);
	desc.label = label;
	return wgpuDeviceCreateShaderModule(device, &desc);
}

/**
 * Helper to create a buffer.
 *
 * \param[in] data pointer to the start of the raw data
 * \param[in] size number of bytes in \a data
 * \param[in] usage type of buffer
 */
static WGPUBuffer createBuffer(const void* data, size_t size, WGPUBufferUsage usage) {
	WGPUBufferDescriptor desc = {};
	desc.usage = WGPUBufferUsage_CopyDst | usage;
	desc.size  = size;
	WGPUBuffer buffer = wgpuDeviceCreateBuffer(device, &desc);
	wgpuQueueWriteBuffer(queue, buffer, 0, data, size);
	return buffer;
}

/**
 * Create bind group layout.
 */
static WGPUBindGroupLayout createBindGroupLayout() {
    WGPUBufferBindingLayout buf = {};
    buf.type = WGPUBufferBindingType_Uniform;

    // bind group layout (used by both the pipeline layout and uniform bind group, released at the end of this function)
    WGPUBindGroupLayoutEntry bglEntry = {};
    bglEntry.binding = 0;
    bglEntry.visibility = WGPUShaderStage_Vertex;
    bglEntry.buffer = buf;

    WGPUBindGroupLayoutDescriptor bglDesc = {};
    bglDesc.entryCount = 1;
    bglDesc.entries = &bglEntry;
    return wgpuDeviceCreateBindGroupLayout(device, &bglDesc);
}

/**
 * Create uniform buffer.
 */
static WGPUBuffer createUniformBuffer() {
	// create the uniform bind group (note 'rotDeg' is copied here, not bound in any way)
	return createBuffer(&rotDeg, sizeof(rotDeg), WGPUBufferUsage_Uniform);
}

/**
 * Create vertex buffer.
 */
static WGPUBuffer createVertexBuffer() {
    // create the buffers (x, y, r, g, b)
    float const vertData[] = {
        -0.8f, -0.8f, 0.0f, 0.0f, 1.0f, // BL
         0.8f, -0.8f, 0.0f, 1.0f, 0.0f, // BR
        -0.0f,  0.8f, 1.0f, 0.0f, 0.0f, // top
    };
	return createBuffer(vertData, sizeof(vertData), WGPUBufferUsage_Vertex);
}

/**
 * Create index buffer.
 */
static WGPUBuffer createIndexBuffer() {
    uint16_t const indxData[] = {
        0, 1, 2,
        0 // padding (better way of doing this?)
    };
	return createBuffer(indxData, sizeof(indxData), WGPUBufferUsage_Index);
}

/**
 * Create render pipeline.
 */
static WGPURenderPipeline createRenderPipeline(WGPUBindGroupLayout bindGroupLayout) {
    // compile shaders
    // NOTE: these are now the WGSL shaders (tested with Dawn and Chrome Canary)
    WGPUShaderModule vertMod = createShader(triangle_vert_wgsl);
    WGPUShaderModule fragMod = createShader(triangle_frag_wgsl);

    // pipeline layout (used by the render pipeline, released after its creation)
    WGPUPipelineLayoutDescriptor layoutDesc = {};
    layoutDesc.bindGroupLayoutCount = 1;
    layoutDesc.bindGroupLayouts = &bindGroupLayout;
    WGPUPipelineLayout pipelineLayout = wgpuDeviceCreatePipelineLayout(device, &layoutDesc);

    // describe buffer layouts
    WGPUVertexAttribute vertAttrs[2] = {};
    vertAttrs[0].format = WGPUVertexFormat_Float32x2;
    vertAttrs[0].offset = 0;
    vertAttrs[0].shaderLocation = 0;
    vertAttrs[1].format = WGPUVertexFormat_Float32x3;
    vertAttrs[1].offset = 2 * sizeof(float);
    vertAttrs[1].shaderLocation = 1;
    WGPUVertexBufferLayout vertexBufferLayout = {};
    vertexBufferLayout.arrayStride = 5 * sizeof(float);
    vertexBufferLayout.attributeCount = 2;
    vertexBufferLayout.attributes = vertAttrs;

    // Fragment state
    WGPUBlendState blend = {};
    blend.color.operation = WGPUBlendOperation_Add;
    blend.color.srcFactor = WGPUBlendFactor_One;
    blend.color.dstFactor = WGPUBlendFactor_One;
    blend.alpha.operation = WGPUBlendOperation_Add;
    blend.alpha.srcFactor = WGPUBlendFactor_One;
    blend.alpha.dstFactor = WGPUBlendFactor_One;

    WGPUColorTargetState colorTarget = {};
    colorTarget.format = webgpu::getSwapChainFormat(device);
    colorTarget.blend = &blend;
    colorTarget.writeMask = WGPUColorWriteMask_All;

    WGPUFragmentState fragment = {};
    fragment.module = fragMod;
    fragment.entryPoint = "main";
    fragment.targetCount = 1;
    fragment.targets = &colorTarget;

    WGPURenderPipelineDescriptor desc = {};
    desc.fragment = &fragment;

    // Other state
    desc.layout = pipelineLayout;
    desc.depthStencil = nullptr;

    desc.vertex.module = vertMod;
    desc.vertex.entryPoint = "main";
    desc.vertex.bufferCount = 1;//0;
    desc.vertex.buffers = &vertexBufferLayout;

    desc.multisample.count = 1;
    desc.multisample.mask = 0xFFFFFFFF;
    desc.multisample.alphaToCoverageEnabled = false;

    desc.primitive.frontFace = WGPUFrontFace_CCW;
    desc.primitive.cullMode = WGPUCullMode_None;
    desc.primitive.topology = WGPUPrimitiveTopology_TriangleList;
    desc.primitive.stripIndexFormat = WGPUIndexFormat_Undefined;

    WGPURenderPipeline renderPipeline = wgpuDeviceCreateRenderPipeline(device, &desc);

    // partial clean-up (just move to the end, no?)
    wgpuPipelineLayoutRelease(pipelineLayout);

    // last bit of clean-up
    wgpuShaderModuleRelease(fragMod);
    wgpuShaderModuleRelease(vertMod);

	return renderPipeline;
}

/**
 * Create bind group using the bind group layout and list of buffers.
 */
static WGPUBindGroup createBindGroup(WGPUBindGroupLayout bindGroupLayout, const WGPUBuffer& uniformBuffer, const WGPUBuffer& /*vertexBuffer*/, const WGPUBuffer& /*indexBuffer*/) {
	// Note: Only 'uniform' buffer.
	std::vector<WGPUBuffer> buffers;
	buffers.push_back(uniformBuffer);

    WGPUBindGroupDescriptor bgDesc = {};
    bgDesc.layout = bindGroupLayout;
    bgDesc.entryCount = buffers.size();

    // Assuming that buffers are ordered correctly for binding
    std::vector<WGPUBindGroupEntry> bgEntries(buffers.size());
    for (size_t i = 0; i < buffers.size(); ++i) {
        bgEntries[i].binding = static_cast<uint32_t>(i);
        bgEntries[i].buffer = buffers[i];
        bgEntries[i].offset = 0;
        bgEntries[i].size = sizeof(rotDeg);
    }
    bgDesc.entries = bgEntries.data();

    return wgpuDeviceCreateBindGroup(device, &bgDesc);
}

/**
 * Draws using the above pipeline and buffers.
 */
static bool redraw() {
	WGPUTextureView backBufView = wgpuSwapChainGetCurrentTextureView(swapchain);			// create textureView

	WGPURenderPassColorAttachment colorDesc = {};
	colorDesc.view    = backBufView;
	colorDesc.loadOp  = WGPULoadOp_Clear;
	colorDesc.storeOp = WGPUStoreOp_Store;

#ifdef __EMSCRIPTEN__
	// Dawn has both clearValue/clearColor but only Color works; Emscripten only has Value
	colorDesc.clearValue.r = 0.3f;
	colorDesc.clearValue.g = 0.3f;
	colorDesc.clearValue.b = 0.3f;
	colorDesc.clearValue.a = 1.0f;
#else
	colorDesc.clearColor.r = 0.3f;
	colorDesc.clearColor.g = 0.3f;
	colorDesc.clearColor.b = 0.3f;
	colorDesc.clearColor.a = 1.0f;
#endif

	WGPURenderPassDescriptor renderPassDesc = {};
	renderPassDesc.colorAttachmentCount = 1;
	renderPassDesc.colorAttachments = &colorDesc;

	WGPUCommandEncoder encoder = wgpuDeviceCreateCommandEncoder(device, nullptr);
	WGPURenderPassEncoder renderPassEncoder = wgpuCommandEncoderBeginRenderPass(encoder, &renderPassDesc);

	// update the rotation
	rotDeg += 0.1f;
	wgpuQueueWriteBuffer(queue, uniformBuffer, 0, &rotDeg, sizeof(rotDeg));

	// draw the triangle (comment these five lines to simply clear the screen)
	wgpuRenderPassEncoderSetPipeline(renderPassEncoder, renderPipeline);
	wgpuRenderPassEncoderSetBindGroup(renderPassEncoder, 0, bindGroup, 0, 0);
	wgpuRenderPassEncoderSetVertexBuffer(renderPassEncoder, 0, vertexBuffer, 0, WGPU_WHOLE_SIZE);
	wgpuRenderPassEncoderSetIndexBuffer(renderPassEncoder, indexBuffer, WGPUIndexFormat_Uint16, 0, WGPU_WHOLE_SIZE);
	wgpuRenderPassEncoderDrawIndexed(renderPassEncoder, 3, 1, 0, 0, 0);

	wgpuRenderPassEncoderEnd(renderPassEncoder);
	wgpuRenderPassEncoderRelease(renderPassEncoder);										// release pass
	WGPUCommandBuffer commands = wgpuCommandEncoderFinish(encoder, nullptr);				// create commands
	wgpuCommandEncoderRelease(encoder);														// release encoder

	wgpuQueueSubmit(queue, 1, &commands);
	wgpuCommandBufferRelease(commands);														// release commands
#ifndef __EMSCRIPTEN__
	/*
	 * TODO: wgpuSwapChainPresent is unsupported in Emscripten, so what do we do?
	 */
	wgpuSwapChainPresent(swapchain);
#endif
	wgpuTextureViewRelease(backBufView);													// release textureView

	return true;
}

extern "C" int __main__(int /*argc*/, char* /*argv*/[]) {
	if (window::Handle wHnd = window::create()) {
		if ((device = webgpu::create(wHnd))) {
			queue = wgpuDeviceGetQueue(device);
			swapchain = webgpu::createSwapChain(device, 1920, 1080);

			uniformBuffer = createUniformBuffer();
			vertexBuffer = createVertexBuffer();
			indexBuffer = createIndexBuffer();
			bindGroupLayout = createBindGroupLayout();
			renderPipeline = createRenderPipeline(bindGroupLayout);
			bindGroup = createBindGroup(bindGroupLayout, uniformBuffer, vertexBuffer, indexBuffer);

			window::show(wHnd);
			window::loop(wHnd, redraw);

		#ifndef __EMSCRIPTEN__
			wgpuBindGroupRelease(bindGroup);
			wgpuRenderPipelineRelease(renderPipeline);
			wgpuBindGroupLayoutRelease(bindGroupLayout);

			wgpuBufferRelease(indexBuffer);
			wgpuBufferRelease(vertexBuffer);
			wgpuBufferRelease(uniformBuffer);

			wgpuSwapChainRelease(swapchain);
			wgpuQueueRelease(queue);
			wgpuDeviceRelease(device);
		#endif
		}
	#ifndef __EMSCRIPTEN__
		window::destroy(wHnd);
	#endif
	}
	return 0;
}
