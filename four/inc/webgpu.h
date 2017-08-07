/**
 * \file webgpu.h
 * WebGPU/Dawn wrapper.
 */
#pragma once

#include <webgpu/webgpu.h>

#include "window.h"

namespace webgpu {
WGPUDevice create(window::Handle window, WGPUBackendType type = WGPUBackendType_Force32);

WGPUSwapChain createSwapChain(WGPUDevice device, unsigned width, unsigned height);

/**
 * See \c #createSwapChain();
 */
WGPUTextureFormat getSwapChainFormat(WGPUDevice device);
}
