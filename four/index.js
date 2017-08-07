"use strict";var Module=typeof Module!="undefined"?Module:{};var moduleOverrides=Object.assign({},Module);var arguments_=[];var thisProgram="./this.program";var quit_=(status,toThrow)=>{throw toThrow};var ENVIRONMENT_IS_WEB=true;var ENVIRONMENT_IS_WORKER=false;var scriptDirectory="";function locateFile(path){return scriptDirectory+path}var read_,readAsync,readBinary;if(ENVIRONMENT_IS_WEB||ENVIRONMENT_IS_WORKER){if(ENVIRONMENT_IS_WORKER){scriptDirectory=self.location.href}else if(typeof document!="undefined"&&document.currentScript){scriptDirectory=document.currentScript.src}if(scriptDirectory.indexOf("blob:")!==0){scriptDirectory=scriptDirectory.substr(0,scriptDirectory.replace(/[?#].*/,"").lastIndexOf("/")+1)}else{scriptDirectory=""}{read_=url=>{var xhr=new XMLHttpRequest;xhr.open("GET",url,false);xhr.send(null);return xhr.responseText};if(ENVIRONMENT_IS_WORKER){readBinary=url=>{var xhr=new XMLHttpRequest;xhr.open("GET",url,false);xhr.responseType="arraybuffer";xhr.send(null);return new Uint8Array(xhr.response)}}readAsync=(url,onload,onerror)=>{var xhr=new XMLHttpRequest;xhr.open("GET",url,true);xhr.responseType="arraybuffer";xhr.onload=()=>{if(xhr.status==200||xhr.status==0&&xhr.response){onload(xhr.response);return}onerror()};xhr.onerror=onerror;xhr.send(null)}}}else{}var out=console.log.bind(console);var err=console.error.bind(console);Object.assign(Module,moduleOverrides);moduleOverrides=null;var wasmBinary;if(typeof WebAssembly!="object"){abort("no native wasm support detected")}var wasmMemory;var ABORT=false;var EXITSTATUS;var HEAP8,HEAPU8,HEAP16,HEAPU16,HEAP32,HEAPU32,HEAPF32,HEAPF64;function updateMemoryViews(){var b=wasmMemory.buffer;Module["HEAP8"]=HEAP8=new Int8Array(b);Module["HEAP16"]=HEAP16=new Int16Array(b);Module["HEAPU8"]=HEAPU8=new Uint8Array(b);Module["HEAPU16"]=HEAPU16=new Uint16Array(b);Module["HEAP32"]=HEAP32=new Int32Array(b);Module["HEAPU32"]=HEAPU32=new Uint32Array(b);Module["HEAPF32"]=HEAPF32=new Float32Array(b);Module["HEAPF64"]=HEAPF64=new Float64Array(b)}var __ATPRERUN__=[];var __ATINIT__=[];var __ATMAIN__=[];var __ATPOSTRUN__=[];var runtimeInitialized=false;function preRun(){callRuntimeCallbacks(__ATPRERUN__)}function initRuntime(){runtimeInitialized=true;callRuntimeCallbacks(__ATINIT__)}function preMain(){callRuntimeCallbacks(__ATMAIN__)}function postRun(){callRuntimeCallbacks(__ATPOSTRUN__)}function addOnInit(cb){__ATINIT__.unshift(cb)}var runDependencies=0;var runDependencyWatcher=null;var dependenciesFulfilled=null;function addRunDependency(id){runDependencies++}function removeRunDependency(id){runDependencies--;if(runDependencies==0){if(runDependencyWatcher!==null){clearInterval(runDependencyWatcher);runDependencyWatcher=null}if(dependenciesFulfilled){var callback=dependenciesFulfilled;dependenciesFulfilled=null;callback()}}}function abort(what){what="Aborted("+what+")";err(what);ABORT=true;EXITSTATUS=1;what+=". Build with -sASSERTIONS for more info.";var e=new WebAssembly.RuntimeError(what);throw e}var dataURIPrefix="data:application/octet-stream;base64,";var isDataURI=filename=>filename.startsWith(dataURIPrefix);var wasmBinaryFile;wasmBinaryFile="index.wasm";if(!isDataURI(wasmBinaryFile)){wasmBinaryFile=locateFile(wasmBinaryFile)}function getBinarySync(file){if(file==wasmBinaryFile&&wasmBinary){return new Uint8Array(wasmBinary)}if(readBinary){return readBinary(file)}throw"both async and sync fetching of the wasm failed"}function getBinaryPromise(binaryFile){if(!wasmBinary&&(ENVIRONMENT_IS_WEB||ENVIRONMENT_IS_WORKER)){if(typeof fetch=="function"){return fetch(binaryFile,{credentials:"same-origin"}).then(response=>{if(!response["ok"]){throw"failed to load wasm binary file at '"+binaryFile+"'"}return response["arrayBuffer"]()}).catch(()=>getBinarySync(binaryFile))}}return Promise.resolve().then(()=>getBinarySync(binaryFile))}function instantiateArrayBuffer(binaryFile,imports,receiver){return getBinaryPromise(binaryFile).then(binary=>WebAssembly.instantiate(binary,imports)).then(instance=>instance).then(receiver,reason=>{err(`failed to asynchronously prepare wasm: ${reason}`);abort(reason)})}function instantiateAsync(binary,binaryFile,imports,callback){if(!binary&&typeof WebAssembly.instantiateStreaming=="function"&&!isDataURI(binaryFile)&&typeof fetch=="function"){return fetch(binaryFile,{credentials:"same-origin"}).then(response=>{var result=WebAssembly.instantiateStreaming(response,imports);return result.then(callback,function(reason){err(`wasm streaming compile failed: ${reason}`);err("falling back to ArrayBuffer instantiation");return instantiateArrayBuffer(binaryFile,imports,callback)})})}return instantiateArrayBuffer(binaryFile,imports,callback)}function createWasm(){var info={"a":wasmImports};function receiveInstance(instance,module){wasmExports=instance.exports;wasmMemory=wasmExports["G"];updateMemoryViews();wasmTable=wasmExports["N"];addOnInit(wasmExports["H"]);removeRunDependency("wasm-instantiate");return wasmExports}addRunDependency("wasm-instantiate");function receiveInstantiationResult(result){receiveInstance(result["instance"])}instantiateAsync(wasmBinary,wasmBinaryFile,info,receiveInstantiationResult);return{}}function glue_preint(){var entry=__glue_main_;if(entry){if(navigator["gpu"]){navigator["gpu"]["requestAdapter"]().then(function(adapter){adapter["requestDevice"]().then(function(device){Module["preinitializedWebGPUDevice"]=device;entry()})},function(){console.error("No WebGPU adapter; not starting")})}else{console.error("No support for WebGPU; not starting")}}else{console.error("Entry point not found; unable to start")}}function ExitStatus(status){this.name="ExitStatus";this.message=`Program terminated with exit(${status})`;this.status=status}var callRuntimeCallbacks=callbacks=>{while(callbacks.length>0){callbacks.shift()(Module)}};var noExitRuntime=true;var _abort=()=>{abort("")};var wasmTableMirror=[];var wasmTable;var getWasmTableEntry=funcPtr=>{var func=wasmTableMirror[funcPtr];if(!func){if(funcPtr>=wasmTableMirror.length)wasmTableMirror.length=funcPtr+1;wasmTableMirror[funcPtr]=func=wasmTable.get(funcPtr)}return func};var _emscripten_request_animation_frame_loop=(cb,userData)=>{function tick(timeStamp){if(getWasmTableEntry(cb)(timeStamp,userData)){requestAnimationFrame(tick)}}return requestAnimationFrame(tick)};var _emscripten_resize_heap=requestedSize=>{var oldSize=HEAPU8.length;requestedSize>>>=0;return false};var withStackSave=f=>{var stack=stackSave();var ret=f();stackRestore(stack);return ret};var lengthBytesUTF8=str=>{var len=0;for(var i=0;i<str.length;++i){var c=str.charCodeAt(i);if(c<=127){len++}else if(c<=2047){len+=2}else if(c>=55296&&c<=57343){len+=4;++i}else{len+=3}}return len};var stringToUTF8Array=(str,heap,outIdx,maxBytesToWrite)=>{if(!(maxBytesToWrite>0))return 0;var startIdx=outIdx;var endIdx=outIdx+maxBytesToWrite-1;for(var i=0;i<str.length;++i){var u=str.charCodeAt(i);if(u>=55296&&u<=57343){var u1=str.charCodeAt(++i);u=65536+((u&1023)<<10)|u1&1023}if(u<=127){if(outIdx>=endIdx)break;heap[outIdx++]=u}else if(u<=2047){if(outIdx+1>=endIdx)break;heap[outIdx++]=192|u>>6;heap[outIdx++]=128|u&63}else if(u<=65535){if(outIdx+2>=endIdx)break;heap[outIdx++]=224|u>>12;heap[outIdx++]=128|u>>6&63;heap[outIdx++]=128|u&63}else{if(outIdx+3>=endIdx)break;heap[outIdx++]=240|u>>18;heap[outIdx++]=128|u>>12&63;heap[outIdx++]=128|u>>6&63;heap[outIdx++]=128|u&63}}heap[outIdx]=0;return outIdx-startIdx};var stringToUTF8=(str,outPtr,maxBytesToWrite)=>stringToUTF8Array(str,HEAPU8,outPtr,maxBytesToWrite);var stringToUTF8OnStack=str=>{var size=lengthBytesUTF8(str)+1;var ret=stackAlloc(size);stringToUTF8(str,ret,size);return ret};var UTF8Decoder=new TextDecoder("utf8");var UTF8ToString=(ptr,maxBytesToRead)=>{if(!ptr)return"";var maxPtr=ptr+maxBytesToRead;for(var end=ptr;!(end>=maxPtr)&&HEAPU8[end];)++end;return UTF8Decoder.decode(HEAPU8.subarray(ptr,end))};var WebGPU={errorCallback:(callback,type,message,userdata)=>{withStackSave(()=>{var messagePtr=stringToUTF8OnStack(message);getWasmTableEntry(callback)(type,messagePtr,userdata)})},initManagers:()=>{if(WebGPU.mgrDevice)return;function Manager(){this.objects={};this.nextId=1;this.create=function(object,wrapper={}){var id=this.nextId++;wrapper.refcount=1;wrapper.object=object;this.objects[id]=wrapper;return id};this.get=function(id){if(!id)return undefined;var o=this.objects[id];return o.object};this.reference=function(id){var o=this.objects[id];o.refcount++};this.release=function(id){var o=this.objects[id];o.refcount--;if(o.refcount<=0){delete this.objects[id]}}}WebGPU.mgrSurface=WebGPU.mgrSurface||new Manager;WebGPU.mgrSwapChain=WebGPU.mgrSwapChain||new Manager;WebGPU.mgrAdapter=WebGPU.mgrAdapter||new Manager;WebGPU.mgrDevice=WebGPU.mgrDevice||new Manager;WebGPU.mgrQueue=WebGPU.mgrQueue||new Manager;WebGPU.mgrCommandBuffer=WebGPU.mgrCommandBuffer||new Manager;WebGPU.mgrCommandEncoder=WebGPU.mgrCommandEncoder||new Manager;WebGPU.mgrRenderPassEncoder=WebGPU.mgrRenderPassEncoder||new Manager;WebGPU.mgrComputePassEncoder=WebGPU.mgrComputePassEncoder||new Manager;WebGPU.mgrBindGroup=WebGPU.mgrBindGroup||new Manager;WebGPU.mgrBuffer=WebGPU.mgrBuffer||new Manager;WebGPU.mgrSampler=WebGPU.mgrSampler||new Manager;WebGPU.mgrTexture=WebGPU.mgrTexture||new Manager;WebGPU.mgrTextureView=WebGPU.mgrTextureView||new Manager;WebGPU.mgrQuerySet=WebGPU.mgrQuerySet||new Manager;WebGPU.mgrBindGroupLayout=WebGPU.mgrBindGroupLayout||new Manager;WebGPU.mgrPipelineLayout=WebGPU.mgrPipelineLayout||new Manager;WebGPU.mgrRenderPipeline=WebGPU.mgrRenderPipeline||new Manager;WebGPU.mgrComputePipeline=WebGPU.mgrComputePipeline||new Manager;WebGPU.mgrShaderModule=WebGPU.mgrShaderModule||new Manager;WebGPU.mgrRenderBundleEncoder=WebGPU.mgrRenderBundleEncoder||new Manager;WebGPU.mgrRenderBundle=WebGPU.mgrRenderBundle||new Manager},makeColor:ptr=>({"r":HEAPF64[ptr>>3],"g":HEAPF64[ptr+8>>3],"b":HEAPF64[ptr+16>>3],"a":HEAPF64[ptr+24>>3]}),makeExtent3D:ptr=>({"width":HEAPU32[ptr>>2],"height":HEAPU32[ptr+4>>2],"depthOrArrayLayers":HEAPU32[ptr+8>>2]}),makeOrigin3D:ptr=>({"x":HEAPU32[ptr>>2],"y":HEAPU32[ptr+4>>2],"z":HEAPU32[ptr+8>>2]}),makeImageCopyTexture:ptr=>({"texture":WebGPU.mgrTexture.get(HEAPU32[ptr+4>>2]),"mipLevel":HEAPU32[ptr+8>>2],"origin":WebGPU.makeOrigin3D(ptr+12),"aspect":WebGPU.TextureAspect[HEAPU32[ptr+24>>2]]}),makeTextureDataLayout:ptr=>{var bytesPerRow=HEAPU32[ptr+16>>2];var rowsPerImage=HEAPU32[ptr+20>>2];return{"offset":HEAPU32[ptr+4+8>>2]*4294967296+HEAPU32[ptr+8>>2],"bytesPerRow":bytesPerRow===4294967295?undefined:bytesPerRow,"rowsPerImage":rowsPerImage===4294967295?undefined:rowsPerImage}},makeImageCopyBuffer:ptr=>{var layoutPtr=ptr+8;var bufferCopyView=WebGPU.makeTextureDataLayout(layoutPtr);bufferCopyView["buffer"]=WebGPU.mgrBuffer.get(HEAPU32[ptr+32>>2]);return bufferCopyView},makePipelineConstants:(constantCount,constantsPtr)=>{if(!constantCount)return;var constants={};for(var i=0;i<constantCount;++i){var entryPtr=constantsPtr+16*i;var key=UTF8ToString(HEAPU32[entryPtr+4>>2]);constants[key]=HEAPF64[entryPtr+8>>3]}return constants},makePipelineLayout:layoutPtr=>{if(!layoutPtr)return"auto";return WebGPU.mgrPipelineLayout.get(layoutPtr)},makeProgrammableStageDescriptor:ptr=>{if(!ptr)return undefined;var desc={"module":WebGPU.mgrShaderModule.get(HEAPU32[ptr+4>>2]),"constants":WebGPU.makePipelineConstants(HEAPU32[ptr+12>>2],HEAPU32[ptr+16>>2])};var entryPointPtr=HEAPU32[ptr+8>>2];if(entryPointPtr)desc["entryPoint"]=UTF8ToString(entryPointPtr);return desc},DeviceLostReason:{undefined:0,destroyed:1},PreferredFormat:{rgba8unorm:18,bgra8unorm:23},BufferMapState:["unmapped","pending","mapped"],AddressMode:["repeat","mirror-repeat","clamp-to-edge"],BlendFactor:["zero","one","src","one-minus-src","src-alpha","one-minus-src-alpha","dst","one-minus-dst","dst-alpha","one-minus-dst-alpha","src-alpha-saturated","constant","one-minus-constant"],BlendOperation:["add","subtract","reverse-subtract","min","max"],BufferBindingType:[,"uniform","storage","read-only-storage"],CompareFunction:[,"never","less","less-equal","greater","greater-equal","equal","not-equal","always"],CompilationInfoRequestStatus:["success","error","device-lost","unknown"],CullMode:["none","front","back"],ErrorFilter:["validation","out-of-memory","internal"],FeatureName:[,"depth-clip-control","depth32float-stencil8","timestamp-query","texture-compression-bc","texture-compression-etc2","texture-compression-astc","indirect-first-instance","shader-f16","rg11b10ufloat-renderable","bgra8unorm-storage","float32filterable"],FilterMode:["nearest","linear"],FrontFace:["ccw","cw"],IndexFormat:[,"uint16","uint32"],LoadOp:[,"clear","load"],MipmapFilterMode:["nearest","linear"],PowerPreference:[,"low-power","high-performance"],PrimitiveTopology:["point-list","line-list","line-strip","triangle-list","triangle-strip"],QueryType:["occlusion","timestamp"],SamplerBindingType:[,"filtering","non-filtering","comparison"],StencilOperation:["keep","zero","replace","invert","increment-clamp","decrement-clamp","increment-wrap","decrement-wrap"],StorageTextureAccess:[,"write-only"],StoreOp:[,"store","discard"],TextureAspect:["all","stencil-only","depth-only"],TextureDimension:["1d","2d","3d"],TextureFormat:[,"r8unorm","r8snorm","r8uint","r8sint","r16uint","r16sint","r16float","rg8unorm","rg8snorm","rg8uint","rg8sint","r32float","r32uint","r32sint","rg16uint","rg16sint","rg16float","rgba8unorm","rgba8unorm-srgb","rgba8snorm","rgba8uint","rgba8sint","bgra8unorm","bgra8unorm-srgb","rgb10a2unorm","rg11b10ufloat","rgb9e5ufloat","rg32float","rg32uint","rg32sint","rgba16uint","rgba16sint","rgba16float","rgba32float","rgba32uint","rgba32sint","stencil8","depth16unorm","depth24plus","depth24plus-stencil8","depth32float","depth32float-stencil8","bc1-rgba-unorm","bc1-rgba-unorm-srgb","bc2-rgba-unorm","bc2-rgba-unorm-srgb","bc3-rgba-unorm","bc3-rgba-unorm-srgb","bc4-r-unorm","bc4-r-snorm","bc5-rg-unorm","bc5-rg-snorm","bc6h-rgb-ufloat","bc6h-rgb-float","bc7-rgba-unorm","bc7-rgba-unorm-srgb","etc2-rgb8unorm","etc2-rgb8unorm-srgb","etc2-rgb8a1unorm","etc2-rgb8a1unorm-srgb","etc2-rgba8unorm","etc2-rgba8unorm-srgb","eac-r11unorm","eac-r11snorm","eac-rg11unorm","eac-rg11snorm","astc-4x4-unorm","astc-4x4-unorm-srgb","astc-5x4-unorm","astc-5x4-unorm-srgb","astc-5x5-unorm","astc-5x5-unorm-srgb","astc-6x5-unorm","astc-6x5-unorm-srgb","astc-6x6-unorm","astc-6x6-unorm-srgb","astc-8x5-unorm","astc-8x5-unorm-srgb","astc-8x6-unorm","astc-8x6-unorm-srgb","astc-8x8-unorm","astc-8x8-unorm-srgb","astc-10x5-unorm","astc-10x5-unorm-srgb","astc-10x6-unorm","astc-10x6-unorm-srgb","astc-10x8-unorm","astc-10x8-unorm-srgb","astc-10x10-unorm","astc-10x10-unorm-srgb","astc-12x10-unorm","astc-12x10-unorm-srgb","astc-12x12-unorm","astc-12x12-unorm-srgb"],TextureSampleType:[,"float","unfilterable-float","depth","sint","uint"],TextureViewDimension:[,"1d","2d","2d-array","cube","cube-array","3d"],VertexFormat:[,"uint8x2","uint8x4","sint8x2","sint8x4","unorm8x2","unorm8x4","snorm8x2","snorm8x4","uint16x2","uint16x4","sint16x2","sint16x4","unorm16x2","unorm16x4","snorm16x2","snorm16x4","float16x2","float16x4","float32","float32x2","float32x3","float32x4","uint32","uint32x2","uint32x3","uint32x4","sint32","sint32x2","sint32x3","sint32x4"],VertexStepMode:["vertex","instance"],FeatureNameString2Enum:{undefined:"0","depth-clip-control":"1","depth32float-stencil8":"2","timestamp-query":"3","texture-compression-bc":"4","texture-compression-etc2":"5","texture-compression-astc":"6","indirect-first-instance":"7","shader-f16":"8","rg11b10ufloat-renderable":"9","bgra8unorm-storage":"10",float32filterable:"11"}};var _emscripten_webgpu_get_device=()=>{if(WebGPU.preinitializedDeviceId===undefined){var device=Module["preinitializedWebGPUDevice"];var deviceWrapper={queueId:WebGPU.mgrQueue.create(device["queue"])};WebGPU.preinitializedDeviceId=WebGPU.mgrDevice.create(device,deviceWrapper)}WebGPU.mgrDevice.reference(WebGPU.preinitializedDeviceId);return WebGPU.preinitializedDeviceId};var _wgpuCommandBufferRelease=id=>WebGPU.mgrCommandBuffer.release(id);var _wgpuCommandEncoderBeginRenderPass=(encoderId,descriptor)=>{function makeColorAttachment(caPtr){var viewPtr=HEAPU32[caPtr+4>>2];if(viewPtr===0){return undefined}var loadOpInt=HEAPU32[caPtr+12>>2];var storeOpInt=HEAPU32[caPtr+16>>2];var clearValue=WebGPU.makeColor(caPtr+24);return{"view":WebGPU.mgrTextureView.get(viewPtr),"resolveTarget":WebGPU.mgrTextureView.get(HEAPU32[caPtr+8>>2]),"clearValue":clearValue,"loadOp":WebGPU.LoadOp[loadOpInt],"storeOp":WebGPU.StoreOp[storeOpInt]}}function makeColorAttachments(count,caPtr){var attachments=[];for(var i=0;i<count;++i){attachments.push(makeColorAttachment(caPtr+56*i))}return attachments}function makeDepthStencilAttachment(dsaPtr){if(dsaPtr===0)return undefined;return{"view":WebGPU.mgrTextureView.get(HEAPU32[dsaPtr>>2]),"depthClearValue":HEAPF32[dsaPtr+12>>2],"depthLoadOp":WebGPU.LoadOp[HEAPU32[dsaPtr+4>>2]],"depthStoreOp":WebGPU.StoreOp[HEAPU32[dsaPtr+8>>2]],"depthReadOnly":HEAP8[dsaPtr+16>>0]!==0,"stencilClearValue":HEAPU32[dsaPtr+28>>2],"stencilLoadOp":WebGPU.LoadOp[HEAPU32[dsaPtr+20>>2]],"stencilStoreOp":WebGPU.StoreOp[HEAPU32[dsaPtr+24>>2]],"stencilReadOnly":HEAP8[dsaPtr+32>>0]!==0}}function makeRenderPassTimestampWrites(twPtr){if(twPtr===0)return undefined;return{"querySet":WebGPU.mgrQuerySet.get(HEAPU32[twPtr>>2]),"beginningOfPassWriteIndex":HEAPU32[twPtr+4>>2],"endOfPassWriteIndex":HEAPU32[twPtr+8>>2]}}function makeRenderPassDescriptor(descriptor){var nextInChainPtr=HEAPU32[descriptor>>2];var maxDrawCount=undefined;if(nextInChainPtr!==0){var sType=HEAPU32[nextInChainPtr+4>>2];var renderPassDescriptorMaxDrawCount=nextInChainPtr;maxDrawCount=HEAPU32[renderPassDescriptorMaxDrawCount+4+8>>2]*4294967296+HEAPU32[renderPassDescriptorMaxDrawCount+8>>2]}var desc={"label":undefined,"colorAttachments":makeColorAttachments(HEAPU32[descriptor+8>>2],HEAPU32[descriptor+12>>2]),"depthStencilAttachment":makeDepthStencilAttachment(HEAPU32[descriptor+16>>2]),"occlusionQuerySet":WebGPU.mgrQuerySet.get(HEAPU32[descriptor+20>>2]),"timestampWrites":makeRenderPassTimestampWrites(HEAPU32[descriptor+24>>2]),"maxDrawCount":maxDrawCount};var labelPtr=HEAPU32[descriptor+4>>2];if(labelPtr)desc["label"]=UTF8ToString(labelPtr);return desc}var desc=makeRenderPassDescriptor(descriptor);var commandEncoder=WebGPU.mgrCommandEncoder.get(encoderId);return WebGPU.mgrRenderPassEncoder.create(commandEncoder["beginRenderPass"](desc))};var _wgpuCommandEncoderFinish=(encoderId,descriptor)=>{var commandEncoder=WebGPU.mgrCommandEncoder.get(encoderId);return WebGPU.mgrCommandBuffer.create(commandEncoder["finish"]())};var _wgpuCommandEncoderRelease=id=>WebGPU.mgrCommandEncoder.release(id);var readI53FromI64=ptr=>HEAPU32[ptr>>2]+HEAP32[ptr+4>>2]*4294967296;var _wgpuDeviceCreateBindGroup=(deviceId,descriptor)=>{function makeEntry(entryPtr){var bufferId=HEAPU32[entryPtr+8>>2];var samplerId=HEAPU32[entryPtr+32>>2];var textureViewId=HEAPU32[entryPtr+36>>2];var binding=HEAPU32[entryPtr+4>>2];if(bufferId){var size=readI53FromI64(entryPtr+24);if(size==-1)size=undefined;return{"binding":binding,"resource":{"buffer":WebGPU.mgrBuffer.get(bufferId),"offset":HEAPU32[entryPtr+4+16>>2]*4294967296+HEAPU32[entryPtr+16>>2],"size":size}}}else if(samplerId){return{"binding":binding,"resource":WebGPU.mgrSampler.get(samplerId)}}else{return{"binding":binding,"resource":WebGPU.mgrTextureView.get(textureViewId)}}}function makeEntries(count,entriesPtrs){var entries=[];for(var i=0;i<count;++i){entries.push(makeEntry(entriesPtrs+40*i))}return entries}var desc={"label":undefined,"layout":WebGPU.mgrBindGroupLayout.get(HEAPU32[descriptor+8>>2]),"entries":makeEntries(HEAPU32[descriptor+12>>2],HEAPU32[descriptor+16>>2])};var labelPtr=HEAPU32[descriptor+4>>2];if(labelPtr)desc["label"]=UTF8ToString(labelPtr);var device=WebGPU.mgrDevice.get(deviceId);return WebGPU.mgrBindGroup.create(device["createBindGroup"](desc))};var _wgpuDeviceCreateBindGroupLayout=(deviceId,descriptor)=>{function makeBufferEntry(entryPtr){var typeInt=HEAPU32[entryPtr+4>>2];if(!typeInt)return undefined;return{"type":WebGPU.BufferBindingType[typeInt],"hasDynamicOffset":HEAP8[entryPtr+8>>0]!==0,"minBindingSize":HEAPU32[entryPtr+4+16>>2]*4294967296+HEAPU32[entryPtr+16>>2]}}function makeSamplerEntry(entryPtr){var typeInt=HEAPU32[entryPtr+4>>2];if(!typeInt)return undefined;return{"type":WebGPU.SamplerBindingType[typeInt]}}function makeTextureEntry(entryPtr){var sampleTypeInt=HEAPU32[entryPtr+4>>2];if(!sampleTypeInt)return undefined;return{"sampleType":WebGPU.TextureSampleType[sampleTypeInt],"viewDimension":WebGPU.TextureViewDimension[HEAPU32[entryPtr+8>>2]],"multisampled":HEAP8[entryPtr+12>>0]!==0}}function makeStorageTextureEntry(entryPtr){var accessInt=HEAPU32[entryPtr+4>>2];if(!accessInt)return undefined;return{"access":WebGPU.StorageTextureAccess[accessInt],"format":WebGPU.TextureFormat[HEAPU32[entryPtr+8>>2]],"viewDimension":WebGPU.TextureViewDimension[HEAPU32[entryPtr+12>>2]]}}function makeEntry(entryPtr){return{"binding":HEAPU32[entryPtr+4>>2],"visibility":HEAPU32[entryPtr+8>>2],"buffer":makeBufferEntry(entryPtr+16),"sampler":makeSamplerEntry(entryPtr+40),"texture":makeTextureEntry(entryPtr+48),"storageTexture":makeStorageTextureEntry(entryPtr+64)}}function makeEntries(count,entriesPtrs){var entries=[];for(var i=0;i<count;++i){entries.push(makeEntry(entriesPtrs+80*i))}return entries}var desc={"entries":makeEntries(HEAPU32[descriptor+8>>2],HEAPU32[descriptor+12>>2])};var labelPtr=HEAPU32[descriptor+4>>2];if(labelPtr)desc["label"]=UTF8ToString(labelPtr);var device=WebGPU.mgrDevice.get(deviceId);return WebGPU.mgrBindGroupLayout.create(device["createBindGroupLayout"](desc))};var _wgpuDeviceCreateBuffer=(deviceId,descriptor)=>{var mappedAtCreation=HEAP8[descriptor+24>>0]!==0;var desc={"label":undefined,"usage":HEAPU32[descriptor+8>>2],"size":HEAPU32[descriptor+4+16>>2]*4294967296+HEAPU32[descriptor+16>>2],"mappedAtCreation":mappedAtCreation};var labelPtr=HEAPU32[descriptor+4>>2];if(labelPtr)desc["label"]=UTF8ToString(labelPtr);var device=WebGPU.mgrDevice.get(deviceId);var bufferWrapper={};var id=WebGPU.mgrBuffer.create(device["createBuffer"](desc),bufferWrapper);if(mappedAtCreation){bufferWrapper.mapMode=2;bufferWrapper.onUnmap=[]}return id};var _wgpuDeviceCreateCommandEncoder=(deviceId,descriptor)=>{var desc;if(descriptor){desc={"label":undefined};var labelPtr=HEAPU32[descriptor+4>>2];if(labelPtr)desc["label"]=UTF8ToString(labelPtr)}var device=WebGPU.mgrDevice.get(deviceId);return WebGPU.mgrCommandEncoder.create(device["createCommandEncoder"](desc))};var _wgpuDeviceCreatePipelineLayout=(deviceId,descriptor)=>{var bglCount=HEAPU32[descriptor+8>>2];var bglPtr=HEAPU32[descriptor+12>>2];var bgls=[];for(var i=0;i<bglCount;++i){bgls.push(WebGPU.mgrBindGroupLayout.get(HEAPU32[bglPtr+4*i>>2]))}var desc={"label":undefined,"bindGroupLayouts":bgls};var labelPtr=HEAPU32[descriptor+4>>2];if(labelPtr)desc["label"]=UTF8ToString(labelPtr);var device=WebGPU.mgrDevice.get(deviceId);return WebGPU.mgrPipelineLayout.create(device["createPipelineLayout"](desc))};var generateRenderPipelineDesc=descriptor=>{function makePrimitiveState(rsPtr){if(!rsPtr)return undefined;return{"topology":WebGPU.PrimitiveTopology[HEAPU32[rsPtr+4>>2]],"stripIndexFormat":WebGPU.IndexFormat[HEAPU32[rsPtr+8>>2]],"frontFace":WebGPU.FrontFace[HEAPU32[rsPtr+12>>2]],"cullMode":WebGPU.CullMode[HEAPU32[rsPtr+16>>2]]}}function makeBlendComponent(bdPtr){if(!bdPtr)return undefined;return{"operation":WebGPU.BlendOperation[HEAPU32[bdPtr>>2]],"srcFactor":WebGPU.BlendFactor[HEAPU32[bdPtr+4>>2]],"dstFactor":WebGPU.BlendFactor[HEAPU32[bdPtr+8>>2]]}}function makeBlendState(bsPtr){if(!bsPtr)return undefined;return{"alpha":makeBlendComponent(bsPtr+12),"color":makeBlendComponent(bsPtr+0)}}function makeColorState(csPtr){var formatInt=HEAPU32[csPtr+4>>2];return formatInt===0?undefined:{"format":WebGPU.TextureFormat[formatInt],"blend":makeBlendState(HEAPU32[csPtr+8>>2]),"writeMask":HEAPU32[csPtr+12>>2]}}function makeColorStates(count,csArrayPtr){var states=[];for(var i=0;i<count;++i){states.push(makeColorState(csArrayPtr+16*i))}return states}function makeStencilStateFace(ssfPtr){return{"compare":WebGPU.CompareFunction[HEAPU32[ssfPtr>>2]],"failOp":WebGPU.StencilOperation[HEAPU32[ssfPtr+4>>2]],"depthFailOp":WebGPU.StencilOperation[HEAPU32[ssfPtr+8>>2]],"passOp":WebGPU.StencilOperation[HEAPU32[ssfPtr+12>>2]]}}function makeDepthStencilState(dssPtr){if(!dssPtr)return undefined;return{"format":WebGPU.TextureFormat[HEAPU32[dssPtr+4>>2]],"depthWriteEnabled":HEAP8[dssPtr+8>>0]!==0,"depthCompare":WebGPU.CompareFunction[HEAPU32[dssPtr+12>>2]],"stencilFront":makeStencilStateFace(dssPtr+16),"stencilBack":makeStencilStateFace(dssPtr+32),"stencilReadMask":HEAPU32[dssPtr+48>>2],"stencilWriteMask":HEAPU32[dssPtr+52>>2],"depthBias":HEAP32[dssPtr+56>>2],"depthBiasSlopeScale":HEAPF32[dssPtr+60>>2],"depthBiasClamp":HEAPF32[dssPtr+64>>2]}}function makeVertexAttribute(vaPtr){return{"format":WebGPU.VertexFormat[HEAPU32[vaPtr>>2]],"offset":HEAPU32[vaPtr+4+8>>2]*4294967296+HEAPU32[vaPtr+8>>2],"shaderLocation":HEAPU32[vaPtr+16>>2]}}function makeVertexAttributes(count,vaArrayPtr){var vas=[];for(var i=0;i<count;++i){vas.push(makeVertexAttribute(vaArrayPtr+i*24))}return vas}function makeVertexBuffer(vbPtr){if(!vbPtr)return undefined;var stepModeInt=HEAPU32[vbPtr+8>>2];return stepModeInt===2?null:{"arrayStride":HEAPU32[vbPtr+4>>2]*4294967296+HEAPU32[vbPtr>>2],"stepMode":WebGPU.VertexStepMode[stepModeInt],"attributes":makeVertexAttributes(HEAPU32[vbPtr+12>>2],HEAPU32[vbPtr+16>>2])}}function makeVertexBuffers(count,vbArrayPtr){if(!count)return undefined;var vbs=[];for(var i=0;i<count;++i){vbs.push(makeVertexBuffer(vbArrayPtr+i*24))}return vbs}function makeVertexState(viPtr){if(!viPtr)return undefined;var desc={"module":WebGPU.mgrShaderModule.get(HEAPU32[viPtr+4>>2]),"constants":WebGPU.makePipelineConstants(HEAPU32[viPtr+12>>2],HEAPU32[viPtr+16>>2]),"buffers":makeVertexBuffers(HEAPU32[viPtr+20>>2],HEAPU32[viPtr+24>>2])};var entryPointPtr=HEAPU32[viPtr+8>>2];if(entryPointPtr)desc["entryPoint"]=UTF8ToString(entryPointPtr);return desc}function makeMultisampleState(msPtr){if(!msPtr)return undefined;return{"count":HEAPU32[msPtr+4>>2],"mask":HEAPU32[msPtr+8>>2],"alphaToCoverageEnabled":HEAP8[msPtr+12>>0]!==0}}function makeFragmentState(fsPtr){if(!fsPtr)return undefined;var desc={"module":WebGPU.mgrShaderModule.get(HEAPU32[fsPtr+4>>2]),"constants":WebGPU.makePipelineConstants(HEAPU32[fsPtr+12>>2],HEAPU32[fsPtr+16>>2]),"targets":makeColorStates(HEAPU32[fsPtr+20>>2],HEAPU32[fsPtr+24>>2])};var entryPointPtr=HEAPU32[fsPtr+8>>2];if(entryPointPtr)desc["entryPoint"]=UTF8ToString(entryPointPtr);return desc}var desc={"label":undefined,"layout":WebGPU.makePipelineLayout(HEAPU32[descriptor+8>>2]),"vertex":makeVertexState(descriptor+12),"primitive":makePrimitiveState(descriptor+40),"depthStencil":makeDepthStencilState(HEAPU32[descriptor+60>>2]),"multisample":makeMultisampleState(descriptor+64),"fragment":makeFragmentState(HEAPU32[descriptor+80>>2])};var labelPtr=HEAPU32[descriptor+4>>2];if(labelPtr)desc["label"]=UTF8ToString(labelPtr);return desc};var _wgpuDeviceCreateRenderPipeline=(deviceId,descriptor)=>{var desc=generateRenderPipelineDesc(descriptor);var device=WebGPU.mgrDevice.get(deviceId);return WebGPU.mgrRenderPipeline.create(device["createRenderPipeline"](desc))};var _wgpuDeviceCreateShaderModule=(deviceId,descriptor)=>{var nextInChainPtr=HEAPU32[descriptor>>2];var sType=HEAPU32[nextInChainPtr+4>>2];var desc={"label":undefined,"code":""};var labelPtr=HEAPU32[descriptor+4>>2];if(labelPtr)desc["label"]=UTF8ToString(labelPtr);switch(sType){case 5:{var count=HEAPU32[nextInChainPtr+8>>2];var start=HEAPU32[nextInChainPtr+12>>2];var offset=start>>2;desc["code"]=HEAPU32.subarray(offset,offset+count);break}case 6:{var sourcePtr=HEAPU32[nextInChainPtr+8>>2];if(sourcePtr){desc["code"]=UTF8ToString(sourcePtr)}break}}var device=WebGPU.mgrDevice.get(deviceId);return WebGPU.mgrShaderModule.create(device["createShaderModule"](desc))};var _wgpuDeviceCreateSwapChain=(deviceId,surfaceId,descriptor)=>{var device=WebGPU.mgrDevice.get(deviceId);var context=WebGPU.mgrSurface.get(surfaceId);var canvasSize=[HEAPU32[descriptor+16>>2],HEAPU32[descriptor+20>>2]];if(canvasSize[0]!==0){context["canvas"]["width"]=canvasSize[0]}if(canvasSize[1]!==0){context["canvas"]["height"]=canvasSize[1]}var configuration={"device":device,"format":WebGPU.TextureFormat[HEAPU32[descriptor+12>>2]],"usage":HEAPU32[descriptor+8>>2],"alphaMode":"opaque"};context["configure"](configuration);return WebGPU.mgrSwapChain.create(context)};var _wgpuDeviceGetQueue=deviceId=>{var queueId=WebGPU.mgrDevice.objects[deviceId].queueId;WebGPU.mgrQueue.reference(queueId);return queueId};var maybeCStringToJsString=cString=>cString>2?UTF8ToString(cString):cString;var specialHTMLTargets=[0,document,window];var findEventTarget=target=>{target=maybeCStringToJsString(target);var domElement=specialHTMLTargets[target]||document.querySelector(target);return domElement};var findCanvasEventTarget=target=>findEventTarget(target);var _wgpuInstanceCreateSurface=(instanceId,descriptor)=>{var nextInChainPtr=HEAPU32[descriptor>>2];var descriptorFromCanvasHTMLSelector=nextInChainPtr;var selectorPtr=HEAPU32[descriptorFromCanvasHTMLSelector+8>>2];var canvas=findCanvasEventTarget(selectorPtr);var context=canvas.getContext("webgpu");if(!context)return 0;var labelPtr=HEAPU32[descriptor+4>>2];if(labelPtr)context.surfaceLabelWebGPU=UTF8ToString(labelPtr);return WebGPU.mgrSurface.create(context)};var _wgpuPipelineLayoutRelease=id=>WebGPU.mgrPipelineLayout.release(id);var _wgpuQueueSubmit=(queueId,commandCount,commands)=>{var queue=WebGPU.mgrQueue.get(queueId);var cmds=Array.from(HEAP32.subarray(commands>>2,commands+commandCount*4>>2),function(id){return WebGPU.mgrCommandBuffer.get(id)});queue["submit"](cmds)};var convertI32PairToI53Checked=(lo,hi)=>hi+2097152>>>0<4194305-!!lo?(lo>>>0)+hi*4294967296:NaN;function _wgpuQueueWriteBuffer(queueId,bufferId,bufferOffset_low,bufferOffset_high,data,size){var bufferOffset=convertI32PairToI53Checked(bufferOffset_low,bufferOffset_high);var queue=WebGPU.mgrQueue.get(queueId);var buffer=WebGPU.mgrBuffer.get(bufferId);var subarray=HEAPU8.subarray(data,data+size);queue["writeBuffer"](buffer,bufferOffset,subarray,0,size)}var _wgpuRenderPassEncoderDrawIndexed=(passId,indexCount,instanceCount,firstIndex,baseVertex,firstInstance)=>{var pass=WebGPU.mgrRenderPassEncoder.get(passId);pass["drawIndexed"](indexCount,instanceCount,firstIndex,baseVertex,firstInstance)};var _wgpuRenderPassEncoderEnd=encoderId=>{var encoder=WebGPU.mgrRenderPassEncoder.get(encoderId);encoder["end"]()};var _wgpuRenderPassEncoderRelease=id=>WebGPU.mgrRenderPassEncoder.release(id);var _wgpuRenderPassEncoderSetBindGroup=(passId,groupIndex,groupId,dynamicOffsetCount,dynamicOffsetsPtr)=>{var pass=WebGPU.mgrRenderPassEncoder.get(passId);var group=WebGPU.mgrBindGroup.get(groupId);if(dynamicOffsetCount==0){pass["setBindGroup"](groupIndex,group)}else{var offsets=[];for(var i=0;i<dynamicOffsetCount;i++,dynamicOffsetsPtr+=4){offsets.push(HEAPU32[dynamicOffsetsPtr>>2])}pass["setBindGroup"](groupIndex,group,offsets)}};function _wgpuRenderPassEncoderSetIndexBuffer(passId,bufferId,format,offset_low,offset_high,size_low,size_high){var offset=convertI32PairToI53Checked(offset_low,offset_high);var size=convertI32PairToI53Checked(size_low,size_high);var pass=WebGPU.mgrRenderPassEncoder.get(passId);var buffer=WebGPU.mgrBuffer.get(bufferId);if(size==-1)size=undefined;pass["setIndexBuffer"](buffer,WebGPU.IndexFormat[format],offset,size)}var _wgpuRenderPassEncoderSetPipeline=(passId,pipelineId)=>{var pass=WebGPU.mgrRenderPassEncoder.get(passId);var pipeline=WebGPU.mgrRenderPipeline.get(pipelineId);pass["setPipeline"](pipeline)};function _wgpuRenderPassEncoderSetVertexBuffer(passId,slot,bufferId,offset_low,offset_high,size_low,size_high){var offset=convertI32PairToI53Checked(offset_low,offset_high);var size=convertI32PairToI53Checked(size_low,size_high);var pass=WebGPU.mgrRenderPassEncoder.get(passId);var buffer=WebGPU.mgrBuffer.get(bufferId);if(size==-1)size=undefined;pass["setVertexBuffer"](slot,buffer,offset,size)}var _wgpuShaderModuleRelease=id=>WebGPU.mgrShaderModule.release(id);var _wgpuSwapChainGetCurrentTextureView=swapChainId=>{var context=WebGPU.mgrSwapChain.get(swapChainId);return WebGPU.mgrTextureView.create(context["getCurrentTexture"]()["createView"]())};var _wgpuTextureViewRelease=id=>WebGPU.mgrTextureView.release(id);var runtimeKeepaliveCounter=0;var keepRuntimeAlive=()=>noExitRuntime||runtimeKeepaliveCounter>0;var SYSCALLS={varargs:undefined,get(){var ret=HEAP32[+SYSCALLS.varargs>>2];SYSCALLS.varargs+=4;return ret},getp(){return SYSCALLS.get()},getStr(ptr){var ret=UTF8ToString(ptr);return ret}};var _proc_exit=code=>{EXITSTATUS=code;if(!keepRuntimeAlive()){ABORT=true}quit_(code,new ExitStatus(code))};var exitJS=(status,implicit)=>{EXITSTATUS=status;_proc_exit(status)};var handleException=e=>{if(e instanceof ExitStatus||e=="unwind"){return EXITSTATUS}quit_(1,e)};WebGPU.initManagers();var wasmImports={l:_abort,n:_emscripten_request_animation_frame_loop,m:_emscripten_resize_heap,q:_emscripten_webgpu_get_device,r:glue_preint,t:_wgpuCommandBufferRelease,D:_wgpuCommandEncoderBeginRenderPass,x:_wgpuCommandEncoderFinish,w:_wgpuCommandEncoderRelease,e:_wgpuDeviceCreateBindGroup,u:_wgpuDeviceCreateBindGroupLayout,a:_wgpuDeviceCreateBuffer,E:_wgpuDeviceCreateCommandEncoder,j:_wgpuDeviceCreatePipelineLayout,g:_wgpuDeviceCreateRenderPipeline,b:_wgpuDeviceCreateShaderModule,o:_wgpuDeviceCreateSwapChain,F:_wgpuDeviceGetQueue,p:_wgpuInstanceCreateSurface,f:_wgpuPipelineLayoutRelease,v:_wgpuQueueSubmit,k:_wgpuQueueWriteBuffer,A:_wgpuRenderPassEncoderDrawIndexed,z:_wgpuRenderPassEncoderEnd,y:_wgpuRenderPassEncoderRelease,B:_wgpuRenderPassEncoderSetBindGroup,h:_wgpuRenderPassEncoderSetIndexBuffer,C:_wgpuRenderPassEncoderSetPipeline,i:_wgpuRenderPassEncoderSetVertexBuffer,c:_wgpuShaderModuleRelease,d:_wgpuSwapChainGetCurrentTextureView,s:_wgpuTextureViewRelease};var wasmExports=createWasm();var ___wasm_call_ctors=()=>(___wasm_call_ctors=wasmExports["H"])();var __glue_main_=Module["__glue_main_"]=()=>(__glue_main_=Module["__glue_main_"]=wasmExports["I"])();var _main=Module["_main"]=(a0,a1)=>(_main=Module["_main"]=wasmExports["J"])(a0,a1);var stackSave=()=>(stackSave=wasmExports["K"])();var stackRestore=a0=>(stackRestore=wasmExports["L"])(a0);var stackAlloc=a0=>(stackAlloc=wasmExports["M"])(a0);var ___start_em_js=Module["___start_em_js"]=2004;var ___stop_em_js=Module["___stop_em_js"]=2457;var calledRun;dependenciesFulfilled=function runCaller(){if(!calledRun)run();if(!calledRun)dependenciesFulfilled=runCaller};function callMain(args=[]){var entryFunction=_main;args.unshift(thisProgram);var argc=args.length;var argv=stackAlloc((argc+1)*4);var argv_ptr=argv;args.forEach(arg=>{HEAPU32[argv_ptr>>2]=stringToUTF8OnStack(arg);argv_ptr+=4});HEAPU32[argv_ptr>>2]=0;try{var ret=entryFunction(argc,argv);exitJS(ret,true);return ret}catch(e){return handleException(e)}}function run(args=arguments_){if(runDependencies>0){return}preRun();if(runDependencies>0){return}function doRun(){if(calledRun)return;calledRun=true;Module["calledRun"]=true;if(ABORT)return;initRuntime();preMain();if(shouldRunNow)callMain(args);postRun()}{doRun()}}var shouldRunNow=true;run();