package main

import "base:intrinsics"
import "base:runtime"
import "core:c"
import "core:fmt"
import "core:io"
import "core:log"
import "core:math"
import "core:math/linalg"
import "core:mem"
import "core:os"
import "core:slice"
import "core:strings"

import glfw "vendor:glfw"
import stbi "vendor:stb/image"
import vk "vendor:vulkan"

m4f :: matrix[4, 4]f32
v2f :: [2]f32
v3f :: [3]f32
v4f :: [4]f32
qf :: quaternion128

rad :: distinct f32
deg :: distinct f32
turn :: distinct f32
rad_from_deg :: proc(v: deg) -> rad {return rad(math.to_radians_f32(f32(v)))}
rad_from_turn :: proc(v: turn) -> rad {return rad(f32(v) * math.TAU)}
deg_from_rad :: proc(v: rad) -> deg {return deg(math.to_degrees_f32(f32(v)))}
deg_from_turn :: proc(v: turn) -> deg {return deg(f32(v) * 360)}
turn_from_rad :: proc(v: rad) -> turn {return turn(f32(v) / math.TAU)}
turn_from_deg :: proc(v: deg) -> turn {return turn(f32(v) / 360.0)}
to_rad :: proc {
	rad_from_deg,
	rad_from_turn,
}
to_deg :: proc {
	deg_from_rad,
	deg_from_turn,
}
to_turn :: proc {
	turn_from_rad,
	turn_from_deg,
}

when ODIN_OS == .Darwin {
	@(require, extra_linker_flags = "-rpath /Users/oliverjorgensen/Tools/vulkan/macOS/lib/")
	foreign import __ "system:System.framework"
}

Camera :: struct {
	pos:          v3f,
	rot:          qf,
	zoom:         f32,
	aspect_ratio: f32,
	fov:          rad,
	near:         f32,
	far:          f32,
}

camera_perspective :: proc "contextless" (cam: ^Camera) -> m4f {
	return linalg.matrix4_perspective_f32(
		f32(cam.fov),
		cam.aspect_ratio,
		cam.near,
		cam.far,
		flip_z_axis = false,
	)
}

camera_view :: proc "contextless" (cam: ^Camera) -> m4f {
	t := linalg.matrix4_translate_f32(cam.pos)
	r := linalg.matrix4_from_quaternion_f32(cam.rot)
	s: m4f = cam.zoom
	return linalg.matrix4_inverse_f32(t * r * s)
}

GlfwState :: struct {
	resized:                bool,
	camera_rot_active:      bool,
	camera_rot_just_active: bool,
	cursor_rot_active_pos:  v2f,
	cursor_rot_on_active:   qf,
	cursor_pos:             v2f,
	wasd:                   v2f,
}

VertexUniforms :: struct {
	model: m4f,
	view:  m4f,
	proj:  m4f,
}

Vertex :: struct {
	pos:   v2f,
	color: v3f,
}
Vertex_Binding_Description :: vk.VertexInputBindingDescription {
	binding   = 0,
	stride    = size_of(Vertex),
	inputRate = .VERTEX,
}
Vertex_Binding_Attributes :: [?]vk.VertexInputAttributeDescription {
	{location = 0, binding = 0, format = .R32G32_SFLOAT, offset = 0},
	{location = 1, binding = 0, format = .R32G32B32_SFLOAT, offset = size_of(f32) * 2},
}

RendererFrame :: struct {
	command_buf:         vk.CommandBuffer,
	sem_image_available: vk.Semaphore,
	sem_render_complete: vk.Semaphore,
	fen_in_flight:       vk.Fence,
	descriptor_set:      vk.DescriptorSet,
}
RendererBuffer :: struct {
	buffer: vk.Buffer,
	memory: vk.DeviceMemory,
	size:   u64,
}
RendererTexture :: struct {
	image:  vk.Image,
	memory: vk.DeviceMemory,
	format: vk.Format,
	width:  u32,
	height: u32,
}
Renderer :: struct {
	instance:              vk.Instance,
	physical_device:       vk.PhysicalDevice,
	device:                vk.Device,
	surface:               vk.SurfaceKHR,
	swapchain:             vk.SwapchainKHR,
	queue_graphics:        vk.Queue,
	queue_transfer:        vk.Queue,
	queue_compute:         vk.Queue,
	vertex_buffer:         RendererBuffer,
	index_buffer:          RendererBuffer,
	uniform_buffer:        RendererBuffer,
	render_pass:           vk.RenderPass,
	descriptor_pool:       vk.DescriptorPool,
	command_pool:          vk.CommandPool,
	pipeline:              vk.Pipeline,
	descriptor_layout:     vk.DescriptorSetLayout,
	pipeline_layout:       vk.PipelineLayout,
	alloc:                 ^vk.AllocationCallbacks,
	format:                vk.SurfaceFormatKHR,
	present_mode:          vk.PresentModeKHR,
	extent:                vk.Extent2D,
	swapchain_images:      [dynamic]vk.Image,
	swapchain_image_views: [dynamic]vk.ImageView,
	framebuffers:          [dynamic]vk.Framebuffer,
	frame_data:            [dynamic]RendererFrame,
	frame_count:           u64,
	window_handle:         glfw.WindowHandle,
}

renderer_init_swapchain :: proc(renderer: ^Renderer) {
	assert(renderer.window_handle != nil)
	assert(renderer.physical_device != nil)
	assert(renderer.device != nil)
	assert(renderer.surface > 0)

	surface_caps: vk.SurfaceCapabilitiesKHR
	vk.GetPhysicalDeviceSurfaceCapabilitiesKHR(
		renderer.physical_device,
		renderer.surface,
		&surface_caps,
	)

	swapchain_extent: vk.Extent2D
	if surface_caps.currentExtent.width != 0xFFFFFFFF {
		swapchain_extent = surface_caps.currentExtent
	} else {
		width, height := glfw.GetFramebufferSize(renderer.window_handle)

		swapchain_extent.width = clamp(
			u32(width),
			surface_caps.minImageExtent.width,
			surface_caps.maxImageExtent.width,
		)
		swapchain_extent.height = clamp(
			u32(height),
			surface_caps.minImageExtent.height,
			surface_caps.maxImageExtent.height,
		)
	}
	renderer.extent = swapchain_extent

	swapchain_info: vk.SwapchainCreateInfoKHR
	swapchain_info.sType = .SWAPCHAIN_CREATE_INFO_KHR
	swapchain_info.surface = renderer.surface
	swapchain_info.minImageCount = surface_caps.minImageCount + 1
	if surface_caps.maxImageCount != 0 &&
	   surface_caps.maxImageCount > swapchain_info.minImageCount {
		swapchain_info.minImageCount = surface_caps.maxImageCount
	}
	swapchain_info.imageUsage = {.COLOR_ATTACHMENT}
	swapchain_info.presentMode = renderer.present_mode
	swapchain_info.imageFormat = renderer.format.format
	swapchain_info.imageColorSpace = renderer.format.colorSpace
	swapchain_info.imageArrayLayers = 1
	swapchain_info.imageExtent = renderer.extent
	swapchain_info.imageSharingMode = .EXCLUSIVE
	swapchain_info.preTransform = surface_caps.currentTransform
	swapchain_info.compositeAlpha = {.OPAQUE}
	swapchain_info.clipped = true
	swapchain: vk.SwapchainKHR
	if r := vk.CreateSwapchainKHR(renderer.device, &swapchain_info, renderer.alloc, &swapchain);
	   r != .SUCCESS {
		log.fatal("Failed to create swapchain: %d", r)
		assert(false)
	}
	renderer.swapchain = swapchain

	swapchain_image_count: u32
	vk.GetSwapchainImagesKHR(renderer.device, swapchain, &swapchain_image_count, nil)
	resize(&renderer.swapchain_images, swapchain_image_count)
	vk.GetSwapchainImagesKHR(
		renderer.device,
		swapchain,
		&swapchain_image_count,
		raw_data(renderer.swapchain_images),
	)
	resize(&renderer.swapchain_image_views, swapchain_image_count)
	for img, i in renderer.swapchain_images {
		view_info: vk.ImageViewCreateInfo
		view_info.sType = .IMAGE_VIEW_CREATE_INFO
		view_info.image = auto_cast img
		view_info.viewType = .D2
		view_info.format = renderer.format.format
		view_info.components.r = .IDENTITY
		view_info.components.g = .IDENTITY
		view_info.components.b = .IDENTITY
		view_info.components.a = .IDENTITY
		view_info.subresourceRange.layerCount = 1
		view_info.subresourceRange.baseMipLevel = 0
		view_info.subresourceRange.aspectMask = {.COLOR}
		view_info.subresourceRange.levelCount = 1
		view_info.subresourceRange.baseArrayLayer = 0
		if r := vk.CreateImageView(
			renderer.device,
			&view_info,
			renderer.alloc,
			&renderer.swapchain_image_views[i],
		); r != .SUCCESS {
			log.fatalf("Failed to create image view: %d", r)
			assert(false)
		}
	}
}

renderer_texture_create :: proc(renderer: ^Renderer, path: string) -> RendererTexture {
	tex: RendererTexture
	data, valid := os.read_entire_file_from_filename(path, context.temp_allocator)
	if valid {
		w, h, channels: c.int
		file_data := stbi.load_from_memory(raw_data(data), i32(len(data)), &w, &h, &channels, 4)
		assert(file_data != nil)
		image_bytes := w * h * 4

		staging_buf := renderer_buffer_create(
			renderer,
			u64(image_bytes),
			{.TRANSFER_SRC},
			{.HOST_VISIBLE, .HOST_COHERENT},
		)
		defer renderer_buffer_destroy(renderer, &staging_buf)
		renderer_buffer_map(renderer, &staging_buf, file_data[0:image_bytes])

		tex_img: vk.Image
		tex_memory: vk.DeviceMemory

		img_info: vk.ImageCreateInfo
		img_info.sType = .IMAGE_CREATE_INFO
		img_info.imageType = .D2
		img_info.extent.width = u32(w)
		img_info.extent.height = u32(h)
		img_info.extent.depth = 1
		img_info.mipLevels = 1
		img_info.arrayLayers = 1
		img_info.format = .R8G8B8A8_SRGB
		img_info.tiling = .OPTIMAL
		img_info.initialLayout = .UNDEFINED
		img_info.usage = {.TRANSFER_DST, .SAMPLED}
		img_info.sharingMode = .EXCLUSIVE
		img_info.samples = {._1}

		if r := vk.CreateImage(renderer.device, &img_info, renderer.alloc, &tex_img);
		   r != .SUCCESS {
			// NOTE: Maybe handle other formats than r8g8b8a8_srgb
			// Just in case the implementation does not support it?
			// What are the odds of that?
			log.fatalf("Failed to create image: %d", r)
			assert(false)
		}

		mem_req: vk.MemoryRequirements
		vk.GetImageMemoryRequirements(renderer.device, tex_img, &mem_req)
		mem_props: vk.PhysicalDeviceMemoryProperties
		vk.GetPhysicalDeviceMemoryProperties(renderer.physical_device, &mem_props)

		alloc_info: vk.MemoryAllocateInfo
		alloc_info.sType = .MEMORY_ALLOCATE_INFO
		alloc_info.allocationSize = mem_req.size
		memory_type, valid := renderer_find_memory_type(
			renderer,
			mem_props,
			mem_req.memoryTypeBits,
			{.DEVICE_LOCAL},
		)
		assert(valid)
		alloc_info.memoryTypeIndex = memory_type
		if r := vk.AllocateMemory(renderer.device, &alloc_info, renderer.alloc, &tex_memory);
		   r != .SUCCESS {
			log.fatalf("Failed to allocate memory: %d", r)
			assert(false)
		}
		vk.BindImageMemory(renderer.device, tex_img, tex_memory, 0)
		tex.image = tex_img
		tex.memory = tex_memory
		tex.format = img_info.format
		tex.width = u32(w)
		tex.height = u32(h)

		renderer_texture_transition_layout(renderer, &tex, .UNDEFINED, .TRANSFER_DST_OPTIMAL)
		renderer_texture_copy_from_buffer(renderer, &tex, &staging_buf)
		renderer_texture_transition_layout(
			renderer,
			&tex,
			.TRANSFER_DST_OPTIMAL,
			.SHADER_READ_ONLY_OPTIMAL,
		)
	} else {
		log.errorf("Tried to creature texture using '%s', but file couldn't be loaded.", path)
	}

	return tex
}

renderer_texture_destroy :: proc "contextless" (renderer: ^Renderer, tex: RendererTexture) {
	vk.DestroyImage(renderer.device, tex.image, renderer.alloc)
	vk.FreeMemory(renderer.device, tex.memory, renderer.alloc)
}

renderer_buffer_copy_to_texture :: proc(
	renderer: ^Renderer,
	buf: ^RendererBuffer,
	tex: ^RendererTexture,
) {
	renderer_texture_copy_from_buffer(renderer, tex, buf)
}
renderer_texture_copy_from_buffer :: proc(
	renderer: ^Renderer,
	tex: ^RendererTexture,
	buf: ^RendererBuffer,
) {
	cb := renderer_command_buf_begin_single(renderer)
	defer renderer_command_buf_end_single(renderer, cb)

	region: vk.BufferImageCopy
	region.bufferOffset = 0
	region.bufferRowLength = 0
	region.bufferImageHeight = 0
	region.imageSubresource.aspectMask = {.COLOR}
	region.imageSubresource.mipLevel = 0
	region.imageSubresource.baseArrayLayer = 0
	region.imageSubresource.layerCount = 1
	region.imageOffset = {0, 0, 0}
	region.imageExtent = {tex.width, tex.height, 1}
	vk.CmdCopyBufferToImage(cb, buf.buffer, tex.image, .TRANSFER_DST_OPTIMAL, 1, &region)
}

renderer_texture_transition_layout :: proc(
	renderer: ^Renderer,
	tex: ^RendererTexture,
	old: vk.ImageLayout,
	new: vk.ImageLayout,
) {
	cb := renderer_command_buf_begin_single(renderer)
	defer renderer_command_buf_end_single(renderer, cb)

	barrier: vk.ImageMemoryBarrier
	barrier.sType = .IMAGE_MEMORY_BARRIER
	barrier.oldLayout = old
	barrier.newLayout = new
	barrier.srcQueueFamilyIndex = vk.QUEUE_FAMILY_IGNORED
	barrier.dstQueueFamilyIndex = vk.QUEUE_FAMILY_IGNORED
	barrier.image = tex.image
	barrier.subresourceRange.aspectMask = {.COLOR}
	barrier.subresourceRange.baseArrayLayer = 0
	barrier.subresourceRange.baseMipLevel = 0
	barrier.subresourceRange.levelCount = 1
	barrier.subresourceRange.layerCount = 1

	srcStageMask: vk.PipelineStageFlags = {}
	dstStageMask: vk.PipelineStageFlags = {}
	// TODO: Refactor this into just separate function calls?
	if old == .UNDEFINED && new == .TRANSFER_DST_OPTIMAL {
		barrier.dstAccessMask = {.TRANSFER_WRITE}
		srcStageMask = {.TOP_OF_PIPE}
		dstStageMask = {.TRANSFER}
	} else if old == .TRANSFER_DST_OPTIMAL && new == .SHADER_READ_ONLY_OPTIMAL {
		barrier.srcAccessMask = {.TRANSFER_WRITE}
		barrier.dstAccessMask = {.SHADER_READ}
		srcStageMask = {.TRANSFER}
		dstStageMask = {.FRAGMENT_SHADER}
	} else {
		assert(false)
	}
	vk.CmdPipelineBarrier(
		cb,
		srcStageMask = srcStageMask,
		dstStageMask = dstStageMask,
		dependencyFlags = {},
		memoryBarrierCount = 0,
		pMemoryBarriers = nil,
		bufferMemoryBarrierCount = 0,
		pBufferMemoryBarriers = nil,
		imageMemoryBarrierCount = 1,
		pImageMemoryBarriers = &barrier,
	)
}

renderer_find_memory_type :: proc "contextless" (
	renderer: ^Renderer,
	mem_props: vk.PhysicalDeviceMemoryProperties,
	req_types: u32,
	req_props: vk.MemoryPropertyFlags,
) -> (
	u32,
	bool,
) {
	for i := intrinsics.count_trailing_zeros(req_types); i < mem_props.memoryTypeCount; i += 1 {
		if req_props <= mem_props.memoryTypes[i].propertyFlags {
			return i, true
		}
	}

	return 0, false
}

// TODO: Maybe just move to init unless it needs to be called multiple times
renderer_buffer_create :: proc(
	renderer: ^Renderer,
	size: u64,
	usage: vk.BufferUsageFlags,
	properties: vk.MemoryPropertyFlags,
) -> RendererBuffer {
	create_info: vk.BufferCreateInfo
	create_info.sType = .BUFFER_CREATE_INFO
	create_info.size = auto_cast size
	create_info.usage = usage
	create_info.sharingMode = .EXCLUSIVE
	buffer: vk.Buffer
	if r := vk.CreateBuffer(renderer.device, &create_info, renderer.alloc, &buffer);
	   r != .SUCCESS {
		log.fatalf("Failed to create buffer: %d", r)
		assert(false)
	}

	mem_requirements: vk.MemoryRequirements
	vk.GetBufferMemoryRequirements(renderer.device, buffer, &mem_requirements)

	mem_properties: vk.PhysicalDeviceMemoryProperties
	vk.GetPhysicalDeviceMemoryProperties(renderer.physical_device, &mem_properties)
	memory_type, valid := renderer_find_memory_type(
		renderer,
		mem_properties,
		mem_requirements.memoryTypeBits,
		properties,
	)
	assert(valid)
	alloc_info: vk.MemoryAllocateInfo
	alloc_info.sType = .MEMORY_ALLOCATE_INFO
	alloc_info.allocationSize = mem_requirements.size
	alloc_info.memoryTypeIndex = memory_type
	buffer_mem: vk.DeviceMemory
	if r := vk.AllocateMemory(renderer.device, &alloc_info, renderer.alloc, &buffer_mem);
	   r != .SUCCESS {
		log.fatalf("Failed to allocate vertex buffer memory: %d", r)
		assert(false)
	}
	vk.BindBufferMemory(renderer.device, buffer, buffer_mem, 0)
	return RendererBuffer{buffer, buffer_mem, size}
}

renderer_buffer_destroy :: proc "contextless" (renderer: ^Renderer, buffer: ^RendererBuffer) {
	assert_contextless(renderer.device != nil)
	vk.FreeMemory(renderer.device, buffer.memory, renderer.alloc)
	vk.DestroyBuffer(renderer.device, buffer.buffer, renderer.alloc)
}

renderer_buffer_map :: proc(renderer: ^Renderer, buf: ^RendererBuffer, data: []byte) {
	data_len := u64(len(data))
	if data_len > buf.size {
		log.fatalf("Tried to write to buffer with data larger than the buffer size")
		assert(false)
		return
	}

	buf_data: rawptr
	vk.MapMemory(renderer.device, buf.memory, 0, auto_cast data_len, {}, &buf_data)
	mem.copy(buf_data, raw_data(data), len(data))
	vk.UnmapMemory(renderer.device, buf.memory)
}

renderer_command_buf_begin_single :: proc "contextless" (renderer: ^Renderer) -> vk.CommandBuffer {
	alloc_info: vk.CommandBufferAllocateInfo
	alloc_info.sType = .COMMAND_BUFFER_ALLOCATE_INFO
	alloc_info.level = .PRIMARY
	alloc_info.commandPool = renderer.command_pool
	alloc_info.commandBufferCount = 1
	cb: vk.CommandBuffer
	vk.AllocateCommandBuffers(renderer.device, &alloc_info, &cb)

	begin_info: vk.CommandBufferBeginInfo
	begin_info.sType = .COMMAND_BUFFER_BEGIN_INFO
	begin_info.flags = {.ONE_TIME_SUBMIT}
	vk.BeginCommandBuffer(cb, &begin_info)
	return cb
}
renderer_command_buf_end_single :: proc(renderer: ^Renderer, command_buffer: vk.CommandBuffer) {
	cb := command_buffer

	vk.EndCommandBuffer(cb)
	submit_info: vk.SubmitInfo
	submit_info.sType = .SUBMIT_INFO
	submit_info.commandBufferCount = 1
	submit_info.pCommandBuffers = &cb
	//NOTE: In the future it might be worth it to combine transfers
	// and then just use a fence to wait for all of them instead of waiting
	// on the queue for each transfer
	vk.QueueSubmit(renderer.queue_transfer, 1, &submit_info, 0)
	vk.QueueWaitIdle(renderer.queue_transfer)
	vk.FreeCommandBuffers(renderer.device, renderer.command_pool, 1, &cb)
}

renderer_buffer_copy :: proc(
	renderer: ^Renderer,
	dst: ^RendererBuffer,
	src: ^RendererBuffer,
	size: u64,
) {
	cb := renderer_command_buf_begin_single(renderer)
	defer renderer_command_buf_end_single(renderer, cb)

	size := min(src.size, dst.size)
	copy_reg: vk.BufferCopy
	copy_reg.size = auto_cast size
	vk.CmdCopyBuffer(cb, src.buffer, dst.buffer, 1, &copy_reg)
}

renderer_create :: proc(window_handle: glfw.WindowHandle) -> Renderer {
	renderer: Renderer
	renderer.window_handle = window_handle
	renderer.swapchain_images = make_dynamic_array([dynamic]vk.Image)
	renderer.swapchain_image_views = make_dynamic_array([dynamic]vk.ImageView)
	renderer.framebuffers = make_dynamic_array([dynamic]vk.Framebuffer)
	renderer.frame_data = make_dynamic_array([dynamic]RendererFrame)

	context.allocator = context.temp_allocator
	defer free_all(context.temp_allocator)

	vk.load_proc_addresses(rawptr(glfw.GetInstanceProcAddress))
	assert(vk.CreateInstance != nil, "vulkan function pointers not loaded")

	create_info := vk.InstanceCreateInfo {
		sType            = .INSTANCE_CREATE_INFO,
		pApplicationInfo = &vk.ApplicationInfo {
			sType = .APPLICATION_INFO,
			pApplicationName = "Hello Triangle",
			applicationVersion = vk.MAKE_VERSION(1, 0, 0),
			pEngineName = "No Engine",
			engineVersion = vk.MAKE_VERSION(1, 0, 0),
			apiVersion = vk.API_VERSION_1_0,
		},
	}

	extensions := slice.clone_to_dynamic(glfw.GetRequiredInstanceExtensions())

	when ODIN_OS == .Darwin {
		create_info.flags |= {.ENUMERATE_PORTABILITY_KHR}
		append(&extensions, vk.KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME)
		append(&extensions, "VK_KHR_get_physical_device_properties2")
	}

	when ODIN_DEBUG {
		create_info.ppEnabledLayerNames = raw_data([]cstring{"VK_LAYER_KHRONOS_validation"})
		create_info.enabledLayerCount = 1

		append(&extensions, vk.EXT_DEBUG_UTILS_EXTENSION_NAME)

		// Severity based on logger level.
		severity: vk.DebugUtilsMessageSeverityFlagsEXT
		if context.logger.lowest_level <= .Error {
			severity |= {.ERROR}
		}
		if context.logger.lowest_level <= .Warning {
			severity |= {.WARNING}
		}
		if context.logger.lowest_level <= .Info {
			severity |= {.INFO}
		}
		if context.logger.lowest_level <= .Debug {
			severity |= {.VERBOSE}
		}

		dbg_create_info := vk.DebugUtilsMessengerCreateInfoEXT {
			sType           = .DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT,
			messageSeverity = severity,
			messageType     = {.GENERAL, .VALIDATION, .PERFORMANCE, .DEVICE_ADDRESS_BINDING}, // all of them.
			pfnUserCallback = vk_message_callback,
		}
		create_info.pNext = &dbg_create_info
	}

	create_info.enabledExtensionCount = u32(len(extensions))
	create_info.ppEnabledExtensionNames = raw_data(extensions)

	instance: vk.Instance
	if r := vk.CreateInstance(&create_info, renderer.alloc, &instance); r != .SUCCESS {
		log.fatal("Failed to create instance: %d", r)
		assert(false)
	}
	assert(instance != nil)
	renderer.instance = instance
	vk.load_proc_addresses_instance(instance)

	when ODIN_OS == .Darwin {
		surface: vk.SurfaceKHR
		if r := glfw.CreateWindowSurface(instance, window_handle, renderer.alloc, &surface);
		   r != .SUCCESS {
			log.fatal("Failed to create surface: %d", r)
			assert(false)
		}
	}
	renderer.surface = surface

	physical_device_count: u32
	assert(vk.EnumeratePhysicalDevices != nil)
	vk.EnumeratePhysicalDevices(instance, &physical_device_count, nil)
	physical_devices := make_slice([]vk.PhysicalDevice, physical_device_count)
	vk.EnumeratePhysicalDevices(instance, &physical_device_count, raw_data(physical_devices))
	physical_device: vk.PhysicalDevice
	graphics_queue, transfer_queue, compute_queue: u32

	required_device_extensions: [dynamic]cstring
	append(&required_device_extensions, vk.KHR_SWAPCHAIN_EXTENSION_NAME)
	when ODIN_DEBUG {
		append(&required_device_extensions, "VK_KHR_portability_subset")
	}

	device_extensions: [dynamic]vk.ExtensionProperties
	formats: [dynamic]vk.SurfaceFormatKHR
	present_modes: [dynamic]vk.PresentModeKHR
	surface_caps: vk.SurfaceCapabilitiesKHR
	for d in physical_devices {
		d_features: vk.PhysicalDeviceFeatures
		d_props: vk.PhysicalDeviceProperties
		vk.GetPhysicalDeviceProperties(d, &d_props)
		vk.GetPhysicalDeviceFeatures(d, &d_features)
		// TODO: Apply some limits here

		d_exts_count: u32
		vk.EnumerateDeviceExtensionProperties(d, nil, &d_exts_count, nil)
		resize(&device_extensions, d_exts_count)
		d_exts := make_slice([]vk.ExtensionProperties, d_exts_count)
		vk.EnumerateDeviceExtensionProperties(d, nil, &d_exts_count, raw_data(d_exts))
		has_all_extensions := true
		for re in required_device_extensions {
			has_extension := false
			sre := string(re)
			for de in d_exts {
				name := de.extensionName
				// NOTE: Can this be done prettier?
				has_extension |= mem.compare(name[:len(sre)], raw_data(sre)[:len(sre)]) == 0
			}
			has_all_extensions &= has_extension
			if (!has_extension) {
				log.infof("Couldn't find extension: '%s' on device", re)
			}
		}

		if has_all_extensions {
			vk.GetPhysicalDeviceSurfaceCapabilitiesKHR(d, surface, &surface_caps)
			format_count: u32
			vk.GetPhysicalDeviceSurfaceFormatsKHR(d, surface, &format_count, nil)
			present_mode_count: u32
			vk.GetPhysicalDeviceSurfacePresentModesKHR(d, surface, &present_mode_count, nil)
			if present_mode_count > 0 && format_count > 0 {
				resize(&formats, format_count)
				vk.GetPhysicalDeviceSurfaceFormatsKHR(d, surface, &format_count, raw_data(formats))
				resize(&present_modes, present_mode_count)
				vk.GetPhysicalDeviceSurfacePresentModesKHR(
					d,
					surface,
					&present_mode_count,
					raw_data(present_modes),
				)

				d_queue_family_count: u32
				vk.GetPhysicalDeviceQueueFamilyProperties(d, &d_queue_family_count, nil)
				queue_families := make_slice([]vk.QueueFamilyProperties, d_queue_family_count)
				vk.GetPhysicalDeviceQueueFamilyProperties(
					d,
					&d_queue_family_count,
					raw_data(queue_families),
				)
				found_graphics, found_transfer, found_computer: bool
				// TODO: Try to split out the queues in cases where there are multiply family 
				// So one family does graphics, one transfer and another computer
				for family, index in queue_families {
					if vk.QueueFlag.GRAPHICS in family.queueFlags {
						graphics_queue = u32(index)
						found_graphics = true
					}
					if vk.QueueFlag.TRANSFER in family.queueFlags {
						transfer_queue = u32(index)
						found_transfer = true
					}
					if vk.QueueFlag.COMPUTE in family.queueFlags {
						compute_queue = u32(index)
						found_computer = true
					}
				}

				if found_computer && found_transfer && found_graphics {
					physical_device = d
					break
				}
			}
		}
		clear_dynamic_array(&device_extensions)
		clear_dynamic_array(&formats)
		clear_dynamic_array(&present_modes)
	}
	assert(physical_device != nil)
	renderer.physical_device = physical_device

	queue_infos: [3]vk.DeviceQueueCreateInfo
	queue_count: u32 = 1

	queue_priority: f32 = 1.0
	queue_infos[0].sType = .DEVICE_QUEUE_CREATE_INFO
	queue_infos[0].pQueuePriorities = &queue_priority
	queue_infos[0].queueFamilyIndex = graphics_queue
	queue_infos[0].queueCount = 1
	transfer_index: u32 = 0
	if (graphics_queue != transfer_queue) {
		transfer_index = 1
		queue_infos[1].sType = .DEVICE_QUEUE_CREATE_INFO
		queue_infos[1].pQueuePriorities = &queue_priority
		queue_infos[1].queueFamilyIndex = transfer_queue
		queue_infos[1].queueCount = 1
		queue_count += 1
	}
	compute_index: u32 = 0
	if (graphics_queue != compute_queue) {
		if (transfer_queue != compute_queue) {
			queue_infos[2].sType = .DEVICE_QUEUE_CREATE_INFO
			queue_infos[2].pQueuePriorities = &queue_priority
			queue_infos[2].queueFamilyIndex = compute_queue
			queue_infos[2].queueCount = 1
			queue_count += 1
			compute_index = 2
		} else {
			compute_index = 1
		}
	}

	required_features: vk.PhysicalDeviceFeatures
	device_create_info: vk.DeviceCreateInfo
	device_create_info.sType = .DEVICE_CREATE_INFO
	device_create_info.pEnabledFeatures = &required_features
	device_create_info.pQueueCreateInfos = raw_data(&queue_infos)
	device_create_info.queueCreateInfoCount = queue_count
	device_create_info.ppEnabledExtensionNames = raw_data(required_device_extensions[:])
	device_create_info.enabledExtensionCount = u32(len(required_device_extensions))
	device_create_info.enabledLayerCount = create_info.enabledLayerCount
	device_create_info.ppEnabledLayerNames = create_info.ppEnabledLayerNames
	device: vk.Device
	if r := vk.CreateDevice(physical_device, &device_create_info, renderer.alloc, &device);
	   r != .SUCCESS {
		log.fatal("Failed to create device: %d", r)
		assert(false)
	}
	vk.load_proc_addresses_device(device)
	renderer.device = device
	vk.GetDeviceQueue(device, graphics_queue, 0, &renderer.queue_graphics)
	vk.GetDeviceQueue(device, transfer_queue, 0, &renderer.queue_transfer)
	vk.GetDeviceQueue(device, compute_queue, 0, &renderer.queue_compute)

	renderer.format = formats[0]
	renderer.present_mode = .FIFO
	for f in formats {
		// TODO: Investigate this, and see if there isn't a better format for macs
		if f.format == .B8G8R8A8_SRGB && f.colorSpace == .SRGB_NONLINEAR {
			renderer.format = f
			break
		}
	}
	for pm in present_modes {
		if pm == .MAILBOX {
			renderer.present_mode = pm
			break
		}
	}
	renderer_init_swapchain(&renderer)
	swapchain := renderer.swapchain
	swapchain_extent := renderer.extent

	vert_shader_code := #load("vert.spv")
	vert_info: vk.ShaderModuleCreateInfo
	vert_info.sType = .SHADER_MODULE_CREATE_INFO
	vert_info.codeSize = len(vert_shader_code)
	vert_info.pCode = cast(^u32)raw_data(vert_shader_code)
	vert_shader: vk.ShaderModule
	if r := vk.CreateShaderModule(device, &vert_info, renderer.alloc, &vert_shader);
	   r != .SUCCESS {
		log.fatal("Failed to create vertex shader: %d", r)
		assert(false)
	}
	defer vk.DestroyShaderModule(device, vert_shader, renderer.alloc)
	frag_shader_code := #load("frag.spv")
	frag_info: vk.ShaderModuleCreateInfo
	frag_info.sType = .SHADER_MODULE_CREATE_INFO
	frag_info.codeSize = len(frag_shader_code)
	frag_info.pCode = cast(^u32)raw_data(frag_shader_code)
	frag_shader: vk.ShaderModule
	if r := vk.CreateShaderModule(device, &frag_info, renderer.alloc, &frag_shader);
	   r != .SUCCESS {
		log.fatal("Failed to create fragment shader: %d", r)
		assert(false)
	}
	defer vk.DestroyShaderModule(device, frag_shader, renderer.alloc)

	shader_stages := [?]vk.PipelineShaderStageCreateInfo {
		{
			sType = .PIPELINE_SHADER_STAGE_CREATE_INFO,
			module = vert_shader,
			stage = {.VERTEX},
			pName = "main",
		},
		{
			sType = .PIPELINE_SHADER_STAGE_CREATE_INFO,
			module = frag_shader,
			stage = {.FRAGMENT},
			pName = "main",
		},
	}

	color_attachment: vk.AttachmentDescription
	color_attachment.format = renderer.format.format
	color_attachment.samples = {._1}
	color_attachment.loadOp = .CLEAR
	color_attachment.storeOp = .STORE
	color_attachment.stencilLoadOp = .DONT_CARE
	color_attachment.stencilStoreOp = .DONT_CARE
	color_attachment.initialLayout = .UNDEFINED
	color_attachment.finalLayout = .PRESENT_SRC_KHR
	color_attachment_ref: vk.AttachmentReference
	color_attachment_ref.attachment = 0
	color_attachment_ref.layout = .COLOR_ATTACHMENT_OPTIMAL

	subpass_dependency: vk.SubpassDependency
	subpass_dependency.srcSubpass = vk.SUBPASS_EXTERNAL
	subpass_dependency.dstSubpass = 0
	subpass_dependency.srcStageMask = {.COLOR_ATTACHMENT_OUTPUT}
	subpass_dependency.srcAccessMask = {}
	subpass_dependency.dstStageMask = {.COLOR_ATTACHMENT_OUTPUT}
	subpass_dependency.dstAccessMask = {.COLOR_ATTACHMENT_WRITE}

	subpass: vk.SubpassDescription
	subpass.pipelineBindPoint = .GRAPHICS
	subpass.colorAttachmentCount = 1
	subpass.pColorAttachments = &color_attachment_ref

	render_pass_info: vk.RenderPassCreateInfo
	render_pass_info.sType = .RENDER_PASS_CREATE_INFO
	render_pass_info.attachmentCount = 1
	render_pass_info.pAttachments = &color_attachment
	render_pass_info.subpassCount = 1
	render_pass_info.pSubpasses = &subpass
	render_pass_info.dependencyCount = 1
	render_pass_info.pDependencies = &subpass_dependency
	render_pass: vk.RenderPass
	if r := vk.CreateRenderPass(device, &render_pass_info, renderer.alloc, &render_pass);
	   r != .SUCCESS {
		log.fatalf("Failed to create render pass: %d", r)
		assert(false)
	}
	renderer.render_pass = render_pass

	vertex_input_info: vk.PipelineVertexInputStateCreateInfo
	vertex_input_info.sType = .PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO
	vertex_binding_desc := Vertex_Binding_Description
	vertex_input_info.vertexBindingDescriptionCount = 1
	vertex_input_info.pVertexBindingDescriptions = &vertex_binding_desc
	vertex_binding_attributes := Vertex_Binding_Attributes
	vertex_input_info.vertexAttributeDescriptionCount = len(vertex_binding_attributes)
	vertex_input_info.pVertexAttributeDescriptions = raw_data(&vertex_binding_attributes)

	pipeline_input_assembly: vk.PipelineInputAssemblyStateCreateInfo
	pipeline_input_assembly.sType = .PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO
	pipeline_input_assembly.topology = .TRIANGLE_LIST
	pipeline_input_assembly.primitiveRestartEnable = false

	dynamic_states := [?]vk.DynamicState{vk.DynamicState.VIEWPORT}
	dynamic_state: vk.PipelineDynamicStateCreateInfo
	dynamic_state.sType = .PIPELINE_DYNAMIC_STATE_CREATE_INFO
	dynamic_state.dynamicStateCount = len(dynamic_states)
	dynamic_state.pDynamicStates = raw_data(&dynamic_states)

	viewport: vk.Viewport
	viewport.x = 0.0
	viewport.y = 0.0
	viewport.width = f32(swapchain_extent.width)
	viewport.height = f32(swapchain_extent.height)
	viewport.minDepth = 0.0
	viewport.maxDepth = 1.0

	scissor: vk.Rect2D
	scissor.extent.width = swapchain_extent.width
	scissor.extent.height = swapchain_extent.height

	pipeline_viewport_info: vk.PipelineViewportStateCreateInfo
	pipeline_viewport_info.sType = .PIPELINE_VIEWPORT_STATE_CREATE_INFO
	pipeline_viewport_info.viewportCount = 1
	pipeline_viewport_info.pViewports = &viewport
	pipeline_viewport_info.scissorCount = 1
	pipeline_viewport_info.pScissors = &scissor

	rasterizer: vk.PipelineRasterizationStateCreateInfo
	rasterizer.sType = .PIPELINE_RASTERIZATION_STATE_CREATE_INFO
	rasterizer.depthClampEnable = false
	rasterizer.rasterizerDiscardEnable = false
	rasterizer.polygonMode = .FILL
	rasterizer.lineWidth = 1.0
	//rasterizer.cullMode = {.BACK}
	rasterizer.frontFace = .COUNTER_CLOCKWISE
	rasterizer.depthBiasEnable = false

	multisample: vk.PipelineMultisampleStateCreateInfo
	multisample.sType = .PIPELINE_MULTISAMPLE_STATE_CREATE_INFO
	multisample.sampleShadingEnable = false
	multisample.rasterizationSamples = {._1}
	multisample.minSampleShading = 1.0
	multisample.pSampleMask = nil
	multisample.alphaToOneEnable = false
	multisample.alphaToCoverageEnable = false

	color_blend_attachment: vk.PipelineColorBlendAttachmentState
	color_blend_attachment.colorWriteMask = {.R, .G, .B, .A}
	color_blend_attachment.blendEnable = true
	color_blend_attachment.colorBlendOp = .ADD
	color_blend_attachment.srcColorBlendFactor = .SRC_ALPHA
	color_blend_attachment.dstColorBlendFactor = .ONE_MINUS_SRC_ALPHA
	color_blend_attachment.alphaBlendOp = .ADD
	color_blend_attachment.srcAlphaBlendFactor = .ONE
	color_blend_attachment.dstAlphaBlendFactor = .ZERO
	color_blend: vk.PipelineColorBlendStateCreateInfo
	color_blend.sType = .PIPELINE_COLOR_BLEND_STATE_CREATE_INFO
	color_blend.logicOpEnable = false
	color_blend.logicOp = .COPY
	color_blend.logicOp = .COPY
	color_blend.attachmentCount = 1
	color_blend.pAttachments = &color_blend_attachment

	vert_ubo_binding: vk.DescriptorSetLayoutBinding
	vert_ubo_binding.binding = 0
	vert_ubo_binding.descriptorType = .UNIFORM_BUFFER
	vert_ubo_binding.descriptorCount = 1
	vert_ubo_binding.stageFlags = {.VERTEX, .FRAGMENT}
	vert_ubo_binding.pImmutableSamplers = nil
	descriptor_set_layout_info: vk.DescriptorSetLayoutCreateInfo
	descriptor_set_layout_info.sType = .DESCRIPTOR_SET_LAYOUT_CREATE_INFO
	descriptor_set_layout_info.bindingCount = 1
	descriptor_set_layout_info.pBindings = &vert_ubo_binding
	desc_set_layout: vk.DescriptorSetLayout
	if r := vk.CreateDescriptorSetLayout(
		device,
		&descriptor_set_layout_info,
		renderer.alloc,
		&desc_set_layout,
	); r != .SUCCESS {
		log.fatalf("Failed to create descriptor set layout: %d", r)
		assert(false)
	}
	renderer.descriptor_layout = desc_set_layout

	pipeline_layout_info: vk.PipelineLayoutCreateInfo
	pipeline_layout_info.sType = .PIPELINE_LAYOUT_CREATE_INFO
	pipeline_layout_info.setLayoutCount = 1
	pipeline_layout_info.pSetLayouts = &desc_set_layout
	pipeline_layout_info.pushConstantRangeCount = 0
	pipeline_layout_info.pPushConstantRanges = nil

	pipeline_layout: vk.PipelineLayout
	if r := vk.CreatePipelineLayout(
		device,
		&pipeline_layout_info,
		renderer.alloc,
		&pipeline_layout,
	); r != .SUCCESS {
		log.fatalf("Failed to create pipeline layout: %d", r)
		assert(false)
	}
	renderer.pipeline_layout = pipeline_layout

	pipeline_info: vk.GraphicsPipelineCreateInfo
	pipeline_info.sType = .GRAPHICS_PIPELINE_CREATE_INFO
	pipeline_info.stageCount = len(shader_stages)
	pipeline_info.pStages = raw_data(&shader_stages)
	pipeline_info.pVertexInputState = &vertex_input_info
	pipeline_info.pInputAssemblyState = &pipeline_input_assembly
	pipeline_info.pViewportState = &pipeline_viewport_info
	pipeline_info.pRasterizationState = &rasterizer
	pipeline_info.pMultisampleState = &multisample
	pipeline_info.pDepthStencilState = nil
	pipeline_info.pColorBlendState = &color_blend
	pipeline_info.pDynamicState = &dynamic_state
	pipeline_info.layout = pipeline_layout
	pipeline_info.renderPass = render_pass
	pipeline_info.subpass = 0
	pipeline: vk.Pipeline
	if r := vk.CreateGraphicsPipelines(device, 0, 1, &pipeline_info, renderer.alloc, &pipeline);
	   r != .SUCCESS {
		log.fatalf("Failed to create pipeline: %d", r)
		assert(false)
	}
	renderer.pipeline = pipeline

	resize(&renderer.framebuffers, len(renderer.swapchain_image_views))
	for view, i in renderer.swapchain_image_views {
		fb_attachments := [?]vk.ImageView{view}
		fb_info: vk.FramebufferCreateInfo
		fb_info.sType = .FRAMEBUFFER_CREATE_INFO
		fb_info.renderPass = render_pass
		fb_info.attachmentCount = len(fb_attachments)
		fb_info.pAttachments = raw_data(&fb_attachments)
		fb_info.width = swapchain_extent.width
		fb_info.height = swapchain_extent.height
		fb_info.layers = 1

		if r := vk.CreateFramebuffer(device, &fb_info, renderer.alloc, &renderer.framebuffers[i]);
		   r != .SUCCESS {
			log.fatalf("Failed to create framebuffer: %d", r)
			assert(false)
		}
	}

	command_pool_info: vk.CommandPoolCreateInfo
	command_pool_info.sType = .COMMAND_POOL_CREATE_INFO
	command_pool_info.flags = {.RESET_COMMAND_BUFFER}
	command_pool_info.queueFamilyIndex = graphics_queue
	command_pool: vk.CommandPool
	if r := vk.CreateCommandPool(device, &command_pool_info, renderer.alloc, &command_pool);
	   r != .SUCCESS {
		log.fatalf("Failed to create command pool: %d", r)
		assert(false)
	}
	renderer.command_pool = command_pool

	resize(&renderer.frame_data, len(renderer.swapchain_image_views))
	for &fd in renderer.frame_data {
		command_buf_info: vk.CommandBufferAllocateInfo
		command_buf_info.sType = .COMMAND_BUFFER_ALLOCATE_INFO
		command_buf_info.commandPool = command_pool
		command_buf_info.level = .PRIMARY
		command_buf_info.commandBufferCount = 1
		command_buf: vk.CommandBuffer
		if r := vk.AllocateCommandBuffers(device, &command_buf_info, &command_buf); r != .SUCCESS {
			log.fatalf("Failed to allocate command buffer: %d", r)
			assert(false)
		}
		fd.command_buf = command_buf

		sem_info: vk.SemaphoreCreateInfo
		sem_info.sType = .SEMAPHORE_CREATE_INFO
		sem_image_available, sem_render_complete: vk.Semaphore
		if r := vk.CreateSemaphore(device, &sem_info, renderer.alloc, &sem_image_available);
		   r != .SUCCESS {
			log.fatalf("Failed to create image available semaphore: %d", r)
			assert(false)
		}
		if r := vk.CreateSemaphore(device, &sem_info, renderer.alloc, &sem_render_complete);
		   r != .SUCCESS {
			log.fatalf("Failed to create render commplete semaphore: %d", r)
			assert(false)
		}
		fd.sem_image_available = sem_image_available
		fd.sem_render_complete = sem_render_complete

		fen_info: vk.FenceCreateInfo
		fen_info.sType = .FENCE_CREATE_INFO
		fen_info.flags = {.SIGNALED}
		fen_in_flight: vk.Fence
		if r := vk.CreateFence(device, &fen_info, renderer.alloc, &fen_in_flight); r != .SUCCESS {
			log.fatalf("Failed to create in flight fence: %d", r)
			assert(false)
		}
		fd.fen_in_flight = fen_in_flight
	}
	assert(renderer.frame_data[0].fen_in_flight > 0)

	vertex_buffer_data := [?]Vertex {
		{pos = {-1.0, -1.0}, color = {0.1, 0.1, 0.0}},
		{pos = {1.0, -1.0}, color = {0.0, 1.0, 0.0}},
		{pos = {1.0, 1.0}, color = {0.0, 0.0, 1.0}},
		{pos = {-1.0, 1.0}, color = {0.4, 0.0, 1.0}},
	}
	index_buffer_data := [?]u32{2, 1, 0, 3, 2, 0}
	default_vertex_uniforms := VertexUniforms {
		proj  = 1.0,
		view  = 1.0,
		model = 1.0,
	}

	assert(size_of(Vertex) * len(vertex_buffer_data) == size_of(vertex_buffer_data))
	vertex_buffer_size: u64 = size_of(vertex_buffer_data)
	index_buffer_size: u64 = size_of(index_buffer_data)
	uniform_buffer_size: u64 = size_of(VertexUniforms)

	renderer.uniform_buffer = renderer_buffer_create(
		&renderer,
		size_of(VertexUniforms),
		{.TRANSFER_DST, .UNIFORM_BUFFER},
		{.DEVICE_LOCAL},
	)
	renderer.vertex_buffer = renderer_buffer_create(
		&renderer,
		vertex_buffer_size,
		{.VERTEX_BUFFER, .TRANSFER_DST},
		{.DEVICE_LOCAL},
	)
	renderer.index_buffer = renderer_buffer_create(
		&renderer,
		index_buffer_size,
		{.TRANSFER_DST, .INDEX_BUFFER},
		{.DEVICE_LOCAL},
	)

	staging_buf := renderer_buffer_create(
		&renderer,
		max(vertex_buffer_size, index_buffer_size, uniform_buffer_size),
		{.TRANSFER_SRC},
		{.HOST_VISIBLE, .HOST_COHERENT},
	)
	defer renderer_buffer_destroy(&renderer, &staging_buf)
	// NOTE: Make it possible to map, and later unmap
	// NOTE: Make a helper function that creates a staging buffer, maps memory, and be done with it 
	renderer_buffer_map(
		&renderer,
		&staging_buf,
		slice.bytes_from_ptr(&vertex_buffer_data, int(vertex_buffer_size)),
	)
	renderer_buffer_copy(&renderer, &renderer.vertex_buffer, &staging_buf, vertex_buffer_size)
	renderer_buffer_map(
		&renderer,
		&staging_buf,
		slice.bytes_from_ptr(&index_buffer_data, int(index_buffer_size)),
	)
	renderer_buffer_copy(&renderer, &renderer.index_buffer, &staging_buf, index_buffer_size)
	renderer_buffer_map(
		&renderer,
		&staging_buf,
		slice.bytes_from_ptr(&default_vertex_uniforms, int(uniform_buffer_size)),
	)
	renderer_buffer_copy(&renderer, &renderer.uniform_buffer, &staging_buf, uniform_buffer_size)

	desc_pool_size: vk.DescriptorPoolSize
	desc_pool_size.type = .UNIFORM_BUFFER
	desc_pool_size.descriptorCount = auto_cast len(renderer.frame_data)
	desc_pool_info: vk.DescriptorPoolCreateInfo
	desc_pool_info.sType = .DESCRIPTOR_POOL_CREATE_INFO
	desc_pool_info.poolSizeCount = 1
	desc_pool_info.pPoolSizes = &desc_pool_size
	desc_pool_info.maxSets = auto_cast len(renderer.frame_data)
	desc_pool: vk.DescriptorPool
	if r := vk.CreateDescriptorPool(renderer.device, &desc_pool_info, renderer.alloc, &desc_pool);
	   r != .SUCCESS {
		log.fatalf("Failed to create desc pool: %d", r)
		assert(false)
	}
	renderer.descriptor_pool = desc_pool

	for &fd in renderer.frame_data {
		alloc_info: vk.DescriptorSetAllocateInfo
		alloc_info.sType = .DESCRIPTOR_SET_ALLOCATE_INFO
		alloc_info.descriptorPool = desc_pool
		alloc_info.descriptorSetCount = 1
		alloc_info.pSetLayouts = &desc_set_layout
		desc_set: vk.DescriptorSet
		if r := vk.AllocateDescriptorSets(renderer.device, &alloc_info, &desc_set); r != .SUCCESS {
			log.fatalf("Failed to create descriptor set: %d", r)
			assert(false)
		}

		buf_info: vk.DescriptorBufferInfo
		buf_info.buffer = renderer.uniform_buffer.buffer
		buf_info.offset = 0
		buf_info.range = auto_cast vertex_buffer_size

		desc_write: vk.WriteDescriptorSet
		desc_write.sType = .WRITE_DESCRIPTOR_SET
		desc_write.dstSet = desc_set
		desc_write.dstBinding = 0
		desc_write.dstArrayElement = 0
		desc_write.descriptorType = .UNIFORM_BUFFER
		desc_write.descriptorCount = 1
		desc_write.pBufferInfo = &buf_info
		vk.UpdateDescriptorSets(renderer.device, 1, &desc_write, 0, nil)

		fd.descriptor_set = desc_set
	}

	return renderer
}

renderer_reconfigure :: proc(renderer: ^Renderer) {
	assert(renderer.window_handle != nil)
	assert(renderer.device != nil)
	assert(renderer.extent.width > 0)
	assert(renderer.extent.height > 0)
	assert(renderer.render_pass > 0)
	assert(len(renderer.swapchain_image_views) > 0)
	assert(len(renderer.swapchain_image_views) == len(renderer.framebuffers))

	vk.DeviceWaitIdle(renderer.device)

	// Cleanup swapchain
	for f in renderer.framebuffers {
		vk.DestroyFramebuffer(renderer.device, f, renderer.alloc)
	}
	for view in renderer.swapchain_image_views {
		vk.DestroyImageView(renderer.device, view, renderer.alloc)
	}
	vk.DestroySwapchainKHR(renderer.device, renderer.swapchain, renderer.alloc)

	// Recreate swapchain and framebuffers
	renderer_init_swapchain(renderer)
	resize(&renderer.framebuffers, len(renderer.swapchain_image_views))
	for view, i in renderer.swapchain_image_views {
		fb_attachments := [?]vk.ImageView{view}
		fb_info: vk.FramebufferCreateInfo
		fb_info.sType = .FRAMEBUFFER_CREATE_INFO
		fb_info.renderPass = renderer.render_pass
		fb_info.attachmentCount = len(fb_attachments)
		fb_info.pAttachments = raw_data(&fb_attachments)
		fb_info.width = renderer.extent.width
		fb_info.height = renderer.extent.height
		fb_info.layers = 1

		if r := vk.CreateFramebuffer(
			renderer.device,
			&fb_info,
			renderer.alloc,
			&renderer.framebuffers[i],
		); r != .SUCCESS {
			log.fatalf("Failed to create framebuffer: %d", r)
			assert(false)
		}
	}
}

renderer_render :: proc(renderer: ^Renderer) {
	frame_index := renderer.frame_count % u64(len(renderer.frame_data))
	frame_data := renderer.frame_data[frame_index]
	vk.WaitForFences(renderer.device, 1, &frame_data.fen_in_flight, true, 0xFFFFFFFFFFFFFFFF)

	image_index: u32
	if r := vk.AcquireNextImageKHR(
		renderer.device,
		renderer.swapchain,
		0xFFFFFFFFFFFFFFFF,
		frame_data.sem_image_available,
		0,
		&image_index,
	); r != .SUCCESS {
		if r == .SUBOPTIMAL_KHR {
			log.infof("Suboptimal image, reconfiguring...")
			renderer_reconfigure(renderer)
			return
		} else {
			log.fatalf("Failed to acquire next swapchain image: %d", r)
			assert(false)
		}
	}

	vk.ResetFences(renderer.device, 1, &frame_data.fen_in_flight)

	cb := frame_data.command_buf
	vk.ResetCommandBuffer(cb, {})
	begin_info: vk.CommandBufferBeginInfo
	begin_info.sType = .COMMAND_BUFFER_BEGIN_INFO
	if r := vk.BeginCommandBuffer(cb, &begin_info); r != .SUCCESS {
		log.fatalf("Failed to begin command buffer: %d", r)
		assert(false)
	}

	rp_info: vk.RenderPassBeginInfo
	rp_info.sType = .RENDER_PASS_BEGIN_INFO
	rp_info.renderPass = renderer.render_pass
	rp_info.framebuffer = renderer.framebuffers[image_index]
	rp_info.renderArea.offset = {0, 0}
	rp_info.renderArea.extent = renderer.extent
	clear_color := vk.ClearColorValue {
		float32 = {0.0, 0.0, 0.0, 1.0},
	}
	rp_info.clearValueCount = 1
	rp_info.pClearValues = auto_cast &clear_color
	vk.CmdBeginRenderPass(cb, &rp_info, .INLINE)
	vk.CmdBindPipeline(cb, .GRAPHICS, renderer.pipeline)
	vertex_buffers := [?]vk.Buffer{renderer.vertex_buffer.buffer}
	vertex_buffers_offset := [?]vk.DeviceSize{0}
	vk.CmdBindVertexBuffers(
		cb,
		0,
		len(vertex_buffers),
		raw_data(&vertex_buffers),
		raw_data(&vertex_buffers_offset),
	)
	vk.CmdBindIndexBuffer(cb, renderer.index_buffer.buffer, 0, .UINT32)
	viewport: vk.Viewport
	viewport.x = 0.0
	viewport.y = 0.0
	viewport.width = f32(renderer.extent.width)
	viewport.height = f32(renderer.extent.height)
	viewport.minDepth = 0.0
	viewport.maxDepth = 1.0
	vk.CmdSetViewport(cb, 0, 1, &viewport)
	vk.CmdBindDescriptorSets(
		cb,
		.GRAPHICS,
		renderer.pipeline_layout,
		0,
		1,
		&frame_data.descriptor_set,
		0,
		nil,
	)
	vk.CmdDrawIndexed(cb, u32(renderer.index_buffer.size / size_of(u32)), 1, 0, 0, 0)
	vk.CmdEndRenderPass(cb)
	if r := vk.EndCommandBuffer(cb); r != .SUCCESS {
		log.fatalf("Failed to finalize command buffer: %d", r)
		assert(false)
	}

	submit_info: vk.SubmitInfo
	submit_info.sType = .SUBMIT_INFO
	wait_sems := [?]vk.Semaphore{frame_data.sem_image_available}
	wait_stages := [?]vk.PipelineStageFlags{{vk.PipelineStageFlag.COLOR_ATTACHMENT_OUTPUT}}
	submit_info.waitSemaphoreCount = len(wait_sems)
	submit_info.pWaitSemaphores = raw_data(&wait_sems)
	submit_info.pWaitDstStageMask = raw_data(&wait_stages)
	submit_info.commandBufferCount = 1
	submit_info.pCommandBuffers = &cb
	signal_sems := [?]vk.Semaphore{frame_data.sem_render_complete}
	submit_info.signalSemaphoreCount = len(signal_sems)
	submit_info.pSignalSemaphores = raw_data(&signal_sems)
	if r := vk.QueueSubmit(renderer.queue_graphics, 1, &submit_info, frame_data.fen_in_flight);
	   r != .SUCCESS {
		log.fatalf("Failed to submit command buffers: %d", r)
		assert(false)
	}

	present_info: vk.PresentInfoKHR
	present_info.sType = .PRESENT_INFO_KHR
	present_info.waitSemaphoreCount = len(signal_sems)
	present_info.pWaitSemaphores = raw_data(&signal_sems)
	present_info.pImageIndices = &image_index
	present_info.swapchainCount = 1
	present_info.pSwapchains = &renderer.swapchain
	present_info.pResults = nil
	if r := vk.QueuePresentKHR(renderer.queue_graphics, &present_info); r != .SUCCESS {
		if r == .SUBOPTIMAL_KHR {
			log.infof("Suboptimal present, reconfiguring...")
			renderer_reconfigure(renderer)
		} else {
			log.fatalf("Error when presenting queue: %d", r)
			assert(false)
		}
	}

	renderer.frame_count += 1
}

renderer_update_vertex_uniforms :: proc(renderer: ^Renderer, vertex_ubo: ^VertexUniforms) {
	staging_buf := renderer_buffer_create(
		renderer,
		size_of(VertexUniforms),
		{.TRANSFER_SRC},
		{.HOST_VISIBLE, .HOST_COHERENT},
	)
	renderer_buffer_map(
		renderer,
		&staging_buf,
		slice.bytes_from_ptr(vertex_ubo, size_of(VertexUniforms)),
	)
	renderer_buffer_copy(renderer, &renderer.uniform_buffer, &staging_buf, size_of(VertexUniforms))

	defer renderer_buffer_destroy(renderer, &staging_buf)
}

renderer_destroy :: proc "contextless" (renderer: ^Renderer) {
	vk.DeviceWaitIdle(renderer.device)

	renderer_buffer_destroy(renderer, &renderer.vertex_buffer)
	renderer_buffer_destroy(renderer, &renderer.index_buffer)
	renderer_buffer_destroy(renderer, &renderer.uniform_buffer)

	for fd in renderer.frame_data {
		vk.DestroySemaphore(renderer.device, fd.sem_image_available, renderer.alloc)
		vk.DestroySemaphore(renderer.device, fd.sem_render_complete, renderer.alloc)
		vk.DestroyFence(renderer.device, fd.fen_in_flight, renderer.alloc)
	}
	vk.DestroyDescriptorPool(renderer.device, renderer.descriptor_pool, renderer.alloc)
	vk.DestroyCommandPool(renderer.device, renderer.command_pool, renderer.alloc)
	for f in renderer.framebuffers {
		vk.DestroyFramebuffer(renderer.device, f, renderer.alloc)
	}
	vk.DestroyPipeline(renderer.device, renderer.pipeline, renderer.alloc)
	vk.DestroyRenderPass(renderer.device, renderer.render_pass, renderer.alloc)
	vk.DestroyDescriptorSetLayout(renderer.device, renderer.descriptor_layout, renderer.alloc)
	vk.DestroyPipelineLayout(renderer.device, renderer.pipeline_layout, renderer.alloc)
	for view in renderer.swapchain_image_views {
		vk.DestroyImageView(renderer.device, view, renderer.alloc)
	}
	vk.DestroySwapchainKHR(renderer.device, renderer.swapchain, renderer.alloc)
	vk.DestroyDevice(renderer.device, renderer.alloc)
	vk.DestroySurfaceKHR(renderer.instance, renderer.surface, renderer.alloc)
	vk.DestroyInstance(renderer.instance, renderer.alloc)
}

main :: proc() {
	context.logger = log.create_console_logger()

	if !glfw.Init() {
		log.fatal("Failed to init glfw")
		return
	}
	defer glfw.Terminate()

	glfw.WindowHint(glfw.CLIENT_API, glfw.NO_API)
	glfw.WindowHint(glfw.RESIZABLE, glfw.TRUE)

	window := glfw.CreateWindow(1280 * 2, 720 * 2, "Claykan", nil, nil)
	defer glfw.DestroyWindow(window)
	assert(glfw.VulkanSupported() == true)

	glfw_state: GlfwState
	glfw.SetWindowUserPointer(window, &glfw_state)
	framebuffer_size_changed :: proc "c" (window: glfw.WindowHandle, width, height: c.int) {
		state: ^GlfwState = auto_cast glfw.GetWindowUserPointer(window)
		assert_contextless(state != nil)
		state.resized = true
	}
	glfw.SetFramebufferSizeCallback(window, framebuffer_size_changed)

	renderer := renderer_create(window)

	camera: Camera
	camera.pos = {0, -0.3, -2}
	camera.rot = linalg.QUATERNIONF32_IDENTITY
	camera.fov = to_rad(deg(60))
	camera.aspect_ratio = f32(renderer.extent.width) / f32(renderer.extent.height)
	camera.near = 0.1
	camera.far = 10
	camera.zoom = 1

	v_uniforms: VertexUniforms
	v_uniforms.model = 1.0
	v_uniforms.proj = camera_perspective(&camera)
	v_uniforms.view = camera_view(&camera)

	key_callback :: proc "c" (window: glfw.WindowHandle, key, scancode, action, mods: c.int) {
		state: ^GlfwState = auto_cast glfw.GetWindowUserPointer(window)
		assert_contextless(state != nil)
		if action == glfw.PRESS {
			switch (scancode) {
			case glfw.GetKeyScancode(glfw.KEY_W):
				state.wasd.y += 1.0
			case glfw.GetKeyScancode(glfw.KEY_S):
				state.wasd.y -= 1.0
			case glfw.GetKeyScancode(glfw.KEY_D):
				state.wasd.x += 1.0
			case glfw.GetKeyScancode(glfw.KEY_A):
				state.wasd.x -= 1.0
			}
		}
		if action == glfw.RELEASE {
			switch (scancode) {
			case glfw.GetKeyScancode(glfw.KEY_W):
				state.wasd.y -= 1.0
			case glfw.GetKeyScancode(glfw.KEY_S):
				state.wasd.y += 1.0
			case glfw.GetKeyScancode(glfw.KEY_D):
				state.wasd.x -= 1.0
			case glfw.GetKeyScancode(glfw.KEY_A):
				state.wasd.x += 1.0
			}
		}
	}
	glfw.SetKeyCallback(window, key_callback)
	cursor_callback :: proc "c" (window: glfw.WindowHandle, xpos, ypos: f64) {
		state: ^GlfwState = auto_cast glfw.GetWindowUserPointer(window)
		assert_contextless(state != nil)
		pos := v2f{f32(xpos), f32(ypos)}
		state.cursor_pos = pos
	}
	glfw.SetCursorPosCallback(window, cursor_callback)
	mouse_button_callback :: proc "c" (window: glfw.WindowHandle, button, action, mods: c.int) {
		state: ^GlfwState = auto_cast glfw.GetWindowUserPointer(window)
		assert_contextless(state != nil)
		if action == glfw.PRESS {
			switch (button) {
			case glfw.MOUSE_BUTTON_LEFT:
				state.camera_rot_active = true
				state.camera_rot_just_active = true
			}
		}
		if action == glfw.RELEASE {
			switch (button) {
			case glfw.MOUSE_BUTTON_LEFT:
				state.camera_rot_active = false
			}
		}
	}
	glfw.SetMouseButtonCallback(window, mouse_button_callback)

	current_time := glfw.GetTime()
	delta_time: f64 = 0.0
	for !glfw.WindowShouldClose(window) {
		glfw_state.resized = false
		glfw_state.camera_rot_just_active = false
		glfw.PollEvents()
		if glfw_state.camera_rot_just_active {
			glfw_state.cursor_rot_on_active = camera.rot
			glfw_state.cursor_rot_active_pos = glfw_state.cursor_pos
		}

		if glfw_state.camera_rot_active {
			ROTATE_SPEED :: 0.01
			delta_pos := glfw_state.cursor_rot_active_pos - glfw_state.cursor_pos
			rot_right := -delta_pos.x * f32(ROTATE_SPEED)
			rot_up := delta_pos.y * f32(ROTATE_SPEED)
			camera.rot =
				glfw_state.cursor_rot_on_active *
				linalg.quaternion_from_euler_angle_y_f32(rot_right) *
				linalg.quaternion_from_euler_angle_x_f32(rot_up)
			//camera.rot = linalg.quaternion_normalize(camera.rot)
		}
		if linalg.vector_length2(glfw_state.wasd) > 0.1 {
			MOVE_SPEED :: 1.0
			right := glfw_state.wasd.x * v3f{1, 0, 0}
			forward := glfw_state.wasd.y * v3f{0, 0, 1}
			cam_move := right + forward
			cam_move = linalg.quaternion_mul_vector3(camera.rot, cam_move)
			cam_move = linalg.vector_normalize(cam_move)
			cam_move *= f32(MOVE_SPEED * delta_time)
			camera.pos += cam_move
		}
		v_uniforms.view = camera_view(&camera)
		renderer_update_vertex_uniforms(&renderer, &v_uniforms)

		if (glfw_state.resized) {
			renderer_reconfigure(&renderer)
		}

		renderer_render(&renderer)

		new_time := glfw.GetTime()
		delta_time = new_time - current_time
		current_time = new_time
	}
	defer renderer_destroy(&renderer)
}

vk_message_callback :: proc "system" (
	messageSeverity: vk.DebugUtilsMessageSeverityFlagsEXT,
	messageTypes: vk.DebugUtilsMessageTypeFlagsEXT,
	pCallbackData: ^vk.DebugUtilsMessengerCallbackDataEXT,
	pUserData: rawptr,
) -> b32 {
	context = runtime.default_context()
	if vk.DebugUtilsMessageSeverityFlagEXT.ERROR in messageSeverity {
		log.errorf(string(pCallbackData.pMessage))
	} else if vk.DebugUtilsMessageSeverityFlagEXT.WARNING in messageSeverity {
		log.warnf(string(pCallbackData.pMessage))
	} else if vk.DebugUtilsMessageSeverityFlagEXT.INFO in messageSeverity {
		log.infof(string(pCallbackData.pMessage))
	} else {
		log.debugf(string(pCallbackData.pMessage))
	}
	return true
}
