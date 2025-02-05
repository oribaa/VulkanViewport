#version 450

layout(location = 0) in vec2 in_pos;
layout(location = 1) in vec3 in_color;
layout(location = 0) out vec3 frag_color;
layout(location = 1) out vec3 near_point;
layout(location = 2) out vec3 far_point;

layout(binding = 0) uniform VertexUniforms {
	mat4 model;
	mat4 view;
	mat4 proj;
} ubo;

vec3 unproject_point(vec3 p, mat4 view, mat4 proj) {
	mat4 viewInv = inverse(view);
	mat4 projInv = inverse(proj);
	vec4 unproject_point = viewInv * projInv * vec4(p, 1.0);
	return unproject_point.xyz / unproject_point.w;
}

void main() {
	//gl_Position = ubo.proj * ubo.view * ubo.model *vec4(in_pos, 0.0, 1.0);
	gl_Position = vec4(in_pos, 0.0, 1.0);
	near_point = unproject_point(vec3(in_pos, 0.0), ubo.view, ubo.proj);
	far_point = unproject_point(vec3(in_pos, 1.0), ubo.view, ubo.proj);
	frag_color = in_color;
}

