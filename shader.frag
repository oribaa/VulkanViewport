#version 450

layout(location = 0) in vec3 frag_color;
layout(location = 1) in vec3 near_point;
layout(location = 2) in vec3 far_point;
layout(location = 0) out vec4 out_color;

layout(binding = 0) uniform VertexUniforms {
	mat4 model;
	mat4 view;
	mat4 proj;
} ubo;

vec4 grid(vec3 fragPos3D, float scale) {
    vec2 coord = fragPos3D.xz * scale; // use the scale variable to set the distance between the lines
    vec2 derivative = fwidth(coord);
    vec2 grid = abs(fract(coord - 0.5) - 0.5) / derivative;
    float line = min(grid.x, grid.y);
    float minimumz = min(derivative.y, 1);
    float minimumx = min(derivative.x, 1);
    vec4 color = vec4(0.2, 0.2, 0.2, 1.0 - min(line, 1.0));
    // z axis
    if(fragPos3D.x > -0.1 * minimumx && fragPos3D.x < 0.1 * minimumx)
        color.z = 1.0;
    // x axis
    if(fragPos3D.z > -0.1 * minimumz && fragPos3D.z < 0.1 * minimumz)
        color.x = 1.0;
    return color;
}

void main() {
	float t = (-near_point.y) / (far_point.y - near_point.y);
	if (t > 0) {
		vec3 frag_pos = near_point + t * (far_point - near_point);
		vec4 color = grid(frag_pos, 10.0); //+ grid(frag_pos, 1.0);

    		vec4 clip_space_pos = ubo.proj * ubo.view * vec4(frag_pos.xyz, 1.0);
		clip_space_pos /= clip_space_pos.w;
    		vec4 near_clip_space_pos = ubo.proj * ubo.view * vec4(near_point.xyz, 1.0);
		near_clip_space_pos /= near_clip_space_pos.w;
    		vec4 far_clip_space_pos = ubo.proj * ubo.view * vec4(far_point.xyz, 1.0);
		far_clip_space_pos /= far_clip_space_pos.w;

		float depth = (clip_space_pos.z - near_clip_space_pos.z) / (far_clip_space_pos.z - near_clip_space_pos.z);

		gl_FragDepth = depth;

		depth *= depth;
		if (depth > 0.9) { 
			discard;
		} else {
			color.a *= 0.9 - depth;
			out_color = color;
		}
	} else {
		discard;
	}
}
