if glslc shader.vert -o vert.spv && glslc shader.frag -o frag.spv; then
	odin run . -debug -o:none
fi
