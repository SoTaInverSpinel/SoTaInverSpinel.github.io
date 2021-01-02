//Reference:aadebdeb,"Gray-Scott Reaction Diffusion",https://neort.io/art/bilttjs3p9f9psc9oafg?index=0&origin=user_like
(function() {

    function createShader(gl, source, type) {
      const shader = gl.createShader(type);
      gl.shaderSource(shader, source);
      gl.compileShader(shader);
      if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
        throw new Error(gl.getShaderInfoLog(shader) + source);
      }
      return shader;
    }
    
    function createProgramFromSource(gl, vertexShaderSource, fragmentShaderSource) {
      const program = gl.createProgram();
      gl.attachShader(program, createShader(gl, vertexShaderSource, gl.VERTEX_SHADER));
      gl.attachShader(program, createShader(gl, fragmentShaderSource, gl.FRAGMENT_SHADER));
      gl.linkProgram(program);
      if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
        throw new Error(gl.getProgramInfoLog(program));
      }
      return program;
    }
  
    function createVbo(gl, array, usage) {
      const vbo = gl.createBuffer();
      gl.bindBuffer(gl.ARRAY_BUFFER, vbo);
      gl.bufferData(gl.ARRAY_BUFFER, array, usage !== undefined ? usage : gl.STATIC_DRAW);
      gl.bindBuffer(gl.ARRAY_BUFFER, null);
      return vbo;
    }
  
    function createIbo(gl, array) {
      const ibo = gl.createBuffer();
      gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, ibo);
      gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, array, gl.STATIC_DRAW);
      gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, null);
      return ibo;
    }
  
    function getUniformLocations(gl, program, keys) {
      const locations = {};
      keys.forEach(key => {
          locations[key] = gl.getUniformLocation(program, key);
      });
      return locations;
    }
  
    function createFramebuffer(gl, sizeX, sizeY) {
      const framebuffer = gl.createFramebuffer();
      gl.bindFramebuffer(gl.FRAMEBUFFER, framebuffer);
      const texture = gl.createTexture();
      gl.bindTexture(gl.TEXTURE_2D, texture);
      //gl.texImage2D(gl.TEXTURE_2D, 0, gl.RG32F, sizeX, sizeY, 0, gl.RG, gl.FLOAT, null);
      //set size and format of texture
      gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA32F, sizeX, sizeY, 0, gl.RGBA, gl.FLOAT, null);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
      //bind texture to framebuffer as color
      gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, texture, 0);
      gl.bindFramebuffer(gl.FRAMEBUFFER, null);
      gl.bindTexture(gl.TEXTURE_2D, null);
      return {
        framebuffer: framebuffer,
        texture: texture
      };
    }
  
    const VERTICES_POSITION = new Float32Array([
      -1.0, -1.0,
      1.0, -1.0,
      -1.0,  1.0,
      1.0,  1.0
    ]);
  
    const VERTICES_INDEX = new Int16Array([
      0, 1, 2,
      3, 2, 1
    ]);
  
  
    const FILL_SCREEN_VERTEX_SHADER_SOURCE =
  `#version 300 es
  
  layout (location = 0) in vec2 position;
  
  void main(void) {
    gl_Position = vec4(position, 0.0, 1.0);
  }
  `;
  
    const INITIALIZE_FRAGMENT_SHADER_SOURCE =
  `#version 300 es
  
  precision highp float;
  
  out vec3 o_state;
  
  uniform vec2 u_resolution;
  uniform vec2 u_randomSeed;
  
  
  float random(vec2 x){
    return fract(sin(dot(x,vec2(12.9898, 78.233))) * 43758.5453);
  }
  
  
  float valuenoise(vec2 x) {
      vec2 i = floor(x);
      vec2 f = fract(x);
  
      vec2 u = f * f * (3.0 - 2.0 * f);
  
      return mix(
          mix(random(i), random(i + vec2(1.0, 0.0)), u.x),
          mix(random(i + vec2(0.0, 1.0)), random(vec2(i + vec2(1.0, 1.0))), u.x),
          u.y
      );
  }
  
  float fbm(vec2 x) {
    float sum = 0.0;
    float amp = 0.5;
    for (int i = 0; i < 5; i++) {
      sum += amp * valuenoise(x);
      amp *= 0.5;
      x *= 2.01;
    }
    return sum;
  }
  
  void main(void) {
    vec2 st = (2.0 * gl_FragCoord.xy - u_resolution) / min(u_resolution.x, u_resolution.y);
  
    o_state=vec3(random(st),0.5,0.5);
  
  }
  `;
  
    const UPDATE_FRAGMENT_SHADER_SOURCE =
  `#version 300 es
  
  precision highp float;
  
  layout (location =0)out vec3 o_state;
  
  uniform sampler2D u_stateTexture;
  uniform float u_timeStep;
  uniform float u_spaceStep;
  uniform vec2 u_diffusion;
  //uniform float u_feed;
  //uniform float u_kill;
  
  void main(void) {
    ivec2 coord = ivec2(gl_FragCoord.xy);
    ivec2 stateTextureSize = textureSize(u_stateTexture, 0);
  
    vec3 state = texelFetch(u_stateTexture, coord, 0).xyz;
  
    vec3 left = texelFetch(u_stateTexture, ivec2(
      coord.x != 0 ? coord.x - 1 : stateTextureSize.x - 1,
      coord.y), 0).xyz;
  
    vec3 right = texelFetch(u_stateTexture, ivec2(
      coord.x != stateTextureSize.x - 1 ? coord.x + 1 : 0,
      coord.y), 0).xyz;
  
    vec3 down = texelFetch(u_stateTexture, ivec2(
      coord.x,
      coord.y != 0 ? coord.y - 1 : stateTextureSize.y - 1), 0).xyz;
  
    vec3 up = texelFetch(u_stateTexture, ivec2(
      coord.x,
      coord.y != stateTextureSize.y - 1 ? coord.y + 1 : 0), 0).xyz;
  
    vec3 upleft = texelFetch(u_stateTexture, ivec2(
        coord.x != 0 ? coord.x - 1 : stateTextureSize.x - 1,
        coord.y != stateTextureSize.y - 1 ? coord.y + 1 : 0), 0).xyz;
  
    vec3 upright = texelFetch(u_stateTexture, ivec2(
        coord.x != stateTextureSize.x - 1 ? coord.x + 1 : 0,
        coord.y != stateTextureSize.y - 1 ? coord.y + 1 : 0), 0).xyz;
  
    vec3 downleft = texelFetch(u_stateTexture, ivec2(
        coord.x != 0 ? coord.x - 1 : stateTextureSize.x - 1,
        coord.y != 0 ? coord.y - 1 : stateTextureSize.y - 1), 0).xyz;
  
    vec3 downright = texelFetch(u_stateTexture, ivec2(
        coord.x != stateTextureSize.x - 1 ? coord.x + 1 : 0,
        coord.y !=  0 ? coord.y - 1 : stateTextureSize.y - 1), 0).xyz;
  
    vec3 conc = (left + right + up + down + upleft + upright + downleft + downright +state) / 9.0f;
  
    o_state = clamp(conc + vec3(
      conc.x * (conc.y - conc.z),
      conc.y * (conc.z - conc.x),
      conc.z * (conc.x - conc.y)
    ),0.0,1.0);
  }
  `
  
    const RENDER_FRAGMNET_SHADER_SOURCE =
  `#version 300 es
  
  precision highp float;
  
  out vec4 o_color;
  
  uniform sampler2D u_stateTexture;
  uniform int u_target;
  uniform int u_rendering;
  uniform float u_spaceStep;
  
  float getValue(ivec2 coord) {
    vec3 state = texelFetch(u_stateTexture, ivec2(coord), 0).xyz;
  
    if (u_target == 0) {
      return state.x;
    } else if (u_target == 1) {
      return state.y;
    } else if (u_target == 2){
      return state.z;
    } else {
      return abs(state.x - state.y);
    }
  }
  
  vec3 render2d(ivec2 coord) {
    return vec3(getValue(coord));
  }
  
  vec3 lambert(vec3 color, vec3 normal, vec3 lightDir) {
    return color * max(dot(normal, lightDir), 0.0);
  }
  
  vec3 render3d(ivec2 coord) {
    ivec2 stateTextureSize = textureSize(u_stateTexture, 0);
    float state = getValue(coord);
    float left = getValue(ivec2(coord.x != 0 ? coord.x - 1 : stateTextureSize.x - 1, coord.y));
    float right = getValue(ivec2(coord.x != stateTextureSize.x - 1 ? coord.x + 1 : 0, coord.y));
    float down = getValue(ivec2(coord.x, coord.y != 0 ? coord.y - 1 : stateTextureSize.y - 1));
    float up = getValue(ivec2(coord.x, coord.y != stateTextureSize.y - 1 ? coord.y + 1 : 0));
  
    vec3 dx = vec3(2.0 * u_spaceStep, 0.0, (right - left) / (2.0 * u_spaceStep));
    vec3 dy = vec3(0.0, 2.0 * u_spaceStep, (up - down) / (2.0 * u_spaceStep));
  
    vec3 normal = mix(normalize(cross(dx, dy)), vec3(0.0, 0.0, 1.0), 0.5);
  
    vec3 baseColor = mix(vec3(0.0, 0.05, 0.1), vec3(0.7, 1.0, 0.1), clamp(state, 0.0, 1.00));
    vec3 color = vec3(0.0);
    color += baseColor * lambert(vec3(1.5), normal, vec3(1.0, 1.0, 1.0));
    color += baseColor * lambert(vec3(0.5), normal, vec3(-1.0, -1.0, 0.3));
    return color;
  }
  
  void main(void) {
    vec3 state = texelFetch(u_stateTexture, ivec2(gl_FragCoord.xy), 0).xyz;
  
    if (u_rendering == 0) {
      //o_color = vec4(state.xyz, 1.0);
      o_color = vec4(render2d(ivec2(gl_FragCoord.xy)), 1.0);
    } else {
      o_color = vec4(render3d(ivec2(gl_FragCoord.xy)), 1.0);
    }
  }
  `;
  
    const parameters = {
      'diffusion U': 0.0002,
      'diffusion V': 0.01,
      
      'space step': 0.04,
      'time step': 0.015,
      'time scale': 1.0,
      'target': 1,
      'rendering': 1,
      reset: _ => reset()
    };
  
    const canvas = document.getElementById('canvas');
    const gl = canvas.getContext('webgl2');
    gl.getExtension('EXT_color_buffer_float');
  
    const initializeProgram = createProgramFromSource(gl, FILL_SCREEN_VERTEX_SHADER_SOURCE, INITIALIZE_FRAGMENT_SHADER_SOURCE);
    const updateProgram = createProgramFromSource(gl, FILL_SCREEN_VERTEX_SHADER_SOURCE, UPDATE_FRAGMENT_SHADER_SOURCE);
    const renderProgram = createProgramFromSource(gl, FILL_SCREEN_VERTEX_SHADER_SOURCE, RENDER_FRAGMNET_SHADER_SOURCE);
    const initializeUniforms = getUniformLocations(gl, initializeProgram, ['u_resolution','u_randomSeed']);
    const updateUniforms = getUniformLocations(gl, updateProgram, ['u_stateTexture', 'u_diffusion', 'u_timeStep', 'u_spaceStep']);
    const renderUniforms = getUniformLocations(gl, renderProgram, ['u_stateTexture', 'u_target', 'u_rendering', 'u_spaceStep']);
  
    const vao = gl.createVertexArray();
    gl.bindVertexArray(vao);
    gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, createIbo(gl, VERTICES_INDEX));
    gl.bindBuffer(gl.ARRAY_BUFFER, createVbo(gl, VERTICES_POSITION));
    gl.enableVertexAttribArray(0);
    gl.vertexAttribPointer(0, 2, gl.FLOAT, false, 0, 0);
    //gl.vertexAttribPointer(0, 3, gl.FLOAT, false, 0, 0);
    gl.bindVertexArray(null);
  
    const renderToFillScreen = function() {
      gl.bindVertexArray(vao);
      gl.drawElements(gl.TRIANGLES, 6, gl.UNSIGNED_SHORT, 0);
      gl.bindVertexArray(null);
    };
  
    let animationId = null;
    const reset = function() {
      if (animationId !== null) {
        cancelAnimationFrame(animationId);
      }
  
      canvas.width = window.innerWidth;
      canvas.height = window.innerHeight;
      gl.viewport(0.0, 0.0, canvas.width, canvas.height);
  
      let stateFbObjR = createFramebuffer(gl, canvas.width, canvas.height);//Read
      let stateFbObjW = createFramebuffer(gl, canvas.width, canvas.height);//Write
      const swapFramebuffer = function() {
        const tmp = stateFbObjR;
        stateFbObjR = stateFbObjW;
        stateFbObjW = tmp;
      };
  
      const initialize = function() {
        gl.bindFramebuffer(gl.FRAMEBUFFER, stateFbObjW.framebuffer);
        gl.useProgram(initializeProgram);
        gl.uniform2f(initializeUniforms['u_resolution'], canvas.width, canvas.height);
        gl.uniform2f(initializeUniforms['u_randomSeed'], Math.random() * 1000.0, Math.random() * 1000.0);
        renderToFillScreen();
        gl.bindFramebuffer(gl.FRAMEBUFFER, null);
        swapFramebuffer();
      };
  
      const update = function(deltaTime) {
        gl.bindFramebuffer(gl.FRAMEBUFFER, stateFbObjW.framebuffer)
        gl.useProgram(updateProgram);
        gl.activeTexture(gl.TEXTURE0);
        gl.bindTexture(gl.TEXTURE_2D, stateFbObjR.texture);
        gl.uniform1i(updateUniforms['u_stateTexture'], 0);
        gl.uniform2f(updateUniforms['u_diffusion'], parameters['diffusion U'], parameters['diffusion V']);
        gl.uniform1f(updateUniforms['u_timeStep'], deltaTime);
        gl.uniform1f(updateUniforms['u_spaceStep'], parameters['space step']);
        renderToFillScreen();
        gl.bindFramebuffer(gl.FRAMEBUFFER, null);
        swapFramebuffer();
      };
  
      const render = function() {
        gl.useProgram(renderProgram);
        gl.activeTexture(gl.TEXTURE0);
        gl.bindTexture(gl.TEXTURE_2D, stateFbObjR.texture);
        gl.uniform1i(renderUniforms['u_stateTexture'], 0);
        gl.uniform1i(renderUniforms['u_target'], parameters['target']);
        gl.uniform1i(renderUniforms['u_rendering'], parameters['rendering']);
        gl.uniform1f(renderUniforms['u_spaceStep'], parameters['space step']);
        renderToFillScreen();
      }
  
      initialize();
      let simulationSeconds = 0.0;
      let previousRealSeconds = performance.now() * 0.001;
      const loop = function() {
  
        const currentRealSeconds = performance.now() * 0.001;
        const nextSimulationSeconds = simulationSeconds + parameters['time scale'] * Math.min(0.2, currentRealSeconds - previousRealSeconds);
        previousRealSeconds = currentRealSeconds;
  
        const timeStep = parameters['time step'];
        while(nextSimulationSeconds - simulationSeconds > timeStep) {
          update(timeStep);
          simulationSeconds += timeStep;
        }
        render();
  
        animationId = requestAnimationFrame(loop);
      }
      loop();
    };
    reset();
  
  }());
