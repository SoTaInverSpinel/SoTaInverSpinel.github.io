//refernce:https://neort.io/art/bioab2k3p9f9psc9oev0
class Vector2 {
    constructor(x, y) {
      this.x = x;
      this.y = y;
    }
  
    static get zero() {
      return new Vector2(0.0, 0.0);
    }
  
    static add(v1, v2) {
      return new Vector2(v1.x + v2.x, v1.y + v2.y);
    }
  
    static sub(v1, v2) {
      return new Vector2(v1.x - v2.x, v1.y - v2.y);
    }
  
    static mul(v, s) {
      return new Vector2(v.x * s, v.y * s);
    }
  
    static div(v, s) {
      return new Vector2(v.x / s, v.y / s);
    }
  
    static norm(v) {
      const m = v.mag();
      return new Vector2(v.x / m, v.y / m);
    }
  
    static dot(v1, v2) {
      return v1.x * v2.x + v1.y * v2.y;
    }
  
    static dist(v1, v2) {
      const v = Vector2.sub(v1, v2);
      return v.mag();
    }
  
    add(v) {
      this.x += v.x;
      this.y += v.y;
      return this;
    }
  
    sub(v) {
      this.x -= v.x;
      this.y -= v.y;
      return this;
    }
  
    mul(s) {
      this.x *= s;
      this.y *= s;
      return this;
    }
  
    div(s) {
      this.x /= s;
      this.y /= s;
      return this;
    }
  
    ceil() {
      this.x = Math.ceil(this.x);
      this.y = Math.ceil(this.y);
      return this;
    }
  
    mag() {
      return Math.sqrt(this.sqMag());
    }
  
    sqMag() {
      return this.x * this.x + this.y * this.y;
    }
  
    norm() {
      return Vector2.norm(this);
    }
  
    dot(v) {
      return Vector3.dot(this, v);
    }
  
  }
  
  function createShader(gl, source, type) {
    const shader = gl.createShader(type);
    gl.shaderSource(shader, source);
    gl.compileShader(shader);
    if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
      throw new Error(gl.getShaderInfoLog(shader) + source);
    }
    return shader;
  }
  
  function createProgram(gl, vertexShader, fragmentShader) {
    const program = gl.createProgram();
    gl.attachShader(program, vertexShader);
    gl.attachShader(program, fragmentShader);
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
  
  function createVao(gl, vboObjs, ibo) {
    const vao = gl.createVertexArray();
    gl.bindVertexArray(vao);
    if (ibo !== undefined) {
      gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, ibo);
    }
    vboObjs.forEach((vboObj) => {
      gl.bindBuffer(gl.ARRAY_BUFFER, vboObj.buffer);
      gl.enableVertexAttribArray(vboObj.index);
      gl.vertexAttribPointer(vboObj.index, vboObj.size, gl.FLOAT, false, 0, 0);
    });
    gl.bindVertexArray(null);
    if (ibo !== undefined) {
      gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, null);
    }
    gl.bindBuffer(gl.ARRAY_BUFFER, null);
    return vao;
  }
  
  function createTexture(gl, size, internalFormat, format, type) {
    const texture = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, texture);
    gl.texImage2D(gl.TEXTURE_2D, 0, internalFormat, size, size, 0, format, type, null);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    gl.bindTexture(gl.TEXTURE_2D, null);
    return texture;
  }
  
  function getUniformLocations(gl, program, keys) {
    const locations = {};
    keys.forEach(key => {
        locations[key] = gl.getUniformLocation(program, key);
    });
    return locations;
  }
  
  function setTextureAsUniform(gl, index, texture, location) {
    gl.activeTexture(gl.TEXTURE0 + index);
    gl.bindTexture(gl.TEXTURE_2D, texture);
    gl.uniform1i(location, index);
  }
  
  (function() {
  
    const FILL_SCREEN_VERTEX_SHADER_SOURCE =
  `#version 300 es
  
  in vec2 position;
  
  void main(void) {
    gl_Position = vec4(position, 0.0, 1.0);
  }
  `
  
    const INITIALIZE_PARTICLE_FRAGMENT_SHADER_SOURCE =
  `#version 300 es
  
  precision highp float;
  
  #define PI 3.14159265359;
  
  layout (location = 0) out vec4 o_positionAndPhase;
  
  uniform vec2 u_randomSeed;
  uniform uint u_particleNum;//need
  uniform uint u_particleTextureSize;
  uniform vec2 u_simulationSpace;//need
  
  uint convertCoordToIndex(uvec2 coord, uint sizeX) {
    return coord.x + sizeX * coord.y;
  }
  
  float random(vec2 x){
    return fract(sin(dot(x,vec2(12.9898, 78.233))) * 43758.5453);
  }
  
  void main(void) {
    uint index = convertCoordToIndex(uvec2(gl_FragCoord.xy), u_particleTextureSize);
  
    if (index >= u_particleNum) { // unused pixels
      o_positionAndPhase = vec4(0.0);
      return;
    }
  
    vec2 position=vec2(
      random(gl_FragCoord.xy * 0.013 + random(u_randomSeed * vec2(32.19, 27.51) * 1000.0)),
      random(gl_FragCoord.xy * 0.029 + random(u_randomSeed * vec2(19.56, 11.34) * 1000.0))
      )* (u_simulationSpace - 1e-5) + 1e-5 * 0.5;
  
    float phase = random(gl_FragCoord.xy * 0.020 + random(u_randomSeed * vec2(19.56, 11.34) * 1000.0))*2.0 *PI;
  
    o_positionAndPhase = vec4(
      position,
      0.0,
      phase
    );
  }
  `
  
    const UPDATE_PARTICLE_FRAGMENT_SHADER_SOURCE = 
  `#version 300 es
  
  precision highp float;
  
  layout (location = 0) out vec4 o_positionAndPhase;
  
  #define PI_2 6.28318530718
  
  uniform sampler2D u_positionTexture;
  uniform sampler2D u_deltaPosAndPhaseTexture;
  uniform float u_deltaTime;
  uniform float u_maxSpeed;//no need
  uniform uint u_particleNum;
  uniform uint u_particleTextureSize;
  uniform vec2 u_simulationSpace;
  
  uint convertCoordToIndex(uvec2 coord, uint sizeX) {
    return coord.x + sizeX * coord.y;
  }
  
  vec2 limit(vec2 v, float max) {
    if (length(v) < max) {
      return normalize(v) * max;
    }
    return v;
  }
  
  void main(void) {
    ivec2 coord = ivec2(gl_FragCoord.xy);
  
    uint index = convertCoordToIndex(uvec2(coord), u_particleTextureSize);
    if (index >= u_particleNum) { // unused pixels
      o_positionAndPhase = vec4(0.0);
      //o_velocity = vec4(0.0);
      return;
    }
  
    vec2 position = texelFetch(u_positionTexture, coord, 0).xy;
    vec2 deltaPosAndPhase = texelFetch(u_deltaPosAndPhaseTexture, coord, 0).xy;
    float phase = texelFetch(u_positionTexture, coord, 0).w;
    float deltaPhase = texelFetch(u_deltaPosAndPhaseTexture, coord, 0).w;
  
  
    vec2 nextPosition = mod(position + u_deltaTime * deltaPosAndPhase, u_simulationSpace);
    float nextPhase = mod((phase + u_deltaTime * deltaPhase), PI_2);
   
    o_positionAndPhase = vec4(nextPosition,0.0,nextPhase);
    
  }
  `
  
  const COMPUTE_DELTA_FRAGMENT_SHADER_SOURCE =
  `#version 300 es
  
  precision highp float;
  precision highp isampler2D;
  precision highp usampler2D;
  
  // 4294967295 = 2^32 - 1
  #define MAX_32UI 4294967295u
  
  out vec4 o_deltaPositionAndPhase;
  
  uniform sampler2D u_positionTexture;
  uniform usampler2D u_bucketTexture;
  uniform usampler2D u_bucketReferrerTexture;
  
  uniform float u_radius;
  uniform uint u_particleNum;
  uniform float u_bucketSize;
  uniform ivec2 u_bucketNum;
  uniform vec2 u_simulationSpace;
  
  uniform float u_alpha;
  uniform float u_c1;
  uniform float u_c2;
  uniform float u_c3;
  
  uint convertCoordToIndex(uvec2 coord, uint sizeX) {
    return coord.x + sizeX * coord.y;
  }
  
  uvec2 convertIndexToCoord(uint index, uint sizeX) {
    return uvec2(index % sizeX, index / sizeX);
  }
  
  //In ONE bucket, find neighbors using bucketReferrer and add deltaPos,deltaPhase
  void findNeighbors(vec2 position, float phase, ivec2 bucketPosition, ivec2 bucketNum, uint particleTextureSizeX, uint bucketReferrerTextureSizeX, inout vec2 deltaPos, inout float deltaPhase) {
    if (bucketPosition.x < 0 || bucketPosition.x >= bucketNum.x ||
        bucketPosition.y < 0 || bucketPosition.y >= bucketNum.y) {
      return;
    }
    uint bucketIndex = uint(bucketPosition.x + bucketNum.x * bucketPosition.y);
    ivec2 coord = ivec2(convertIndexToCoord(bucketIndex, bucketReferrerTextureSizeX));
  
    uvec2 bucketReferrer = texelFetch(u_bucketReferrerTexture, coord, 0).xy;
  
    if (bucketReferrer.x == MAX_32UI || bucketReferrer.y == MAX_32UI) {
      return;
    }
  
    for (uint i = bucketReferrer.x; i <= bucketReferrer.y; i++) {
      uint ParticleIndex = texelFetch(u_bucketTexture, ivec2(convertIndexToCoord(i, particleTextureSizeX)), 0).y;
  
      ivec2 ParticleCoord = ivec2(convertIndexToCoord(ParticleIndex, particleTextureSizeX));
      vec2 otherPos = texelFetch(u_positionTexture, ParticleCoord, 0).xy;
      vec2 diffPos = otherPos - position;
      float dist = length(diffPos);
  
      float otherPhase = texelFetch(u_positionTexture, ParticleCoord,0 ).w;
      float diffPhase = otherPhase - phase;
  
      if (dist == 0.0) {
        continue;
      }
  
      if(dist < u_radius){
        deltaPos += normalize(diffPos) * exp(-dist) * sin(diffPhase + u_alpha * dist - u_c2);
        deltaPhase += exp(-dist) * sin(diffPhase + u_alpha * dist - u_c1);
      }
    }
  }
  
  vec4 computeDeltaPosAndPhase(vec2 position, float phase, uint particleTextureSizeX) {
    uint bucketReferrerTextureSizeX = uint(textureSize(u_bucketReferrerTexture, 0).x);
  
    vec2 bucketPosition = position / u_bucketSize;
    int xOffset = fract(bucketPosition.x) < 0.5 ? -1 : 1;
    int yOffset = fract(bucketPosition.y) < 0.5 ? -1 : 1;
  
    ivec2 bucketPosition00 = ivec2(bucketPosition);
    ivec2 bucketPosition10 = bucketPosition00 + ivec2(xOffset, 0);
    ivec2 bucketPosition01 = bucketPosition00 + ivec2(0, yOffset);
    ivec2 bucketPosition11 = bucketPosition00 + ivec2(xOffset, yOffset);
  
    vec2 deltaPos = vec2(0.0);
    float deltaPhase = 0.0f;
    
    findNeighbors(position, phase, bucketPosition00, u_bucketNum, particleTextureSizeX, bucketReferrerTextureSizeX, deltaPos, deltaPhase);
    findNeighbors(position, phase, bucketPosition10, u_bucketNum, particleTextureSizeX, bucketReferrerTextureSizeX, deltaPos, deltaPhase);
    findNeighbors(position, phase, bucketPosition01, u_bucketNum, particleTextureSizeX, bucketReferrerTextureSizeX, deltaPos, deltaPhase);
    findNeighbors(position, phase, bucketPosition11, u_bucketNum, particleTextureSizeX, bucketReferrerTextureSizeX, deltaPos, deltaPhase);
    //return separationForce + alignmentForce + cohesionForce;
  
    deltaPos*= u_c3;
    return vec4(deltaPos, 0.0, deltaPhase);
  
  }
  
  vec4 computeDeltaPositionAndPhase(uint particleTextureSizeX) {
  
    ivec2 coord = ivec2(gl_FragCoord.xy);
    vec2 position = texelFetch(u_positionTexture, coord, 0).xy;
    float phase = texelFetch(u_positionTexture, coord, 0).w;
    
    vec4 deltaPosAndPhase = computeDeltaPosAndPhase(position, phase, particleTextureSizeX);
    
    return deltaPosAndPhase;
  }
  
  void main(void) {
    uint particleTextureSizeX = uint(textureSize(u_positionTexture, 0).x);
    uint index = convertCoordToIndex(uvec2(gl_FragCoord.xy), particleTextureSizeX);
    if (index >= u_particleNum) { // unused pixels
      o_deltaPositionAndPhase = vec4(0.0);
      
      return;
    }
    
    o_deltaPositionAndPhase= computeDeltaPositionAndPhase(particleTextureSizeX);
  }
  `;
  
    const RENDER_Particle_VERTEX_SHADER_SOURCE =
  `#version 300 es
  
  precision highp isampler2D;
  precision highp usampler2D;
  
  const float PI = 3.14159265359;
  
  out vec4 v_color;
  
  uniform sampler2D u_positionTexture;
  uniform vec2 u_canvasSize;
  uniform vec2 u_simulationSpace;
  uniform float u_particleSize;
  
  vec3 hsv2rgb(vec3 color){
    vec4 K=vec4(1.0, 2.0/3.0, 1.0/3.0, 3.0);
    vec3 p = abs(fract(color.xxx + K.xyz) * 6.0 - K.www);
    return color.z * mix(K.xxx, clamp(p - K.xxx, 0.0, 1.0), color.y);
  }
  
  ivec2 convertIndexToCoord(int index, int sizeX) {
    return ivec2(index % sizeX, index / sizeX);
  }
  
  void main(void) {
    ivec2 coord = convertIndexToCoord(gl_VertexID, textureSize(u_positionTexture, 0).x);
    vec2 position = texelFetch(u_positionTexture, coord, 0).xy;
    
    vec2 scale = min(u_canvasSize.x, u_canvasSize.y) / u_canvasSize;
    float minSimulationSpace = min(u_simulationSpace.x, u_simulationSpace.y);
    vec2 canvasPos = scale * ((position / minSimulationSpace) * 2.0 - u_simulationSpace / minSimulationSpace);
    gl_Position = vec4(canvasPos, 0.0, 1.0);
    gl_PointSize = u_particleSize;
  
    float phase=texelFetch(u_positionTexture, coord,0).w;
    //0-2PI=>0-1
    v_color=vec4(hsv2rgb(vec3(phase/2.0/PI,1.0,1.0)),1.0);
  }
  `
  
    const RENDER_Particle_FRAGMENT_SHADER_SOURCE =
  `#version 300 es
  
  precision highp float;
  
  in vec4 v_color;
  out vec4 o_color;
  
  void main(void) {
    //circle shape
    vec2 p = ((gl_PointCoord - 0.5) * 2.0).xy;
    float dst= 1.0 - dot(p.xy,p.xy);
    if(dst < 0.0)discard;
    
    o_color = v_color;
  }
  `
  
  const INITIALIZE_BUCKET_FRAGMENT_SHADER_SOURCE =
  `#version 300 es
  
  precision highp float;
  
  // 4294967295 = 2^32 - 1
  #define MAX_32UI 4294967295u
  
  out uvec2 o_bucket;
  
  uniform sampler2D u_positionTexture;
  uniform float u_bucketSize;
  uniform uint u_particleNum;
  uniform uvec2 u_bucketNum;
  
  uint convertCoordToIndex(uvec2 coord, uint sizeX) {
    return coord.x + sizeX * coord.y;
  }
  
  uint getBucketIndex(vec2 position) {
    uvec2 bucketCoord = uvec2(position / u_bucketSize);
    return bucketCoord.x + bucketCoord.y * u_bucketNum.x;
  }
  
  void main(void) {
    uint positionTextureSizeX = uint(textureSize(u_positionTexture, 0).x);
    uint ParticleIndex = convertCoordToIndex(uvec2(gl_FragCoord.xy), positionTextureSizeX);
    if (ParticleIndex >= u_particleNum) {
      o_bucket = uvec2(MAX_32UI, 0); // = uvec2(2^32 - 1, 0)
    }
    vec2 position = texelFetch(u_positionTexture, ivec2(gl_FragCoord.xy), 0).xy;
    uint bucketIndex = getBucketIndex(position);
    o_bucket = uvec2(bucketIndex, ParticleIndex);
  }
  `
  
    const SWAP_BUCKET_INDEX_FRAGMENT_SHADER_SOURCE = 
  `#version 300 es
  
  precision highp float;
  precision highp usampler2D;
  
  out uvec2 o_bucket;
  
  uniform usampler2D u_bucketTexture;
  uniform uint u_size;
  uniform uint u_blockStep;
  uniform uint u_subBlockStep;
  
  uint convertCoordToIndex(uvec2 coord, uint sizeX) {
    return coord.x + sizeX * coord.y;
  }
  
  uvec2 convertIndexToCoord(uint index, uint sizeX) {
    return uvec2(index % sizeX, index / sizeX);
  }
  
  void main(void) {
    uint index = convertCoordToIndex(uvec2(gl_FragCoord.xy), u_size);
    uint d = 1u << (u_blockStep - u_subBlockStep);
  
    bool up = ((index >> u_blockStep) & 2u) == 0u;
  
    uint targetIndex;
    bool first = (index & d) == 0u;
    if (first) {
      targetIndex = index | d;
    } else {
      targetIndex = index & ~d;
      up = !up;
    }
  
    uvec2 a = texelFetch(u_bucketTexture, ivec2(gl_FragCoord.xy), 0).xy;
    uvec2 b = texelFetch(u_bucketTexture, ivec2(convertIndexToCoord(targetIndex, u_size)), 0).xy;
  
    if (a.x == b.x || (a.x >= b.x) == up) {
      o_bucket = b;
    } else {
      o_bucket = a;
    }
  }
  
  `;
  
    const INITIALIZE_BUCKET_REFERRER_FRAGMENT_SHADER_SOURCE =
  `#version 300 es
  
  precision highp float;
  precision highp usampler2D;
  
  // 4294967295 = 2^32 - 1
  #define MAX_32UI 4294967295u
  
  out uvec2 o_referrer;
  
  uniform uvec2 u_bucketReferrerTextureSize;
  uniform usampler2D u_bucketTexture;
  uniform uint u_particleNumN;
  uniform uvec2 u_bucketNum;
  
  uint convertCoordToIndex(uvec2 coord, uint sizeX) {
    return coord.x + sizeX * coord.y;
  }
  
  uvec2 convertIndexToCoord(uint index, uint sizeX) {
    return uvec2(index % sizeX, index / sizeX);
  }
  
  uint getBucketIndex(uint particleIndex, uint particleTextureSizeX) {
    return texelFetch(u_bucketTexture, ivec2(convertIndexToCoord(particleIndex, particleTextureSizeX)), 0).x;
  }
  
  
  uint binarySearchMinIndex(uint target, uint from, uint to, uint particleTextureSizeX) {
    for (uint i = 0u; i < u_particleNumN + 1u; i++) {
      uint middle = from + (to - from) / 2u;
      uint bucketIndex = getBucketIndex(middle, particleTextureSizeX);
      if (bucketIndex < target) {
        from = middle + 1u;
      } else {
        to = middle;
      }
      if (from == to) {
        if (getBucketIndex(from, particleTextureSizeX) == target) {
          return from;
        } else {
          return MAX_32UI;
        }
      }
    }
    return MAX_32UI;
  }
  
  uint binarySearchMaxIndex(uint target, uint from, uint to, uint particleTextureSizeX) {
    for (uint i = 0u; i < u_particleNumN + 1u; i++) {
      uint middle = from + (to - from) / 2u + 1u;
      uint bucketIndex = getBucketIndex(middle, particleTextureSizeX);
      if (bucketIndex > target) {
        to = middle - 1u;
      } else {
        from = middle;
      }
      if (from == to) {
        if (getBucketIndex(from, particleTextureSizeX) == target) {
          return from;
        } else {
          return MAX_32UI;
        }
      }
    }
    return MAX_32UI;
  }
  
  uvec2 binarySearchRange(uint target, uint from, uint to) {
    uint particleTextureSizeX = uint(textureSize(u_bucketTexture, 0).x);
    from =  binarySearchMinIndex(target, from, to, particleTextureSizeX);
    to = from == MAX_32UI ? MAX_32UI : binarySearchMaxIndex(target, from, to, particleTextureSizeX);
    return uvec2(from, to);
  }
  
  void main(void) {
    uint bucketIndex = convertCoordToIndex(uvec2(gl_FragCoord.xy), u_bucketReferrerTextureSize.x);
    uint maxBucketIndex = u_bucketNum.x * u_bucketNum.y;
  
    if (bucketIndex >= maxBucketIndex) {
      o_referrer = uvec2(MAX_32UI, MAX_32UI);
      return;
    }
  
    uvec2 particleTextureSize = uvec2(textureSize(u_bucketTexture, 0));
    uint particleNum = particleTextureSize.x * particleTextureSize.y;
  
    o_referrer = binarySearchRange(bucketIndex, 0u, particleNum - 1u);
  }
  `
  
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
  
  
    function createInitializeParticleProgram(gl) {
      const vertexShader = createShader(gl, FILL_SCREEN_VERTEX_SHADER_SOURCE, gl.VERTEX_SHADER);
      const fragmentShader = createShader(gl, INITIALIZE_PARTICLE_FRAGMENT_SHADER_SOURCE, gl.FRAGMENT_SHADER);
      return createProgram(gl, vertexShader, fragmentShader);
    }
  
    function createUpdateParticleProgram(gl) {
      const vertexShader = createShader(gl, FILL_SCREEN_VERTEX_SHADER_SOURCE, gl.VERTEX_SHADER);
      const fragmentShader = createShader(gl, UPDATE_PARTICLE_FRAGMENT_SHADER_SOURCE, gl.FRAGMENT_SHADER);
      return createProgram(gl, vertexShader, fragmentShader);
    }
  
    function createconputeDeltaPosAndPhaseProgram(gl) {
      const vertexShader = createShader(gl, FILL_SCREEN_VERTEX_SHADER_SOURCE, gl.VERTEX_SHADER);
      const fragmentShader = createShader(gl, COMPUTE_DELTA_FRAGMENT_SHADER_SOURCE, gl.FRAGMENT_SHADER);
      return createProgram(gl, vertexShader, fragmentShader);
    }
  
    function createRenderParticleProgram(gl) {
      const vertexShader = createShader(gl, RENDER_Particle_VERTEX_SHADER_SOURCE, gl.VERTEX_SHADER);
      const fragmentShader = createShader(gl, RENDER_Particle_FRAGMENT_SHADER_SOURCE, gl.FRAGMENT_SHADER);
      return createProgram(gl, vertexShader, fragmentShader);
    }
  
    function createInitializeBucketProgram(gl) {
      const vertexShader = createShader(gl, FILL_SCREEN_VERTEX_SHADER_SOURCE, gl.VERTEX_SHADER);
      const fragmentShader = createShader(gl, INITIALIZE_BUCKET_FRAGMENT_SHADER_SOURCE, gl.FRAGMENT_SHADER);
      return createProgram(gl, vertexShader, fragmentShader);
    }
  
    function createSwapBucketIndexProgram(gl) {
      const vertexShader = createShader(gl, FILL_SCREEN_VERTEX_SHADER_SOURCE, gl.VERTEX_SHADER);
      const fragmentShader = createShader(gl, SWAP_BUCKET_INDEX_FRAGMENT_SHADER_SOURCE, gl.FRAGMENT_SHADER);
      return createProgram(gl, vertexShader, fragmentShader);
    }
  
    function createInitializeBucketReferrerProgram(gl) {
      const vertexShader = createShader(gl, FILL_SCREEN_VERTEX_SHADER_SOURCE, gl.VERTEX_SHADER);
      const fragmentShader = createShader(gl, INITIALIZE_BUCKET_REFERRER_FRAGMENT_SHADER_SOURCE, gl.FRAGMENT_SHADER);
      return createProgram(gl, vertexShader, fragmentShader);
    }
  
    function createParticleFramebuffer(gl, size) {
      const framebuffer = gl.createFramebuffer();
      gl.bindFramebuffer(gl.FRAMEBUFFER, framebuffer);
      const positionTexture = createTexture(gl, size, gl.RGBA32F, gl.RGBA, gl.FLOAT);
      gl.bindTexture(gl.TEXTURE_2D, positionTexture);
      gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, positionTexture, 0);
      gl.bindFramebuffer(gl.FRAMEBUFFER, null);
      gl.bindTexture(gl.TEXTURE_2D, null);
      return {
        framebuffer: framebuffer,
        positionTexture: positionTexture
      };
    }
  
    function createForceFramebuffer(gl, size) {
      const framebuffer = gl.createFramebuffer();
      gl.bindFramebuffer(gl.FRAMEBUFFER, framebuffer);
      const deltaPosAndPhaseTexture = createTexture(gl, size, gl.RGBA32F, gl.RGBA, gl.FLOAT);
      gl.bindTexture(gl.TEXTURE_2D, deltaPosAndPhaseTexture);
      gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, deltaPosAndPhaseTexture, 0);
      gl.bindFramebuffer(gl.FRAMEBUFFER, null);
      gl.bindTexture(gl.TEXTURE_2D, null);
      return {
        framebuffer: framebuffer,
        deltaPosAndPhaseTexture: deltaPosAndPhaseTexture
      };
    }
  
    function createBucketFramebuffer(gl, size) {
      const framebuffer = gl.createFramebuffer();
      gl.bindFramebuffer(gl.FRAMEBUFFER, framebuffer);
      const bucketTexture = createTexture(gl, size, gl.RG32UI, gl.RG_INTEGER, gl.UNSIGNED_INT);
      gl.bindTexture(gl.TEXTURE_2D, bucketTexture);
      gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, bucketTexture, 0);
      gl.bindFramebuffer(gl.FRAMEBUFFER, null);
      gl.bindTexture(gl.TEXTURE_2D, null);
      return {
        framebuffer: framebuffer,
        bucketTexture: bucketTexture
      };
    }
  
    function createBucketReferrerFramebuffer(gl, size) {
      const framebuffer = gl.createFramebuffer();
      gl.bindFramebuffer(gl.FRAMEBUFFER, framebuffer);
      const bucketReferrerTexture = createTexture(gl, size, gl.RG32UI, gl.RG_INTEGER, gl.UNSIGNED_INT);
      gl.bindTexture(gl.TEXTURE_2D, bucketReferrerTexture);
      gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, bucketReferrerTexture, 0);
      gl.bindFramebuffer(gl.FRAMEBUFFER, null);
      gl.bindTexture(gl.TEXTURE_2D, null);
      return {
        framebuffer: framebuffer,
        bucketReferrerTexture: bucketReferrerTexture
      };
    }
  
    const canvas = document.getElementById('canvas');
    const resizeCanvas = function() {
      canvas.width = window.innerWidth;
      canvas.height = window.innerHeight;
    };
    window.addEventListener('resize', resizeCanvas);
    resizeCanvas();
  
    const gl = canvas.getContext('webgl2');
    gl.getExtension('EXT_color_buffer_float');
  
    const initializeParticleProgram = createInitializeParticleProgram(gl);
    const updateParticleProgram = createUpdateParticleProgram(gl);
    const conputeDeltaPosAndPhaseProgram = createconputeDeltaPosAndPhaseProgram(gl);
    const renderParticleProgram = createRenderParticleProgram(gl);
    const initializeBucketProgram = createInitializeBucketProgram(gl);
    const swapBucketIndexProgram = createSwapBucketIndexProgram(gl);
    const initializeBucketReferrerProgram = createInitializeBucketReferrerProgram(gl);
  
    const initializeParticleUniforms = getUniformLocations(gl, initializeParticleProgram, ['u_randomSeed', 'u_particleNum', 'u_particleTextureSize', 'u_simulationSpace']);
    const updateParticleUniforms = getUniformLocations(gl, updateParticleProgram, ['u_positionTexture', 'u_deltaPosAndPhaseTexture', 'u_deltaTime', 'u_particleNum', 'u_particleTextureSize', 'u_maxSpeed', 'u_simulationSpace']);
    const conputeDeltaPosAndPhaseUniforms = getUniformLocations(gl, conputeDeltaPosAndPhaseProgram,
      ['u_positionTexture', 'u_bucketTexture', 'u_bucketReferrerTexture', 'u_particleNum', 'u_radius', 'u_bucketSize', 'u_bucketNum', 'u_simulationSpace','u_alpha', 'u_c1','u_c2','u_c3']);
    const renderParticleUniforms = getUniformLocations(gl, renderParticleProgram, ['u_positionTexture', 'u_canvasSize', 'u_simulationSpace', 'u_particleSize']);
    const initializeBucketUniforms = getUniformLocations(gl, initializeBucketProgram, ['u_positionTexture', 'u_bucketSize', 'u_particleNum', 'u_bucketNum']);
    const swapBucketIndexUniforms = getUniformLocations(gl, swapBucketIndexProgram, ['u_bucketTexture', 'u_size', 'u_blockStep', 'u_subBlockStep']);
    const initializeBucketReferrerUniforms = getUniformLocations(gl, initializeBucketReferrerProgram, ['u_bucketTexture', 'u_bucketReferrerTextureSize', 'u_particleNumN', 'u_bucketNum']);
  
    const fillScreenVao = createVao(gl,
      [{buffer: createVbo(gl, VERTICES_POSITION), size: 2, index: 0}],
      createIbo(gl, VERTICES_INDEX)
    );
  
    const MAX_PARTICLE_NUM = 4294967295; // = 2^32 - 1
  　const MAX_BUCKET_NUM = 4294967295; // = 2^32 - 1
  
    const parameters = {
      dynamic: {
        'alpha':0.5,
        'c1':1.0,
        'c2':0.5,
        'c3':2.0,
        'particle size': 6.0,
      },
      static: {
      'particle num': 2000,
      'effective radius':8.0,
      },
      'reset': () => reset()
    };
  
    let requestId = null;
    function reset() {
      if (requestId !== null) {
        cancelAnimationFrame(requestId);
      }
  
      const particleNum = parameters.static['particle num'];
      if (particleNum > MAX_PARTICLE_NUM) {
        throw new Error(`number of Particles must be less than ${MAX_PARTICLE_NUM}. current value is ${particleNum}.`);
      }
      let particleTextureSize;
      let particleNumN;
      for (let i = 0; ; i++) {
        particleTextureSize = 2 ** i;
        if (particleTextureSize * particleTextureSize > particleNum) {
          particleNumN = i * 2;
          break;
        }
      }
  
      const neighborRadius = parameters.static['effective radius'];
  
      const bucketSize = 2.0 * neighborRadius;
      const simulationSpace = new Vector2(100.0,60.0);
      const bucketNum = Vector2.div(simulationSpace, bucketSize).ceil().add(new Vector2(1, 1));
      const totalBuckets = bucketNum.x * bucketNum.y;
      if (totalBuckets > MAX_BUCKET_NUM) {
        throw new Error(`number of buckets must be less than ${MAX_BUCKET_NUM}. current value is ${totalBuckets}.`);
      }
      let bucketReferrerTextureSize;
      for (let i = 0; ; i++) {
        bucketReferrerTextureSize = 2 ** i;
        if (bucketReferrerTextureSize * bucketReferrerTextureSize > totalBuckets) {
          break;
        }
      }
  
      let ParticleFbObjR = createParticleFramebuffer(gl, particleTextureSize);
      let ParticleFbObjW = createParticleFramebuffer(gl, particleTextureSize);
      const swapParticleFbObj = function() {
        const tmp = ParticleFbObjR;
        ParticleFbObjR = ParticleFbObjW;
        ParticleFbObjW = tmp;
      };
      let forceFbObj = createForceFramebuffer(gl, particleTextureSize);
      let bucketFbObjR = createBucketFramebuffer(gl, particleTextureSize);
      let bucketFbObjW = createBucketFramebuffer(gl, particleTextureSize);
      const swapBucketFbObj = function() {
        const tmp = bucketFbObjR;
        bucketFbObjR = bucketFbObjW;
        bucketFbObjW = tmp;
      }
      const bucketReferrerFbObj = createBucketReferrerFramebuffer(gl, bucketReferrerTextureSize);
  
      const initializeParticles = function() {
        gl.bindFramebuffer(gl.FRAMEBUFFER, ParticleFbObjW.framebuffer);
        gl.viewport(0.0, 0.0, particleTextureSize, particleTextureSize);
        gl.useProgram(initializeParticleProgram);
        gl.uniform2f(initializeParticleUniforms['u_randomSeed'], Math.random() * 1000.0, Math.random() * 1000.0);
        gl.uniform1ui(initializeParticleUniforms['u_particleNum'], particleNum);
        gl.uniform1ui(initializeParticleUniforms['u_particleTextureSize'], particleTextureSize)
        gl.uniform2f(initializeParticleUniforms['u_simulationSpace'], simulationSpace.x, simulationSpace.y);
        gl.bindVertexArray(fillScreenVao);
        gl.drawElements(gl.TRIANGLES, 6, gl.UNSIGNED_SHORT, 0);
        gl.bindVertexArray(null);
    
        gl.bindFramebuffer(gl.FRAMEBUFFER, null);
        swapParticleFbObj();
      };
    
      const initializeBucket = function() {
        gl.bindFramebuffer(gl.FRAMEBUFFER, bucketFbObjW.framebuffer);
        gl.viewport(0.0, 0.0, particleTextureSize, particleTextureSize);
        gl.useProgram(initializeBucketProgram);
        setTextureAsUniform(gl, 0, ParticleFbObjR.positionTexture, initializeBucketUniforms['u_positionTexture']);
        gl.uniform1f(initializeBucketUniforms['u_bucketSize'], bucketSize);
        gl.uniform1ui(initializeBucketUniforms['u_particleNum'], particleNum);
        gl.uniform2ui(initializeBucketUniforms['u_bucketNum'], bucketNum.x, bucketNum.y);
        gl.bindVertexArray(fillScreenVao);
        gl.drawElements(gl.TRIANGLES, 6, gl.UNSIGNED_SHORT, 0);
        gl.bindVertexArray(null);
        gl.bindFramebuffer(gl.FRAMEBUFFER, null);
        swapBucketFbObj();
      }
    
      const swapBucketIndex = function(i, j) {
        gl.bindFramebuffer(gl.FRAMEBUFFER, bucketFbObjW.framebuffer);
        gl.viewport(0.0, 0.0, particleTextureSize, particleTextureSize);
        gl.useProgram(swapBucketIndexProgram);
        setTextureAsUniform(gl, 0, bucketFbObjR.bucketTexture, swapBucketIndexUniforms['u_bucketTexture']);
        gl.uniform1ui(swapBucketIndexUniforms['u_size'], particleTextureSize, particleTextureSize);
        gl.uniform1ui(swapBucketIndexUniforms['u_blockStep'], i);
        gl.uniform1ui(swapBucketIndexUniforms['u_subBlockStep'], j);
        gl.bindVertexArray(fillScreenVao);
        gl.drawElements(gl.TRIANGLES, 6, gl.UNSIGNED_SHORT, 0);
        gl.bindVertexArray(null);
        gl.bindFramebuffer(gl.FRAMEBUFFER, null);
        swapBucketFbObj();
      }
    
      const initializeBucketRefrrer = function() {
        gl.bindFramebuffer(gl.FRAMEBUFFER, bucketReferrerFbObj.framebuffer);
        gl.viewport(0.0, 0.0, bucketReferrerTextureSize, bucketReferrerTextureSize);
        gl.useProgram(initializeBucketReferrerProgram);
        setTextureAsUniform(gl, 0, bucketFbObjR.bucketTexture, initializeBucketReferrerUniforms['u_bucketTexture']);
        gl.uniform1ui(initializeBucketReferrerUniforms['u_particleNumN'], particleNumN);
        gl.uniform2ui(initializeBucketReferrerUniforms['u_bucketReferrerTextureSize'], bucketReferrerTextureSize, bucketReferrerTextureSize);
        gl.uniform2ui(initializeBucketReferrerUniforms['u_bucketNum'], bucketNum.x, bucketNum.y);
        gl.bindVertexArray(fillScreenVao);
        gl.drawElements(gl.TRIANGLES, 6, gl.UNSIGNED_SHORT, 0);
        gl.bindVertexArray(null);
        gl.bindFramebuffer(gl.FRAMEBUFFER, null);
      }
    
      const constructBuckets = function() {
        initializeBucket();
        // sort by bitonic sort
        for (let i = 0; i < particleNumN; i++) {
          for (let j = 0; j <= i; j++) {
            swapBucketIndex(i, j);
          }
        }
        initializeBucketRefrrer();
      }
  
      const conputeDeltaPosAndPhases = function() {
        gl.bindFramebuffer(gl.FRAMEBUFFER, forceFbObj.framebuffer);
        gl.viewport(0.0, 0.0, particleTextureSize, particleTextureSize);
        gl.useProgram(conputeDeltaPosAndPhaseProgram);
        setTextureAsUniform(gl, 0, ParticleFbObjR.positionTexture, conputeDeltaPosAndPhaseUniforms['u_positionTexture']);
        setTextureAsUniform(gl, 1, bucketFbObjR.bucketTexture, conputeDeltaPosAndPhaseUniforms['u_bucketTexture']);
        setTextureAsUniform(gl, 2, bucketReferrerFbObj.bucketReferrerTexture, conputeDeltaPosAndPhaseUniforms['u_bucketReferrerTexture']);
        gl.uniform1ui(conputeDeltaPosAndPhaseUniforms['u_particleNum'], particleNum);
        gl.uniform1f(conputeDeltaPosAndPhaseUniforms['u_bucketSize'], bucketSize);
        gl.uniform2i(conputeDeltaPosAndPhaseUniforms['u_bucketNum'], bucketNum.x, bucketNum.y);
        gl.uniform2f(conputeDeltaPosAndPhaseUniforms['u_simulationSpace'], simulationSpace.x, simulationSpace.y);
        gl.uniform1f(conputeDeltaPosAndPhaseUniforms['u_radius'], neighborRadius);
        gl.uniform1f(conputeDeltaPosAndPhaseUniforms['u_alpha'],parameters.dynamic['alpha']);
        gl.uniform1f(conputeDeltaPosAndPhaseUniforms['u_c1'],parameters.dynamic['c1']);
        gl.uniform1f(conputeDeltaPosAndPhaseUniforms['u_c2'],parameters.dynamic['c2']);
        gl.uniform1f(conputeDeltaPosAndPhaseUniforms['u_c3'],parameters.dynamic['c3']);
  
        gl.bindVertexArray(fillScreenVao);
        gl.drawElements(gl.TRIANGLES, 6, gl.UNSIGNED_SHORT, 0);
        gl.bindVertexArray(null);
        gl.bindFramebuffer(gl.FRAMEBUFFER, null);
      }
  
      const updateParticles = function(deltaTime) {
        gl.bindFramebuffer(gl.FRAMEBUFFER, ParticleFbObjW.framebuffer);
        gl.viewport(0.0, 0.0, particleTextureSize, particleTextureSize);
        gl.useProgram(updateParticleProgram);
        setTextureAsUniform(gl, 0, ParticleFbObjR.positionTexture, updateParticleUniforms['u_positionTexture']);
        setTextureAsUniform(gl, 1, forceFbObj.deltaPosAndPhaseTexture, updateParticleUniforms['u_deltaPosAndPhaseTexture']);
        gl.uniform1f(updateParticleUniforms['u_deltaTime'], deltaTime);
        gl.uniform1ui(updateParticleUniforms['u_particleNum'], particleNum);
        gl.uniform1ui(updateParticleUniforms['u_particleTextureSize'], particleTextureSize);
        gl.uniform2f(updateParticleUniforms['u_simulationSpace'], simulationSpace.x, simulationSpace.y);
        gl.bindVertexArray(fillScreenVao);
        gl.drawElements(gl.TRIANGLES, 6, gl.UNSIGNED_SHORT, 0);
        gl.bindVertexArray(null);
        gl.bindFramebuffer(gl.FRAMEBUFFER, null);
        swapParticleFbObj();
      }
  
      const stepSimulation = function(deltaTime) {
        constructBuckets();
        conputeDeltaPosAndPhases();
        updateParticles(deltaTime);
      };
  
      const renderParticles = function() {
        gl.viewport(0.0, 0.0, canvas.width, canvas.height);
        gl.clear(gl.COLOR_BUFFER_BIT);
  
        gl.useProgram(renderParticleProgram);
        setTextureAsUniform(gl, 0, ParticleFbObjR.positionTexture, renderParticleUniforms['u_positionTexture']);
        gl.uniform2f(renderParticleUniforms['u_canvasSize'], canvas.width, canvas.height);
        gl.uniform2f(renderParticleUniforms['u_simulationSpace'], simulationSpace.x, simulationSpace.y);
        gl.uniform1f(renderParticleUniforms['u_particleSize'], parameters.dynamic['particle size']);
  
        gl.drawArrays(gl.POINTS, 0, particleNum);
      }
  
      initializeParticles();
    
      gl.clearColor(0.2, 0.2, 0.2, 1.0);
      let previousTime = performance.now();
      const render = function() {
    
        const currentTime = performance.now();
        const deltaTime = 20.0*Math.min(0.01, (currentTime - previousTime) * 0.001);
        previousTime = currentTime;
  
        stepSimulation(deltaTime);
        renderParticles();
  
        requestId = requestAnimationFrame(render);
      };
      render();
    };
    reset();
  
  
  }());