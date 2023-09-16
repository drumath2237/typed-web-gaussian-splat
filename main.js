import "/src/style.css";

import { preDeclCamera } from "/src/preDeclCameras.ts";
import {
  getProjectionMatrix,
  getViewMatrix,
  invert4,
  multiply4,
  rotate4,
  translate4,
} from "/src/mathUtils.ts";

import {
  vertexShaderSource,
  fragmentShaderSource,
} from "/src/shaders/shader.ts";

let cameras = preDeclCamera;

const camera = cameras[0];

function createWorker(self) {
  let buffer;
  let vertexCount = 0;
  let viewProj;
  // 6*4 + 4 + 4 = 8*4
  // XYZ - Position (Float32)
  // XYZ - Scale (Float32)
  // RGBA - colors (uint8)
  // IJKL - quaternion/rot (uint8)
  const rowLength = 3 * 4 + 3 * 4 + 4 + 4;
  let depthMix = new BigInt64Array();
  let lastProj = [];

  const runSort = (viewProj) => {
    if (!buffer) return;

    const f_buffer = new Float32Array(buffer);
    const u_buffer = new Uint8Array(buffer);

    const quat = new Float32Array(4 * vertexCount);
    const scale = new Float32Array(3 * vertexCount);
    const center = new Float32Array(3 * vertexCount);
    const color = new Float32Array(4 * vertexCount);

    if (depthMix.length !== vertexCount) {
      depthMix = new BigInt64Array(vertexCount);
      const indexMix = new Uint32Array(depthMix.buffer);
      for (let j = 0; j < vertexCount; j++) {
        indexMix[2 * j] = j;
      }
    } else {
      let dot =
        lastProj[2] * viewProj[2] +
        lastProj[6] * viewProj[6] +
        lastProj[10] * viewProj[10];
      if (Math.abs(dot - 1) < 0.01) {
        return;
      }
    }
    // console.time("sort");

    const floatMix = new Float32Array(depthMix.buffer);
    const indexMix = new Uint32Array(depthMix.buffer);

    for (let j = 0; j < vertexCount; j++) {
      let i = indexMix[2 * j];
      floatMix[2 * j + 1] =
        10000 +
        viewProj[2] * f_buffer[8 * i + 0] +
        viewProj[6] * f_buffer[8 * i + 1] +
        viewProj[10] * f_buffer[8 * i + 2];
    }

    lastProj = viewProj;

    depthMix.sort();

    for (let j = 0; j < vertexCount; j++) {
      const i = indexMix[2 * j];

      quat[4 * j + 0] = (u_buffer[32 * i + 28 + 0] - 128) / 128;
      quat[4 * j + 1] = (u_buffer[32 * i + 28 + 1] - 128) / 128;
      quat[4 * j + 2] = (u_buffer[32 * i + 28 + 2] - 128) / 128;
      quat[4 * j + 3] = (u_buffer[32 * i + 28 + 3] - 128) / 128;

      center[3 * j + 0] = f_buffer[8 * i + 0];
      center[3 * j + 1] = f_buffer[8 * i + 1];
      center[3 * j + 2] = f_buffer[8 * i + 2];

      color[4 * j + 0] = u_buffer[32 * i + 24 + 0] / 255;
      color[4 * j + 1] = u_buffer[32 * i + 24 + 1] / 255;
      color[4 * j + 2] = u_buffer[32 * i + 24 + 2] / 255;
      color[4 * j + 3] = u_buffer[32 * i + 24 + 3] / 255;

      scale[3 * j + 0] = f_buffer[8 * i + 3 + 0];
      scale[3 * j + 1] = f_buffer[8 * i + 3 + 1];
      scale[3 * j + 2] = f_buffer[8 * i + 3 + 2];
    }

    self.postMessage({ quat, center, color, scale, viewProj }, [
      quat.buffer,
      center.buffer,
      color.buffer,
      scale.buffer,
    ]);

    // console.timeEnd("sort");
  };

  function processPlyBuffer(inputBuffer) {
    const ubuf = new Uint8Array(inputBuffer);
    // 10KB ought to be enough for a header...
    const header = new TextDecoder().decode(ubuf.slice(0, 1024 * 10));
    const header_end = "end_header\n";
    const header_end_index = header.indexOf(header_end);
    if (header_end_index < 0)
      throw new Error("Unable to read .ply file header");
    const vertexCount = parseInt(/element vertex (\d+)\n/.exec(header)[1]);
    console.log("Vertex Count", vertexCount);
    let row_offset = 0,
      offsets = {},
      types = {};
    const TYPE_MAP = {
      double: "getFloat64",
      int: "getInt32",
      uint: "getUint32",
      float: "getFloat32",
      short: "getInt16",
      ushort: "getUint16",
      uchar: "getUint8",
    };
    for (let prop of header
      .slice(0, header_end_index)
      .split("\n")
      .filter((k) => k.startsWith("property "))) {
      const [p, type, name] = prop.split(" ");
      const arrayType = TYPE_MAP[type] || "getInt8";
      types[name] = arrayType;
      offsets[name] = row_offset;
      row_offset += parseInt(arrayType.replace(/[^\d]/g, "")) / 8;
    }
    console.log("Bytes per row", row_offset, types, offsets);

    let dataView = new DataView(
      inputBuffer,
      header_end_index + header_end.length
    );
    let row = 0;
    const attrs = new Proxy(
      {},
      {
        get(target, prop) {
          if (!types[prop]) throw new Error(prop + " not found");
          return dataView[types[prop]](row * row_offset + offsets[prop], true);
        },
      }
    );

    console.time("calculate importance");
    let sizeList = new Float32Array(vertexCount);
    let sizeIndex = new Uint32Array(vertexCount);
    for (row = 0; row < vertexCount; row++) {
      sizeIndex[row] = row;
      if (!types["scale_0"]) continue;
      const size =
        Math.exp(attrs.scale_0) *
        Math.exp(attrs.scale_1) *
        Math.exp(attrs.scale_2);
      const opacity = 1 / (1 + Math.exp(-attrs.opacity));
      sizeList[row] = size * opacity;
    }
    console.timeEnd("calculate importance");

    console.time("sort");
    sizeIndex.sort((b, a) => sizeList[a] - sizeList[b]);
    console.timeEnd("sort");

    // 6*4 + 4 + 4 = 8*4
    // XYZ - Position (Float32)
    // XYZ - Scale (Float32)
    // RGBA - colors (uint8)
    // IJKL - quaternion/rot (uint8)
    const rowLength = 3 * 4 + 3 * 4 + 4 + 4;
    const buffer = new ArrayBuffer(rowLength * vertexCount);

    console.time("build buffer");
    for (let j = 0; j < vertexCount; j++) {
      row = sizeIndex[j];

      const position = new Float32Array(buffer, j * rowLength, 3);
      const scales = new Float32Array(buffer, j * rowLength + 4 * 3, 3);
      const rgba = new Uint8ClampedArray(
        buffer,
        j * rowLength + 4 * 3 + 4 * 3,
        4
      );
      const rot = new Uint8ClampedArray(
        buffer,
        j * rowLength + 4 * 3 + 4 * 3 + 4,
        4
      );

      if (types["scale_0"]) {
        const qlen = Math.sqrt(
          attrs.rot_0 ** 2 +
            attrs.rot_1 ** 2 +
            attrs.rot_2 ** 2 +
            attrs.rot_3 ** 2
        );

        rot[0] = (attrs.rot_0 / qlen) * 128 + 128;
        rot[1] = (attrs.rot_1 / qlen) * 128 + 128;
        rot[2] = (attrs.rot_2 / qlen) * 128 + 128;
        rot[3] = (attrs.rot_3 / qlen) * 128 + 128;

        scales[0] = Math.exp(attrs.scale_0);
        scales[1] = Math.exp(attrs.scale_1);
        scales[2] = Math.exp(attrs.scale_2);
      } else {
        scales[0] = 0.01;
        scales[1] = 0.01;
        scales[2] = 0.01;

        rot[0] = 255;
        rot[1] = 0;
        rot[2] = 0;
        rot[3] = 0;
      }

      position[0] = attrs.x;
      position[1] = attrs.y;
      position[2] = attrs.z;

      if (types["f_dc_0"]) {
        const SH_C0 = 0.28209479177387814;
        rgba[0] = (0.5 + SH_C0 * attrs.f_dc_0) * 255;
        rgba[1] = (0.5 + SH_C0 * attrs.f_dc_1) * 255;
        rgba[2] = (0.5 + SH_C0 * attrs.f_dc_2) * 255;
      } else {
        rgba[0] = attrs.red;
        rgba[1] = attrs.green;
        rgba[2] = attrs.blue;
      }
      if (types["opacity"]) {
        rgba[3] = (1 / (1 + Math.exp(-attrs.opacity))) * 255;
      } else {
        rgba[3] = 255;
      }
    }
    console.timeEnd("build buffer");
    return buffer;
  }

  const throttledSort = () => {
    if (!sortRunning) {
      sortRunning = true;
      let lastView = viewProj;
      runSort(lastView);
      setTimeout(() => {
        sortRunning = false;
        if (lastView !== viewProj) {
          throttledSort();
        }
      }, 0);
    }
  };

  let sortRunning;
  self.onmessage = (e) => {
    if (e.data.ply) {
      vertexCount = 0;
      runSort(viewProj);
      buffer = processPlyBuffer(e.data.ply);
      vertexCount = Math.floor(buffer.byteLength / rowLength);
      postMessage({ buffer: buffer });
    } else if (e.data.buffer) {
      buffer = e.data.buffer;
      vertexCount = e.data.vertexCount;
    } else if (e.data.vertexCount) {
      vertexCount = e.data.vertexCount;
    } else if (e.data.view) {
      viewProj = e.data.view;
      throttledSort();
    }
  };
}

let defaultViewMatrix = [
  0.47, 0.04, 0.88, 0, -0.11, 0.99, 0.02, 0, -0.88, -0.11, 0.47, 0, 0.07, 0.03,
  6.55, 1,
];
let viewMatrix = defaultViewMatrix;

async function main() {
  let carousel = true;
  const params = new URLSearchParams(location.search);
  try {
    viewMatrix = JSON.parse(decodeURIComponent(location.hash.slice(1)));
    carousel = false;
  } catch (err) {}
  const url = new URL(
    // "nike.splat",
    // location.href,
    params.get("url") || "train.splat",
    "https://huggingface.co/cakewalk/splat-data/resolve/main/"
  );
  const req = await fetch(url, {
    mode: "cors", // no-cors, *cors, same-origin
    credentials: "omit", // include, *same-origin, omit
  });
  console.log(req);
  if (req.status != 200)
    throw new Error(req.status + " Unable to load " + req.url);

  const rowLength = 3 * 4 + 3 * 4 + 4 + 4;
  const reader = req.body.getReader();
  let splatData = new Uint8Array(req.headers.get("content-length"));

  const downsample = splatData.length / rowLength > 500000 ? 2 : 1;
  console.log(splatData.length / rowLength, downsample);

  const worker = new Worker(
    URL.createObjectURL(
      new Blob(["(", createWorker.toString(), ")(self)"], {
        type: "application/javascript",
      })
    )
  );

  const canvas = document.getElementById("canvas");
  canvas.width = innerWidth / downsample;
  canvas.height = innerHeight / downsample;

  const fps = document.getElementById("fps");

  let projectionMatrix = getProjectionMatrix(
    camera.fx / downsample,
    camera.fy / downsample,
    canvas.width,
    canvas.height
  );

  const gl = canvas.getContext("webgl");
  const ext = gl.getExtension("ANGLE_instanced_arrays");

  const vertexShader = gl.createShader(gl.VERTEX_SHADER);
  gl.shaderSource(vertexShader, vertexShaderSource);
  gl.compileShader(vertexShader);
  if (!gl.getShaderParameter(vertexShader, gl.COMPILE_STATUS))
    console.error(gl.getShaderInfoLog(vertexShader));

  const fragmentShader = gl.createShader(gl.FRAGMENT_SHADER);
  gl.shaderSource(fragmentShader, fragmentShaderSource);
  gl.compileShader(fragmentShader);
  if (!gl.getShaderParameter(fragmentShader, gl.COMPILE_STATUS))
    console.error(gl.getShaderInfoLog(fragmentShader));

  const program = gl.createProgram();
  gl.attachShader(program, vertexShader);
  gl.attachShader(program, fragmentShader);
  gl.linkProgram(program);
  gl.useProgram(program);

  if (!gl.getProgramParameter(program, gl.LINK_STATUS))
    console.error(gl.getProgramInfoLog(program));

  gl.disable(gl.DEPTH_TEST); // Disable depth testing

  // Enable blending
  gl.enable(gl.BLEND);

  // Set blending function
  gl.blendFuncSeparate(
    gl.ONE_MINUS_DST_ALPHA,
    gl.ONE,
    gl.ONE_MINUS_DST_ALPHA,
    gl.ONE
  );

  // Set blending equation
  gl.blendEquationSeparate(gl.FUNC_ADD, gl.FUNC_ADD);

  // projection
  const u_projection = gl.getUniformLocation(program, "projection");
  gl.uniformMatrix4fv(u_projection, false, projectionMatrix);

  // viewport
  const u_viewport = gl.getUniformLocation(program, "viewport");
  gl.uniform2fv(u_viewport, new Float32Array([canvas.width, canvas.height]));

  // focal
  const u_focal = gl.getUniformLocation(program, "focal");
  gl.uniform2fv(
    u_focal,
    new Float32Array([camera.fx / downsample, camera.fy / downsample])
  );

  // view
  const u_view = gl.getUniformLocation(program, "view");
  gl.uniformMatrix4fv(u_view, false, viewMatrix);

  // positions
  const triangleVertices = new Float32Array([1, -1, 1, 1, -1, 1, -1, -1]);
  const vertexBuffer = gl.createBuffer();
  gl.bindBuffer(gl.ARRAY_BUFFER, vertexBuffer);
  gl.bufferData(gl.ARRAY_BUFFER, triangleVertices, gl.STATIC_DRAW);
  const a_position = gl.getAttribLocation(program, "position");
  gl.enableVertexAttribArray(a_position);
  gl.bindBuffer(gl.ARRAY_BUFFER, vertexBuffer);
  gl.vertexAttribPointer(a_position, 2, gl.FLOAT, false, 0, 0);

  // center
  const centerBuffer = gl.createBuffer();
  // gl.bindBuffer(gl.ARRAY_BUFFER, centerBuffer);
  // gl.bufferData(gl.ARRAY_BUFFER, center, gl.STATIC_DRAW);
  const a_center = gl.getAttribLocation(program, "center");
  gl.enableVertexAttribArray(a_center);
  gl.bindBuffer(gl.ARRAY_BUFFER, centerBuffer);
  gl.vertexAttribPointer(a_center, 3, gl.FLOAT, false, 0, 0);
  ext.vertexAttribDivisorANGLE(a_center, 1); // Use the extension here

  // color
  const colorBuffer = gl.createBuffer();
  // gl.bindBuffer(gl.ARRAY_BUFFER, colorBuffer);
  // gl.bufferData(gl.ARRAY_BUFFER, color, gl.STATIC_DRAW);
  const a_color = gl.getAttribLocation(program, "color");
  gl.enableVertexAttribArray(a_color);
  gl.bindBuffer(gl.ARRAY_BUFFER, colorBuffer);
  gl.vertexAttribPointer(a_color, 4, gl.FLOAT, false, 0, 0);
  ext.vertexAttribDivisorANGLE(a_color, 1); // Use the extension here

  // quat
  const quatBuffer = gl.createBuffer();
  // gl.bindBuffer(gl.ARRAY_BUFFER, quatBuffer);
  // gl.bufferData(gl.ARRAY_BUFFER, quat, gl.STATIC_DRAW);
  const a_quat = gl.getAttribLocation(program, "quat");
  gl.enableVertexAttribArray(a_quat);
  gl.bindBuffer(gl.ARRAY_BUFFER, quatBuffer);
  gl.vertexAttribPointer(a_quat, 4, gl.FLOAT, false, 0, 0);
  ext.vertexAttribDivisorANGLE(a_quat, 1); // Use the extension here

  // scale
  const scaleBuffer = gl.createBuffer();
  // gl.bindBuffer(gl.ARRAY_BUFFER, scaleBuffer);
  // gl.bufferData(gl.ARRAY_BUFFER, scale, gl.STATIC_DRAW);
  const a_scale = gl.getAttribLocation(program, "scale");
  gl.enableVertexAttribArray(a_scale);
  gl.bindBuffer(gl.ARRAY_BUFFER, scaleBuffer);
  gl.vertexAttribPointer(a_scale, 3, gl.FLOAT, false, 0, 0);
  ext.vertexAttribDivisorANGLE(a_scale, 1); // Use the extension here

  let lastProj = [];
  let lastData;

  worker.onmessage = (e) => {
    if (e.data.buffer) {
      splatData = new Uint8Array(e.data.buffer);
      const blob = new Blob([splatData.buffer], {
        type: "application/octet-stream",
      });
      const link = document.createElement("a");
      link.download = "model.splat";
      link.href = URL.createObjectURL(blob);
      document.body.appendChild(link);
      link.click();
    } else {
      let { quat, scale, center, color, viewProj } = e.data;
      lastData = e.data;

      lastProj = viewProj;
      vertexCount = quat.length / 4;

      gl.bindBuffer(gl.ARRAY_BUFFER, centerBuffer);
      gl.bufferData(gl.ARRAY_BUFFER, center, gl.STATIC_DRAW);

      gl.bindBuffer(gl.ARRAY_BUFFER, colorBuffer);
      gl.bufferData(gl.ARRAY_BUFFER, color, gl.STATIC_DRAW);

      gl.bindBuffer(gl.ARRAY_BUFFER, quatBuffer);
      gl.bufferData(gl.ARRAY_BUFFER, quat, gl.STATIC_DRAW);

      gl.bindBuffer(gl.ARRAY_BUFFER, scaleBuffer);
      gl.bufferData(gl.ARRAY_BUFFER, scale, gl.STATIC_DRAW);
    }
  };

  let activeKeys = [];

  window.addEventListener("keydown", (e) => {
    if (document.activeElement != document.body) return;
    carousel = false;
    if (!activeKeys.includes(e.key)) activeKeys.push(e.key);
    if (/\d/.test(e.key)) {
      viewMatrix = getViewMatrix(cameras[parseInt(e.key)]);
    }
    if (e.key == "v") {
      location.hash =
        "#" + JSON.stringify(viewMatrix.map((k) => Math.round(k * 100) / 100));
    } else if (e.key === "p") {
      carousel = true;
    }
  });
  window.addEventListener("keyup", (e) => {
    activeKeys = activeKeys.filter((k) => k !== e.key);
  });
  window.addEventListener("blur", () => {
    activeKeys = [];
  });

  window.addEventListener(
    "wheel",
    (e) => {
      carousel = false;
      e.preventDefault();
      const lineHeight = 10;
      const scale =
        e.deltaMode == 1 ? lineHeight : e.deltaMode == 2 ? innerHeight : 1;
      let inv = invert4(viewMatrix);
      if (e.shiftKey) {
        inv = translate4(
          inv,
          (e.deltaX * scale) / innerWidth,
          (e.deltaY * scale) / innerHeight,
          0
        );
      } else if (e.ctrlKey || e.metaKey) {
        // inv = rotate4(inv,  (e.deltaX * scale) / innerWidth,  0, 0, 1);
        // inv = translate4(inv,  0, (e.deltaY * scale) / innerHeight, 0);
        let preY = inv[13];
        inv = translate4(inv, 0, 0, (-10 * (e.deltaY * scale)) / innerHeight);
        inv[13] = preY;
      } else {
        let d = 4;
        inv = translate4(inv, 0, 0, d);
        inv = rotate4(inv, -(e.deltaX * scale) / innerWidth, 0, 1, 0);
        inv = rotate4(inv, (e.deltaY * scale) / innerHeight, 1, 0, 0);
        inv = translate4(inv, 0, 0, -d);
      }

      viewMatrix = invert4(inv);
    },
    { passive: false }
  );

  let startX, startY, down;
  canvas.addEventListener("mousedown", (e) => {
    carousel = false;
    e.preventDefault();
    startX = e.clientX;
    startY = e.clientY;
    down = e.ctrlKey || e.metaKey ? 2 : 1;
  });
  canvas.addEventListener("contextmenu", (e) => {
    carousel = false;
    e.preventDefault();
    startX = e.clientX;
    startY = e.clientY;
    down = 2;
  });

  canvas.addEventListener("mousemove", (e) => {
    e.preventDefault();
    if (down == 1) {
      let inv = invert4(viewMatrix);
      let dx = (5 * (e.clientX - startX)) / innerWidth;
      let dy = (5 * (e.clientY - startY)) / innerHeight;
      let d = 4;

      inv = translate4(inv, 0, 0, d);
      inv = rotate4(inv, dx, 0, 1, 0);
      inv = rotate4(inv, -dy, 1, 0, 0);
      inv = translate4(inv, 0, 0, -d);
      // let postAngle = Math.atan2(inv[0], inv[10])
      // inv = rotate4(inv, postAngle - preAngle, 0, 0, 1)
      // console.log(postAngle)
      viewMatrix = invert4(inv);

      startX = e.clientX;
      startY = e.clientY;
    } else if (down == 2) {
      let inv = invert4(viewMatrix);
      // inv = rotateY(inv, );
      let preY = inv[13];
      inv = translate4(
        inv,
        (-10 * (e.clientX - startX)) / innerWidth,
        0,
        (10 * (e.clientY - startY)) / innerHeight
      );
      inv[13] = preY;
      viewMatrix = invert4(inv);

      startX = e.clientX;
      startY = e.clientY;
    }
  });
  canvas.addEventListener("mouseup", (e) => {
    e.preventDefault();
    down = false;
    startX = 0;
    startY = 0;
  });

  let altX = 0,
    altY = 0;
  canvas.addEventListener(
    "touchstart",
    (e) => {
      e.preventDefault();
      if (e.touches.length === 1) {
        carousel = false;
        startX = e.touches[0].clientX;
        startY = e.touches[0].clientY;
        down = 1;
      } else if (e.touches.length === 2) {
        // console.log('beep')
        carousel = false;
        startX = e.touches[0].clientX;
        altX = e.touches[1].clientX;
        startY = e.touches[0].clientY;
        altY = e.touches[1].clientY;
        down = 1;
      }
    },
    { passive: false }
  );
  canvas.addEventListener(
    "touchmove",
    (e) => {
      e.preventDefault();
      if (e.touches.length === 1 && down) {
        let inv = invert4(viewMatrix);
        let dx = (4 * (e.touches[0].clientX - startX)) / innerWidth;
        let dy = (4 * (e.touches[0].clientY - startY)) / innerHeight;

        let d = 4;
        inv = translate4(inv, 0, 0, d);
        // inv = translate4(inv,  -x, -y, -z);
        // inv = translate4(inv,  x, y, z);
        inv = rotate4(inv, dx, 0, 1, 0);
        inv = rotate4(inv, -dy, 1, 0, 0);
        inv = translate4(inv, 0, 0, -d);

        viewMatrix = invert4(inv);

        startX = e.touches[0].clientX;
        startY = e.touches[0].clientY;
      } else if (e.touches.length === 2) {
        // alert('beep')
        const dtheta =
          Math.atan2(startY - altY, startX - altX) -
          Math.atan2(
            e.touches[0].clientY - e.touches[1].clientY,
            e.touches[0].clientX - e.touches[1].clientX
          );
        const dscale =
          Math.hypot(startX - altX, startY - altY) /
          Math.hypot(
            e.touches[0].clientX - e.touches[1].clientX,
            e.touches[0].clientY - e.touches[1].clientY
          );
        const dx =
          (e.touches[0].clientX + e.touches[1].clientX - (startX + altX)) / 2;
        const dy =
          (e.touches[0].clientY + e.touches[1].clientY - (startY + altY)) / 2;
        let inv = invert4(viewMatrix);
        // inv = translate4(inv,  0, 0, d);
        inv = rotate4(inv, dtheta, 0, 0, 1);

        inv = translate4(inv, -dx / innerWidth, -dy / innerHeight, 0);

        let preY = inv[13];
        inv = translate4(inv, 0, 0, 3 * (1 - dscale));
        inv[13] = preY;

        viewMatrix = invert4(inv);

        startX = e.touches[0].clientX;
        altX = e.touches[1].clientX;
        startY = e.touches[0].clientY;
        altY = e.touches[1].clientY;
      }
    },
    { passive: false }
  );
  canvas.addEventListener(
    "touchend",
    (e) => {
      e.preventDefault();
      down = false;
      startX = 0;
      startY = 0;
    },
    { passive: false }
  );

  let jumpDelta = 0;
  let vertexCount = 0;

  let lastFrame = 0;
  let avgFps = 0;
  let start = 0;

  const frame = (now) => {
    let inv = invert4(viewMatrix);

    if (activeKeys.includes("ArrowUp")) {
      if (activeKeys.includes("Shift")) {
        inv = translate4(inv, 0, -0.03, 0);
      } else {
        let preY = inv[13];
        inv = translate4(inv, 0, 0, 0.1);
        inv[13] = preY;
      }
    }
    if (activeKeys.includes("ArrowDown")) {
      if (activeKeys.includes("Shift")) {
        inv = translate4(inv, 0, 0.03, 0);
      } else {
        let preY = inv[13];
        inv = translate4(inv, 0, 0, -0.1);
        inv[13] = preY;
      }
    }
    if (activeKeys.includes("ArrowLeft")) inv = translate4(inv, -0.03, 0, 0);
    //
    if (activeKeys.includes("ArrowRight")) inv = translate4(inv, 0.03, 0, 0);
    // inv = rotate4(inv, 0.01, 0, 1, 0);
    if (activeKeys.includes("a")) inv = rotate4(inv, -0.01, 0, 1, 0);
    if (activeKeys.includes("d")) inv = rotate4(inv, 0.01, 0, 1, 0);
    if (activeKeys.includes("q")) inv = rotate4(inv, 0.01, 0, 0, 1);
    if (activeKeys.includes("e")) inv = rotate4(inv, -0.01, 0, 0, 1);
    if (activeKeys.includes("w")) inv = rotate4(inv, 0.005, 1, 0, 0);
    if (activeKeys.includes("s")) inv = rotate4(inv, -0.005, 1, 0, 0);

    if (["j", "k", "l", "i"].some((k) => activeKeys.includes(k))) {
      let d = 4;
      inv = translate4(inv, 0, 0, d);
      inv = rotate4(
        inv,
        activeKeys.includes("j") ? -0.05 : activeKeys.includes("l") ? 0.05 : 0,
        0,
        1,
        0
      );
      inv = rotate4(
        inv,
        activeKeys.includes("i") ? 0.05 : activeKeys.includes("k") ? -0.05 : 0,
        1,
        0,
        0
      );
      inv = translate4(inv, 0, 0, -d);
    }

    // inv[13] = preY;
    viewMatrix = invert4(inv);

    if (carousel) {
      let inv = invert4(defaultViewMatrix);

      const t = Math.sin((Date.now() - start) / 5000);
      inv = translate4(inv, 2.5 * t, 0, 6 * (1 - Math.cos(t)));
      inv = rotate4(inv, -0.6 * t, 0, 1, 0);

      viewMatrix = invert4(inv);
    }

    if (activeKeys.includes(" ")) {
      jumpDelta = Math.min(1, jumpDelta + 0.05);
    } else {
      jumpDelta = Math.max(0, jumpDelta - 0.05);
    }

    let inv2 = invert4(viewMatrix);
    inv2[13] -= jumpDelta;
    inv2 = rotate4(inv2, -0.1 * jumpDelta, 1, 0, 0);
    let actualViewMatrix = invert4(inv2);

    const viewProj = multiply4(projectionMatrix, actualViewMatrix);
    worker.postMessage({ view: viewProj });

    const currentFps = 1000 / (now - lastFrame) || 0;
    avgFps = avgFps * 0.9 + currentFps * 0.1;

    if (vertexCount > 0) {
      document.getElementById("spinner").style.display = "none";
      gl.uniformMatrix4fv(u_view, false, actualViewMatrix);
      ext.drawArraysInstancedANGLE(gl.TRIANGLE_STRIP, 0, 4, vertexCount);
    } else {
      gl.clear(gl.COLOR_BUFFER_BIT);
      document.getElementById("spinner").style.display = "";
      start = Date.now() + 2000;
    }
    const progress = (100 * vertexCount) / (splatData.length / rowLength);
    if (progress < 100) {
      document.getElementById("progress").style.width = progress + "%";
    } else {
      document.getElementById("progress").style.display = "none";
    }
    fps.innerText = Math.round(avgFps) + " fps";
    lastFrame = now;
    requestAnimationFrame(frame);
  };

  frame();

  const selectFile = (file) => {
    const fr = new FileReader();
    if (/\.json$/i.test(file.name)) {
      fr.onload = () => {
        cameras = JSON.parse(fr.result);
        viewMatrix = getViewMatrix(cameras[0]);
        projectionMatrix = getProjectionMatrix(
          camera.fx / downsample,
          camera.fy / downsample,
          canvas.width,
          canvas.height
        );
        gl.uniformMatrix4fv(u_projection, false, projectionMatrix);

        console.log("Loaded Cameras");
      };
      fr.readAsText(file);
    } else {
      stopLoading = true;
      fr.onload = () => {
        splatData = new Uint8Array(fr.result);
        console.log("Loaded", Math.floor(splatData.length / rowLength));

        if (
          splatData[0] == 112 &&
          splatData[1] == 108 &&
          splatData[2] == 121 &&
          splatData[3] == 10
        ) {
          // ply file magic header means it should be handled differently
          worker.postMessage({ ply: splatData.buffer });
        } else {
          worker.postMessage({
            buffer: splatData.buffer,
            vertexCount: Math.floor(splatData.length / rowLength),
          });
        }
      };
      fr.readAsArrayBuffer(file);
    }
  };

  window.addEventListener("hashchange", (e) => {
    try {
      viewMatrix = JSON.parse(decodeURIComponent(location.hash.slice(1)));
      carousel = false;
    } catch (err) {}
  });

  const preventDefault = (e) => {
    e.preventDefault();
    e.stopPropagation();
  };
  document.addEventListener("dragenter", preventDefault);
  document.addEventListener("dragover", preventDefault);
  document.addEventListener("dragleave", preventDefault);
  document.addEventListener("drop", (e) => {
    e.preventDefault();
    e.stopPropagation();
    selectFile(e.dataTransfer.files[0]);
  });

  let bytesRead = 0;
  let lastVertexCount = -1;
  let stopLoading = false;

  while (true) {
    const { done, value } = await reader.read();
    if (done || stopLoading) break;

    splatData.set(value, bytesRead);
    bytesRead += value.length;

    if (vertexCount > lastVertexCount) {
      worker.postMessage({
        buffer: splatData.buffer,
        vertexCount: Math.floor(bytesRead / rowLength),
      });
      lastVertexCount = vertexCount;
    }
  }
  if (!stopLoading)
    worker.postMessage({
      buffer: splatData.buffer,
      vertexCount: Math.floor(bytesRead / rowLength),
    });
}

main().catch((err) => {
  document.getElementById("spinner").style.display = "none";
  document.getElementById("message").innerText = err.toString();
});

if (location.host.includes("hf.space")) document.body.classList.add("nohf");
