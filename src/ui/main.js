import "./styles.css";
import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { RapierPhysics } from "three/addons/physics/RapierPhysics.js";
import {
  EmbodiedMemory,
  LEARNING_CONFIG,
  chooseAction,
  estimateDepthFromDisparity,
  readTactileSensors,
  shouldDodgeReflex,
} from "./learning.js";
import { configureEyeCamera, projectWorldBoundsToFrame } from "./vision.js";

const canvas = document.querySelector("#scene");
const selectedLabel = document.querySelector("#selected-label");
const riskMeter = document.querySelector("#risk-meter");
const confidenceMeter = document.querySelector("#confidence-meter");
const pressureMeter = document.querySelector("#pressure-meter");
const painMeter = document.querySelector("#pain-meter");
const sensorState = document.querySelector("#sensor-state");
const statusAction = document.querySelector("#status-action");
const statusReason = document.querySelector("#status-reason");
const memoryList = document.querySelector("#memory-list");
const memoryState = document.querySelector("#memory-state");
const resetMemoryButton = document.querySelector("#reset-memory");
const exportMemoryButton = document.querySelector("#export-memory");
const replayContactButton = document.querySelector("#replay-contact");
const sensorToggle = document.querySelector("#sensor-toggle");
const cameraToggle = document.querySelector("#camera-toggle");
const motorToggle = document.querySelector("#motor-toggle");
const visionRig = document.querySelector("#vision-rig");

const eyeElements = {
  left: {
    view: document.querySelector("#left-eye-view"),
    box: document.querySelector("#left-eye-box"),
    meta: document.querySelector("#left-eye-meta"),
    depth: document.querySelector("#left-eye-depth"),
    side: -1,
  },
  right: {
    view: document.querySelector("#right-eye-view"),
    box: document.querySelector("#right-eye-box"),
    meta: document.querySelector("#right-eye-meta"),
    depth: document.querySelector("#right-eye-depth"),
    side: 1,
  },
};

const renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: false });
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
renderer.shadowMap.enabled = true;
renderer.shadowMap.type = THREE.PCFSoftShadowMap;

for (const eye of Object.values(eyeElements)) {
  eye.camera = new THREE.PerspectiveCamera(58, 1, 0.05, 22);
  eye.renderer = createEyeRenderer(eye.view);
}

const scene = new THREE.Scene();
scene.background = new THREE.Color(0x151b21);
scene.fog = new THREE.Fog(0x151b21, 9, 18);

const camera = new THREE.PerspectiveCamera(44, 1, 0.1, 50);
camera.position.set(0.25, 4.85, 9.7);

const cameraTarget = new THREE.Vector3(0.25, 0.55, 0);
const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = false;
controls.target.copy(cameraTarget);
controls.update();

const pointer = new THREE.Vector2();
const raycaster = new THREE.Raycaster();
const dragPlane = new THREE.Plane(new THREE.Vector3(0, 1, 0), -0.58);
const planeHit = new THREE.Vector3();
const dragOffset = new THREE.Vector3();
const worldBounds = Object.freeze({
  minX: -5.2,
  maxX: 3.15,
  minY: -0.78,
  maxY: 0.78,
  minZ: -2.35,
  maxZ: 2.35,
  floorY: -0.62,
});
const dragYLimits = Object.freeze({
  min: worldBounds.minY,
  max: worldBounds.maxY,
});
const PHYSICS = Object.freeze({
  gravity: -5.8,
  maxFallSpeed: 2.8,
  linearDamping: 0.92,
  restitution: 0.04,
  iterations: 8,
  heldMassMultiplier: 15,
  wallDamping: 0.16,
  penetrationBias: 1.0005,
  contactTolerance: 0.0012,
  obbEpsilon: 1e-6,
  collisionSteps: 72,
  collisionSubstepDistance: 0.025,
  heldSubstepDistance: 0.02,
  collisionEps: 1e-3,
  characterStep: 1 / 60,
});
const LIMB = Object.freeze({
  mass: 16,
  damping: 0.84,
  restitution: 0.02,
  impulseScale: 0.4,
  maxImpulse: 0.55,
  maxOffset: 0.32,
  springReturn: 0.14,
});
const LIMB_CONTROLLER = Object.freeze({
  halfHeight: 0.5,
  radius: 0.3,
  mass: 3,
  stepHeight: 0.01,
  moveSpeed: 2.5,
});
const limbPhysics = {
  ready: false,
  engine: null,
  controller: null,
  collider: null,
  colliderOffset: new THREE.Vector3(),
};
const limbPhysicsScratch = {
  movement: new THREE.Vector3(),
  nextColliderPosition: { x: 0, y: 0, z: 0 },
  colliderPosition: { x: 0, y: 0, z: 0 },
};
const physicsScratch = {
  collisionAxis: new THREE.Vector3(),
  sensorPosition: new THREE.Vector3(),
  sensorScale: new THREE.Vector3(),
  meshWorldScale: new THREE.Vector3(),
  sensorClosestPoint: new THREE.Vector3(),
  localPoint: new THREE.Vector3(),
  localOffset: new THREE.Vector3(),
  localClampedPoint: new THREE.Vector3(),
  localNormal: new THREE.Vector3(),
  normalMatrix: new THREE.Matrix3(),
  obbRotation: new Float32Array(9),
  obbAbsRotation: new Float32Array(9),
  obbCenterDelta: new THREE.Vector3(),
};
const pointyAvoidanceThreshold = 0.68;
const pointySidestepX = 1.16;
const pointySidestepZ = 1.03;
const pointySidestepXBoost = 1.28;
const pointySidestepZBoost = 1.14;
const dragVerticalScale = 0.0055;
const pointySlowApproachX = 0.42;

const learnedMemory = new EmbodiedMemory(LEARNING_CONFIG, { storageKey: "limb.visual.embodiedMemory.v2" });
const restingBasePosition = new THREE.Vector3(3.85, -0.12, 0);
const depthStereoBaseline = 0.18;
const eyeCameraSeparation = 0.72;
const stereoFocal = 1.8;
const eyeHandPosition = new THREE.Vector3();
const visibleObjectBounds = new THREE.Box3();
const meshWorldBounds = new THREE.Box3();
const emptySensors = {
  contact: 0,
  pressure: 0,
  puncture: 0,
  pain: 0,
  safeTouch: false,
  painful: false,
  approachingSensor: false,
  closingSpeed: 0,
  distance: Infinity,
  sensorName: "sensor",
  contactName: "object",
};

const objectSpecs = [
  {
    key: "fork",
    label: "fork",
    x: -4.6,
    z: -1.6,
    kind: "fork",
    cueVector: [0.92, 0.28, 0.18, 0.88, 0.22, 0.76],
    tactile: { tipSharpness: 0.88, compliance: 0.12, contactArea: 0.18 },
    contactPoints: [
      { x: 0.82, y: -0.01, z: -0.13, name: "tine", sharpnessScale: 1.05, areaScale: 0.8 },
      { x: 0.82, y: -0.01, z: -0.04, name: "tine", sharpnessScale: 1.05, areaScale: 0.8 },
      { x: 0.82, y: -0.01, z: 0.05, name: "tine", sharpnessScale: 1.05, areaScale: 0.8 },
      { x: 0.82, y: -0.01, z: 0.14, name: "tine", sharpnessScale: 1.05, areaScale: 0.8 },
    ],
  },
  {
    key: "knife",
    label: "knife",
    x: -3.35,
    z: -1.35,
    kind: "knife",
    cueVector: [0.98, 0.18, 0.1, 0.96, 0.08, 0.82],
    tactile: { tipSharpness: 0.96, compliance: 0.08, contactArea: 0.12 },
    contactPoints: [{ x: 0.86, y: -0.01, z: 0, name: "blade tip", sharpnessScale: 1.08, areaScale: 0.72 }],
  },
  {
    key: "spoon",
    label: "spoon",
    x: -4.55,
    z: 0.05,
    kind: "spoon",
    cueVector: [0.74, 0.5, 0.62, 0.12, 0.72, 0.65],
    tactile: { tipSharpness: 0.08, compliance: 0.34, contactArea: 0.76 },
    contactPoints: [{ x: 0.54, y: 0.03, z: 0, name: "bowl", sharpnessScale: 0.5, areaScale: 1.18 }],
  },
  {
    key: "cup",
    label: "cup",
    x: -3.2,
    z: 0.1,
    kind: "cup",
    cueVector: [0.38, 0.82, 0.75, 0.04, 0.84, 0.24],
    tactile: { tipSharpness: 0.03, compliance: 0.42, contactArea: 0.86 },
    contactPoints: [{ x: 0.25, y: 0.22, z: 0, name: "rim", sharpnessScale: 0.45, areaScale: 1.1 }],
  },
  {
    key: "ball",
    label: "ball",
    x: -4.35,
    z: 1.55,
    kind: "ball",
    cueVector: [0.2, 1.0, 0.96, 0.02, 0.92, 0.08],
    tactile: { tipSharpness: 0.01, compliance: 0.72, contactArea: 0.94 },
    contactPoints: [{ x: 0, y: 0.21, z: 0, name: "surface", sharpnessScale: 0.3, areaScale: 1.2 }],
  },
  {
    key: "block",
    label: "block",
    x: -3.1,
    z: 1.45,
    kind: "block",
    cueVector: [0.35, 0.7, 0.36, 0.1, 0.38, 0.18],
    tactile: { tipSharpness: 0.07, compliance: 0.28, contactArea: 0.88 },
    contactPoints: [
      { x: 0.26, y: 0.21, z: -0.26, name: "corner", sharpnessScale: 0.7, areaScale: 0.9 },
      { x: 0.26, y: 0.21, z: 0.26, name: "corner", sharpnessScale: 0.7, areaScale: 0.9 },
    ],
  },
  {
    key: "metal",
    label: "unknown tool",
    x: -1.9,
    z: 1.55,
    kind: "metal",
    cueVector: [0.85, 0.22, 0.18, 0.78, 0.2, 0.8],
    tactile: { tipSharpness: 0.78, compliance: 0.14, contactArea: 0.2 },
    contactPoints: [{ x: 0.72, y: -0.01, z: 0, name: "unknown tip", sharpnessScale: 1.0, areaScale: 0.82 }],
  },
];

const draggableObjects = [];
const depthHistory = new Map();
const contactLearningState = new Map();
const uiTextCache = new Map();
let sensorsEnabled = true;
let cameraEnabled = true;
let motorEnabled = true;
let selected = null;
let dragging = false;
let activePointerId = null;
let dragPointerY = 0;
let lastFrameTime = 0;
let reflexPulse = 0;
let memoryRenderSignature = "";
let currentDecision = {
  action: "HOLD",
  reason: "Select an object and move it toward the limb.",
  confidence: 0,
  risk: 0,
  className: "unknown",
};
let currentSensors = { ...emptySensors };

function setSensorsEnabled(enabled) {
  sensorsEnabled = Boolean(enabled);
  if (sensorToggle) {
    sensorToggle.checked = sensorsEnabled;
  }
  if (!sensorsEnabled) {
    currentSensors = { ...emptySensors };
    depthHistory.clear();
    contactLearningState.clear();
    limb.reactionOffset.set(0, 0, 0);
    limb.reactionVelocity.set(0, 0, 0);
  }
}

function setMotorEnabled(enabled) {
  motorEnabled = Boolean(enabled);
  if (motorToggle) {
    motorToggle.checked = motorEnabled;
  }
  if (!motorEnabled) {
    currentDecision = {
      action: "HOLD",
      reason: "Motor disabled",
      confidence: 0,
      risk: 0,
      className: "unknown",
    };
    limb.reactionOffset.set(0, 0, 0);
    limb.reactionVelocity.set(0, 0, 0);
  }
}

function setCameraEnabled(enabled) {
  cameraEnabled = Boolean(enabled);
  if (cameraToggle) {
    cameraToggle.checked = cameraEnabled;
  }
  if (visionRig) {
    visionRig.classList.toggle("is-off", !cameraEnabled);
  }
  if (!cameraEnabled) {
    setEyeCameraOffline(eyeElements.left);
    setEyeCameraOffline(eyeElements.right);
  }
}

setupScene();
purgeUnexpectedCenterDividerArtifacts();
const limb = createLimb();
scene.add(limb.group);

for (const spec of objectSpecs) {
  const object = createObject(spec);
  scene.add(object.group);
  draggableObjects.push(object);
}
void initPhysicsEngine().finally(() => {
  purgeUnexpectedCenterDividerArtifacts();
});

window.limbSimDebug = {
  objectScreenPosition,
  worldScreenPosition,
  decision: () => ({ ...currentDecision }),
  sensors: () => ({ ...currentSensors }),
  memoryEntries: () => learnedMemory.entries.map((entry) => ({ ...entry, cueVector: [...entry.cueVector] })),
  limbPosition: () => ({
    x: limb.group.position.x,
    y: limb.group.position.y,
    z: limb.group.position.z,
  }),
  handPosition: () => handWorldPosition(),
  selectedObject: () => {
    if (!selected) return null;
    const dragTarget = selected.dragTarget;
    const velocity = selected.velocity;
    return {
      key: selected.spec.key,
      label: selected.spec.label,
      world: {
        x: selected.group.position.x,
        y: selected.group.position.y,
        z: selected.group.position.z,
      },
      dragTarget: {
        x: dragTarget.x,
        y: dragTarget.y,
        z: dragTarget.z,
      },
      isHeld: selected.isHeld,
      isRigid: Boolean(selected.group.userData.physics?.body),
      velocity: {
        x: velocity.x,
        y: velocity.y,
        z: velocity.z,
      },
    };
  },
  limbPhysicsState: () => {
    if (!limbPhysics.ready || !limbPhysics.collider) {
      return {
        ready: false,
        controller: false,
      };
    }

    const colliderPosition = limbPhysics.collider.translation();
    return {
      ready: true,
      controller: Boolean(limbPhysics.controller),
      colliderPosition: {
        x: colliderPosition.x,
        y: colliderPosition.y,
        z: colliderPosition.z,
      },
      colliderOffset: {
        x: limbPhysics.colliderOffset.x,
        y: limbPhysics.colliderOffset.y,
        z: limbPhysics.colliderOffset.z,
      },
      handOffsetDelta: {
        x: colliderPosition.x - limb.group.position.x,
        y: colliderPosition.y - limb.group.position.y,
        z: colliderPosition.z - limb.group.position.z,
      },
    };
  },
  sceneRenderableSummary: () => {
    const objects = [];
    const bounds = new THREE.Box3();
    const boundsSize = new THREE.Vector3();
    const worldCenter = new THREE.Vector3();

    scene.traverse((object) => {
      if (!object.isMesh || !object.geometry) return;
      bounds.setFromObject(object);
      if (bounds.isEmpty()) return;
      bounds.getSize(boundsSize);
      object.getWorldPosition(worldCenter);
      objects.push({
        name: object.name || "",
        userSource: object.userData?.source || "",
        geometryType: object.geometry?.type || "",
        materialType: object.material?.type || "",
        materialColor: object.material?.color ? object.material.color.getHexString() : "",
        position: {
          x: Number(worldCenter.x.toFixed(3)),
          y: Number(worldCenter.y.toFixed(3)),
          z: Number(worldCenter.z.toFixed(3)),
        },
        size: {
          x: Number(boundsSize.x.toFixed(3)),
          y: Number(boundsSize.y.toFixed(3)),
          z: Number(boundsSize.z.toFixed(3)),
        },
      });
    });

    return objects;
  },
};

renderMemory();
resize();
window.addEventListener("resize", resize);
canvas.addEventListener("pointerdown", onPointerDown);
canvas.addEventListener("pointermove", onPointerMove);
canvas.addEventListener("pointerup", onPointerUp);
canvas.addEventListener("pointercancel", onPointerUp);
canvas.addEventListener("wheel", onPointerWheel, { passive: false });
resetMemoryButton?.addEventListener("click", () => resetLearnedMemory());
exportMemoryButton?.addEventListener("click", () => exportLearnedMemory());
replayContactButton?.addEventListener("click", () => replayLastContact());
sensorToggle?.addEventListener("change", (event) => {
  setSensorsEnabled(Boolean(event.currentTarget.checked));
});
cameraToggle?.addEventListener("change", (event) => {
  setCameraEnabled(Boolean(event.currentTarget.checked));
});
motorToggle?.addEventListener("change", (event) => {
  setMotorEnabled(Boolean(event.currentTarget.checked));
});
setSensorsEnabled(sensorToggle?.checked ?? true);
setCameraEnabled(cameraToggle?.checked ?? true);
setMotorEnabled(motorToggle?.checked ?? true);

renderer.setAnimationLoop(animate);

function createEyeRenderer(view) {
  const eyeRenderer = new THREE.WebGLRenderer({ canvas: view, antialias: true, alpha: false });
  eyeRenderer.setPixelRatio(Math.min(window.devicePixelRatio, 1.5));
  eyeRenderer.setClearColor(0x071014, 1);
  eyeRenderer.shadowMap.enabled = false;
  return eyeRenderer;
}

function setupScene() {
  const hemi = new THREE.HemisphereLight(0xd8f0ff, 0x28313a, 2.4);
  scene.add(hemi);

  const key = new THREE.DirectionalLight(0xffffff, 3.6);
  key.position.set(-3.5, 7, 4);
  key.castShadow = true;
  key.shadow.mapSize.set(2048, 2048);
  key.shadow.camera.left = -8;
  key.shadow.camera.right = 8;
  key.shadow.camera.top = 7;
  key.shadow.camera.bottom = -7;
  scene.add(key);

  const fill = new THREE.PointLight(0x80caff, 1.8, 10);
  fill.position.set(3, 3, 2);
  scene.add(fill);

  const floor = new THREE.Mesh(
    new THREE.PlaneGeometry(12, 7),
    new THREE.MeshStandardMaterial({ color: 0x202832, roughness: 0.74, metalness: 0.05 }),
  );
  floor.rotation.x = -Math.PI / 2;
  floor.position.set(0, worldBounds.floorY - 0.06, 0);
  floor.receiveShadow = true;
	floor.userData.physics = { mass: 0, restitution: 0 };
  scene.add(floor);

  const shelf = new THREE.Mesh(
    new THREE.BoxGeometry(3.9, 0.08, 4.8),
	  new THREE.MeshStandardMaterial({ color: 0x303b44, roughness: 0.68 }),
	);
	shelf.position.set(-3.45, -0.54, 0);
	shelf.receiveShadow = true;
	shelf.userData.physics = { mass: 0, restitution: 0.02 };
	scene.add(shelf);

  const limbPad = new THREE.Mesh(
    new THREE.BoxGeometry(2.7, 0.08, 4.8),
	  new THREE.MeshStandardMaterial({ color: 0x273139, roughness: 0.7 }),
	);
  limbPad.position.set(3.25, -0.53, 0);
  limbPad.receiveShadow = true;
  limbPad.userData.physics = { mass: 0, restitution: 0.02 };
  scene.add(limbPad);
}

function purgeUnexpectedCenterDividerArtifacts() {
  const queue = [scene];
  const localBounds = new THREE.Box3();
  const localSize = new THREE.Vector3();
  const worldPosition = new THREE.Vector3();
  let removed = 0;

  while (queue.length > 0) {
    const object = queue.pop();
    for (const child of object.children) {
      queue.push(child);
    }

    if (object.userData?.source === "sim-divider") continue;
    const isRenderable = object.isMesh || object.isLine || object.isPoints;
    if (!isRenderable || !object.geometry) continue;

    object.getWorldPosition(worldPosition);
    const isCentered = Math.abs(worldPosition.x) <= 1.25 && Math.abs(worldPosition.z) <= 1.5 && worldPosition.y >= -1.25 && worldPosition.y <= 1.3;
    if (!isCentered) continue;

    localBounds.setFromObject(object);
    if (localBounds.isEmpty()) continue;
    const size = localBounds.getSize(localSize);

    const spans = [Math.abs(size.x), Math.abs(size.y), Math.abs(size.z)].sort((left, right) => left - right);
    const thinSpan = spans[0];
    const mediumSpan = spans[1];
    const longSpan = spans[2];

    const isLikelyDivider = (
      thinSpan <= 0.2 &&
      mediumSpan <= 0.95 &&
      longSpan >= 4.0 &&
      longSpan <= 12.0 &&
      mediumSpan / longSpan < 0.12
    );

    if (!isLikelyDivider) continue;

    const parent = object.parent;
    if (!parent) continue;
    parent.remove(object);
    removed += 1;
    object.traverse((child) => {
      if (child.isMesh && child.geometry) child.geometry.dispose?.();
      if (child.isMesh && child.material) {
        const materials = Array.isArray(child.material) ? child.material : [child.material];
        for (const material of materials) material?.dispose?.();
      }
    });
  }
  return removed;
}

function createLimb() {
  const group = new THREE.Group();
  group.position.copy(restingBasePosition);
  const colliders = [];

  const base = new THREE.Mesh(
    new THREE.CylinderGeometry(0.38, 0.48, 0.36, 32),
    new THREE.MeshStandardMaterial({ color: 0x606a72, roughness: 0.42, metalness: 0.18 }),
  );
  base.castShadow = true;
  base.receiveShadow = true;
  group.add(base);
  addLimbCollider(colliders, base, "base");

  const shoulder = new THREE.Group();
  shoulder.position.y = 0.38;
  group.add(shoulder);

  const upper = new THREE.Mesh(
    new THREE.CapsuleGeometry(0.16, 1.45, 8, 18),
    new THREE.MeshStandardMaterial({ color: 0x8ca0ad, roughness: 0.38, metalness: 0.24 }),
  );
  upper.rotation.z = Math.PI / 2;
  upper.position.x = -0.78;
  upper.castShadow = true;
  shoulder.add(upper);
  addLimbCollider(colliders, upper, "upper");

  const elbow = new THREE.Group();
  elbow.position.x = -1.55;
  shoulder.add(elbow);

  const joint = new THREE.Mesh(
    new THREE.SphereGeometry(0.23, 28, 16),
    new THREE.MeshStandardMaterial({ color: 0x77a9c4, roughness: 0.34, metalness: 0.2 }),
  );
  joint.castShadow = true;
  elbow.add(joint);
  addLimbCollider(colliders, joint, "joint");

  const forearm = new THREE.Mesh(
    new THREE.CapsuleGeometry(0.14, 1.25, 8, 18),
    new THREE.MeshStandardMaterial({ color: 0xb2c1c9, roughness: 0.36, metalness: 0.14 }),
  );
  forearm.rotation.z = Math.PI / 2;
  forearm.position.x = -0.68;
  forearm.castShadow = true;
  elbow.add(forearm);
  addLimbCollider(colliders, forearm, "forearm");

  const hand = new THREE.Group();
  hand.position.x = -1.36;
  elbow.add(hand);

  const palm = new THREE.Mesh(
    new THREE.BoxGeometry(0.36, 0.2, 0.48),
    new THREE.MeshStandardMaterial({ color: 0xe1d1bd, roughness: 0.48 }),
  );
  palm.castShadow = true;
  hand.add(palm);
  addLimbCollider(colliders, palm, "palm");

  const fingerMat = new THREE.MeshStandardMaterial({ color: 0xe6d7c4, roughness: 0.5 });
  const sensorMeshes = [];
  for (const [index, z] of [-0.17, 0, 0.17].entries()) {
    const finger = new THREE.Mesh(new THREE.CapsuleGeometry(0.045, 0.34, 6, 10), fingerMat);
    finger.rotation.z = Math.PI / 2;
    finger.position.set(-0.27, 0.01, z);
    finger.castShadow = true;
    hand.add(finger);
    addLimbCollider(colliders, finger, `finger-${index + 1}`);

    const sensor = new THREE.Mesh(
      new THREE.SphereGeometry(0.035, 14, 10),
      new THREE.MeshStandardMaterial({
        color: 0x7ee7bf,
        emissive: 0x103b31,
        roughness: 0.38,
      }),
    );
    sensor.position.set(-0.47, 0.01, z);
    sensor.geometry.computeBoundingSphere();
    sensor.userData.touchRadius = sensor.geometry.boundingSphere?.radius ?? 0.035;
    sensor.userData.sensorName = `finger ${index + 1}`;
    sensor.castShadow = true;
    hand.add(sensor);
    sensorMeshes.push(sensor);
  }

  const palmSensor = new THREE.Mesh(
    new THREE.SphereGeometry(0.045, 16, 10),
    new THREE.MeshStandardMaterial({
      color: 0x7ee7bf,
      emissive: 0x103b31,
      roughness: 0.38,
    }),
  );
  palmSensor.position.set(-0.14, 0.12, 0);
  palmSensor.geometry.computeBoundingSphere();
  palmSensor.userData.touchRadius = palmSensor.geometry.boundingSphere?.radius ?? 0.045;
  palmSensor.userData.sensorName = "palm";
  palmSensor.castShadow = true;
  hand.add(palmSensor);
  sensorMeshes.push(palmSensor);

  return {
    group,
    shoulder,
    elbow,
    hand,
    sensors: sensorMeshes,
    colliders,
    reactionOffset: new THREE.Vector3(),
    reactionVelocity: new THREE.Vector3(),
  };
}

function addLimbCollider(colliders, mesh, label) {
  if (!mesh.geometry?.boundingBox) mesh.geometry.computeBoundingBox();
  const localBox = mesh.geometry.boundingBox.clone();
  const localCenter = new THREE.Vector3();
  const localHalfExtents = new THREE.Vector3();
  localBox.getCenter(localCenter);
  localBox.getSize(localHalfExtents).multiplyScalar(0.5);
  colliders.push({
    mesh,
    label,
    localBox,
    localCenter,
    localHalfExtents,
    worldCenter: new THREE.Vector3(),
    worldHalfExtents: new THREE.Vector3(),
    worldBox: new THREE.Box3(),
    worldAxes: [new THREE.Vector3(), new THREE.Vector3(), new THREE.Vector3()],
    active: true,
  });
}

function createObject(spec) {
  const collider = getObjectHalfExtents(spec);
  const group = new THREE.Mesh(
    new THREE.BoxGeometry(collider.x * 2, collider.y * 2, collider.z * 2),
    new THREE.MeshBasicMaterial({ transparent: true, opacity: 0, depthWrite: false }),
  );
  group.position.set(spec.x, -0.39, spec.z);
  group.userData.spec = spec;
  const baseMass = getObjectMass(spec);
  const collisionMeshes = [];
  const colliderWorldBounds = new THREE.Box3();
  const colliderWorldCenter = new THREE.Vector3();
  const colliderWorldHalfExtents = new THREE.Vector3();
  group.raycast = () => null;

  if (spec.kind === "fork") {
    addFork(group);
  } else if (spec.kind === "knife") {
    addKnife(group);
  } else if (spec.kind === "spoon") {
    addSpoon(group);
  } else if (spec.kind === "cup") {
    addCup(group);
  } else if (spec.kind === "ball") {
    addBall(group);
  } else if (spec.kind === "block") {
    addBlock(group);
  } else {
    addUnknownMetal(group);
  }

  const hitTarget = new THREE.Mesh(
    new THREE.BoxGeometry(1.15, 0.7, 0.85),
    new THREE.MeshBasicMaterial({ transparent: true, opacity: 0, depthWrite: false }),
  );
  hitTarget.position.y = 0.16;
  hitTarget.userData.draggableRoot = group;
  group.add(hitTarget);

  group.traverse((child) => {
    if (child.isMesh) {
      if (child === group || child === hitTarget) return;
      if (!child.geometry) return;
      addObjectCollisionMesh(
        collisionMeshes,
        child,
        child.userData.touchName || child.name || `${spec.label}-${collisionMeshes.length + 1}`,
      );
      child.castShadow = true;
      child.receiveShadow = true;
      child.userData.draggableRoot = group;
    }
  });
  hitTarget.castShadow = false;
  hitTarget.receiveShadow = false;
  group.userData.physics = { mass: baseMass, restitution: PHYSICS.restitution };
  group.userData.draggableRoot = group;
  group.frustumCulled = false;
  group.castShadow = false;
  group.receiveShadow = false;

  return {
    group,
    spec,
    home: new THREE.Vector3(spec.x, -0.39, spec.z),
    velocity: new THREE.Vector3(),
    dragTarget: new THREE.Vector3(spec.x, -0.39, spec.z),
    collider,
    baseMass,
    mass: baseMass,
    restitution: PHYSICS.restitution,
    collisionMeshes,
    colliderWorldBounds,
    colliderWorldCenter,
    colliderWorldHalfExtents,
    isHeld: false,
  };
}

async function initPhysicsEngine() {
  try {
    const engine = await RapierPhysics();
    limbPhysics.engine = engine;
    engine.addScene(scene);
    initializeLimbCharacterController();
    limbPhysics.ready = true;
  } catch (error) {
    limbPhysics.ready = false;
    console.warn("Rapier physics unavailable, falling back to legacy collision solver.", error);
  }
}

function initializeLimbCharacterController() {
  if (!limbPhysics.engine || !limb.group) return;

  const { world, RAPIER } = limbPhysics.engine;
  syncLimbColliderOffset();

  const characterController = world.createCharacterController(LIMB_CONTROLLER.stepHeight);
  characterController.setApplyImpulsesToDynamicBodies(true);
  characterController.setCharacterMass(LIMB_CONTROLLER.mass);
  const colliderPosition = new THREE.Vector3();
  colliderPosition.copy(limb.group.position).add(limbPhysics.colliderOffset);
  const colliderDesc = RAPIER.ColliderDesc
    .capsule(LIMB_CONTROLLER.halfHeight, LIMB_CONTROLLER.radius)
    .setTranslation(colliderPosition.x, colliderPosition.y, colliderPosition.z);
  const limbCollider = world.createCollider(colliderDesc);

  limbPhysics.controller = characterController;
  limbPhysics.collider = limbCollider;
}

function syncLimbColliderOffset() {
  const rootWorld = new THREE.Vector3();
  const handWorld = new THREE.Vector3();
  limb.group.getWorldPosition(rootWorld);
  limb.hand.getWorldPosition(handWorld);
  limbPhysics.colliderOffset.subVectors(handWorld, rootWorld);
  limbPhysics.colliderOffset.y -= 0.08;
}

function syncLimbToPhysicsCollider() {
  if (!limbPhysics.ready || !limbPhysics.collider) return;

  const translation = limbPhysics.collider.translation();
  limb.group.position.set(
    translation.x - limbPhysics.colliderOffset.x,
    translation.y - limbPhysics.colliderOffset.y,
    translation.z - limbPhysics.colliderOffset.z,
  );
}

function addObjectCollisionMesh(colliders, mesh, label) {
  if (!mesh.geometry?.boundingBox) mesh.geometry.computeBoundingBox();
  if (!mesh.geometry?.boundingSphere) mesh.geometry.computeBoundingSphere();
  const localBox = mesh.geometry.boundingBox.clone();
  const localCenter = new THREE.Vector3();
  const localHalfExtents = new THREE.Vector3();
  localBox.getCenter(localCenter);
  localBox.getSize(localHalfExtents).multiplyScalar(0.5);
  colliders.push({
    mesh,
    label,
    localBox,
    localCenter,
    localHalfExtents,
    localRadius: mesh.geometry.boundingSphere.radius,
    worldBox: new THREE.Box3(),
    worldCenter: new THREE.Vector3(),
    worldHalfExtents: new THREE.Vector3(),
    worldMatrix: new THREE.Matrix4(),
    worldMatrixInverse: new THREE.Matrix4(),
    worldAxes: [new THREE.Vector3(), new THREE.Vector3(), new THREE.Vector3()],
    worldRadius: mesh.geometry.boundingSphere.radius,
    active: true,
  });
}

function addFork(group) {
  const metal = new THREE.MeshStandardMaterial({ color: 0xcfd8dc, metalness: 0.65, roughness: 0.22 });
  const handle = new THREE.Mesh(new THREE.CapsuleGeometry(0.055, 0.92, 8, 14), metal);
  handle.rotation.z = Math.PI / 2;
  group.add(handle);
  for (const z of [-0.13, -0.04, 0.05, 0.14]) {
    const tine = new THREE.Mesh(new THREE.BoxGeometry(0.5, 0.045, 0.028), metal);
    tine.position.set(0.55, 0, z);
    group.add(tine);
  }
}

function addKnife(group) {
  const metal = new THREE.MeshStandardMaterial({ color: 0xdce5e8, metalness: 0.72, roughness: 0.2 });
  const handleMat = new THREE.MeshStandardMaterial({ color: 0x313840, roughness: 0.55 });
  const handle = new THREE.Mesh(new THREE.BoxGeometry(0.58, 0.16, 0.18), handleMat);
  handle.position.x = -0.28;
  group.add(handle);
  const blade = new THREE.Mesh(new THREE.ConeGeometry(0.16, 0.9, 3), metal);
  blade.rotation.z = -Math.PI / 2;
  blade.scale.z = 0.28;
  blade.position.x = 0.42;
  group.add(blade);
}

function addSpoon(group) {
  const metal = new THREE.MeshStandardMaterial({ color: 0xd7e0e4, metalness: 0.58, roughness: 0.26 });
  const handle = new THREE.Mesh(new THREE.CapsuleGeometry(0.05, 0.88, 8, 14), metal);
  handle.rotation.z = Math.PI / 2;
  handle.position.x = -0.22;
  group.add(handle);
  const bowl = new THREE.Mesh(new THREE.SphereGeometry(0.24, 28, 14), metal);
  bowl.scale.set(1.25, 0.18, 0.82);
  bowl.position.x = 0.52;
  group.add(bowl);
}

function addCup(group) {
  const mat = new THREE.MeshStandardMaterial({ color: 0x62b4d8, roughness: 0.46, metalness: 0.05 });
  const cup = new THREE.Mesh(new THREE.CylinderGeometry(0.27, 0.22, 0.5, 28, 1, true), mat);
  cup.position.y = 0.22;
  group.add(cup);
  const handle = new THREE.Mesh(new THREE.TorusGeometry(0.16, 0.035, 8, 18, Math.PI * 1.35), mat);
  handle.rotation.y = Math.PI / 2;
  handle.position.set(0.28, 0.22, 0);
  group.add(handle);
}

function addBall(group) {
  const ball = new THREE.Mesh(
    new THREE.SphereGeometry(0.25, 32, 18),
    new THREE.MeshStandardMaterial({ color: 0x76f4be, roughness: 0.42 }),
  );
  ball.position.y = 0.21;
  group.add(ball);
}

function addBlock(group) {
  const block = new THREE.Mesh(
    new THREE.BoxGeometry(0.48, 0.42, 0.48),
    new THREE.MeshStandardMaterial({ color: 0xf2b766, roughness: 0.55 }),
  );
  block.position.y = 0.2;
  group.add(block);
}

function addUnknownMetal(group) {
  const mat = new THREE.MeshStandardMaterial({ color: 0xb4b8bd, metalness: 0.55, roughness: 0.28 });
  const body = new THREE.Mesh(new THREE.CapsuleGeometry(0.08, 0.84, 7, 12), mat);
  body.rotation.z = Math.PI / 2;
  group.add(body);
  const tip = new THREE.Mesh(new THREE.ConeGeometry(0.12, 0.28, 16), mat);
  tip.rotation.z = -Math.PI / 2;
  tip.position.x = 0.56;
  group.add(tip);
}

function getObjectMass(spec) {
  const massByKind = {
    fork: 1.05,
    knife: 0.95,
    spoon: 0.86,
    cup: 0.9,
    ball: 0.75,
    block: 1.2,
    metal: 1.08,
  };
  return massByKind[spec.kind] ?? 1.0;
}

function getObjectHalfExtents(spec) {
  const byKind = {
    fork: { x: 0.78, y: 0.11, z: 0.2 },
    knife: { x: 0.6, y: 0.12, z: 0.12 },
    spoon: { x: 0.58, y: 0.11, z: 0.25 },
    cup: { x: 0.33, y: 0.24, z: 0.23 },
    ball: { x: 0.25, y: 0.25, z: 0.25 },
    block: { x: 0.24, y: 0.21, z: 0.24 },
    metal: { x: 0.66, y: 0.11, z: 0.17 },
  };
  const half = byKind[spec.kind] ?? { x: 0.4, y: 0.2, z: 0.28 };
  return new THREE.Vector3(half.x, half.y, half.z);
}

function onPointerDown(event) {
  updatePointer(event);
  raycaster.setFromCamera(pointer, camera);
  const hits = raycaster.intersectObjects(scene.children, true);
  const hit = hits.find((item) => item.object.userData.draggableRoot);
  const root = hit?.object.userData.draggableRoot || nearestScreenObject(event.clientX, event.clientY);
  if (!root) return;

  selected = draggableObjects.find((item) => item.group === root);
  if (!selected) return;

  selected.isHeld = true;
  selected.mass = Math.max(selected.baseMass * PHYSICS.heldMassMultiplier, selected.baseMass);
  selected.velocity.set(0, 0, 0);
  selected.dragTarget.copy(selected.group.position);
  dragging = true;
  activePointerId = event.pointerId;
  controls.enabled = false;
  canvas.setPointerCapture(activePointerId);
  dragPointerY = event.clientY;
  dragPlane.setFromNormalAndCoplanarPoint(new THREE.Vector3(0, 1, 0), selected.group.position);
  raycaster.ray.intersectPlane(dragPlane, planeHit);
  dragOffset.copy(selected.group.position).sub(planeHit);
}

function onPointerMove(event) {
  if (!dragging || event.pointerId !== activePointerId || !selected) return;
  updatePointer(event);
  const pointerDeltaY = event.clientY - dragPointerY;
  dragPointerY = event.clientY;
  const canLift = event.ctrlKey;

  if (canLift) {
    selected.dragTarget.y = THREE.MathUtils.clamp(
      selected.dragTarget.y - pointerDeltaY * dragVerticalScale,
      dragYLimits.min,
      dragYLimits.max,
    );
    return;
  }

  dragPlane.setFromNormalAndCoplanarPoint(new THREE.Vector3(0, 1, 0), selected.group.position);
  raycaster.setFromCamera(pointer, camera);
  if (raycaster.ray.intersectPlane(dragPlane, planeHit)) {
    selected.dragTarget.set(
      THREE.MathUtils.clamp(planeHit.x + dragOffset.x, worldBounds.minX, worldBounds.maxX),
      selected.dragTarget.y,
      THREE.MathUtils.clamp(planeHit.z + dragOffset.z, worldBounds.minZ, worldBounds.maxZ),
    );
  }
}

function onPointerUp(event) {
  if (activePointerId !== event.pointerId) return;
  if (selected) {
    selected.isHeld = false;
    selected.mass = selected.baseMass;
  }
  dragging = false;
  activePointerId = null;
  dragPointerY = 0;
  controls.enabled = true;
  try {
    canvas.releasePointerCapture(event.pointerId);
  } catch {
    // Pointer capture may already be released after a cancel event.
  }
}

function onPointerWheel(event) {
  if (!dragging || !selected) return;
  event.preventDefault();
  const lift = -event.deltaY * 0.004;
  selected.dragTarget.y = THREE.MathUtils.clamp(selected.dragTarget.y + lift, dragYLimits.min, dragYLimits.max);
}

function updatePointer(event) {
  const rect = canvas.getBoundingClientRect();
  pointer.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
  pointer.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;
}

function nearestScreenObject(clientX, clientY) {
  let nearest = null;
  let nearestDistance = Infinity;
  for (const item of draggableObjects) {
    const screen = objectScreenPosition(item.spec.key);
    if (!screen) continue;
    const distance = Math.hypot(screen.x - clientX, screen.y - clientY);
    if (distance < nearestDistance) {
      nearestDistance = distance;
      nearest = item.group;
    }
  }
  return nearestDistance < 80 ? nearest : null;
}

function animate(timeMs) {
  const time = timeMs / 1000;
  const deltaSeconds = lastFrameTime > 0 ? Math.min(0.05, time - lastFrameTime) : 0.016;
  lastFrameTime = time;
  simulatePhysics(deltaSeconds);
  purgeUnexpectedCenterDividerArtifacts();
  const tracked = selected || nearestObjectToLimb();
  currentSensors = tracked ? readSensorsForItem(tracked, time) : { ...emptySensors };
  if (motorEnabled) {
    currentDecision = decide(tracked, currentSensors);
    if (currentDecision.action === "WITHDRAW_FAST") reflexPulse = Math.max(reflexPulse, 1);

    if (sensorsEnabled && tracked && shouldRecordExperience(tracked, currentSensors, time)) {
      learnedMemory.recordExperience({
        cueVector: tracked.spec.cueVector,
        label: tracked.spec.label,
        action: currentDecision.action,
        sensors: currentSensors,
        time,
        metadata: { objectKey: tracked.spec.key },
      });
    }
  } else if (currentDecision.className !== "unknown" || currentDecision.action !== "HOLD") {
    currentDecision = {
      action: "HOLD",
      reason: "Motor disabled",
      confidence: 0,
      risk: 0,
      className: "unknown",
    };
  }

  if (motorEnabled) {
    animateLimb(time, deltaSeconds, tracked, currentDecision, currentSensors);
  }
  updateSensorIndicators(currentSensors);

  for (const item of draggableObjects) {
    if (item !== selected && !dragging) {
      if (!limbPhysics.ready) {
        item.group.rotation.y += (0 - item.group.rotation.y) * 0.08;
      }
    }
    if (!limbPhysics.ready) {
      item.group.rotation.z = Math.sin(time * 1.2 + item.home.x) * 0.025;
    }
  }

  updateStereoEyes(tracked, currentDecision);
  updateUi(tracked, currentDecision, currentSensors);

  controls.update();
  renderer.render(scene, camera);
  reflexPulse = Math.max(0, reflexPulse - deltaSeconds * 3.8);
}

function simulatePhysics(deltaSeconds) {
  if (limbPhysics.ready && limbPhysics.engine && limbPhysics.collider) {
    simulatePhysicsWithRapier(deltaSeconds);
  } else {
    simulatePhysicsLegacy(deltaSeconds);
  }
}

function simulatePhysicsWithRapier(deltaSeconds) {
  for (const item of draggableObjects) {
    const body = item.group.userData.physics?.body;
    if (!limbPhysics.engine) continue;
    const hasPhysicsBody = Boolean(body);

    if (item.isHeld) {
      item.group.position.copy(item.dragTarget);
      enforceWorldBounds(item);
      if (!item.group.position.equals(item.dragTarget)) {
        item.dragTarget.copy(item.group.position);
      }
      const bodyPosition = {
        x: item.group.position.x,
        y: item.group.position.y,
        z: item.group.position.z,
      };
      if (hasPhysicsBody) {
        limbPhysics.engine.setMeshPosition(item.group, bodyPosition);
        limbPhysics.engine.setMeshVelocity(item.group, { x: 0, y: 0, z: 0 });
      }
      item.velocity.set(0, 0, 0);
    } else if (body) {
      const bodyPosition = body.translation();
      const bodyVelocity = body.linvel();
      item.group.position.set(bodyPosition.x, bodyPosition.y, bodyPosition.z);
      item.velocity.set(bodyVelocity.x, bodyVelocity.y, bodyVelocity.z);
    } else {
      item.velocity.multiplyScalar(0);
    }

    const preClamp = {
      x: item.group.position.x,
      y: item.group.position.y,
      z: item.group.position.z,
    };
    enforceWorldBounds(item);
    if (
      preClamp.x !== item.group.position.x ||
      preClamp.y !== item.group.position.y ||
      preClamp.z !== item.group.position.z
    ) {
      if (hasPhysicsBody) {
        limbPhysics.engine.setMeshPosition(item.group, {
          x: item.group.position.x,
          y: item.group.position.y,
          z: item.group.position.z,
        });
        limbPhysics.engine.setMeshVelocity(item.group, { x: 0, y: 0, z: 0 });
      }
      item.velocity.set(0, 0, 0);
    }
    refreshObjectColliderStates(item);
  }

  if (motorEnabled) {
    updateLimbReaction(deltaSeconds);
  } else {
    limb.reactionOffset.set(0, 0, 0);
    limb.reactionVelocity.set(0, 0, 0);
  }
  syncLimbToPhysicsCollider();
}

function simulatePhysicsLegacy(deltaSeconds) {
  const dt = Math.min(0.05, Math.max(1 / 120, deltaSeconds));
  let maxVelocity = 0;
  let maxHeldDrag = 0;
  for (const item of draggableObjects) {
    const speed = item.velocity.length();
    if (speed > maxVelocity) maxVelocity = speed;
    if (item.isHeld) {
      const dragDeltaX = item.dragTarget.x - item.group.position.x;
      const dragDeltaY = item.dragTarget.y - item.group.position.y;
      const dragDeltaZ = item.dragTarget.z - item.group.position.z;
      const dragDistance = Math.hypot(dragDeltaX, dragDeltaY, dragDeltaZ);
      if (dragDistance > maxHeldDrag) maxHeldDrag = dragDistance;
    }
  }

  const targetTravel = Math.max(PHYSICS.collisionSubstepDistance * 0.5, maxVelocity * dt);
  const travelSteps = Math.ceil(targetTravel / PHYSICS.collisionSubstepDistance);
  const dragSteps = Math.ceil(maxHeldDrag / PHYSICS.heldSubstepDistance);
  const baseFrameSteps = Math.max(2, Math.ceil(dt / (1 / 40)));
  const requestedSubsteps = Math.max(baseFrameSteps, travelSteps, dragSteps);
  const clampedSubsteps = Math.max(2, Math.min(PHYSICS.collisionSteps, requestedSubsteps));
  const stepDt = dt / clampedSubsteps;
  const velocityDamping = Math.pow(PHYSICS.linearDamping, 1 / clampedSubsteps);

  for (let step = 0; step < clampedSubsteps; step += 1) {
    refreshLimbColliderWorldBounds();
    for (const item of draggableObjects) {
      if (!item.isHeld) {
        item.velocity.y += PHYSICS.gravity * stepDt;
        item.velocity.y = Math.max(item.velocity.y, -PHYSICS.maxFallSpeed);
        item.group.position.addScaledVector(item.velocity, stepDt);
        if (item.group.position.y - item.collider.y < worldBounds.floorY) {
          item.group.position.y = worldBounds.floorY + item.collider.y;
          item.velocity.y = 0;
        }
      } else {
        item.velocity.set(0, 0, 0);
        const dragDeltaX = item.dragTarget.x - item.group.position.x;
        const dragDeltaY = item.dragTarget.y - item.group.position.y;
        const dragDeltaZ = item.dragTarget.z - item.group.position.z;
        const dragDeltaLengthSq = dragDeltaX * dragDeltaX + dragDeltaY * dragDeltaY + dragDeltaZ * dragDeltaZ;
        if (dragDeltaLengthSq > 0) {
          const dragDeltaLength = Math.sqrt(dragDeltaLengthSq);
          const remainingSteps = Math.max(1, clampedSubsteps - step);
          const maxStepDistance = Math.min(PHYSICS.heldSubstepDistance, dragDeltaLength / remainingSteps);
          const dragStep = maxStepDistance / dragDeltaLength;
          item.group.position.x += dragDeltaX * dragStep;
          item.group.position.y += dragDeltaY * dragStep;
          item.group.position.z += dragDeltaZ * dragStep;
        }
      }

      item.velocity.x *= velocityDamping;
      item.velocity.z *= velocityDamping;
      item.velocity.y *= velocityDamping;
      enforceWorldBounds(item);
      refreshObjectColliderStates(item);
    }

    for (let pass = 0; pass < PHYSICS.iterations; pass++) {
      let adjusted = false;

      for (let i = 0; i < draggableObjects.length; i++) {
        const left = draggableObjects[i];
        for (let j = i + 1; j < draggableObjects.length; j++) {
          const right = draggableObjects[j];
          adjusted = resolveDynamicBodyCollision(left, right) || adjusted;
        }
      }

      for (const item of draggableObjects) {
        adjusted = resolveBodyLimbCollision(item) || adjusted;
        if (sensorsEnabled) {
          adjusted = resolveBodySensorCollisions(item) || adjusted;
        }
        enforceWorldBounds(item);
      }

      if (!adjusted) break;
    }
  }

  if (motorEnabled) {
    updateLimbReaction(dt);
  } else {
    limb.reactionOffset.set(0, 0, 0);
    limb.reactionVelocity.set(0, 0, 0);
  }
}

function resolveDynamicBodyCollision(left, right) {
  let adjusted = false;
  if (!left.collisionMeshes?.length || !right.collisionMeshes?.length) return false;
  for (const leftCollider of left.collisionMeshes) {
    if (!leftCollider.active || leftCollider.worldHalfExtents.lengthSq() === 0) continue;
    for (const rightCollider of right.collisionMeshes) {
      if (!rightCollider.active || rightCollider.worldHalfExtents.lengthSq() === 0) continue;
      const contact = getObbContact(leftCollider, rightCollider);
      if (!contact || contact.penetration <= PHYSICS.contactTolerance) continue;

      const leftInvMass = 1 / Math.max(left.mass, 0.0001);
      const rightInvMass = 1 / Math.max(right.mass, 0.0001);
      const inverseMassSum = leftInvMass + rightInvMass;
      if (inverseMassSum <= 0) continue;

      const normal = contact.normal;
      const correction = contact.penetration * PHYSICS.penetrationBias;
      const leftShare = correction * (leftInvMass / inverseMassSum);
      const rightShare = correction * (rightInvMass / inverseMassSum);

      left.group.position.addScaledVector(normal, leftShare);
      right.group.position.addScaledVector(normal, -rightShare);

      const closingSpeed = normal.dot(left.velocity) - normal.dot(right.velocity);
      if (closingSpeed < 0) {
        const restitution = Math.min(left.restitution, right.restitution);
        const impulse = -(1 + restitution) * closingSpeed / inverseMassSum;
        left.velocity.addScaledVector(normal, impulse * leftInvMass);
        right.velocity.addScaledVector(normal, -impulse * rightInvMass);
      }

      refreshObjectColliderStates(left);
      refreshObjectColliderStates(right);
      adjusted = true;
    }
  }

  return adjusted;
}

function refreshLimbColliderWorldBounds() {
  for (const collider of limb.colliders) {
    collider.mesh.updateWorldMatrix(true, false);
    const scale = collider.mesh.getWorldScale(physicsScratch.meshWorldScale);
    collider.worldBox.copy(collider.localBox).applyMatrix4(collider.mesh.matrixWorld);
    collider.worldCenter.copy(collider.localCenter).applyMatrix4(collider.mesh.matrixWorld);
    collider.worldHalfExtents.set(
      collider.localHalfExtents.x * Math.max(Math.abs(scale.x), 1e-6),
      collider.localHalfExtents.y * Math.max(Math.abs(scale.y), 1e-6),
      collider.localHalfExtents.z * Math.max(Math.abs(scale.z), 1e-6),
    );
    collider.mesh.matrixWorld.extractBasis(
      collider.worldAxes[0],
      collider.worldAxes[1],
      collider.worldAxes[2],
    );
    collider.worldAxes[0].normalize();
    collider.worldAxes[1].normalize();
    collider.worldAxes[2].normalize();
  }
}

function resolveBodyLimbCollision(body) {
  let adjusted = false;
  if (!body.collisionMeshes?.length) return false;
  for (const objectCollider of body.collisionMeshes) {
    if (!objectCollider.active || objectCollider.worldHalfExtents.lengthSq() === 0) continue;
    for (const limbCollider of limb.colliders) {
      if (!limbCollider.active || limbCollider.worldHalfExtents.lengthSq() === 0) continue;
      adjusted = resolveObbContact(body, objectCollider, limbCollider) || adjusted;
    }
  }

  return adjusted;
}

function resolveObbContact(body, objectCollider, limbCollider) {
  const contact = getObbContact(objectCollider, limbCollider);
  if (!contact || contact.penetration <= PHYSICS.contactTolerance) return false;

  const normal = contact.normal;
  body.group.position.addScaledVector(normal, contact.penetration * PHYSICS.penetrationBias);
  const bodyClosingSpeed = body.velocity.dot(normal);
  if (bodyClosingSpeed < 0) {
    body.velocity.addScaledVector(normal, -(1 + LIMB.restitution) * bodyClosingSpeed);
  }

  if (motorEnabled && sensorsEnabled) {
    const contactStrength = Math.min(
      Math.max(contact.penetration * 3, 0) + Math.max(-bodyClosingSpeed, 0) * 0.65 + (body.isHeld ? 0.35 : 0),
      LIMB.maxImpulse,
    );
    limb.reactionVelocity.addScaledVector(normal, -(contactStrength * LIMB.impulseScale / LIMB.mass));
    limb.reactionOffset.addScaledVector(normal, -(contactStrength * 0.05));
  }

  refreshObjectColliderStates(body);
  return true;
}

function getObbContact(firstCollider, secondCollider) {
  if (!firstCollider.worldBox || !secondCollider.worldBox) return null;
  if (!firstCollider.worldBox.intersectsBox(secondCollider.worldBox)) return null;

  const firstHalf = firstCollider.worldHalfExtents;
  const secondHalf = secondCollider.worldHalfExtents;
  const firstAxes = firstCollider.worldAxes;
  const secondAxes = secondCollider.worldAxes;

  const delta = physicsScratch.obbCenterDelta.copy(firstCollider.worldCenter).sub(secondCollider.worldCenter);
  const tA = [delta.dot(firstAxes[0]), delta.dot(firstAxes[1]), delta.dot(firstAxes[2])];
  const tB = [delta.dot(secondAxes[0]), delta.dot(secondAxes[1]), delta.dot(secondAxes[2])];

  const rotation = physicsScratch.obbRotation;
  const absRotation = physicsScratch.obbAbsRotation;
  const firstE = [firstHalf.x, firstHalf.y, firstHalf.z];
  const secondE = [secondHalf.x, secondHalf.y, secondHalf.z];

  for (let first = 0; first < 3; first += 1) {
    const row = first * 3;
    for (let second = 0; second < 3; second += 1) {
      const value = firstAxes[first].dot(secondAxes[second]);
      rotation[row + second] = value;
      absRotation[row + second] = Math.abs(value) + PHYSICS.obbEpsilon;
    }
  }

  let bestDepth = Infinity;
  const normal = physicsScratch.collisionAxis.set(1, 0, 0);

  // A[i] axes.
  for (let first = 0; first < 3; first += 1) {
    const axis = Math.abs(tA[first]);
    const axisDepth =
      firstE[first] +
      secondE[0] * absRotation[first * 3] +
      secondE[1] * absRotation[first * 3 + 1] +
      secondE[2] * absRotation[first * 3 + 2] -
      axis;
    if (axisDepth <= 0) return null;
    if (axisDepth < bestDepth) {
      bestDepth = axisDepth;
      const normalSign = Math.sign(tA[first]);
      normal.copy(firstAxes[first]).multiplyScalar(
        normalSign !== 0 ? normalSign : delta.dot(firstAxes[first]) >= 0 ? 1 : -1,
      );
    }
  }

  // B[i] axes.
  for (let second = 0; second < 3; second += 1) {
    const axis = Math.abs(tB[second]);
    const axisDepth =
      secondE[second] +
      firstE[0] * absRotation[second] +
      firstE[1] * absRotation[3 + second] +
      firstE[2] * absRotation[6 + second] -
      axis;
    if (axisDepth <= 0) return null;
    if (axisDepth < bestDepth) {
      bestDepth = axisDepth;
      const normalSign = Math.sign(tB[second]);
      normal.copy(secondAxes[second]).multiplyScalar(
        normalSign !== 0 ? normalSign : delta.dot(secondAxes[second]) >= 0 ? 1 : -1,
      );
    }
  }

  // A[i] x B[j] axes.
  for (let first = 0; first < 3; first += 1) {
    const firstNext = (first + 1) % 3;
    const firstOther = (first + 2) % 3;
    for (let second = 0; second < 3; second += 1) {
      const secondNext = (second + 1) % 3;
      const secondOther = (second + 2) % 3;

      const axis = physicsScratch.collisionAxis.copy(firstAxes[first]).cross(secondAxes[second]);
      const axisLen2 = axis.lengthSq();
      if (axisLen2 <= PHYSICS.obbEpsilon) continue;

      const axisLen = Math.sqrt(axisLen2);
      const crossDepth =
        firstE[firstNext] * absRotation[firstOther * 3 + second] +
        firstE[firstOther] * absRotation[firstNext * 3 + second] +
        secondE[secondNext] * absRotation[first * 3 + secondOther] +
        secondE[secondOther] * absRotation[first * 3 + secondNext] -
        Math.abs(tA[firstOther] * rotation[firstNext * 3 + second] - tA[firstNext] * rotation[firstOther * 3 + second]);

      if (crossDepth <= 0) return null;
      const axisPenetration = crossDepth / axisLen;
      if (axisPenetration < bestDepth) {
        bestDepth = axisPenetration;
        normal.copy(axis.multiplyScalar(Math.sign(axis.dot(delta)) || 1).divideScalar(axisLen));
      }
    }
  }

  if (!Number.isFinite(bestDepth) || bestDepth <= 0 || normal.lengthSq() === 0) return null;

  return {
    normal: normal.clone().normalize(),
    penetration: bestDepth,
  };
}

function resolveBodySensorCollisions(body) {
  let adjusted = false;
  for (const sample of collectBodySensorContactSamples(body)) {
    if (sample.penetration <= PHYSICS.contactTolerance) continue;

    const normal = physicsScratch.collisionAxis
      .set(sample.normal.x, sample.normal.y, sample.normal.z)
      .normalize();
    if (!normal.lengthSq()) {
      normal.set(sample.sensor.x - body.group.position.x, sample.sensor.y - body.group.position.y, sample.sensor.z - body.group.position.z).normalize();
    }
    if (!normal.lengthSq()) continue;

    body.group.position.addScaledVector(normal, sample.penetration * PHYSICS.penetrationBias);
    const closingSpeed = body.velocity.dot(normal);
    if (closingSpeed < 0) {
      body.velocity.addScaledVector(normal, -(1 + body.restitution) * closingSpeed);
    }
    if (Math.abs(normal.y) > 0.5) {
      body.velocity.y *= PHYSICS.wallDamping;
    }
    adjusted = true;
  }

  if (adjusted) {
    refreshObjectColliderStates(body);
  }
}

function refreshObjectColliderStates(item) {
  const colliders = item.collisionMeshes;
  if (!colliders?.length) return;

  let hasBounds = false;
  item.colliderWorldBounds.makeEmpty();
  for (const collider of colliders) {
    if (!collider.active) continue;
    collider.mesh.updateWorldMatrix(true, false);
    const scale = collider.mesh.getWorldScale(physicsScratch.meshWorldScale);
    collider.worldMatrix.copy(collider.mesh.matrixWorld);
    collider.worldMatrixInverse.copy(collider.mesh.matrixWorld).invert();
    collider.worldBox.copy(collider.localBox).applyMatrix4(collider.mesh.matrixWorld);
    collider.worldCenter.copy(collider.localCenter).applyMatrix4(collider.mesh.matrixWorld);
    collider.worldHalfExtents.set(
      collider.localHalfExtents.x * Math.max(Math.abs(scale.x), 1e-6),
      collider.localHalfExtents.y * Math.max(Math.abs(scale.y), 1e-6),
      collider.localHalfExtents.z * Math.max(Math.abs(scale.z), 1e-6),
    );
    collider.mesh.matrixWorld.extractBasis(
      collider.worldAxes[0],
      collider.worldAxes[1],
      collider.worldAxes[2],
    );
    collider.worldAxes[0].normalize();
    collider.worldAxes[1].normalize();
    collider.worldAxes[2].normalize();
    collider.worldRadius = collider.localRadius * Math.max(Math.abs(scale.x), Math.abs(scale.y), Math.abs(scale.z), 1e-6);
    if (!hasBounds) {
      item.colliderWorldBounds.copy(collider.worldBox);
      hasBounds = true;
    } else {
      item.colliderWorldBounds.union(collider.worldBox);
    }
  }

  if (!hasBounds) return;
  item.colliderWorldBounds.getCenter(item.colliderWorldCenter);
  item.colliderWorldBounds.getSize(item.colliderWorldHalfExtents);
  item.colliderWorldHalfExtents.multiplyScalar(0.5);
  item.collider.copy(item.colliderWorldHalfExtents);
}

function closestPointOnCollider(sensorPosition, collider, sensorRadius) {
  const localPoint = physicsScratch.localPoint.copy(sensorPosition).applyMatrix4(collider.worldMatrixInverse);
  const localOffsetFromCenter = physicsScratch.localOffset.copy(localPoint).sub(collider.localCenter);

  const clampedLocal = physicsScratch.localClampedPoint
    .set(
      THREE.MathUtils.clamp(localOffsetFromCenter.x, -collider.localHalfExtents.x, collider.localHalfExtents.x),
      THREE.MathUtils.clamp(localOffsetFromCenter.y, -collider.localHalfExtents.y, collider.localHalfExtents.y),
      THREE.MathUtils.clamp(localOffsetFromCenter.z, -collider.localHalfExtents.z, collider.localHalfExtents.z),
    )
    .add(collider.localCenter);

  const closest = physicsScratch.sensorClosestPoint.copy(clampedLocal).applyMatrix4(collider.worldMatrix);
  const localNormal = physicsScratch.localNormal.copy(localPoint).sub(clampedLocal);
  const isInside = localNormal.lengthSq() === 0;
  if (isInside) {
    const distToLeft = collider.localHalfExtents.x - Math.abs(localOffsetFromCenter.x);
    const distToRight = collider.localHalfExtents.y - Math.abs(localOffsetFromCenter.y);
    const distToTop = collider.localHalfExtents.z - Math.abs(localOffsetFromCenter.z);
    const minDist = Math.min(distToLeft, distToRight, distToTop);
    if (minDist === distToLeft) {
      localNormal.set(localOffsetFromCenter.x >= 0 ? 1 : -1, 0, 0);
    } else if (minDist === distToRight) {
      localNormal.set(0, localOffsetFromCenter.y >= 0 ? 1 : -1, 0);
    } else {
      localNormal.set(0, 0, localOffsetFromCenter.z >= 0 ? 1 : -1);
    }
  }

  physicsScratch.normalMatrix.getNormalMatrix(collider.worldMatrix);
  const normal = physicsScratch.localNormal.normalize().applyMatrix3(physicsScratch.normalMatrix);
  const signedDistance = sensorPosition.distanceTo(closest) - sensorRadius;
  return {
    contactPoint: closest,
    distance: Math.max(0, signedDistance),
    signedDistance,
    penetration: Math.max(0, -signedDistance),
    normal,
  };
}

function getSensorWorldRadius(sensor) {
  const localRadius = sensor.userData.touchRadius ?? 0.035;
  const scale = sensor.getWorldScale(physicsScratch.sensorScale);
  return localRadius * Math.max(Math.abs(scale.x), Math.abs(scale.y), Math.abs(scale.z), 1e-6);
}

function collectBodySensorContactSamples(body) {
  const samples = [];
  if (!body.collisionMeshes?.length) return samples;
  for (const sensor of limb.sensors) {
    sensor.getWorldPosition(physicsScratch.sensorPosition);
    const sensorRadius = getSensorWorldRadius(sensor);
    const sensorName = sensor.userData.sensorName || "sensor";
    let best = null;

    for (const meshCollider of body.collisionMeshes) {
      if (!meshCollider.active) continue;
      const sample = closestPointOnCollider(physicsScratch.sensorPosition, meshCollider, sensorRadius);
      const gapDistance = sample.distance;
      if (best === null || sample.signedDistance < best.signedDistance) {
        best = {
          sensorRadius,
          sensor: {
            x: physicsScratch.sensorPosition.x,
            y: physicsScratch.sensorPosition.y,
            z: physicsScratch.sensorPosition.z,
            name: sensorName,
          },
          distance: gapDistance,
          signedDistance: sample.signedDistance,
          penetration: sample.penetration > PHYSICS.contactTolerance ? sample.penetration : 0,
          contact: {
            x: sample.contactPoint.x,
            y: sample.contactPoint.y,
            z: sample.contactPoint.z,
            name: meshCollider.label || body.spec.label,
            sharpnessScale: 1,
            areaScale: 1,
          },
          normal: {
            x: sample.normal.x,
            y: sample.normal.y,
            z: sample.normal.z,
          },
        };
      }
    }

    if (best) {
      samples.push(best);
    }
  }

  return samples;
}

function enforceWorldBounds(item) {
  const minX = worldBounds.minX + item.collider.x;
  const maxX = worldBounds.maxX - item.collider.x;
  const minZ = worldBounds.minZ + item.collider.z;
  const maxZ = worldBounds.maxZ - item.collider.z;
  const minY = worldBounds.floorY + item.collider.y;
  const maxY = worldBounds.maxY - item.collider.y;

  if (item.group.position.x < minX) {
    item.group.position.x = minX;
    if (item.velocity.x < 0) item.velocity.x = 0;
  } else if (item.group.position.x > maxX) {
    item.group.position.x = maxX;
    if (item.velocity.x > 0) item.velocity.x = 0;
  }
  if (item.group.position.z < minZ) {
    item.group.position.z = minZ;
    if (item.velocity.z < 0) item.velocity.z = 0;
  } else if (item.group.position.z > maxZ) {
    item.group.position.z = maxZ;
    if (item.velocity.z > 0) item.velocity.z = 0;
  }
  if (item.group.position.y < minY) {
    item.group.position.y = minY;
    if (item.velocity.y < 0) item.velocity.y = 0;
  } else if (item.group.position.y > maxY) {
    item.group.position.y = maxY;
    if (item.velocity.y > 0) item.velocity.y = 0;
  }
}

function updateLimbReaction(deltaSeconds) {
  limb.reactionOffset.addScaledVector(limb.reactionVelocity, deltaSeconds);
  limb.reactionVelocity.multiplyScalar(Math.pow(LIMB.damping, deltaSeconds * 60));
  limb.reactionOffset.multiplyScalar(Math.max(0, 1 - LIMB.springReturn * deltaSeconds));

  if (limb.reactionOffset.lengthSq() <= 1e-6 && limb.reactionVelocity.lengthSq() <= 1e-6) {
    limb.reactionOffset.set(0, 0, 0);
    limb.reactionVelocity.set(0, 0, 0);
    return;
  }

  if (limb.reactionOffset.length() > LIMB.maxOffset) {
    limb.reactionOffset.normalize().multiplyScalar(LIMB.maxOffset);
  }
}

function nearestObjectToLimb() {
  const handPosition = handWorldPosition();
  let nearest = null;
  let nearestScore = Infinity;
  for (const item of draggableObjects) {
    const score = Math.hypot(
      item.group.position.x - handPosition.x,
      item.group.position.y - handPosition.y,
      item.group.position.z - handPosition.z,
    );
    if (score < nearestScore) {
      nearestScore = score;
      nearest = item;
    }
  }
  return nearest;
}

function decide(item, sensors) {
  if (!item) {
    return {
      action: "HOLD",
      reason: "No object in view.",
      confidence: 0,
      risk: 0,
      className: "unknown",
    };
  }

  const x = item.group.position.x;
  const confidence = THREE.MathUtils.clamp(0.45 + (x + 5.2) / 8.35 * 0.45, 0.35, 0.94);
  return chooseAction({
    itemX: x,
    cueVector: item.spec.cueVector,
    confidence,
    sensors,
    memory: learnedMemory,
  });
}

function readSensorsForItem(item, time) {
  if (!sensorsEnabled) {
    return { ...emptySensors };
  }
  const previous = depthHistory.get(item.spec.key) ?? null;
  const contactSamples = collectBodySensorContactSamples(item);
  const sensors = readTactileSensors({
    sensorPoints: sensorWorldPoints(),
    contactPoints: contactWorldPoints(item),
    contactSamples,
    tactileProfile: item.spec.tactile,
    previousDistance: previous?.distance ?? null,
    deltaSeconds: previous ? Math.max(0, time - previous.time) : 0,
  });
  depthHistory.set(item.spec.key, { distance: sensors.distance, time });
  return sensors;
}

function shouldRecordExperience(item, sensors, time) {
  const key = item.spec.key;
  const previous = contactLearningState.get(key) || {
    inContact: false,
    lastRecordTime: -Infinity,
    painRecorded: false,
  };
  const inContact = sensors.safeTouch || sensors.painful;

  if (!inContact) {
    contactLearningState.set(key, { ...previous, inContact: false });
    return false;
  }

  const painNow = sensors.pain >= LEARNING_CONFIG.painThreshold;
  const enteredContact = !previous.inContact;
  const painReinforcementDue = painNow && time - previous.lastRecordTime >= 0.2;
  const shouldRecord = enteredContact || (painNow && !previous.painRecorded) || painReinforcementDue;

  contactLearningState.set(key, {
    inContact: true,
    lastRecordTime: shouldRecord ? time : previous.lastRecordTime,
    painRecorded: previous.painRecorded || painNow,
  });
  return shouldRecord;
}

function handWorldPosition() {
  const position = new THREE.Vector3();
  limb.hand.getWorldPosition(position);
  return { x: position.x, y: position.y, z: position.z };
}

function sensorWorldPoints() {
  return limb.sensors.map((sensor) => {
    const position = new THREE.Vector3();
    sensor.getWorldPosition(position);
    return {
      x: position.x,
      y: position.y,
      z: position.z,
      name: sensor.userData.sensorName || "sensor",
    };
  });
}

function contactWorldPoints(item) {
  return (item.spec.contactPoints || [{ x: 0, z: 0, name: item.spec.label }]).map((point) => {
    const position = new THREE.Vector3(point.x, point.y ?? 0, point.z);
    item.group.localToWorld(position);
    return {
      x: position.x,
      y: position.y,
      z: position.z,
      name: point.name || item.spec.label,
      sharpnessScale: point.sharpnessScale ?? 1,
      areaScale: point.areaScale ?? 1,
    };
  });
}

function animateLimb(time, deltaSeconds, item, decision, sensors) {
  let shoulderZ = -0.06;
  let elbowZ = -0.18;
  let handY = -0.1;
  let shoulderYaw = 0;
  let alpha = 0.08;
  const baseTarget = restingBasePosition.clone();
  baseTarget.add(limb.reactionOffset);
  const pointyThreat = (item?.spec?.tactile?.tipSharpness ?? 0) >= pointyAvoidanceThreshold;
  const closeEnoughForAvoidance = item ? item.group.position.x > 0.16 : false;
  const pointyThreatAction =
    pointyThreat &&
    ["CAUTIOUS_PROBE", "APPROACH_SLOW", "GENTLE_TOUCH", "ORIENT_CAMERA", "RETRACT", "WITHDRAW_FAST"].includes(
      decision.action,
    );
  const pointySlowApproach =
    pointyThreat &&
    item &&
    pointyThreatAction &&
    item.group.position.x >= pointySlowApproachX &&
    (sensors.approachingSensor || decision.action !== "WITHDRAW_FAST");
  const pointyDistanceBonus = pointyThreatAction && item ? (1 + (Math.max(0, item.group.position.x - 0.28) / 1.5) * 0.25) : 1;
  const shouldSidestep =
    item &&
    closeEnoughForAvoidance &&
    (decision.className === "danger" || pointyThreatAction || pointySlowApproach);

  if (item) {
    shoulderYaw = THREE.MathUtils.clamp(item.group.position.z * 0.12, -0.28, 0.28);
    const objectPressure = THREE.MathUtils.smoothstep(item.group.position.x, 0.15, 2.7);

    if (decision.className === "danger") {
      const fast = decision.action === "WITHDRAW_FAST";
      const dodgeFastApproach = shouldDodgeReflex(decision, sensors, LEARNING_CONFIG);
      alpha = fast ? Math.min(0.94, 0.5 + reflexPulse * 0.42 + deltaSeconds * 2.4) : 0.14;
      shoulderZ = fast ? -1.05 - sensors.pain * 0.22 : -0.62 - sensors.pain * 0.18;
      elbowZ = fast ? 1.2 + sensors.pain * 0.25 : 0.72 + sensors.pain * 0.18;
      handY = fast ? 1.12 : 0.72;
      if (dodgeFastApproach || shouldSidestep) {
        const sideStep = item.group.position.z >= 0 ? -pointySidestepZ : pointySidestepZ;
        const slowApproachBoost = pointySlowApproach ? 1.18 : 1;
        const sidestepBoost = pointyThreat ? pointySidestepXBoost * pointyDistanceBonus * slowApproachBoost : 1;
        baseTarget.x += pointySidestepX * sidestepBoost;
        baseTarget.z += sideStep * (pointyThreat ? pointySidestepZBoost * pointyDistanceBonus * slowApproachBoost : 1);
        baseTarget.y += fast ? 0.52 : pointySlowApproach ? 0.44 : 0.24;
        shoulderYaw = THREE.MathUtils.clamp(sideStep * 0.55, -0.48, 0.48);
        shoulderZ = -1.38;
        elbowZ = 1.55;
        handY = 1.22;
        alpha = Math.max(alpha, 0.92);
      }

      if (!shouldSidestep && decision.action === "WITHDRAW_FAST") {
        baseTarget.y += 0.2;
      } else if (!dodgeFastApproach && sensors.pain > 0) {
        baseTarget.y += 0.11;
      }
    } else if (shouldSidestep) {
      const sideStep = item.group.position.z >= 0 ? -pointySidestepZ : pointySidestepZ;
      const sidestepBoost = pointyThreat
        ? pointySidestepXBoost * pointyDistanceBonus * (pointySlowApproach ? 1.22 : 1)
        : 1;
      baseTarget.x += pointySidestepX * sidestepBoost;
      baseTarget.z += sideStep * 0.78 * sidestepBoost;
      baseTarget.y += pointyThreat ? (pointySlowApproach ? 0.52 : 0.34) : 0.24;
      shoulderYaw = THREE.MathUtils.clamp(sideStep * (pointySlowApproach ? 0.62 : 0.5), -0.5, 0.5);
      shoulderZ = -1.3;
      elbowZ = 1.45;
      handY = 1.2;
      alpha = Math.max(alpha, 0.9);
    } else if (decision.className === "unknown") {
      shoulderZ = -0.24 - objectPressure * 0.08 + Math.sin(time * 5) * 0.018;
      elbowZ = 0.28;
      handY = 0.16;
      baseTarget.y = restingBasePosition.y - 0.05;
    } else {
      shoulderZ = -0.02 + objectPressure * 0.06;
      elbowZ = -0.34 - objectPressure * 0.08;
      handY = -0.12;
      baseTarget.y = restingBasePosition.y - 0.03;
    }
  }

  baseTarget.y = THREE.MathUtils.clamp(baseTarget.y, dragYLimits.min, dragYLimits.max);
  const rootBlend = decision.action === "WITHDRAW_FAST" ? Math.max(0.42, alpha * 0.55) : 0.16;
  const targetRoot = new THREE.Vector3().copy(limb.group.position).lerp(baseTarget, rootBlend);
  moveLimbRootToTarget(targetRoot);
  limb.shoulder.rotation.y += (shoulderYaw - limb.shoulder.rotation.y) * alpha;
  limb.shoulder.rotation.z += (shoulderZ - limb.shoulder.rotation.z) * alpha;
  limb.elbow.rotation.z += (elbowZ - limb.elbow.rotation.z) * alpha;
  limb.hand.rotation.y += (handY - limb.hand.rotation.y) * alpha;
}

function moveLimbRootToTarget(targetRoot) {
  if (!limbPhysics.ready || !limbPhysics.controller || !limbPhysics.collider) {
    limb.group.position.copy(targetRoot);
    return;
  }

  syncLimbColliderOffset();
  limbPhysicsScratch.movement.subVectors(targetRoot, limb.group.position);
  const desiredDistanceSq = limbPhysicsScratch.movement.lengthSq();
  if (desiredDistanceSq < PHYSICS.collisionEps) return;

  const speed = LIMB_CONTROLLER.moveSpeed * PHYSICS.characterStep;
  const desiredDistance = Math.sqrt(desiredDistanceSq);
  const stepScale = Math.min(1, speed / desiredDistance);
  limbPhysicsScratch.movement.multiplyScalar(stepScale);

  const movement = new limbPhysics.engine.RAPIER.Vector3(
    limbPhysicsScratch.movement.x,
    limbPhysicsScratch.movement.y,
    limbPhysicsScratch.movement.z,
  );
  limbPhysics.controller.computeColliderMovement(limbPhysics.collider, movement);
  const computedMovement = limbPhysics.controller.computedMovement();

  const position = limbPhysics.collider.translation();
  limbPhysicsScratch.nextColliderPosition.x = position.x + computedMovement.x;
  limbPhysicsScratch.nextColliderPosition.y = position.y + computedMovement.y;
  limbPhysicsScratch.nextColliderPosition.z = position.z + computedMovement.z;
  limbPhysics.collider.setTranslation(limbPhysicsScratch.nextColliderPosition);

  limb.group.position.set(
    limbPhysicsScratch.nextColliderPosition.x - limbPhysics.colliderOffset.x,
    limbPhysicsScratch.nextColliderPosition.y - limbPhysics.colliderOffset.y,
    limbPhysicsScratch.nextColliderPosition.z - limbPhysics.colliderOffset.z,
  );
}

function updateStereoEyes(item, decision) {
  if (!cameraEnabled) {
    setEyeCameraOffline(eyeElements.left);
    setEyeCameraOffline(eyeElements.right);
    return;
  }
  for (const eye of Object.values(eyeElements)) {
    updateEyeCamera(eye, item);
    renderEyeScene(eye);
  }

  if (!item) {
    setEyeEmpty(eyeElements.left);
    setEyeEmpty(eyeElements.right);
    return;
  }

  const tactileDepth = Number.isFinite(currentSensors.distance) ? currentSensors.distance : objectDistanceToHand(item);
  const disparity = THREE.MathUtils.clamp((depthStereoBaseline * stereoFocal) / Math.max(tactileDepth, 0.25), 0.025, 0.22);
  const depth = THREE.MathUtils.clamp(estimateDepthFromDisparity(disparity, depthStereoBaseline, stereoFocal), 0.25, 2.4);
  const conf = Math.round(decision.confidence * 100);

  for (const eye of Object.values(eyeElements)) {
    const rect = objectProjectionInEye(item, eye.camera);
    const detection = {
      rect,
      depth: eye.side < 0 ? depth : depth * 0.99,
      label: item.spec.label,
      conf,
      className: decision.className,
      closingSpeed: currentSensors.closingSpeed,
    };
    if (rect) {
      setEye(eye, detection);
    } else {
      setEyeNoDetection(eye, detection);
    }
  }
}

function updateEyeCamera(eye, item) {
  limb.hand.getWorldPosition(eyeHandPosition);
  configureEyeCamera(eye.camera, {
    handPosition: eyeHandPosition,
    itemPosition: item?.group.position ?? null,
    side: eye.side,
    stereoBaseline: depthStereoBaseline,
    eyeSeparation: eyeCameraSeparation,
  });
}

function renderEyeScene(eye) {
  resizeEyeRenderer(eye);
  eye.renderer.render(scene, eye.camera);
}

function resizeEyeRenderer(eye) {
  const rect = eye.view.getBoundingClientRect();
  const width = Math.max(1, Math.round(rect.width));
  const height = Math.max(1, Math.round(rect.height));
  const needsResize = eye.viewSize?.width !== width || eye.viewSize?.height !== height;
  if (needsResize) {
    eye.renderer.setSize(width, height, false);
    eye.viewSize = { width, height };
    eye.camera.aspect = width / height;
    eye.camera.updateProjectionMatrix();
  }
}

function objectDistanceToHand(item) {
  limb.hand.getWorldPosition(eyeHandPosition);
  return Math.hypot(
    item.group.position.x - eyeHandPosition.x,
    item.group.position.y - eyeHandPosition.y,
    item.group.position.z - eyeHandPosition.z,
  );
}

function objectProjectionInEye(item, eyeCamera) {
  visibleObjectBounds.makeEmpty();
  item.group.updateWorldMatrix(true, true);
  item.group.traverse((child) => {
    if (!isDetectionMesh(child)) return;
    if (!child.geometry.boundingBox) child.geometry.computeBoundingBox();
    meshWorldBounds.copy(child.geometry.boundingBox).applyMatrix4(child.matrixWorld);
    visibleObjectBounds.union(meshWorldBounds);
  });

  if (visibleObjectBounds.isEmpty()) return null;
  return projectWorldBoundsToFrame(visibleObjectBounds, eyeCamera);
}

function isDetectionMesh(child) {
  if (!child.isMesh || !child.visible || !child.geometry) return false;
  const materials = Array.isArray(child.material) ? child.material : [child.material];
  return !materials.every((material) => material?.transparent && material.opacity <= 0.001);
}

function setEye(eye, detection) {
  eye.box.style.opacity = "1";
  eye.box.style.left = `${detection.rect.x * 100}%`;
  eye.box.style.top = `${detection.rect.y * 100}%`;
  eye.box.style.width = `${detection.rect.width * 100}%`;
  eye.box.style.height = `${detection.rect.height * 100}%`;
  eye.box.className = `box ${detection.className}`;
  setText(eye.depth, `${detection.depth.toFixed(2)}m`);
  setText(eye.meta, `${detection.label} ${detection.conf}% ${detection.closingSpeed.toFixed(2)}m/s`);
}

function setEyeNoDetection(eye, detection) {
  eye.box.style.opacity = "0";
  eye.box.className = `box ${detection.className}`;
  setText(eye.depth, `${detection.depth.toFixed(2)}m`);
  setText(eye.meta, `${detection.label} ${detection.conf}% out of view`);
}

function setEyeEmpty(eye) {
  eye.box.style.opacity = "0";
  setText(eye.depth, "0.00m");
  setText(eye.meta, "no object");
}

function setEyeCameraOffline(eye) {
  eye.box.style.opacity = "0";
  setText(eye.depth, "OFF");
  setText(eye.meta, "camera off");
}

function updateSensorIndicators(sensors) {
  if (!sensorsEnabled) {
    for (const sensor of limb.sensors) {
      const material = sensor.material;
      material.color.setHex(0x7ee7bf);
      material.emissive.setHex(0x103b31);
      material.emissive.multiplyScalar(0.35);
      sensor.scale.setScalar(1);
    }
    return;
  }
  const painColor = new THREE.Color(0xff5f62);
  const pressureColor = new THREE.Color(0xffd16b);
  const clearColor = new THREE.Color(0x7ee7bf);
  for (const sensor of limb.sensors) {
    const material = sensor.material;
    material.color.copy(clearColor).lerp(pressureColor, sensors.pressure).lerp(painColor, sensors.pain);
    material.emissive.copy(material.color).multiplyScalar(0.35 + sensors.pain * 0.85);
    sensor.scale.setScalar(1 + sensors.pressure * 0.9 + sensors.pain * 0.55);
  }
}

function updateUi(item, decision, sensors) {
  setText(selectedLabel, `Object: ${item ? item.spec.label : "none"}`);
  riskMeter.value = decision.risk;
  confidenceMeter.value = decision.confidence;
  pressureMeter.value = sensors.pressure;
  painMeter.value = sensors.pain;
  if (!sensorsEnabled) {
    setText(sensorState, "Sensors: disabled");
  } else {
    setText(
      sensorState,
      sensors.painful
        ? "Sensors: pain reflex"
        : sensors.safeTouch
          ? "Sensors: low-pain contact"
          : sensors.approachingSensor
            ? `Sensors: closing ${sensors.closingSpeed.toFixed(2)}m/s`
            : sensors.contact > 0
              ? `Sensors: ${sensors.sensorName} touching ${sensors.contactName}`
              : "Sensors: clear",
    );
  }
  setText(statusAction, decision.action);
  setText(statusReason, decision.reason);
  renderMemory();
}

function renderMemory() {
  if (learnedMemory.entries.length === 0) {
    const signature = "empty";
    if (memoryRenderSignature === signature) return;
    memoryRenderSignature = signature;
    memoryList.replaceChildren(memoryRow("no learned contacts", "empty"));
    setText(memoryState, "Memory: none yet, persisted locally after contact");
    return;
  }

  const avoidCount = learnedMemory.entries.filter((entry) => entry.risk >= LEARNING_CONFIG.riskWithdrawalThreshold).length;
  const rows = learnedMemory.entries
    .slice()
    .sort((left, right) => right.lastUsed - left.lastUsed)
    .map((entry) => ({
      label: entry.label,
      detail: `${entry.risk >= LEARNING_CONFIG.riskWithdrawalThreshold ? "avoid" : "safe"} r${entry.risk.toFixed(2)} p${entry.painPeak.toFixed(2)} x${entry.contacts}`,
    }));
  const signature = JSON.stringify(rows);
  if (memoryRenderSignature === signature) {
    setText(memoryState, `Memory: ${learnedMemory.entries.length} learned, ${avoidCount} avoid, persisted locally`);
    return;
  }
  memoryRenderSignature = signature;
  setText(memoryState, `Memory: ${learnedMemory.entries.length} learned, ${avoidCount} avoid, persisted locally`);
  memoryList.replaceChildren(
    ...rows.map((row) => memoryRow(row.label, row.detail)),
  );
}

function setText(element, text) {
  if (!element) return;
  const value = String(text);
  if (uiTextCache.get(element) === value) return;
  element.textContent = value;
  uiTextCache.set(element, value);
}

function memoryRow(label, className) {
  const row = document.createElement("div");
  row.className = "memory-row";
  const name = document.createElement("span");
  name.textContent = label;
  const tag = document.createElement("strong");
  tag.textContent = className;
  row.append(name, tag);
  return row;
}

function resetLearnedMemory() {
  learnedMemory.reset();
  depthHistory.clear();
  contactLearningState.clear();
  selected = null;
  memoryRenderSignature = "";
  setText(memoryState, "Memory: reset");
  renderMemory();
}

function exportLearnedMemory() {
  const blob = new Blob([learnedMemory.exportJSON()], { type: "application/json" });
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = `limb-memory-${new Date().toISOString().replace(/[:.]/g, "-")}.json`;
  document.body.append(link);
  link.click();
  link.remove();
  URL.revokeObjectURL(url);
  setText(memoryState, "Memory: exported JSON");
}

function replayLastContact() {
  const latest = learnedMemory.entries.slice().sort((left, right) => right.lastUsed - left.lastUsed)[0];
  const objectKey = latest?.metadata?.objectKey;
  const item = draggableObjects.find((candidate) => candidate.spec.key === objectKey);
  if (!latest || !item) {
    setText(memoryState, "Memory: nothing to replay");
    return;
  }

  item.group.position.set(0.62, -0.39, THREE.MathUtils.clamp(item.group.position.z, -1.8, 1.8));
  if (limbPhysics.ready && limbPhysics.engine && item.group.userData.physics?.body) {
    limbPhysics.engine.setMeshPosition(item.group, {
      x: item.group.position.x,
      y: item.group.position.y,
      z: item.group.position.z,
    });
  }
  item.dragTarget.copy(item.group.position);
  selected = item;
  depthHistory.delete(item.spec.key);
  setText(memoryState, `Memory: replaying ${latest.label}`);
}

function resize() {
  const rect = canvas.getBoundingClientRect();
  const width = Math.max(1, rect.width);
  const height = Math.max(1, rect.height);
  renderer.setSize(width, height, false);
  camera.aspect = width / height;
  controls.update();
  camera.updateProjectionMatrix();
}

function objectScreenPosition(key) {
  const item = draggableObjects.find((candidate) => candidate.spec.key === key);
  if (!item) return null;
  return worldScreenPosition(item.group.position.x, item.group.position.y + 0.12, item.group.position.z);
}

function worldScreenPosition(x, y, z) {
  const rect = canvas.getBoundingClientRect();
  const projected = new THREE.Vector3(x, y, z).project(camera);
  return {
    x: rect.left + ((projected.x + 1) / 2) * rect.width,
    y: rect.top + ((-projected.y + 1) / 2) * rect.height,
  };
}
