import assert from "node:assert/strict";
import test from "node:test";

import * as THREE from "three";

import { configureEyeCamera, projectWorldBoundsToFrame } from "../../src/ui/vision.js";

test("scene eye projection makes the cube grow as it moves toward the arm", () => {
  const leftEye = new THREE.PerspectiveCamera(58, 16 / 9, 0.05, 22);
  const hand = new THREE.Vector3(0.95, -0.27, 0);
  const farCube = cubeBounds(-3.1, 0);
  const nearCube = cubeBounds(0.7, 0);

  configureEyeCamera(leftEye, {
    handPosition: hand,
    itemPosition: new THREE.Vector3(-3.1, -0.39, 0),
    side: -1,
  });
  const farDetection = projectWorldBoundsToFrame(farCube, leftEye);

  configureEyeCamera(leftEye, {
    handPosition: hand,
    itemPosition: new THREE.Vector3(0.7, -0.39, 0),
    side: -1,
  });
  const nearDetection = projectWorldBoundsToFrame(nearCube, leftEye);

  assert.ok(farDetection);
  assert.ok(nearDetection);
  assert.ok(nearDetection.width > farDetection.width * 1.5);
  assert.ok(nearDetection.height > farDetection.height * 2);
});

test("left and right YOLO eyes project the same cube from different viewpoints", () => {
  const leftEye = new THREE.PerspectiveCamera(58, 16 / 9, 0.05, 22);
  const rightEye = new THREE.PerspectiveCamera(58, 16 / 9, 0.05, 22);
  const hand = new THREE.Vector3(0.95, -0.27, 0);
  const cubePosition = new THREE.Vector3(0.7, -0.39, 0);
  const cube = cubeBounds(cubePosition.x, cubePosition.z);

  const leftRig = configureEyeCamera(leftEye, { handPosition: hand, itemPosition: cubePosition, side: -1 });
  const rightRig = configureEyeCamera(rightEye, { handPosition: hand, itemPosition: cubePosition, side: 1 });

  const leftDetection = projectWorldBoundsToFrame(cube, leftEye);
  const rightDetection = projectWorldBoundsToFrame(cube, rightEye);

  assert.ok(leftDetection);
  assert.ok(rightDetection);
  assert.ok(leftRig.position.distanceTo(rightRig.position) >= 0.62);
  assert.ok(Math.abs(leftRig.target.z - rightRig.target.z) > 0.2);
  assert.ok(Math.abs(leftDetection.x - rightDetection.x) > 0.05);
  assert.equal(leftDetection.width.toFixed(3), rightDetection.width.toFixed(3));
});

function cubeBounds(x, z) {
  return new THREE.Box3(
    new THREE.Vector3(x - 0.24, -0.39, z - 0.24),
    new THREE.Vector3(x + 0.24, 0.03, z + 0.24),
  );
}
