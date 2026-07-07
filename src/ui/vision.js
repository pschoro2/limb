import * as THREE from "three";

const eyeCameraPosition = new THREE.Vector3();
const eyeCameraTarget = new THREE.Vector3();
const eyeCameraDirection = new THREE.Vector3();
const projectionCenter = new THREE.Vector3();
const projectedCorner = new THREE.Vector3();

export function configureEyeCamera(camera, {
  handPosition,
  itemPosition = null,
  side = 1,
  stereoBaseline = 0.18,
  eyeSeparation = Math.max(stereoBaseline * 3.4, 0.62),
  targetSeparation = eyeSeparation * 0.42,
}) {
  const eyeOffset = side * eyeSeparation * 0.5;
  eyeCameraPosition.set(
    handPosition.x + 1.65,
    handPosition.y + 0.95,
    handPosition.z + eyeOffset,
  );
  camera.position.copy(eyeCameraPosition);

  const targetOffset = side * targetSeparation * 0.5;
  if (itemPosition) {
    eyeCameraTarget.set(
      THREE.MathUtils.lerp(handPosition.x, itemPosition.x, 0.58),
      -0.24,
      THREE.MathUtils.lerp(handPosition.z, itemPosition.z, 0.38) + targetOffset,
    );
  } else {
    eyeCameraTarget.set(handPosition.x - 2.2, -0.28, handPosition.z + targetOffset);
  }

  camera.lookAt(eyeCameraTarget);
  camera.updateMatrixWorld();
  return {
    position: camera.position.clone(),
    target: eyeCameraTarget.clone(),
  };
}

export function projectWorldBoundsToFrame(bounds, camera, {
  minSize = 0.08,
  maxSize = 0.96,
  padding = 0.015,
} = {}) {
  bounds.getCenter(projectionCenter);
  camera.getWorldDirection(eyeCameraDirection);
  projectedCorner.copy(projectionCenter).sub(camera.position);
  if (eyeCameraDirection.dot(projectedCorner) <= 0) return null;

  let minX = Infinity;
  let minY = Infinity;
  let maxX = -Infinity;
  let maxY = -Infinity;

  for (const x of [bounds.min.x, bounds.max.x]) {
    for (const y of [bounds.min.y, bounds.max.y]) {
      for (const z of [bounds.min.z, bounds.max.z]) {
        projectedCorner.set(x, y, z).project(camera);
        if (!Number.isFinite(projectedCorner.x) || !Number.isFinite(projectedCorner.y)) continue;
        minX = Math.min(minX, projectedCorner.x);
        minY = Math.min(minY, projectedCorner.y);
        maxX = Math.max(maxX, projectedCorner.x);
        maxY = Math.max(maxY, projectedCorner.y);
      }
    }
  }

  if (!Number.isFinite(minX) || maxX < -1 || minX > 1 || maxY < -1 || minY > 1) return null;

  const left = THREE.MathUtils.clamp((minX - padding + 1) / 2, 0, 1);
  const right = THREE.MathUtils.clamp((maxX + padding + 1) / 2, 0, 1);
  const top = THREE.MathUtils.clamp((1 - maxY - padding) / 2, 0, 1);
  const bottom = THREE.MathUtils.clamp((1 - minY + padding) / 2, 0, 1);
  let width = THREE.MathUtils.clamp(right - left, minSize, maxSize);
  let height = THREE.MathUtils.clamp(bottom - top, minSize, maxSize);
  const x = THREE.MathUtils.clamp((left + right) / 2, width / 2, 1 - width / 2);
  const y = THREE.MathUtils.clamp((top + bottom) / 2, height / 2, 1 - height / 2);

  if (height > width * 1.6) height = width * 1.6;
  if (width > height * 2.1) width = height * 2.1;
  return { x, y, width, height };
}
