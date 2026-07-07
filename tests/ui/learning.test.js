import assert from "node:assert/strict";
import test from "node:test";

import {
  EmbodiedMemory,
  LEARNING_CONFIG,
  chooseAction,
  estimateDepthFromDisparity,
  readTactileSensors,
  shouldDodgeReflex,
} from "../../src/ui/learning.js";

const pointyCue = [0.98, 0.18, 0.1, 0.96, 0.08, 0.82];
const smoothCue = [0.2, 1.0, 0.96, 0.02, 0.92, 0.08];
const noContactSensors = {
  contact: 0,
  pressure: 0,
  puncture: 0,
  pain: 0,
  safeTouch: false,
  painful: false,
  approachingSensor: false,
  closingSpeed: 0,
  distance: Infinity,
};

test("pointy object is not avoided before tactile pain is learned", () => {
  const memory = new EmbodiedMemory();

  const decision = chooseAction({
    itemX: 0.8,
    cueVector: pointyCue,
    confidence: 0.78,
    sensors: noContactSensors,
    memory,
  });

  assert.equal(decision.action, "CAUTIOUS_PROBE");
  assert.equal(memory.entries.length, 0);
});

test("painful fingertip contact creates avoidance memory for the next step", () => {
  const memory = new EmbodiedMemory();
  const sensors = readTactileSensors({
    handPosition: { x: 1, z: 0 },
    objectPosition: { x: 1.06, z: 0.02 },
    tactileProfile: { tipSharpness: 0.96, compliance: 0.08, contactArea: 0.12 },
  });

  assert.equal(sensors.painful, true);

  const reflex = chooseAction({
    itemX: 1.2,
    cueVector: pointyCue,
    confidence: 0.86,
    sensors,
    memory,
  });
  assert.equal(reflex.action, "WITHDRAW_FAST");

  memory.recordExperience({
    cueVector: pointyCue,
    label: "novel pointy object",
    action: reflex.action,
    sensors,
    time: 1,
  });

  const nextStep = chooseAction({
    itemX: 0.7,
    cueVector: pointyCue,
    confidence: 0.86,
    sensors: noContactSensors,
    memory,
  });

  assert.equal(nextStep.action, "WITHDRAW_FAST");
  assert.match(nextStep.reason, /maximum avoidance/);
  assert.equal(memory.entries.length, 1);
  assert.ok(memory.entries[0].risk >= LEARNING_CONFIG.riskWithdrawalThreshold);
});

test("first pain memory requires the fifty percent threshold", () => {
  const memory = new EmbodiedMemory(LEARNING_CONFIG, { load: false });
  const belowThreshold = {
    contact: 0.6,
    pressure: 0.5,
    puncture: 0.5,
    pain: 0.49,
    safeTouch: false,
    painful: false,
    approachingSensor: false,
    closingSpeed: 0,
    distance: 0.4,
  };
  const atThreshold = { ...belowThreshold, pain: 0.5, painful: true };

  const warningEntry = memory.recordExperience({
    cueVector: pointyCue,
    label: "below threshold",
    action: "CAUTIOUS_PROBE",
    sensors: belowThreshold,
    time: 1,
  });
  assert.ok(warningEntry.risk < 0.85);

  const painEntry = memory.recordExperience({
    cueVector: pointyCue,
    label: "at threshold",
    action: "WITHDRAW_FAST",
    sensors: atThreshold,
    time: 2,
  });

  assert.ok(painEntry.risk >= 0.85);
  assert.equal(painEntry.painPeak, 0.5);
});

test("one strong pain poke makes later visual recall withdraw fast before contact", () => {
  const memory = new EmbodiedMemory();
  const painfulSensors = readTactileSensors({
    handPosition: { x: 1, z: 0 },
    objectPosition: { x: 1.03, z: 0.01 },
    tactileProfile: { tipSharpness: 0.96, compliance: 0.08, contactArea: 0.12 },
  });

  memory.recordExperience({
    cueVector: pointyCue,
    label: "pointy cue",
    action: "WITHDRAW_FAST",
    sensors: painfulSensors,
    time: 1,
  });

  const visualOnly = chooseAction({
    itemX: 0.35,
    cueVector: pointyCue,
    confidence: 0.88,
    sensors: noContactSensors,
    memory,
  });

  assert.equal(visualOnly.action, "WITHDRAW_FAST");
  assert.match(visualOnly.reason, /maximum avoidance/);
  assert.ok(memory.entries[0].risk >= 0.85);
  assert.equal(memory.entries[0].contacts, 1);
});

test("learned danger only gets a dodge reflex above ten meters per second", () => {
  const memory = new EmbodiedMemory();
  const painfulSensors = readTactileSensors({
    handPosition: { x: 1, z: 0 },
    objectPosition: { x: 1.03, z: 0.01 },
    tactileProfile: { tipSharpness: 0.96, compliance: 0.08, contactArea: 0.12 },
  });

  memory.recordExperience({
    cueVector: pointyCue,
    label: "pointy cue",
    action: "WITHDRAW_FAST",
    sensors: painfulSensors,
    time: 1,
  });

  const decision = chooseAction({
    itemX: 0.7,
    cueVector: pointyCue,
    confidence: 0.88,
    sensors: noContactSensors,
    memory,
  });
  assert.equal(decision.action, "WITHDRAW_FAST");

  assert.equal(
    shouldDodgeReflex(decision, {
      ...noContactSensors,
      approachingSensor: true,
      closingSpeed: LEARNING_CONFIG.dodgeClosingSpeedThreshold,
    }),
    false,
  );
  assert.equal(
    shouldDodgeReflex(decision, {
      ...noContactSensors,
      approachingSensor: true,
      closingSpeed: LEARNING_CONFIG.dodgeClosingSpeedThreshold + 0.01,
    }),
    true,
  );
});

test("two warning contacts can cross the learned avoidance threshold", () => {
  const memory = new EmbodiedMemory();
  const warningSensors = {
    contact: 0.36,
    pressure: 0.45,
    puncture: 0.3,
    pain: LEARNING_CONFIG.warningPainThreshold,
    safeTouch: false,
    painful: false,
    approachingSensor: false,
    closingSpeed: 0,
    distance: 0.52,
  };

  memory.recordExperience({
    cueVector: pointyCue,
    label: "warning cue",
    action: "CAUTIOUS_PROBE",
    sensors: warningSensors,
    time: 1,
  });
  assert.ok(memory.entries[0].risk < LEARNING_CONFIG.riskWithdrawalThreshold);

  memory.recordExperience({
    cueVector: pointyCue,
    label: "warning cue",
    action: "CAUTIOUS_PROBE",
    sensors: warningSensors,
    time: 2,
  });

  const visualOnly = chooseAction({
    itemX: 0.35,
    cueVector: pointyCue,
    confidence: 0.82,
    sensors: noContactSensors,
    memory,
  });

  assert.equal(memory.entries[0].contacts, 2);
  assert.ok(memory.entries[0].risk >= LEARNING_CONFIG.riskWithdrawalThreshold);
  assert.equal(visualOnly.action, "RETRACT");
});

test("low-pain contact becomes a safe reaching memory", () => {
  const memory = new EmbodiedMemory();
  const sensors = readTactileSensors({
    handPosition: { x: 1, z: 0 },
    objectPosition: { x: 1.05, z: 0 },
    tactileProfile: { tipSharpness: 0.01, compliance: 0.72, contactArea: 0.94 },
  });

  assert.equal(sensors.safeTouch, true);
  assert.equal(sensors.painful, false);

  memory.recordExperience({
    cueVector: smoothCue,
    label: "smooth object",
    action: "CAUTIOUS_PROBE",
    sensors,
    time: 1,
  });

  const nextStep = chooseAction({
    itemX: 1.2,
    cueVector: smoothCue,
    confidence: 0.84,
    sensors: noContactSensors,
    memory,
  });

  assert.equal(nextStep.action, "GENTLE_TOUCH");
  assert.match(nextStep.reason, /Learned low-pain contact/);
  assert.ok(memory.entries[0].valence > 0);
});

test("depth perception detects an object moving toward the fingertip sensors", () => {
  const sensors = readTactileSensors({
    handPosition: { x: 1, z: 0 },
    objectPosition: { x: 1.42, z: 0 },
    previousDistance: 0.74,
    deltaSeconds: 0.5,
    tactileProfile: { tipSharpness: 0.1, compliance: 0.4, contactArea: 0.8 },
  });

  assert.equal(sensors.approachingSensor, true);
  assert.ok(sensors.closingSpeed >= LEARNING_CONFIG.depthClosingThreshold);

  const decision = chooseAction({
    itemX: 0.8,
    cueVector: pointyCue,
    confidence: 0.82,
    sensors,
    memory: new EmbodiedMemory(),
  });

  assert.equal(decision.action, "CAUTIOUS_PROBE");
  assert.match(decision.reason, /Depth cue/);
});

test("tactile sensing uses the closest fingertip and contact point", () => {
  const sensors = readTactileSensors({
    sensorPoints: [
      { x: 0, z: -0.5, name: "finger left" },
      { x: 1, z: 0.1, name: "finger center" },
    ],
    contactPoints: [
      { x: 1.06, z: 0.12, name: "blade tip", sharpnessScale: 1.1, areaScale: 0.7 },
      { x: 0, z: 0.8, name: "handle", sharpnessScale: 0.2, areaScale: 1.2 },
    ],
    tactileProfile: { tipSharpness: 0.9, compliance: 0.1, contactArea: 0.2 },
  });

  assert.equal(sensors.sensorName, "finger center");
  assert.equal(sensors.contactName, "blade tip");
  assert.equal(sensors.painful, true);
});

test("tactile sensing is sensitive to vertical separation", () => {
  const levelContact = readTactileSensors({
    handPosition: { x: 1, y: 0, z: 0 },
    objectPosition: { x: 1.06, y: 0, z: 0.01 },
    tactileProfile: { tipSharpness: 0.96, compliance: 0.08, contactArea: 0.12 },
  });
  const raisedContact = readTactileSensors({
    handPosition: { x: 1, y: 0.45, z: 0 },
    objectPosition: { x: 1.06, y: 0, z: 0.01 },
    tactileProfile: { tipSharpness: 0.96, compliance: 0.08, contactArea: 0.12 },
  });
  const loweredContact = readTactileSensors({
    handPosition: { x: 1, y: -0.45, z: 0 },
    objectPosition: { x: 1.06, y: 0, z: 0.01 },
    tactileProfile: { tipSharpness: 0.96, compliance: 0.08, contactArea: 0.12 },
  });

  assert.ok(raisedContact.contact < levelContact.contact);
  assert.ok(loweredContact.contact < levelContact.contact);
});

test("memory persists through storage and can be reset", () => {
  const storage = fakeStorage();
  const memory = new EmbodiedMemory(LEARNING_CONFIG, { storage, storageKey: "test-memory" });
  const sensors = readTactileSensors({
    handPosition: { x: 1, z: 0 },
    objectPosition: { x: 1.03, z: 0.01 },
    tactileProfile: { tipSharpness: 0.96, compliance: 0.08, contactArea: 0.12 },
  });

  memory.recordExperience({
    cueVector: pointyCue,
    label: "stored point",
    action: "WITHDRAW_FAST",
    sensors,
    time: 1,
    metadata: { objectKey: "knife" },
  });

  const reloaded = new EmbodiedMemory(LEARNING_CONFIG, { storage, storageKey: "test-memory" });
  assert.equal(reloaded.entries.length, 1);
  assert.equal(reloaded.entries[0].metadata.objectKey, "knife");

  reloaded.reset();
  assert.equal(storage.getItem("test-memory"), null);
});

test("stereo disparity estimates nearer depth as disparity grows", () => {
  const far = estimateDepthFromDisparity(0.08);
  const near = estimateDepthFromDisparity(0.18);

  assert.ok(near < far);
});

function fakeStorage() {
  const values = new Map();
  return {
    getItem(key) {
      return values.has(key) ? values.get(key) : null;
    },
    setItem(key, value) {
      values.set(key, String(value));
    },
    removeItem(key) {
      values.delete(key);
    },
  };
}
