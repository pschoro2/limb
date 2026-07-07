# Limb YOLO Memory Prototype

This project implements a simulation-first version of the requested architecture:

```text
camera/frame
  -> YOLO detector adapter
  -> rich cue signature
  -> deterministic spike trace
  -> flash + consolidated memory retrieval
  -> valence and uncertainty gate
  -> hard safety supervisor
  -> symbolic motor primitive
```

The core rule is preserved: YOLO identifies and localizes an object, but memory and safety policy decide what that object means for the limb.

## What Is Implemented

- Optional Ultralytics YOLO adapter with a testable detector protocol.
- Cue vectors that combine class probabilities, detection confidence, normalized bbox geometry, depth, tracked motion, crop embedding, current limb pose, and previous action.
- Deterministic rate-coded spike traces for memory/action path recording.
- Flash and consolidated memory entries with cue signature, spike trace, action, valence, risk, timestamps, strength, and source.
- Nearest-neighbor retrieval with top-match similarity, top-2 margin, novelty, and ambiguity handling.
- A fixed emergency supervisor for non-learned stop conditions, with normal approach/avoid choices driven by memory valence.
- A dry-run motor controller boundary where ROS 2 / MoveIt integration can be attached later.
- Symbolic motor primitive selection that learns from tactile pressure, pain, and depth-closing cues.

## Quick Test

```bash
python3 -m pytest
```

Run the 3D visual simulator:

```bash
npm install
npm run dev -- --port 4173
```

Then open `http://127.0.0.1:4173/`. The scene has draggable 3D objects on the left, the limb on the right, and two side-by-side scene-backed YOLO panels for stereo eyesight. Move a fork or knife toward the limb to see withdrawal; move a safe object like a spoon, cup, ball, or block to see approach/touch behavior.

The visual simulator starts with no safe/danger memory. Drag an object into the fingertip/palm sensors and the end effector reads pressure, puncture-like pain, depth, and closing speed. Low-pain contact creates a positive memory. The first contact that reaches 50% pain creates a high-risk negative memory immediately; two weaker warning contacts can also cross the avoidance threshold. When the same visual/tactile cue appears again, vision/depth recall triggers maximum-avoidance fast withdrawal before waiting for another painful touch.

The tactile model uses individual fingertip/palm sensor points against object contact points such as tines, blade tips, rounded surfaces, rims, and corners. You can also adjust object height during drag by holding `Shift`, `Ctrl`, `Alt`, or `⌘` (or using the mouse wheel) while dragging a selected object. Learned memory is saved in browser `localStorage`, so refreshes keep the learned reflex. The side panel includes controls to reset memory, export memory JSON, and replay the latest learned contact. `WITHDRAW_FAST` uses a faster joint impulse than normal retract. On a learned high-risk return, the arm gives the pointier object a wider sidestep even when approach speed is lower, with an amplified dodge when closing fast.

Verify the browser UI:

```bash
npm run build
npm run test:ui
```

For non-browser learning checks only:

```bash
npm run test:learning
```

Run the simulation demo:

```bash
python3 examples/simulate_spoon_fork.py
```

## Optional YOLO Use

Install the optional dependency when you want real image inference:

```bash
python3 -m pip install -e ".[yolo]"
```

Then create a detector:

```python
from limb.detector import UltralyticsYOLODetector

detector = UltralyticsYOLODetector(confidence=0.4)
detections = detector.predict(frame)
```

The detector is local-model only by default. It looks for `yolo5s.pt`, then `yolov5s.pt`, in the current working directory before importing Ultralytics. If neither file is present, it raises `FileNotFoundError` instead of allowing a model download. The current workspace contains `yolov5s.pt`.

The browser UI does not load or download a YOLO model. Its two YOLO panels render the same 3D experience board from stereo eye cameras, with detection boxes, depth, and closing speed toward the end-effector sensors.

The rest of the pipeline accepts the detector output without depending on Ultralytics, which keeps simulation and safety tests fast.

## Safety Boundary

The learned memory system never owns hard safety. `SafetySupervisor` runs before the valence gate can emit an approach or touch command. Emergency stop, force spikes, collisions, high force, human stop, and sensor faults are handled by fixed policy.
