Yes. You're describing the missing spatial-reasoning and state-estimation layer that connects visual understanding to physical motion. And you're right that it can be combined with SayCan-style skill feasibility and a stateful motion controller.

There are three important distinctions:

1. Knowing what an object is — visual recognition and semantic grounding.
2. Knowing where it is in robot coordinates — calibrated geometry and state estimation.
3. Knowing how to reach and manipulate it — reachability checks, IK, motion planning and control.

OneJev can contribute to the first and to high-level decisions. It should not be responsible for doing all three.

## 1. Can one RGB camera produce usable 3D coordinates?

Yes, under additional geometric assumptions. You do not necessarily need stereo cameras, a depth camera, or the dimensions of the entire room.

The main limitation is that a pixel corresponds to a ray through 3D space, not a unique 3D point.

Conceptual diagram: the pixel identifies a viewing ray. A known tabletop surface supplies the additional constraint needed to calculate a point.

The relationship is:

\\[ \lambda \begin{bmatrix}u\\\v\\\1\end{bmatrix} = K \begin{bmatrix}X_c\\\Y_c\\\Z_c\end{bmatrix} \\]

where \\(K\\) contains the camera's intrinsic calibration and \\(\lambda\\) is an unknown depth scale. Without another constraint, infinitely many 3D points could produce the same pixel.

### Three practical monocular methods

| Method                              | What must be known                                                                | What you obtain                                                               |
| ----------------------------------- | --------------------------------------------------------------------------------- | ----------------------------------------------------------------------------- |
| Ray–plane intersection / homography | Camera calibration and a known tabletop plane, or measured planar correspondences | 3D positions constrained to that plane                                        |
| SolvePnP + known object geometry    | Camera intrinsics and matching 2D image points ↔ known 3D points on an object     | The object's 6D pose relative to the camera                                   |
| ArUco marker / calibration board    | Marker dimensions, detected corners and camera intrinsics                         | Marker pose; with known robot-relative placement, camera-to-robot calibration |

OpenCV's `solvePnP` specifically estimates pose from known 3D-to-2D point correspondences. It does not infer arbitrary object depth from a single pixel.&#x20;

[image](https://www.google.com/s2/favicons?domain=https://docs.opencv.org\&sz=32)

docs.opencv.org



### What I'd use for your first real tabletop setup

A fixed USB camera, a calibrated tabletop and a measured camera-to-robot transform.

Then:

1. Detect the object and select its intended contact point.
2. Back-project that pixel to a camera ray.
3. Intersect the ray with the known tabletop plane.
4. Transform the resulting coordinates into the robot's base frame.
5. Account for object dimensions when constructing the grasp target.

One correction: for a ball, the detected image centre is not necessarily a point lying on the table. You must account for its radius, visible geometry or estimated contact point.

ArUco markers are useful for establishing the reference frame and scale during calibration. They do not have to remain attached to every object.

Where monocular vision becomes insufficient: reliably measuring unknown object heights, objects in free space, occluded geometry, or their exact pose during arbitrary 3D motion. Those situations need additional assumptions, changing viewpoints, tracking, known object geometry, or more sensing.

Your MuJoCo approach is well suited to learning this separation because the simulator can provide precise poses as a reference while you develop the image-based estimator independently.

## 2. Will all these calculations still run in real time?

For simple tabletop manipulation, generally yes. But you should not assume that CV + planning will always be faster than OneJev.

These computations have very different characteristics.

| Component                               | Computational behavior                              | Main difficulty                           |
| --------------------------------------- | --------------------------------------------------- | ----------------------------------------- |
| Camera calibration                      | Usually precomputed                                 | Keeping calibration valid                 |
| Object tracking / simple CV             | Often relatively inexpensive                        | Occlusion, appearance changes             |
| Learned object detection / segmentation | Depends on encoder, resolution and hardware         | Visual inference cost                     |
| Pixel → 3D geometry                     | Generally inexpensive after calibration             | Accuracy of the inputs                    |
| FK / local differential IK              | Often suitable for repeated control updates         | Singularities, limits, infeasible targets |
| Collision checking                      | Depends on geometry and number of candidate states  | Scene complexity                          |
| Global path planning (e.g. OMPL)        | Variable and potentially expensive                  | Clutter, narrow passages, difficult paths |
| OneJev decision                         | Depends on model, visual input and serving hardware | Multimodal inference                      |

Mink provides MuJoCo-based differential IK and collision-avoidance constraints suitable for local control. However, it does not guarantee that a global collision-free path exists or will be found.&#x20;

[image](https://www.google.com/s2/favicons?domain=https://github.com\&sz=32)

GitHub



In a simple scene, local geometry and IK might be much faster than OneJev. In a highly cluttered scene, a motion planner may require substantially more computation—or fail to find a valid path within its time budget.

The crucial architecture decision is that OneJev and the controller should not run at the same frequency.

OneJev

Skill decisions and replanning

Task state

Shared, versioned history

Perception + tracking

Fresh visual state

IK + motion control

Continuous local corrections

MuJoCo / physical robot

Execute, observe, update and correct

This pattern is well established. [MoveIt Servo](https://moveit.picknik.ai/main/doc/examples/realtime_servo/realtime_servo_tutorial.html?utm_source=chatgpt.com), for example, accepts joint or end-effector velocity/pose commands and supports collision proximity checks, motion smoothing and singularity handling in a continuous servoing architecture.&#x20;

[image](https://www.google.com/s2/favicons?domain=https://moveit.picknik.ai\&sz=32)

MoveIt Documentation: Rolling documentation



### There's also an alternative to full 3D reconstruction

You should study image-based visual servoing (IBVS).

Instead of always computing a complete 3D target pose, IBVS uses the error between current and desired image features to guide local motion. It may use approximate depth and camera/robot Jacobians without requiring a complete reconstructed scene.

This is particularly interesting for a monocular camera and a gripper aligning with an object.

Read [Peter Corke's visual-servoing chapter](https://petercorke.com/rvc2/v-visual-servoing/15-visual-servoing/?utm_source=chatgpt.com) and the [Intel visual-servoing overview](https://eci.intel.com/embodied-sdk-docs/content/developer_tools_tutorials/model_tutorials/model_cns.html?utm_source=chatgpt.com).&#x20;

[image](https://www.google.com/s2/favicons?domain=https://petercorke.com\&sz=32)

Peter Corke

+1



I would use 3D target-based planning for larger movements and investigate visual servoing for fine alignment near the object. IBVS does not eliminate the need for collision awareness or the underlying robot kinematics.

## 3. Your proposed shared state is exactly where SayCan becomes useful

I would extend your earlier state design into a world-state estimator, sometimes called a belief state when it explicitly represents uncertainty.

It combines visual observations, robot kinematics, object estimates, task progress and recent execution events.

But each subsystem should consume only the fields relevant to its responsibilities.

| Consumer                   | State it needs                                                                                        |
| -------------------------- | ----------------------------------------------------------------------------------------------------- |
| OneJev                     | Images, instruction, object identity, task progress, last skill results, uncertainty                  |
| Feasibility / SayCan layer | Candidate skills, preconditions, grasp availability, reachability, predicted execution feasibility    |
| IK / motion planner        | Joint positions, target poses, coordinate frames, obstacles, collision geometry, relevant uncertainty |
| Skill executor             | Selected trajectory or target, limits, controller state, contact/retention evidence                   |
| Evaluator                  | Actual simulator ground truth, observed outcomes, task success predicates                             |

That last distinction is especially important in simulation: the evaluator can know exactly where every object is, while OneJev is restricted to permitted visual and state inputs. Your implementation plan already explicitly preserves that separation.&#x20;

visual_skill_agent_implementation_plan.pdf



### A concrete state representation

```
observation:
  timestamp: 1842
  camera: overhead
  age_ms: 32

objects:
  ball_3:
    visible: true
    position_base: [0.24, 0.01, 0.03]
    pose_source: monocular_plane_geometry
    uncertainty_m: 0.005
    tracking_status: valid

robot:
  joint_positions: [...]
  gripper_state: open
  end_effector_pose: [...]

task:
  goal: ball_3_inside_bowl_1
  active_skill: approach
  last_skill_result: success
  grasp_confirmed: false

world:
  obstacles: [...]
  planning_scene_version: 42
```

Illustrative values only. In the real system, every coordinate should carry a defined frame and measurement/estimation provenance.

The motion planner shouldn't blindly trust the position estimate just because it is present. For example, it could reject a target if uncertainty exceeds its configured tolerance.

### How SayCan integrates

Google's original SayCan architecture combines two different judgments:

- Usefulness: Is the skill appropriate for accomplishing the goal?
- Affordance / feasibility: Is the skill likely to succeed in the current state?

The original formulation combines language-model skill scores with skill-specific value functions estimating success.&#x20;

[image](https://www.google.com/s2/favicons?domain=https://say-can.github.io\&sz=32)

say-can.github.io



We could construct an analogous design:

\\[ \operatorname{score}(a) = P\_{\text{OneJev}}(a\mid o,g,m) \cdot P\_{\text{skill-success}}(a\mid \hat{s}) \\]

where \\(o\\) is the observation, \\(g\\) the goal, \\(m\\) task memory, and \\(\hat{s}\\) the estimated physical state.

For your initial implementation, the second quantity does not need to be learned. You can start with deterministic predicates such as `reachable`, `grasp_available`, and `path_valid`. Those are binary feasibility checks rather than calibrated success probabilities.

Later, you could learn skill-success estimates from actual simulation outcomes.

Importantly, OneJev's output probabilities and a separately learned affordance model must be validated before multiplying them as meaningful calibrated probabilities. The original SayCan formulation motivates the architecture; it does not automatically validate OneJev's probabilities.

## 4. Is the MotionRequest / MotionSolver design grounded in Google's approaches?

Architecturally, yes. But the exact interface is our proposed software contract, not a published Google API.

[Gemini Robotics-ER 1.5](https://developers.googleblog.com/en/building-the-next-generation-of-physical-agents-with-gemini-robotics-er-15/?utm_source=chatgpt.com) explicitly describes finding object points in images and combining those with 3D sensor information so a motion-planning library can plan movements.&#x20;

[image](https://www.google.com/s2/favicons?domain=https://developers.googleblog.com\&sz=32)

Google Developers Blog



The other strong precedent is MoveIt's Planning Scene architecture, which maintains a representation of the robot, environment geometry and collision objects for motion planning.&#x20;

[image](https://www.google.com/s2/favicons?domain=https://moveit.picknik.ai\&sz=32)

MoveIt Documentation: Rolling documentation



Our interface should follow the same separation:

```
Visual observations
      ↓
State estimation + object tracking
      ↓
Planning scene (3D estimates, obstacles, robot state)
      ↓
OneJev proposes grasp(ball_3)
      ↓
SayCan-style feasibility assessment
      ↓
MotionRequest:
    target_id: ball_3
    target_pose: estimated grasp pose
    frame: robot_base
    world_version: 42
    permitted_contact: gripper ↔ ball_3
      ↓
MotionSolver: IK + collision checks + path
      ↓
Skill executor
      ↓
New observation + execution result
      ↓
Update world and task state
```

The most important addition is `world_version`.

If the ball moves, or an obstacle enters the planned path, the old motion request may no longer be valid. The executor should stop, locally correct or request replanning depending on the disturbance.

This is also why you need a persistent state manager even if OneJev itself does not remember earlier decisions.

## 5. What changes in a dynamic environment?

A static-world pipeline can plan once and execute while monitoring progress.

A dynamic-world pipeline must continually check whether the plan remains valid.

For example:

```
t0: Ball detected at position A.
t1: OneJev selects grasp(ball_3).
t2: Planner computes approach toward A.
t3: Ball moves to position B.
t4: Tracker updates ball position.
t5: Local controller corrects or cancels the approach.
t6: Replan if the original path/skill is no longer valid.
```

You do not need to ask OneJev to choose `grasp(ball_3)` again merely because the ball moved slightly. The skill intent can remain valid while the low-level target is updated.

But if the ball becomes unreachable, another object blocks the approach, or the task itself changes, the high-level selector may need to reconsider its choice.

This is the separation between semantic replanning and geometric replanning.

And it is precisely what makes a stateful, hierarchical architecture attractive for your project.

## 6. One extra opportunity for your camera setup

You previously mentioned having two ordinary USB cameras.

Two USB cameras can potentially be calibrated into a stereo system, provided they have overlapping views, a known relative transform, and adequate temporal synchronization. They do not need to be sold as a dedicated stereo-depth product.

Alternatively, one moving wrist camera can provide multiple viewpoints with known camera poses derived from robot kinematics. Static scene points can then be triangulated across observations.&#x20;

[image](https://www.google.com/s2/favicons?domain=https://docs.opencv.org\&sz=32)

docs.opencv.org



Both routes have practical limitations, especially with moving objects and unsynchronized frames, but they're worth investigating before purchasing additional sensors.

## What I would preserve from your current architecture

I would keep OneJev, the separate skill library, MuJoCo, the independent evaluator and the state manager. The main change is making spatial state estimation and planning-scene maintenance explicit rather than hiding them inside each skill.

Then the full system becomes:

Visual understanding → estimated world state → OneJev skill selection → SayCan-style feasibility → motion planning / servoing → execution → verification → state update.

For your simulation experiments, MuJoCo can provide ground-truth poses to the controller while you establish motion feasibility, then you can replace that input with noisy estimates or monocular geometry and quantify the degradation.

That is a grounded architecture using concepts from SayCan, Gemini Robotics-ER, MoveIt and visual servoing, without requiring an end-to-end VLA.

The three resources I'd prioritize now are [SayCan](https://say-can.github.io/?utm_source=chatgpt.com), [MoveIt Planning Scene Monitor](https://moveit.picknik.ai/main/doc/concepts/planning_scene_monitor.html?utm_source=chatgpt.com), and [Peter Corke's Visual Servoing](https://petercorke.com/rvc2/v-visual-servoing/15-visual-servoing/?utm_source=chatgpt.com). Together they explain the connections between goal reasoning, world-state representation and local motion correction.