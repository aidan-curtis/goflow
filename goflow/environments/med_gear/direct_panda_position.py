from __future__ import annotations

# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
This script demonstrates different single-arm manipulators.

.. code-block:: bash

    # Usage
    ./isaaclab.sh -p source/standalone/demos/arms.py

"""

"""Launch Isaac Sim Simulator first."""
import argparse
import os
from typing import List
from dataclasses import MISSING
import numpy as np
import gymnasium as gym

import omni.isaac.lab.utils.math as math_utils
import omni.kit.commands

from omni.isaac.lab.managers import RewardTermCfg
from omni.isaac.lab.managers import TerminationTermCfg as DoneTerm
from omni.isaac.lab.controllers import DifferentialIKControllerCfg
from omni.isaac.lab.envs import DirectRLEnv, DirectRLEnvCfg
from omni.isaac.lab.assets import ArticulationCfg, AssetBaseCfg
from omni.isaac.lab.assets.rigid_object import RigidObjectCfg
from goflow.utils import solve_ik
from omni.isaac.lab.controllers.differential_ik import DifferentialIKController
from omni.isaac.core.utils.string import find_unique_string_name
import omni.isaac.core.utils.prims as prim_utils
from pxr import UsdPhysics
from pxr import Gf, Sdf
"""Rest everything follows."""

import torch
import omni.isaac.core.utils.stage as stage_utils
import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.assets import Articulation, RigidObject, AssetBase
from omni.isaac.lab.actuators import ImplicitActuatorCfg
from omni.isaac.lab.assets.articulation import ArticulationCfg
from omni.isaac.lab.managers import SceneEntityCfg
from omni.isaac.lab.utils import configclass
import omni.isaac.lab.envs.mdp as mdp
from typing import Sequence
from dataclasses import dataclass
from scipy.spatial.transform import Rotation as R
from omni.isaac.lab.sim import SimulationCfg
from goflow.utils import log_list, task_from_name, ROOT_DIR, wxyz_to_xyzw, xyzw_to_wxyz
import goflow.pb_utils as pbu
from omni.isaac.lab.sensors import TiledCameraCfg, Camera, TiledCamera
from PIL import Image
import time
from omni.isaac.lab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from omni.isaac.lab.scene import InteractiveSceneCfg
from collections import deque

from . import agents

TOOL_BODY = "panda_hand"

NUM_ENVS = 1024
# NUM_ENVS = 10

INITIAL_CFG = task_from_name("med_gear_in_taskboard")
# INITIAL_CFG = task_from_name("large_gear_in_taskboard")
# INITIAL_CFG = task_from_name("small_gear_in_taskboard")
# INITIAL_CFG = task_from_name("rod_in_gear")
# INITIAL_CFG = task_from_name("strut_in_elbow")
# INITIAL_CFG = task_from_name("elbow_in_platform")
# INITIAL_CFG = task_from_name("strut_in_platform")
# INITIAL_CFG = task_from_name("bolt_in_strut")
# INITIAL_CFG = task_from_name("bolt_in_platform")
# INITIAL_CFG = task_from_name("fmb_example")
# INITIAL_CFG = task_from_name("wrench_in_bearing")

USE_CAMERA = False
IMAGE_WIDTH = 60
IMAGE_HEIGHT = 60

ARM_JOINTS = ['panda_joint1', 'panda_joint2', 'panda_joint3', 'panda_joint4', 'panda_joint5', 'panda_joint6', 'panda_joint7', 'panda_finger_joint1', 'panda_finger_joint2']

FORCE_LIMIT_TRANS = 50
VELOCITY_LIMIT_TRANS = 20
VELOCITY_LIMIT_ROT = 1
TORQUE_LIMIT = 1
H = 10
SIM_DT = 1/120
SUBSTEPS = 1
DECIMATION = 5
UNIT_QUAT = torch.tensor([[1, 0, 0, 0]]).type(torch.FloatTensor).cuda()
UNIT_POINT = torch.zeros(1,3).type(torch.FloatTensor).cuda()
POSE_SIZE = 9
FORCE_SIZE = 6

# New hyperparameter for the number of pose histories
NUM_POSE_HISTORY = 10

@dataclass
class IPose():
    pos: torch.tensor
    quat: torch.tensor

    def multiply(self, p:IPose):
        return IPose(*math_utils.combine_frame_transforms(self.pos, self.quat, p.pos, p.quat))
    
    def invert(self):
        t12, q12 = math_utils.subtract_frame_transforms(self.pos, self.quat)
        return IPose(t12, q12)
    
    def to_vec(self):
        return torch.cat([self.pos, self.quat], dim=1)

    def to(self, device):
        return IPose(self.pos.to(device), self.quat.to(device))
    
    @staticmethod
    def from_pose(pose):
        pos, quat = pose
        return IPose(torch.Tensor([pos]).type(torch.FloatTensor).cuda(),
                     torch.Tensor([xyzw_to_wxyz(quat)]).type(torch.FloatTensor).cuda())
    
    def repeat(self, n):
        return IPose(self.pos.repeat((n, 1)), self.quat.repeat((n, 1)))

EFFORT_CFGS = {"robot1": SceneEntityCfg("robot1", body_names=[TOOL_BODY], joint_names=ARM_JOINTS)}
ROBOT1_CAMERA_CFG = SceneEntityCfg("tiled_camera")

def pose_from_state(state, body_id, origins):
    ee_pos_w = state[:, body_id, :3] - origins
    ee_quat_w = state[:, body_id, 3:7]
    ipose = IPose(ee_pos_w, ee_quat_w)
    return ipose

def save_depth(depth_data):
    depth_min = np.min(depth_data)
    depth_max = np.max(depth_data)
    normalized_depth = ((depth_data - depth_min) / (depth_max - depth_min) * 255).astype(np.uint8)

    # Create a PIL Image from the numpy array
    image = Image.fromarray(normalized_depth, mode='L')

    # Save the image as PNG
    image.save('runs/depth_image_{}.png'.format(str(time.time())))


def compute_frame_pose(asset, body_idx, offset_pos, offset_rot) -> tuple[torch.Tensor, torch.Tensor]:
    """Computes the pose of the target frame in the root frame.

    Returns:
        A tuple of the body's position and orientation in the root frame.
    """
    # obtain quantities from simulation
    ee_pose_w = asset.data.body_state_w[:, body_idx, :7]
    root_pose_w = asset.data.root_state_w[:, :7]
    # compute the pose of the body in the root frame
    ee_pose_b, ee_quat_b = math_utils.subtract_frame_transforms(
        root_pose_w[:, 0:3], root_pose_w[:, 3:7], ee_pose_w[:, 0:3], ee_pose_w[:, 3:7]
    )

    ee_pose_b, ee_quat_b = math_utils.combine_frame_transforms(
        ee_pose_b, ee_quat_b, offset_pos, offset_rot
    )
    return ee_pose_b, ee_quat_b


NUM_POSES = POSE_SIZE*INITIAL_CFG.robot_count if not INITIAL_CFG.relative_observations else POSE_SIZE
NUM_OBS = NUM_POSES if not INITIAL_CFG.force_sensing else POSE_SIZE+FORCE_SIZE*INITIAL_CFG.robot_count

DEFAULT_R0_CONF = [0.0, -0.569, 0.0, -2.810, 0.0, 3.037, 0.741, 0.04, 0.04]
@configclass
class MyPandaEnvCfg(DirectRLEnvCfg):
    """Configuration for the cartpole environment."""
    
    # Scene settings
    episode_length_s = 2
    decimation = DECIMATION*SUBSTEPS
    sim = SimulationCfg(dt=SIM_DT/SUBSTEPS)
    scene = InteractiveSceneCfg(num_envs=NUM_ENVS, env_spacing=4)
    num_actions = 3*INITIAL_CFG.robot_count + (3*int(INITIAL_CFG.allow_hole_rotation)) + (3*int(INITIAL_CFG.allow_peg_rotation))
    num_observations = NUM_OBS + 6 + NUM_POSE_HISTORY * POSE_SIZE  # Add NUM_POSE_HISTORY * POSE_SIZE for historical poses
    # trans_bounds = 0.02
    trans_bounds = 0.02

    rot_bounds = np.pi
    dr_ranges = {
        "yaw_offset": (-rot_bounds, rot_bounds),
        "x_offset": (-trans_bounds, trans_bounds),
        "y_offset": (-trans_bounds, trans_bounds),
        # "gear_mass": (0.8, 1.2),  # Add mass scale to DR ranges
    }
    num_states = num_observations+len(dr_ranges)

    
    def __post_init__(self):


        # strong robot
        lower_arm_effort = 100
        upper_arm_effort = 100
        arm_stiffness = 400.0
        arm_damping = 800.0


        # Original panda from isaaclab
        # lower_arm_effort = 87.0
        # upper_arm_effort = 12.0
        # arm_stiffness = 80.0
        # arm_damping = 4.0


        arm_vel_limit = 20.0
        self.parts = []
        self.peripherals = []
        self.articulated_parts = []
        
        if(INITIAL_CFG.moving_peg):
                        
            self.robot1 = ArticulationCfg(
                spawn=sim_utils.UsdFileCfg(
                    usd_path=os.path.join(ROOT_DIR, "models/franka.usd"),
                    activate_contact_sensors=False,
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(
                        disable_gravity=True,
                        max_depenetration_velocity=5.0,
                    ),
                    articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                        enabled_self_collisions=True, solver_position_iteration_count=8, solver_velocity_iteration_count=0
                    ),
                    # collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.005, rest_offset=0.0),
                ),
                init_state=ArticulationCfg.InitialStateCfg(
                    joint_pos={
                        name: value for  name, value in zip(ARM_JOINTS, DEFAULT_R0_CONF)
                    },
                ),
                actuators={
                    "panda_shoulder": ImplicitActuatorCfg(
                        joint_names_expr=["panda_joint[1-4]"],
                        effort_limit=lower_arm_effort,
                        velocity_limit=2.175,
                        stiffness=arm_stiffness,
                        damping=arm_damping,
                    ),
                    "panda_forearm": ImplicitActuatorCfg(
                        joint_names_expr=["panda_joint[5-7]"],
                        effort_limit=upper_arm_effort,
                        velocity_limit=2.61,
                        stiffness=arm_stiffness,
                        damping=arm_damping,
                    ),
                    "panda_hand": ImplicitActuatorCfg(
                        joint_names_expr=["panda_finger_joint.*"],
                        effort_limit=200.0,
                        velocity_limit=0.2,
                        stiffness=2e3,
                        damping=1e2,
                    ),
                },
                soft_joint_pos_limit_factor=1.0,
            ).replace(prim_path="/World/envs/env_.*/robot1")

        self.peripherals.append(
            AssetBaseCfg(
                prim_path="/World/envs/env_.*/table",  
                spawn = sim_utils.UsdFileCfg(
                    usd_path=os.path.join(ROOT_DIR, "models/franka_table_top_collision.usd"), 
                )
            )
        )


        world_T_peg_start = IPose.from_pose(INITIAL_CFG.WORLD_T_PEG_START)

        if(INITIAL_CFG.PEG_TYPE == "part"):
            self.parts.append(
                RigidObjectCfg( 
                    prim_path="/World/envs/env_.*/peg",  
                    spawn=sim_utils.UsdFileCfg(
                        usd_path = INITIAL_CFG.PEG_ASSET_NAME,
                        rigid_props=sim_utils.RigidBodyPropertiesCfg(
                            disable_gravity=True
                        )
                    ),
                    init_state=RigidObjectCfg.InitialStateCfg(pos=world_T_peg_start.pos[0], 
                                                            rot=world_T_peg_start.quat[0]),
                )
            )
        elif(INITIAL_CFG.PEG_TYPE == "peripheral"):
            self.peripherals.append(
                AssetBaseCfg(
                    prim_path="/World/envs/env_.*/peg",
                    spawn = sim_utils.UsdFileCfg(
                        usd_path=INITIAL_CFG.PEG_ASSET_NAME,
                    ),
                    
                    init_state = AssetBaseCfg.InitialStateCfg(world_T_peg_start.pos[0], world_T_peg_start.quat[0])
                )
            )
        else:
            raise NotImplementedError

            
        world_T_hole_start = IPose.from_pose(INITIAL_CFG.WORLD_T_HOLE_START)

        if(INITIAL_CFG.HOLE_TYPE == "peripheral"):
            self.peripherals.append(
                AssetBaseCfg(
                    prim_path="/World/envs/env_.*/hole",
                    spawn = sim_utils.UsdFileCfg(
                        usd_path=INITIAL_CFG.HOLE_ASSET_NAME,
                    ),
                    
                    init_state = AssetBaseCfg.InitialStateCfg(world_T_hole_start.pos[0], world_T_hole_start.quat[0])
                )
            )
        else:
            raise NotImplementedError

        if(USE_CAMERA):
            camera_pos = list(INITIAL_CFG.WORLD_T_PEG_START[0])
            camera_pos[0] -= 0.2

            self.tiled_camera = TiledCameraCfg(
                prim_path="/World/envs/env_.*/camera",
                offset=TiledCameraCfg.OffsetCfg(pos=camera_pos, rot=(0.9945, 0.0, 0.1045, 0.0), convention="world"),
                data_types=["depth"],
                spawn=sim_utils.PinholeCameraCfg(
                    focal_length=24.0,  # Keep this as is, it's actually good for close-up
                    focus_distance=0.1,  # Set to 0.1 meters (about 4 inches)
                    horizontal_aperture=20.955,  # Keep this as is
                    clipping_range=(0.01, 1.0),  # Near plane at 1cm, far plane at 2m
                ),
                width=IMAGE_WIDTH,
                height=IMAGE_HEIGHT,
            ) 
        # Add extra parts
        for ei, (extra_path, extra_pose) in enumerate(zip(INITIAL_CFG.EXTRA_PARTS, INITIAL_CFG.EXTRA_PARTS_STARTING_POSE)):
            ipose = IPose.from_pose(extra_pose)
            self.peripherals.append(
                AssetBaseCfg(
                    prim_path="/World/envs/env_.*/extra"+str(ei),
                    spawn = sim_utils.UsdFileCfg(
                        usd_path=extra_path,
                    ),
                    init_state = AssetBaseCfg.InitialStateCfg(ipose.pos[0], ipose.quat[0])
                )
            )

        # Initialize the gripper to be directly above the target object pose
        # This is where the object is initialized to be reset later inside the fingers

        tool_poses = [None, None]

        if(INITIAL_CFG.moving_peg):
            tool_poses[0] = pbu.multiply(INITIAL_CFG.WORLD_T_PEG_START, INITIAL_CFG.PEG_T_TOOL)
        
        ik_solutions = solve_ik([DEFAULT_R0_CONF], tool_poses)
        ik_solutions[0][-2:] = DEFAULT_R0_CONF[-2:]

        if(INITIAL_CFG.moving_peg):
            for joint_name, joint_angle in zip(ARM_JOINTS, ik_solutions[0]):
                self.robot1.init_state.joint_pos[joint_name] = joint_angle
            

class MyPandaEnv(DirectRLEnv):
    cfg: MyPandaEnvCfg

    cnn: bool = USE_CAMERA
    image_width: int = IMAGE_WIDTH
    image_height: int = IMAGE_HEIGHT

    def __init__(self, cfg: MyPandaEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)


        # self.trans_action_scale = 0.01
        default_scale = 0.05
        trans_scale = 0.05
    
        self.default_action_scale = torch.tensor([default_scale, default_scale, default_scale]).cuda()
        self.trans_action_scale = torch.tensor([trans_scale, trans_scale, trans_scale]).cuda()
        self.rot_action_scale = torch.tensor([0.0, 0.0, 0.0]).cuda()
        self.fixed_quat = {}
        self._ik_controllers = {}
        self._joint_ids = {}  
        self._jacobi_body_idx = {}
        self._body_idx = {}

        for robot_name, robot_asset in self.robots.items():
            self.robots[robot_name] = robot_asset
             # resolve the joints over which the action term is applied
            self._joint_ids[robot_name], _ = robot_asset.find_joints(ARM_JOINTS)
            self._num_joints = len(self._joint_ids[robot_name])

            # parse the body index
            body_ids, _ = robot_asset.find_bodies(TOOL_BODY)
            
            # check if articulation is fixed-base
            # if fixed-base then the jacobian for the base is not computed
            # this means that number of bodies is one less than the articulation's number of bodies
            self._body_idx[robot_name] = body_ids[0]
            if robot_asset.is_fixed_base:
                self._jacobi_body_idx[robot_name] = body_ids[0] - 1
            else:
                self._jacobi_body_idx[robot_name] = body_ids[0]

            # Avoid indexing across all joints for efficiency
            if self._num_joints == robot_asset.num_joints:
                self._joint_ids[robot_name] = slice(None)

            ik_controller_cfg = DifferentialIKControllerCfg(command_type="pose", use_relative_mode=False, ik_method="dls")
            self._ik_controllers[robot_name] = DifferentialIKController(
                cfg=ik_controller_cfg, num_envs=NUM_ENVS, device=self.sim.device
            )
        
        self.context = torch.zeros(self.num_envs, len(self.cfg.dr_ranges), device=self.sim.device)


        if(INITIAL_CFG.moving_peg):
            self.robot1_fixed_joints = self.create_rigid_attachments(robot_path="robot1", attachment_path = INITIAL_CFG.PEG_ATTACHMENT_PRIM, extra_attachments=INITIAL_CFG.EXTRA_ATTACHMENTS)
        
        # Initialize previous pose for velocity calculation
        self.prev_pose = {robot_name: None for robot_name in self.robots.keys()}
        
        # Adjust observation space size
        # self.cfg.num_observations += 6  # 3 for linear velocity, 3 for angular velocity

        # Initialize pose history buffer
        self.pose_history = deque(maxlen=NUM_POSE_HISTORY)
        self.pose_history_size = NUM_POSE_HISTORY * POSE_SIZE

    def update_rigid_attachments(self, robot_name, fixed_joints, obj_T_flange:IPose, rigid_object, env_ids):
        
        cpu_env_ids = env_ids.cpu()
        if(hasattr(self, 'dist')):
            context = self.dist.rsample([cpu_env_ids.shape[0]]).to(self.sim.device).clone()
        else:
            # Uniform sampling for context if no distribution is set
            context = torch.zeros(cpu_env_ids.shape[0], len(self.cfg.dr_ranges), device=self.sim.device)
            for i, (key, (low, high)) in enumerate(self.cfg.dr_ranges.items()):
                context[:, i] = torch.rand(cpu_env_ids.shape[0], device=self.sim.device) * (high - low) + low
        
        self.context[cpu_env_ids, :] = context.clone()

        # Domain randomization using context
        dr_keys = list(self.cfg.dr_ranges.keys())
        
        # Initialize translation and rotation offsets with zeros
        translation_offset = torch.zeros(self.context.shape[0], 3, device=self.context.device)
        rotation_offset = torch.zeros(self.context.shape[0], 3, device=self.context.device)
        
        # Fill in the offsets based on available DR keys
        for i, axis in enumerate(['x_offset', 'y_offset', 'z_offset']):
            if axis in dr_keys:
                translation_offset[:, i] = self.context[:, dr_keys.index(axis)]
        
        # for i, axis in enumerate(['roll_offset', 'pitch_offset', 'yaw_offset']):
        #     if axis in dr_keys:
        #         rotation_offset[:, i] = self.context[:, dr_keys.index(axis)]
        
        # Apply mass scaling if it's in the DR ranges
        if 'gear_mass' in dr_keys:
            mass_key = list(self.get_dr_ranges().keys()).index("gear_mass")
            masses = self.scene.rigid_objects["peg"].root_physx_view.get_masses().clone()
            mass_tensor = torch.tensor(context[:, mass_key]).cpu()
            body_ids, body_names = self.scene.rigid_objects["peg"].find_bodies(".*")
            masses[cpu_env_ids, body_ids[0]] = mass_tensor
            self.scene.rigid_objects["peg"].root_physx_view.set_masses(masses, cpu_env_ids)

        
        # Convert rotation offset to quaternion
        rotation_offset_quat = math_utils.quat_from_euler_xyz(
            rotation_offset[:, 0],
            rotation_offset[:, 1],
            rotation_offset[:, 2]
        )

        for env_id, fixed_joint in enumerate(fixed_joints):

            trans_pose = IPose(pos = translation_offset[env_id:env_id+1, :], quat = UNIT_QUAT)
            rot_pose = IPose(pos = UNIT_POINT, quat=rotation_offset_quat[env_id:env_id+1, :])

            flange_T_obj = trans_pose.multiply(obj_T_flange.invert()).multiply(rot_pose)
            
            # set into the physics simulation
            fixed_joint.GetLocalPos0Attr().Set(Gf.Vec3f(*(flange_T_obj.pos[0].detach().cpu().numpy().tolist())))
            fixed_joint.GetLocalRot0Attr().Set(Gf.Quatf(*(flange_T_obj.quat[0].detach().cpu().numpy().tolist())))
            fixed_joint.GetLocalPos1Attr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
            fixed_joint.GetLocalRot1Attr().Set(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
        
        return fixed_joint

    def create_rigid_attachments(self, attachment_path:str, robot_path:str, extra_attachments=[]):

        fixed_joints = []
        for env_id in range(NUM_ENVS):

            env0_path = f"/World/envs/env_{env_id}/{robot_path}"
            fixed_joint_path = env0_path + "/AssemblerFixedJoint"
            fixed_joint_path = find_unique_string_name(fixed_joint_path, lambda x: not prim_utils.is_prim_path_valid(x))
            stage = stage_utils.get_current_stage()
            fixed_joint = UsdPhysics.FixedJoint.Define(stage, fixed_joint_path)

            target0 = env0_path+"/franka/panda_hand"        
            target1 = f"/World/envs/env_{env_id}/{attachment_path}"

            if target0 is not None:
                fixed_joint.GetBody0Rel().SetTargets([target0])
            if target1 is not None:
                fixed_joint.GetBody1Rel().SetTargets([target1])

            
            fixed_joints.append(fixed_joint)

            for ex_id, (attachment_a, attachment_b) in enumerate(extra_attachments):
                ex_path = f"/World/envs/env_{env_id}/ExFixedJoint_{ex_id}"
                stage = stage_utils.get_current_stage()
                fixed_joint = UsdPhysics.FixedJoint.Define(stage, ex_path)
                if attachment_a is not None:
                    fixed_joint.GetBody0Rel().SetTargets([f"/World/envs/env_{env_id}/{attachment_a}"])
                if attachment_b is not None:
                    fixed_joint.GetBody1Rel().SetTargets([f"/World/envs/env_{env_id}/{attachment_b}"])


        return fixed_joints
    
    @property
    def robots(self):
        return {k:v for k, v in self.scene.articulations.items() if "robot" in k}

    @property
    def articulated_parts(self):
        return {k:v for k, v in self.scene.articulations.items() if "robot" not in k}

    def _setup_scene(self):
        

        if(INITIAL_CFG.moving_peg):
            robot1 = Articulation(self.cfg.robot1) # _asset


        rigid_bodies = [RigidObject(part) for part in self.cfg.parts]
        articulated_parts = [Articulation(apart) for apart in self.cfg.articulated_parts]

        for asset_cfg in self.cfg.peripherals:
            asset_cfg.spawn.func(
                asset_cfg.prim_path,
                asset_cfg.spawn,
                translation=asset_cfg.init_state.pos,
                orientation=asset_cfg.init_state.rot,
            )

        # add ground plane
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())
        
        # clone, filter, and replicate
        self.scene.clone_environments(copy_from_source=False)
        self.scene.filter_collisions(global_prim_paths=[])

        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

        if(INITIAL_CFG.moving_peg):
            self.scene.articulations["robot1"] = robot1
            
        for rigid_body in rigid_bodies:
            self.scene.rigid_objects[rigid_body.cfg.prim_path.split("/")[-1]] = rigid_body

        for articulated_part in articulated_parts:
            self.scene.articulations[articulated_part.cfg.prim_path.split("/")[-1]] = articulated_part

        if(USE_CAMERA):
            self.scene.sensors["camera"] = TiledCamera(self.cfg.tiled_camera)


    def update_dr(self, env_ids: Sequence[int] | None, dr_samples):
        self.dr_samples = dr_samples

    def _reset_idx(self, env_ids: Sequence[int] | None):
        super()._reset_idx(env_ids)

        # if len(env_ids) == self.num_envs:
        #     # Spread out the resets to avoid spikes in training when many environments reset at a similar time
        #     self.episode_length_buf[:] = torch.randint_like(self.episode_length_buf, high=int(self.max_episode_length))

        
        mdp.reset_scene_to_default(self, env_ids)
        # mdp.reset_joints_by_offset(self, env_ids, [-0.025, 0.025], [0, 0], EFFORT_CFGS["robot1"])

        if(INITIAL_CFG.moving_peg):
            self.update_rigid_attachments("robot1", self.robot1_fixed_joints, obj_T_flange = IPose.from_pose(INITIAL_CFG.PEG_T_TOOL), rigid_object=self.scene.rigid_objects["peg"], env_ids=env_ids)
        
        self.step_sim()
        
        if(INITIAL_CFG.moving_peg):
            peg_obj = self.scene.rigid_objects["peg"]
            self.world_T_peg_start = IPose(peg_obj.data.root_state_w[:, :3]-self.scene.env_origins, peg_obj.data.root_state_w[:, 3:7])
        

        self.initialized = False
        for robot_name, robot_asset in self.robots.items():
            _, self.fixed_quat[robot_name] = self._compute_frame_pose(robot_name, robot_asset)

        # Reset previous pose for velocity calculation
        for robot_name in self.robots.keys():
            if self.prev_pose[robot_name] is None:
                self.prev_pose[robot_name] = self._compute_frame_pose(robot_name, self.robots[robot_name])
            else:
                pos, quat = self._compute_frame_pose(robot_name, self.robots[robot_name])
                if env_ids is not None:
                    self.prev_pose[robot_name][0][env_ids] = pos[env_ids]
                    self.prev_pose[robot_name][1][env_ids] = quat[env_ids]
                else:
                    self.prev_pose[robot_name] = (pos, quat)

        # Reset pose history buffer
        if env_ids is None:
            self.pose_history.clear()
        else:
            # Remove poses for reset environments
            self.pose_history = deque([pose for i, pose in enumerate(self.pose_history) if i not in env_ids], maxlen=NUM_POSE_HISTORY)

    def step_sim(self):
        for robot_name, robot_asset in self.robots.items():
            _joint_ids, _ = robot_asset.find_joints(ARM_JOINTS)
            current_joint_pos = robot_asset.data.joint_pos[:, _joint_ids]
            robot_asset.set_joint_position_target(current_joint_pos, joint_ids=_joint_ids)
            robot_asset.set_joint_velocity_target(0, joint_ids=_joint_ids)
            self.scene.write_data_to_sim()
            self.sim.step(render=True)
            self.scene.update(dt=self.physics_dt)
        
    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        log_list(f"isaac_action", actions)

        num_r1_actions = 3*(int(INITIAL_CFG.allow_peg_rotation)+int(INITIAL_CFG.moving_peg))
        if(INITIAL_CFG.moving_peg):
            self._pre_physics_step_sr(actions[:, :num_r1_actions], "robot1", self.robots["robot1"])

    def _pre_physics_step_sr(self, actions, robot_name, robot_asset):
        # Calculate the default action (moving towards the goal)
        default_action = self._compute_default_action(robot_name, robot_asset)
        
        # Prepare the processed actions
        processed_actions = torch.zeros_like(default_action)
        
        # Add the residual action from the policy to the translational part
        processed_actions[:, :3] = default_action[:, :3] * self.default_action_scale + actions[:, :3] * self.trans_action_scale
        rotation_enabled = INITIAL_CFG.allow_peg_rotation

        
        if rotation_enabled:
            # Add the residual action to the rotational part if rotation is enabled
            processed_actions[:, 3:] = default_action[:, 3:] + actions[:, 3:]
            processed_actions[:, 3:] *= self.rot_action_scale

        delta = torch.zeros(NUM_ENVS, 6, device=self.sim.device)
        delta[:, :3] = processed_actions[:, :3]
        
        if rotation_enabled:
            delta[:, 3:6] = processed_actions[:, 3:6]
            delta_rotm = math_utils.matrix_from_euler(delta[:, 3:6], "XYZ")
            delta_rotation = math_utils.quat_from_matrix(delta_rotm)
            
        tool_trans, tool_quat = self._compute_frame_pose(robot_name, robot_asset)
        log_list(f"{robot_name}_isaac_tool_quat", tool_quat)

        # Target absolute position
        control_cmd = torch.zeros(self.num_envs, 7).to(self.sim.device)
        
        control_cmd[:, :3] = tool_trans + delta[:, :3]

        if rotation_enabled:
            control_cmd[:, 3:7] = math_utils.quat_mul(tool_quat, delta_rotation)
        else:
            control_cmd[:, 3:7] = self.fixed_quat[robot_name]

        # obtain quantities from simulation
        ee_pos_curr, ee_quat_curr = self._compute_frame_pose(robot_name, robot_asset)

        # set command into controller
        self._ik_controllers[robot_name].set_command(control_cmd, ee_pos_curr, ee_quat_curr)
        log_list(f"{robot_name}_isaac_cmd", control_cmd)

        self.extras['context'] = self.context.clone()

    def _compute_default_action(self, robot_name, robot_asset):
        # Compute the current pose
        current_pos, current_quat = self._compute_frame_pose(robot_name, robot_asset)
        
        # Compute the goal pose
        if INITIAL_CFG.relative_goal:
            world_T_hole = IPose.from_pose(INITIAL_CFG.WORLD_T_HOLE_START).repeat(NUM_ENVS)
            des_hole_T_peg = IPose.from_pose(INITIAL_CFG.HOLE_T_PEG_GOAL).repeat(NUM_ENVS)
            goal_pose = world_T_hole.multiply(des_hole_T_peg)
        else:
            goal_pose = IPose.from_pose(INITIAL_CFG.PEG_GOAL).repeat(NUM_ENVS)
        
        # Compute the difference between current and goal pose
        pos_diff = goal_pose.pos - current_pos
        quat_diff = math_utils.quat_mul(goal_pose.quat, math_utils.quat_conjugate(current_quat))
        
        # Convert quaternion difference to axis-angle representation
        axis_angle = math_utils.axis_angle_from_quat(quat_diff)
        
        # Combine position and rotation differences
        default_action = torch.cat([pos_diff, axis_angle], dim=1)
        
        # Normalize the action
        default_action = default_action / (torch.norm(default_action, dim=1, keepdim=True) + 1e-8)
        
        return default_action

    def _apply_action(self) -> None:
        for robot_name, robot_asset in self.robots.items():
            # obtain quantities from simulation
            ee_pos_curr, ee_quat_curr = self._compute_frame_pose(robot_name, robot_asset)
            joint_pos = robot_asset.data.joint_pos[:, self._joint_ids[robot_name]]

            # compute the delta in joint-space
            jacobian = self._compute_frame_jacobian(robot_name, robot_asset)
            joint_pos_des = self._ik_controllers[robot_name].compute(ee_pos_curr, ee_quat_curr, jacobian, joint_pos)

            error = torch.abs(joint_pos-joint_pos_des)
            log_list(f"{robot_name}_joint_error", error)

            # set the joint position command
            robot_asset.set_joint_position_target(joint_pos_des, self._joint_ids[robot_name])

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        return time_out, time_out
    
    def ee_pos(self, asset_cfg: SceneEntityCfg):
        """The joint velocities of the asset w.r.t. the default joint velocities.

        Note: Only the joints configured in :attr:`asset_cfg.joint_ids` will have their velocities returned.
        """
        # extract the used quantities (to enable type-hinting)
        asset: Articulation = self.scene[asset_cfg.name]
        
        # obtain quantities from simulation
        body_ids, _ = asset.find_bodies(TOOL_BODY)
        flange_pose = pose_from_state(asset.data.body_state_w, body_ids[0], self.scene.env_origins)
        return flange_pose
    
    def flatten_pose(self, pose: IPose):
        ee_rotm = math_utils.matrix_from_quat(pose.quat)
        total_obs = torch.cat([pose.pos, ee_rotm[:, :3, 0], ee_rotm[:, :3, 1]], axis=1)
        return total_obs
    
    def depth_image(self, tiled_camera: SceneEntityCfg):
        camera_image = tiled_camera.data.output["depth"].clone()
        
        # Save the first camera image for debugging purposes
        # save_depth(torch.squeeze(camera_image[0, :, :]).cpu().numpy())    
        # import sys
        # sys.exit()

        return camera_image


    def get_ee_force(self, robot_name):
        jacobian = self.robots[robot_name]._root_physx_view.get_jacobians()[:, self._jacobi_body_idx[robot_name], :, self._joint_ids[robot_name]]
        jacobian_T = torch.transpose(jacobian, dim0=1, dim1=2)
        
        joint_torques = self.robots[robot_name]._root_physx_view.get_dof_projected_joint_forces()[:, self._joint_ids[robot_name]]
        end_effector_forces = jacobian_T @ joint_torques.unsqueeze(-1)
        end_effector_forces = end_effector_forces.squeeze(-1)
        
        ee_pose = self.ee_pos(EFFORT_CFGS[robot_name])
        linear_force = math_utils.quat_apply(ee_pose.quat, end_effector_forces[:, :3])
        angular_force = math_utils.quat_apply(ee_pose.quat, end_effector_forces[:, 3:])
        return torch.cat((linear_force, angular_force), dim=1)
        
    def _get_observations(self) -> dict:

        all_obs = []
        

        if(INITIAL_CFG.moving_peg):
            ee_pose = self.ee_pos(EFFORT_CFGS["robot1"])
            flattened_pose = self.flatten_pose(ee_pose)
            all_obs.append(flattened_pose)

            # Add current pose to history
            self.pose_history.appendleft(flattened_pose)

            # Compute and add velocity to observations
            linear_velocity, angular_velocity = self._compute_velocity("robot1")
            all_obs.append(linear_velocity)
            all_obs.append(angular_velocity)
        
        
        if(INITIAL_CFG.force_sensing):
            if(INITIAL_CFG.moving_peg):
                ee_force = self.get_ee_force("robot1")
                all_obs.append(ee_force)

        # Add historical poses
        historical_poses = torch.cat(list(self.pose_history), dim=1)
        padding_size = self.pose_history_size - historical_poses.shape[1]
        if padding_size > 0:
            padding = torch.zeros(historical_poses.shape[0], padding_size, device=self.sim.device)
            historical_poses = torch.cat([historical_poses, padding], dim=1)
        all_obs.append(historical_poses)

        obs_noisy = torch.cat(all_obs, dim=1)

        # noise = torch.randn_like(obs) * 0.001
        # obs_noisy = obs + noise

        
        log_list(f"isaac_obs", obs_noisy)
        observations = {"policy": obs_noisy, "critic": torch.cat([obs_noisy, self.context.to(obs_noisy.device)], dim=1)}
        # if(USE_CAMERA):
        #     observations["image"] = self.depth_image(self.scene.sensors["camera"])
        
        return observations
    
    def _compute_frame_pose(self, asset_name, asset) -> tuple[torch.Tensor, torch.Tensor]:
        """Computes the pose of the target frame in the root frame.

        Returns:
            A tuple of the body's position and orientation in the root frame.
        """
        # obtain quantities from simulation
        ee_pos_w = asset.data.body_state_w[:, self._body_idx[asset_name], :3]-self.scene.env_origins
        ee_quat_w = asset.data.body_state_w[:, self._body_idx[asset_name], 3:7]
        return ee_pos_w.clone(), ee_quat_w.clone()


    def _compute_frame_jacobian(self, asset_name, asset):
        """Computes the geometric Jacobian of the target frame in the root frame.

        This function accounts for the target frame offset and applies the necessary transformations to obtain
        the right Jacobian from the parent body Jacobian.
        """
        # read the parent jacobian
        jacobian = asset.root_physx_view.get_jacobians()[:, self._jacobi_body_idx[asset_name], :, self._joint_ids[asset_name]]
        return jacobian
    
    def insert_relative_pose(self) -> torch.Tensor:    
        """Reward the agent for reaching the object using tanh-kernel."""
        peg_distance = None
        hole_distance = None


        if(INITIAL_CFG.moving_peg):
            world_T_peg = IPose(self.scene["peg"].data.root_state_w[:, :3]-self.scene.env_origins, self.scene["peg"].data.root_state_w[:, 3:7])
        else:
            world_T_peg = IPose.from_pose(INITIAL_CFG.WORLD_T_PEG_START).repeat(NUM_ENVS)

        if(INITIAL_CFG.moving_hole):
            world_T_hole = IPose(self.scene["hole"].data.root_state_w[:, :3]-self.scene.env_origins, self.scene["hole"].data.root_state_w[:, 3:7])
        else:
            world_T_hole = IPose.from_pose(INITIAL_CFG.WORLD_T_HOLE_START).repeat(NUM_ENVS)
                    
        if(INITIAL_CFG.relative_goal):
            hole_T_peg = world_T_hole.invert().multiply(world_T_peg)
            des_hole_T_peg = IPose.from_pose(INITIAL_CFG.HOLE_T_PEG_GOAL)
            peg_distance =  des_hole_T_peg.repeat(NUM_ENVS).multiply(hole_T_peg.invert())
        else:
            if(INITIAL_CFG.moving_peg):
                des_world_T_peg =  IPose.from_pose(INITIAL_CFG.PEG_GOAL)
                peg_distance = des_world_T_peg.repeat(NUM_ENVS).multiply(world_T_peg.invert())

            if(INITIAL_CFG.moving_hole):
                des_world_T_hole =  IPose.from_pose(INITIAL_CFG.HOLE_GOAL)
                hole_distance = des_world_T_hole.repeat(NUM_ENVS).multiply(world_T_hole.invert())

        distance = 0
        for (di, (pose_distance, weights)) in enumerate([(peg_distance, INITIAL_CFG.peg_goal_weights), (hole_distance, INITIAL_CFG.hole_goal_weights)]):
            if(pose_distance is not None):
                diff_pos_b = pose_distance.pos
                diff_rot_b = math_utils.axis_angle_from_quat(pose_distance.quat)
                weights = torch.tensor(weights).type(torch.FloatTensor).cuda()
                diff_pose =  torch.cat([diff_pos_b, diff_rot_b], dim=1)*weights
                distance += torch.norm(diff_pose, dim=1)
        
        reward = torch.clip(0.01/distance, -10.0, 10.0)
        log_list("reward", reward)
        return reward


    def set_sampling_dist(self, dist: torch.distributions.Distribution):
        # Uniform distribution 
        self.dist = dist
        
    def get_dr_ranges(self):
        return self.cfg.dr_ranges

    def _get_rewards(self) -> torch.Tensor:
        return self.insert_relative_pose()

    def _compute_velocity(self, robot_name):
        current_pos, current_quat = self._compute_frame_pose(robot_name, self.robots[robot_name])
        prev_pos, prev_quat = self.prev_pose[robot_name]

        # Compute linear velocity
        linear_velocity = (current_pos - prev_pos) / self.physics_dt
        
        # Handle NaN and inf values in linear velocity
        linear_velocity = torch.nan_to_num(linear_velocity, nan=0.0, posinf=1e3, neginf=-1e3)

        # Compute angular velocity
        quat_diff = math_utils.quat_mul(current_quat, math_utils.quat_conjugate(prev_quat))
        angle_diff = 2 * torch.acos(torch.clamp(quat_diff[:, 0], -1.0, 1.0))
        
        # Avoid division by zero
        sin_angle_diff = torch.sin(angle_diff / 2)
        mask = sin_angle_diff.abs() > 1e-6
        axis = torch.zeros_like(quat_diff[:, 1:])
        axis[mask] = quat_diff[mask, 1:] / sin_angle_diff[mask].unsqueeze(1)
        
        angular_velocity = axis * (angle_diff / self.physics_dt).unsqueeze(1)
        
        # Handle NaN and inf values in angular velocity
        angular_velocity = torch.nan_to_num(angular_velocity, nan=0.0, posinf=1e3, neginf=-1e3)

        # Update previous pose
        self.prev_pose[robot_name] = (current_pos, current_quat)

        return linear_velocity, angular_velocity



for dr_method in ["NoDR", "FullDR", "LSDR", "ADR", "GOFLOW", "DORAEMON"]:
    gym.register(
        id=f"Gears-{dr_method}-v0",
        entry_point="goflow.environments.med_gear.direct_panda_position:MyPandaEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": MyPandaEnvCfg,
            "rl_games_cfg_entry_point": f"{agents.__name__}:{dr_method}.yaml",
        },
    )