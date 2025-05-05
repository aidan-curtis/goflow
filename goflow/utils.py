
import json
import logging
from typing import List, Tuple, Dict
import os
import time
import pathlib
import numpy as np
from dataclasses import dataclass, field
import random
import pybullet as p
import numpy as np 
import math
from pathlib import Path
import numpy as np
import torch
import goflow.pb_utils as pbu
import time
import copy
import pybullet_utils.bullet_client as bc

# from ur_ikfast import ur_kinematics

BASE_LINK = -1
ROOT_DIR = Path(__file__).parent.parent

# ur10e_arm = ur_kinematics.URKinematics('ur10e')

def wxyz_to_xyzw(q):
    return [q[1], q[2], q[3], q[0]]

def xyzw_to_wxyz(q):
    return [q[3], q[0], q[1], q[2]]


R0_POSE = ((0,0,0), (0,0,0,1))
IGNORE_COLLISIONS = [(6, 9), (11, 13)]

FLANGE_T_TOOL = ([0,0,0], [0.5, 0.5, 0.5, 0.5])
PLATFORM_POSE = ([[0.005, 0.809, 0.006], [0, 0, -0.9659258, 0.258819]])
TOOL_T_TIP = ([0, 0, 0.115], [0, 0, 0, 1])

PLATFORM_T_MEDGEAR = ([0.2172, 0.0617, 0.2647], [-4.3298e-17, -7.0711e-01,  7.0711e-01, 4.3298e-17])
PLATFORM_T_SMALLGEAR =  ([0.1872, 0.0617, 0.2647], [-4.3298e-17, -7.0711e-01,  7.0711e-01, 4.3298e-17])
PLATFORM_T_LARGEGEAR = ([0.2672, 0.0617, 0.2647], [-4.3298e-17, -7.0711e-01,  7.0711e-01, 4.3298e-17])

TASKBOARD_POSE = ([0.8, 0.1, -0.02], [0.5, -0.5, -0.5, 0.5])

def setup_environment(client=None, **kwargs):
    r0 = client.loadURDF("goflow/franka_panda/panda.urdf", R0_POSE[0], R0_POSE[1], useFixedBase=True)

    # Create the table
    plane_id = client.createCollisionShape(shapeType=p.GEOM_PLANE)
    ground_id = client.createMultiBody(baseCollisionShapeIndex=plane_id)
    pbu.set_pose(ground_id, ((0, 0, -0.01), (0, 0, 0, 1)), client=client)
    obstacles = [ground_id]
    return [r0], obstacles

def load_obj(stl_file_path, mass=1.0, base_position=[0, 0, 0], base_orientation=[0, 0, 0, 1], mesh_scale=[1, 1, 1], client=None):
    """
    Loads an STL file into PyBullet as a multibody object.

    Args:
        stl_file_path (str): Path to the STL file.
        mass (float): Mass of the object.
        base_position (list): Initial position [x, y, z] of the object.
        base_orientation (list): Initial orientation [x, y, z, w] (quaternion) of the object.
        mesh_scale (list): Scale factors [x, y, z] for the mesh.

    Returns:
        int: The body ID of the created object.
    """
    # Create visual shape from the STL
    visual_shape_id = client.createVisualShape(
        shapeType=p.GEOM_MESH,
        fileName=stl_file_path,
        meshScale=mesh_scale
    )

    # Create collision shape from the STL (optional, for physics interactions)
    collision_shape_id = client.createCollisionShape(
        shapeType=p.GEOM_MESH,
        fileName=stl_file_path,
        meshScale=mesh_scale
    )

    # Create the multibody object using the visual and collision shapes
    body_id = client.createMultiBody(
        baseMass=mass,
        baseCollisionShapeIndex=collision_shape_id,
        baseVisualShapeIndex=visual_shape_id,
        basePosition=base_position,
        baseOrientation=base_orientation
    )

    return body_id    

def solve_ik(start_qs, target_poses, tool_name="panda_hand", ee_collisions=True, step_through=False, client=None):

    random.seed(0)
    np.random.seed(0)

    new_client = False
    if(client is None):
        client = bc.BulletClient(connection_mode=p.GUI if step_through else p.DIRECT)
        new_client = True
    
    robots, obstacles = setup_environment(ee_collisions=ee_collisions, client=client)

    for robot, start_q in zip(robots, start_qs):
        pbu.set_joint_positions(robot, pbu.get_movable_joints(robot, client=client), start_q, client=client)

        
    ik_solutions = []
    randomize_seed = [False, False]
    max_attempts = 15000

    for i in range(max_attempts):
        
        if(len (ik_solutions) == len(robots)):
            p.disconnect()
            return ik_solutions

        robot_idx = len(ik_solutions)

        if(i % int(math.sqrt(max_attempts)) == 0):
            ik_solutions = []
            randomize_seed = [True, False]
            continue

        robot = robots[robot_idx]
        target_pose = target_poses[robot_idx]

        if(target_pose is None):
            ik_solutions.append(start_qs[robot_idx])
            continue

        link = pbu.link_from_name(robot, tool_name, client=client)
        joints = pbu.get_movable_joints(robot, client=client)
        ranges = [pbu.get_joint_limits(robot, joint, client=client) for joint in joints]

        # Start with the current joint positions and then randomize within limits after
        if(not randomize_seed[robot_idx]):
            initialization_sample = start_qs[robot_idx]
            randomize_seed[robot_idx] = True
        else:
            initialization_sample = [random.uniform(r[0], r[1]) for r in ranges]
            
        pbu.set_joint_positions(robot, joints, initialization_sample, client=client)

        conf = p.calculateInverseKinematics(
            int(robot), link, target_pose[0], target_pose[1], 
            residualThreshold=0.00001, maxNumIterations=5000
        )
        
        lower, upper = list(zip(*ranges))
        if(not pbu.all_between(lower, conf, upper)):
            print("IK solution outside limits")
            continue
        
        assert len(joints) == len(conf)
        pbu.set_joint_positions(robot, joints, conf, client=client)

        contact_points = []
        for obstacle in obstacles:
            contact_points += p.getClosestPoints(bodyA=obstacle, bodyB=robot, distance = pbu.MAX_DISTANCE)

        for r_idx in range(len(ik_solutions)):
            contact_points += p.getClosestPoints(bodyA=robots[r_idx], bodyB=robot, distance = pbu.MAX_DISTANCE)
        

        all_joints = pbu.get_joints(robot, client = client)
        check_link_pairs = (
            pbu.get_self_link_pairs(robot, all_joints, IGNORE_COLLISIONS, client = client)
        )

        self_collision = False
        for link1, link2 in check_link_pairs:
            if pbu.pairwise_link_collision(robot, link1, robot, link2, client = client):
                print(link1, link2)
                self_collision = True

        if(self_collision):
            print("Self collision")
            continue

        # Print contact points if there are any
        if contact_points:
            print("Collision!")
            # time.sleep(0.5)
            continue

        pose = pbu.get_link_pose(robot, link, client=client)
        trans_diff, rot_diff = pbu.get_pose_distance(target_pose, pose)

        if(trans_diff < 0.001 and rot_diff < 0.01):
            ik_solutions.append(list(conf))
            continue
        else:
            print("IK Error: {}, {}".format(trans_diff, rot_diff))

    if(new_client): 
        client.disconnect()
    
    return None


os.makedirs("logs", exist_ok=True)

class HighPrecisionUnixEpochFormatter(logging.Formatter):
    def formatTime(self, record, datefmt=None):
        # Get the current time in seconds
        current_time = time.time()
        # Convert to milliseconds
        millis = int(round(current_time * 1000))
        return str(millis)

logFormatter = HighPrecisionUnixEpochFormatter('%(asctime)s:%(message)s')
# logFormatter = logging.Formatter("%(asctime)s:%(message)s")
log = logging.getLogger()
log.setLevel(logging.INFO)  # Set the logger level (DEBUG, INFO, WARNING, ERROR, CRITICAL)

logPath = os.path.join(pathlib.Path(__file__).parent.resolve(), "logs")
fileHandler = logging.FileHandler("{0}/{1}.log".format(logPath, str(time.time())))
fileHandler.setFormatter(logFormatter)
log.addHandler(fileHandler)

consoleHandler = logging.StreamHandler()
consoleHandler.setFormatter(logFormatter)
log.addHandler(consoleHandler)

def obj_to_str(d):
    return json.dumps(d)

def listify(v):
    if(isinstance(v, torch.Tensor)):
        return v.cpu().detach().numpy().tolist()
    elif(isinstance(v, np.ndarray)):
        return listify(v.tolist())
    elif(isinstance(v, list)):
        return [listify(ve) for ve in v]
    elif(isinstance(v, float) or isinstance(v, int)):
        return v
    else:
        raise NotImplementedError


CAPTURE_DEBUG_LOG = False
def log_list(name, vector:List[float]):
    if(CAPTURE_DEBUG_LOG):
        vector = listify(vector)
        assert isinstance(vector, list)
        log.info("ADKPlot:"+str(obj_to_str({name: vector})))
        

# Translation 
def get_scripted_actions(num_envs, num_robots=1, repeat=10):
    scripted_actions = [torch.tensor([[0, 0, 1]*num_robots]).repeat([num_envs, 1])]*repeat+\
                       [torch.tensor([[0, 0, -1]*num_robots]).repeat([num_envs, 1])]*repeat+\
                       [torch.tensor([[0, 1, 0]*num_robots]).repeat([num_envs, 1])]*repeat+\
                       [torch.tensor([[0, -1, 0]*num_robots]).repeat([num_envs, 1])]*repeat+\
                       [torch.tensor([[1, 0, 0]*num_robots]).repeat([num_envs, 1])]*repeat+\
                       [torch.tensor([[-1, 0, 0]*num_robots]).repeat([num_envs, 1])]*repeat
    return scripted_actions

@dataclass
class TaskConfig():
    PEG_ASSET_NAME: str = None
    PEG_TYPE: str = "part"
    PEG_ATTACHMENT_PRIM: str = None

    HOLE_ASSET_NAME: str = None
    HOLE_TYPE: str = "part"
    HOLE_ATTACHMENT_PRIM: str = None

    # Add parts to the scene that aren't being manipulated and aren't part of the goal calculation
    EXTRA_PARTS: List = field(default_factory=list)
    EXTRA_PARTS_STARTING_POSE: List = field(default_factory=list)
    EXTRA_ATTACHMENTS: List = field(default_factory=list)

    HOLE_T_PEG_GOAL: Tuple = None
    PEG_GOAL: Tuple = None
    HOLE_GOAL: Tuple = None

    # These are used if the relevant part of the peg/hole is separate from the. 
    # The target could be defined by pose, but this can be used if the pose doesn't matter much
    # In most cases, these can be kept at identity
    PEG_T_ORIGIN_GOAL: Tuple = ([0,0,0], [0,0,0,1])
    HOLE_T_ORIGIN_GOAL: Tuple = ([0,0,0], [0,0,0,1])

    WORLD_T_PEG_START: Tuple = None
    WORLD_T_HOLE_START: Tuple = None

    PEG_T_TIP: Tuple = None
    HOLE_T_TIP: Tuple = None

    WORLD_T_PEG_PICK: Tuple = None
    WORLD_T_HOLE_PICK: Tuple = None

    relative_goal: bool = True

    allow_peg_rotation: bool = False
    allow_hole_rotation: bool = False

    # Weights on the x, y, z, rx, ry, rz components of the error
    peg_goal_weights: List[float] = field(default_factory=lambda: [1, 1, 1, 0, 0, 0])
    # Only used if absolute pose and hole part
    hole_goal_weights: List[float] = field(default_factory=lambda: [1, 1, 1, 0, 0, 0])

    PEG_IK_WORLD_T_TOOL: Tuple = None
    HOLE_IK_WORLD_T_TOOL: Tuple = None
    
    origin_regularization: float = 0

    relative_observations: bool = False
    force_sensing: bool = False

    mask_peg: bool = False
    mask_hole: bool = False

    # Weights on the x, y, z, rx, ry, rz components of domain randomization
    randomizations: List[float] = field(default_factory=lambda: [0, 0, 0, 0, 0, 0])

    @property
    def PEG_T_TOOL(self):
        return pbu.multiply(self.PEG_T_TIP, pbu.invert(TOOL_T_TIP))
    
    @property
    def HOLE_T_TOOL(self):
        return pbu.multiply(self.HOLE_T_TIP, pbu.invert(TOOL_T_TIP))
    
    @property
    def WORLD_T_TOOL_PICK_PEG(self):
        if(self.WORLD_T_PEG_PICK is None):
            return None
        return pbu.multiply(pbu.multiply(self.WORLD_T_PEG_PICK, self.PEG_T_TIP), pbu.invert(TOOL_T_TIP))
    
    @property
    def WORLD_T_TOOL_PICK_HOLE(self):
        if(self.WORLD_T_HOLE_PICK is None):
            return None
        return pbu.multiply(pbu.multiply(self.WORLD_T_HOLE_PICK, self.HOLE_T_TIP), pbu.invert(TOOL_T_TIP))
    
    @property
    def WORLD_T_PEG_TOOL_START(self):
        return pbu.multiply(self.WORLD_T_PEG_START, self.PEG_T_TOOL)

    @property
    def WORLD_T_HOLE_TOOL_START(self):
        return pbu.multiply(self.WORLD_T_HOLE_START, self.HOLE_T_TOOL)

    @property
    def holding_peg(self):
        return self.PEG_T_TIP is not None
    
    @property
    def holding_hole(self):
        return self.HOLE_T_TIP is not None
    
    @property
    def moving_peg(self):
        return self.holding_peg and not self.mask_peg
    
    @property
    def moving_hole(self):
        return self.holding_hole and not self.mask_hole
    
    @property
    def robot_count(self):
        return int(self.moving_peg)+int(self.moving_hole)

def fmb_example():
    PEG_T_TIP = ([0,0,0.02113], [-1, 0, 0, 0])
    HOLE_T_PEG_GOAL = ([-0.06795, -0.08966, 0.02422], [0,0,0,1])
    WORLD_T_HOLE_START = ([0, 0, 0.025], [0, 0, 0, 1])
    WORLD_T_PEG_START = pbu.multiply(([0, 0, 0.1], [0,0,0,1]), pbu.multiply(WORLD_T_HOLE_START, HOLE_T_PEG_GOAL))
    return TaskConfig(PEG_ASSET_NAME = os.path.join(ROOT_DIR, "taskboard/fmb_example/Medium_Short_Hexagon_Green_Updated.usd"),
                      PEG_ATTACHMENT_PRIM = "peg/Medium_Short_Hexagon_Green/Medium_Short_Hexagon_Green/obj1_012_obj1_005",
                      HOLE_ASSET_NAME = os.path.join(ROOT_DIR, "taskboard/fmb_example/Medium_Board_DarkBlue_Updated.usd"),
                      HOLE_TYPE = "peripheral",
                      HOLE_T_PEG_GOAL = HOLE_T_PEG_GOAL,
                      WORLD_T_PEG_START=WORLD_T_PEG_START,
                      WORLD_T_HOLE_START=WORLD_T_HOLE_START,
                      PEG_T_TIP = PEG_T_TIP)

def med_gear_in_taskboard():

    WORLD_T_PEG_START = pbu.multiply(([0, 0.00, 0.05], [0, 0, 0, 1]), pbu.multiply(TASKBOARD_POSE, PLATFORM_T_MEDGEAR))
    PEG_T_TIP = ([0, 0, 0], [0, 0, 0, 1])
    TASKBOARD_T_PICK_PEG = ([0.21818, 0.05686, 0.03195], [-0.0, -0.70711, 0.70711, 0.0])
    WORLD_T_PEG_PICK = pbu.multiply(TASKBOARD_POSE, TASKBOARD_T_PICK_PEG)

    return TaskConfig(PEG_ASSET_NAME = os.path.join(ROOT_DIR, "taskboard/gear_medium.usd"),
                      HOLE_ASSET_NAME = os.path.join(ROOT_DIR, "taskboard/taskboard.usd"),
                      HOLE_TYPE = "peripheral",
                      PEG_ATTACHMENT_PRIM = "peg/GEABP1_0_40_10_B_10_Gear_40teeth/node_/mesh_",
                      HOLE_ATTACHMENT_PRIM = None,
                      HOLE_T_PEG_GOAL = PLATFORM_T_MEDGEAR,
                      WORLD_T_PEG_START = WORLD_T_PEG_START,
                      WORLD_T_HOLE_START = TASKBOARD_POSE,
                      PEG_T_TIP=PEG_T_TIP,
                      WORLD_T_PEG_PICK=WORLD_T_PEG_PICK,
                      peg_goal_weights=[1, 1, 1, 0, 0, 0])

def small_gear_in_taskboard():

    WORLD_T_PEG_START = pbu.multiply(([0, 0.00, 0.05], [0, 0, 0, 1]), pbu.multiply(TASKBOARD_POSE, PLATFORM_T_SMALLGEAR))
    PEG_T_TIP = ([0, 0, 0], [0, 0, 0, 1])
    TASKBOARD_T_PICK_PEG = ([0.14292, 0.05686, -0.03561], [-0.0, -0.70711, 0.70711, 0.0])
    WORLD_T_PEG_PICK = pbu.multiply(TASKBOARD_POSE, TASKBOARD_T_PICK_PEG)

    return TaskConfig(PEG_ASSET_NAME=os.path.join(ROOT_DIR, "taskboard/gear_small.usd"),
                      HOLE_ASSET_NAME=os.path.join(ROOT_DIR, "taskboard/taskboard.usd"),
                      HOLE_TYPE="peripheral",
                      PEG_ATTACHMENT_PRIM="peg/GEABP1_0_20_10_B_10_Gear_20teeth/node_/mesh_",
                      HOLE_ATTACHMENT_PRIM=None,
                      HOLE_T_PEG_GOAL=PLATFORM_T_SMALLGEAR,
                      WORLD_T_PEG_START=WORLD_T_PEG_START,
                      WORLD_T_HOLE_START=TASKBOARD_POSE,
                      PEG_T_TIP=PEG_T_TIP,
                      WORLD_T_PEG_PICK=WORLD_T_PEG_PICK,
                      peg_goal_weights=[1, 1, 1, 0, 0, 0])

def large_gear_in_taskboard():

    WORLD_T_PEG_START = pbu.multiply(([0, 0.00, 0.10], [0, 0, 0, 1]), pbu.multiply(TASKBOARD_POSE, PLATFORM_T_LARGEGEAR))

    PEG_T_TIP = ([0, 0, 0], [0, 0, 0, 1])
    TASKBOARD_T_PICK_PEG = ([0.2926, 0.05686, -0.03619], [-0.0, -0.70711, 0.70711, 0.0])
    WORLD_T_PEG_PICK = pbu.multiply(TASKBOARD_POSE, TASKBOARD_T_PICK_PEG)

    return TaskConfig(PEG_ASSET_NAME=os.path.join(ROOT_DIR, "taskboard/gear_large.usd"),
                      HOLE_ASSET_NAME=os.path.join(ROOT_DIR, "taskboard/taskboard.usd"),
                      HOLE_TYPE="peripheral",
                      PEG_ATTACHMENT_PRIM="peg/GEABP1_0_60_10_B_10_Gear_60teeth/node_/mesh_",
                      HOLE_ATTACHMENT_PRIM=None,
                      HOLE_T_PEG_GOAL=PLATFORM_T_LARGEGEAR,
                      WORLD_T_PEG_START=WORLD_T_PEG_START,
                      WORLD_T_HOLE_START=TASKBOARD_POSE,
                      PEG_T_TIP=PEG_T_TIP,
                      WORLD_T_PEG_PICK=WORLD_T_PEG_PICK)

def task_from_name(name)->TaskConfig:
    return globals()[name]()
    