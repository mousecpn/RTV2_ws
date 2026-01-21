#coding=utf-8

from . import Utils
from . import RobotState as rs
from .RobotState import Action
from . import RobotAssistancePolicy
from . import PrintOnFile as pof
from .PotentialFunction import PotentialFunction
import copy
import numpy as np
from .msg import Pose, GoalArray, PoseStamped, Goal as goalMsg


class PredictorAssistance2:
    """
    PredictorAssistance Class \n
    Args:
        name_srv: name of initPred service
    """
    def __init__(self):
        self._goals = None
        self._user_twist = np.zeros(6)
        self._ee_pose = np.zeros((4, 4)) #zero matrix
        self._ca_twist = np.zeros(6)
        self._file = pof.PrintOnFile()
        self._print_on_file = False
        self._robot_state = None
        self._policy = None
        self._vmax = 2.0
    

    def InitPred(self, goals, ee_pose):
        """
        Service to initialize predictor node \n
        Args: request \n
        Return: response
        """
        ee_pose_msg = Pose()
        ee_pose_msg.position.x = ee_pose[0]
        ee_pose_msg.position.y = ee_pose[1]
        ee_pose_msg.position.z = ee_pose[2]
        ee_pose_msg.orientation.x = 0.0
        ee_pose_msg.orientation.y = 0.0
        ee_pose_msg.orientation.z = 0.0
        ee_pose_msg.orientation.w = 1.0
        goal_msg = GoalArray()
        for k in range(goals.shape[0]):
            pose_msg = goalMsg()
            pose_msg.id = k
            pose_msg.center.position.x = goals[k][0]
            pose_msg.center.position.y = goals[k][1]
            pose_msg.center.position.z = 0
            pose_msg.center.orientation.x = 0
            pose_msg.center.orientation.y = 0
            pose_msg.center.orientation.z = 0
            pose_msg.center.orientation.w = 1
            grasp_point = PoseStamped()
            grasp_point.pose.position.x = goals[k][0]
            grasp_point.pose.position.y = goals[k][1]
            grasp_point.pose.position.z = 0
            grasp_point.pose.orientation.x = 0
            grasp_point.pose.orientation.y = 0
            grasp_point.pose.orientation.z = 0
            grasp_point.pose.orientation.w = 1
            pose_msg.grasping_points.append(grasp_point)
            goal_msg.goal.append(pose_msg)
        self._goals = Utils.getGoal(goal_msg)
        self._ee_pose = Utils.pose_to_mat(ee_pose_msg)        
        self._robot_state = rs.RobotState(self._ee_pose, 0,0,0)
        self._policy = RobotAssistancePolicy.RobotAssistancePolicy(self._goals, self._robot_state, self._print_on_file, self._file)
        
        print("Predictor node received request!")
        return




    def Assistance(self, user_input, ee_pose):
        """
        Service predictor assistance. It receivers twist user and actual EE pose and computes distribution probability and assisted twist. \n
        Args: request \n
        Return: response
        """
        #Set user twist, CA twist and actual EE pose
        self.setUserTwist(user_input)
        self.setEEPose(ee_pose)

        #Update Robot State pose
        #self._robot_state.updateState(self._ee_pose)
        
        #Upadate policy
        action_u = Action(self._user_twist)
        self._policy.update(action_u)
        result_action = self._policy.get_action()

        #Get assistance twist, probability distribution and index of goal with max prob
        assistance_twist = result_action.getTwist()
        assistance_twist_msg = Utils.arrayToTwistMsg(assistance_twist)
        prob_distribution = self._policy.getDistribution()
        self._policy.visualize_prob()
        index_max = self._policy.getIndexMax()

        return assistance_twist_msg, prob_distribution, index_max


    def setUserTwist(self, user_twist):
        """
        Set user twist \n
        Args: 
            user_twist: np.array(6,)
        """
        self._user_twist = user_twist


    def setCATwist(self, ca_twist):
        """
        Set CA twist \n
        Args: 
            ca_twist: np.array(6,)
        """
        self._ca_twist = ca_twist


    def setEEPose(self, pose):
        """
        Set pose matrix of EE \n
        Args: 
            pose: pose message of EE
        """
        self._ee_pose = Utils.pose_to_mat(pose)
    
    def create_potential(self, potential_params, obs_position, goal_pos_list):
        if potential_params is None:
            threshold_distance =  2.5
            attractive_gain =  0.05
            repulsive_gain =  1
            escape_gain = 0.5
            potential_params = [threshold_distance, repulsive_gain, attractive_gain, escape_gain]
        escape_points = []
        escape_gap = 1
        # for p in obs_position:
        #     escape_points.append(p+np.array([escape_gap,escape_gap, 0]))
        #     escape_points.append(p+np.array([escape_gap, -escape_gap, 0]))
        #     escape_points.append(p+np.array([-escape_gap, escape_gap, 0]))
        #     escape_points.append(p+np.array([-escape_gap, -escape_gap, 0]))
        self.pot_func = PotentialFunction(potential_params[0],potential_params[1],potential_params[2], potential_params[3], 
                                obs_position, goal_pos_list, escape_points,  None)
    
    def get_CAtwist(self, user_input, ee_pose):
        _, goal_distrib, index_max = self.Assistance(user_input, ee_pose)
        ee_pose_array = np.array([ee_pose.position.x, ee_pose.position.y, ee_pose.position.z, 0,0,0])
        twist_ca = self.pot_func.getCATwist(ee_pose_array, self._vmax, goal_distrib)
        print("Twist CA: " +str(twist_ca))

        #Final twist
        weight = min(max(goal_distrib), 0.75)
        # final_twist = twist_ca * weight + self._user_twist * (1-weight)
        final_twist = Utils.setTwist(twist_ca + self._user_twist , 2.1) #
        # final_twist = 0.5 * twist_ca + 0.5* self._user_twist
        return final_twist
        