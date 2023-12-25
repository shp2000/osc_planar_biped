import numpy as np
from typing import List, Tuple

from pydrake.multibody.plant import MultibodyPlant
from pydrake.multibody.parsing import Parser
from pydrake.systems.framework import Context, LeafSystem, BasicVector
from pydrake.trajectories import Trajectory, PiecewisePolynomial
from pydrake.solvers import MathematicalProgram, OsqpSolver
from pydrake.common.value import AbstractValue
from pydrake.math import RigidTransform

import importlib
import osc_tracking_objective
importlib.reload(osc_tracking_objective)

from osc_tracking_objective import *

import importlib
import fsm_utils
importlib.reload(fsm_utils)

from fsm_utils import LEFT_STANCE, RIGHT_STANCE, get_fsm


@dataclass
class OscGains:
    kp_com: np.ndarray
    kd_com: np.ndarray
    w_com: np.ndarray
    kp_swing_foot: np.ndarray
    kd_swing_foot: np.ndarray
    w_swing_foot: np.ndarray
    kp_base: np.ndarray
    kd_base: np.ndarray
    w_base: np.ndarray
    w_vdot: float


class OperationalSpaceWalkingController(LeafSystem):
    def __init__(self, gains: OscGains):
        """
            Constructor for the operational space controller (Do Not Modify).
            We load a drake MultibodyPlant representing the planar walker
            to use for kinematics and dynamics calculations.

            We then define tracking objectives, and finally,
            we declare input ports and output ports
        """
        LeafSystem.__init__(self)
        self.gains = gains

        ''' Load the MultibodyPlant '''
        self.plant = MultibodyPlant(0.0)
        self.parser = Parser(self.plant)
        self.parser.AddModels("planar_walker.urdf")
        self.plant.WeldFrames(
            self.plant.world_frame(),
            self.plant.GetBodyByName("base").body_frame(),
            RigidTransform.Identity()
        )
        self.plant.Finalize()
        self.plant_context = self.plant.CreateDefaultContext()

        ''' Assign contact frames '''
        self.contact_points = {
            LEFT_STANCE: PointOnFrame(
                self.plant.GetBodyByName("left_lower_leg").body_frame(),
                np.array([0, 0, -0.5])
            ),
            RIGHT_STANCE: PointOnFrame(
                self.plant.GetBodyByName("right_lower_leg").body_frame(),
                np.array([0, 0, -0.5])
            )
        }
        self.swing_foot_points = {
            LEFT_STANCE: self.contact_points[RIGHT_STANCE],
            RIGHT_STANCE: self.contact_points[LEFT_STANCE]
        }

        ''' Initialize tracking objectives '''
        self.tracking_objectives = {
            "com_traj": CenterOfMassPositionTrackingObjective(
                self.plant, self.plant_context, [LEFT_STANCE, RIGHT_STANCE],
                self.gains.kp_com, self.gains.kd_com
            ),
            "swing_foot_traj": PointPositionTrackingObjective(
                self.plant, self.plant_context, [LEFT_STANCE, RIGHT_STANCE],
                self.gains.kp_swing_foot, self.gains.kd_swing_foot, self.swing_foot_points
            ),
            "base_joint_traj": JointAngleTrackingObjective(
                self.plant, self.plant_context, [LEFT_STANCE, RIGHT_STANCE],
                self.gains.kp_base, self.gains.kd_base, "planar_roty"
            )
        }
        self.tracking_costs = {
            "com_traj": self.gains.w_com,
            "swing_foot_traj": self.gains.w_swing_foot,
            "base_joint_traj": self.gains.w_base
        }
        self.trajs = self.tracking_objectives.keys()

        ''' Declare Input Ports '''
        # State input port
        self.robot_state_input_port_index = self.DeclareVectorInputPort(
            "x", self.plant.num_positions() + self.plant.num_velocities()
        ).get_index()

        # Trajectory Input Ports
        trj = PiecewisePolynomial()
        self.traj_input_ports = {
            "com_traj": self.DeclareAbstractInputPort("com_traj", AbstractValue.Make(trj)).get_index(),
            "swing_foot_traj": self.DeclareAbstractInputPort("swing_foot_traj", AbstractValue.Make(trj)).get_index(),
            "base_joint_traj": self.DeclareAbstractInputPort("base_joint_traj", AbstractValue.Make(trj)).get_index()
        }

        # Define the output ports
        self.torque_output_port = self.DeclareVectorOutputPort(
            "u", self.plant.num_actuators(), self.CalcTorques
        )

        self.u = np.zeros((self.plant.num_actuators()))

    def get_traj_input_port(self, traj_name):
        return self.get_input_port(self.traj_input_ports[traj_name])
    
    def get_state_input_port(self):
        return self.get_input_port(self.robot_state_input_port_index)

    def CalculateContactJacobian(self, fsm: int) -> Tuple[np.ndarray, np.ndarray]:
        """
            For a given finite state, LEFT_STANCE or RIGHT_STANCE, calculate the
            Jacobian terms for the contact constraint, J and Jdot * v.

            As an example, see CalcJ and CalcJdotV in PointPositionTrackingObjective

            use self.contact_points to get the PointOnFrame for the current stance foot
        """
        J = np.zeros((3, self.plant.num_velocities()))
        JdotV = np.zeros((3,))

        # TODO - STUDENT CODE HERE:

        pt_to_track = self.contact_points[fsm]
        J =  self.plant.CalcJacobianTranslationalVelocity(
            self.plant_context, JacobianWrtVariable.kV, pt_to_track.frame,
            pt_to_track.pt, self.plant.world_frame(), self.plant.world_frame()
        )

        JdotV = self.plant.CalcBiasTranslationalAcceleration(
            self.plant_context, JacobianWrtVariable.kV, pt_to_track.frame,
            pt_to_track.pt, self.plant.world_frame(), self.plant.world_frame()
        ).ravel()
        # contact_point = self.contact_points[fsm]

        # # Calculate the translational Jacobian Jc
        # Jc_translational = self.plant.CalcJacobianTranslationalVelocity(
        #     self.plant_context, JacobianWrtVariable.kV,
        #     contact_point.frame, contact_point.pt,
        #     self.plant.world_frame(), self.plant.world_frame()
        # )

        # # Calculate the rotational Jacobian Jc
        # Jc_rotational = self.plant.CalcJacobianAngularVelocity(
        #     self.plant_context, JacobianWrtVariable.kV,
        #     contact_point.frame, self.plant.world_frame(), self.plant.world_frame()
        # )

        # # Combine translational and rotational Jacobians to get the full Jacobian Jc
        # Jc = np.vstack((Jc_translational, Jc_rotational))

        # Calculate the time derivative of the contact constraint term Jcv˙
        # Jcv_dot = self.plant.CalcBiasForJacobianTransposeTimesV(
        #     self.plant_context, Jc,
        #     JacobianWrtVariable.kV,
        #     contact_point.frame, contact_point.pt,
        #     self.plant.world_frame(), self.plant.world_frame()
        # )

#         Jc = self.plant.CalcJacobianCenterOfMassTranslationalVelocity(
#             self.context, JacobianWrtVariable.kV, self.plant.world_frame(), self.plant.world_frame()
#             )
# # Combine translational and rotational accelerations
#         Jcv_dot = 

        return J, JdotV

        #return J, JdotV

    def SetupAndSolveQP(self,  context: Context) -> Tuple[np.ndarray, MathematicalProgram]:

        # First get the state, time, and fsm state
        x = self.EvalVectorInput(context, self.robot_state_input_port_index).get_value()
        t = context.get_time()
        fsm = get_fsm(t)

        # Update the plant context with the current position and velocity
        self.plant.SetPositionsAndVelocities(self.plant_context, x)

        # Update tracking objectives
        for traj_name in self.trajs:
            traj = self.EvalAbstractInput(context, self.traj_input_ports[traj_name]).get_value()
            self.tracking_objectives[traj_name].Update(t, traj, fsm)

        '''Set up and solve the QP '''
        prog = MathematicalProgram()

        # Make decision variables
        u = prog.NewContinuousVariables(self.plant.num_actuators(), "u")
        vdot = prog.NewContinuousVariables(self.plant.num_velocities(), "vdot")
        lambda_c = prog.NewContinuousVariables(3, "lambda_c")

        # Add Quadratic Cost on Desired Acceleration
        for traj_name in self.trajs:
            obj = self.tracking_objectives[traj_name]
            yddot_cmd_i = obj.yddot_cmd
            J_i = obj.J
            JdotV_i = obj.JdotV
            W_i = self.tracking_costs[traj_name]

            # TODO: Add Cost per tracking objective
            #yddot = JdotV_i + (J_i @ vdot)
            Cost = (yddot_cmd_i - (JdotV_i + (J_i @ vdot))).T @ W_i @ (yddot_cmd_i - (JdotV_i + (J_i @ vdot)))
            prog.AddQuadraticCost(Cost)
            #cost_i = np.array(J_i.T @ W_i @ (yddot_cmd_i - JdotV_i))
            # cost_i = (yddot_cmd_i - JdotV_i).T @ W_i @ (yddot_cmd_i - JdotV_i)
            # prog.AddQuadraticCost(cost_i)
            
            

        # TODO: Add Quadratic Cost on vdot using self.gains.w_vdot
        prog.AddQuadraticCost(0.5*(vdot.T @ (self.gains.w_vdot*np.eye(7) @ vdot)))
        
        # prog.AddQuadraticCost(self.gains.w_vdot * vdot.dot(vdot))

        # Calculate terms in the manipulator equation
        J_c, J_c_dot_v = self.CalculateContactJacobian(fsm)
        M = self.plant.CalcMassMatrix(self.plant_context)
        Cv = self.plant.CalcBiasTerm(self.plant_context)
        
        # Drake considers gravity to be an external force (on the right side of the dynamics equations), 
        # so we negate it to match the homework PDF and slides
        G = -self.plant.CalcGravityGeneralizedForces(self.plant_context)
        B = self.plant.MakeActuationMatrix()

        # TODO: Add the dynamics constraint
        prog.AddLinearEqualityConstraint(M @ vdot.reshape(-1,1)+ Cv.reshape(-1,1) + G.reshape(-1,1) - B @ u.reshape(-1,1) - J_c.T @ lambda_c.reshape(-1,1), np.zeros((7,1)))
        # dynamics_constraint = M @ vdot + Cv + G - B @ u - J_c.T @ lambda_c
        # prog.AddLinearEqualityConstraint(dynamics_constraint, [0] * len(dynamics_constraint))

        # TODO: Add Contact Constraint
        prog.AddLinearEqualityConstraint(J_c_dot_v + J_c @ vdot, np.zeros((3,1)))
        # contact_constraint = J_c @ vdot + J_c_dot_v @ v + J_c @ lambda_c
        # prog.AddLinearEqualityConstraint(contact_constraint, [0] * len(contact_constraint))

        # TODO: Add Friction Cone Constraint assuming mu = 1
        
        prog.AddLinearConstraint(lambda_c[0] <= lambda_c[2])
        prog.AddLinearConstraint(lambda_c[0] >= -lambda_c[2])
        # mu = 1  # Coefficient of friction (adjust as needed)
        # lambda_max = 1.0  # Maximum friction force (adjust as needed)
        # A_friction = np.array([[1, 0], [-1, 0], [0, 1], [0, -mu]])
        # b_friction = np.array([lambda_max, lambda_max, lambda_max, 0])
        # prog.AddLinearConstraint(A_friction @ lambda_c <= b_friction)

        prog.AddLinearEqualityConstraint(lambda_c[1] == 0)

        # Solve the QP
        solver = OsqpSolver()
        prog.SetSolverOption(solver.id(), "max_iter", 2000)

        result = solver.Solve(prog)

        # If we exceed iteration limits use the previous solution
        if not result.is_success():
            usol = self.u
        else:
            usol = result.GetSolution(u)
            self.u = usol

        return usol, prog

    def CalcTorques(self, context: Context, output: BasicVector) -> None:
        usol, _ = self.SetupAndSolveQP(context)
        output.SetFromVector(usol)


if __name__ == "__main__":
    gains = OscGains(
        np.eye(3), np.eye(3), np.eye(3),
        np.eye(3), np.eye(3), np.eye(3),
        np.eye(1), np.eye(1), np.eye(1),
        0.001*np.eye(5)
    )
    osc = OperationalSpaceWalkingController(gains)