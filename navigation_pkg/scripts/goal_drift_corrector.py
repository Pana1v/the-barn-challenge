#!/usr/bin/env python3
"""
Drift correction node: periodically computes odom drift using gazebo ground truth
and re-sends a corrected move_base goal to compensate.
"""
import rospy
import tf
import math
import actionlib
from geometry_msgs.msg import Quaternion
from gazebo_msgs.srv import GetModelState
from move_base_msgs.msg import MoveBaseGoal, MoveBaseAction


def main():
    rospy.init_node('goal_drift_corrector', anonymous=True)

    # Get goal from params (set by run.py Section 1)
    goal_x = rospy.get_param('goal_position', [0, 10])[0]
    goal_y = rospy.get_param('goal_position', [0, 10])[1]
    init_pos = rospy.get_param('init_position', [-2.25, 3, 1.57])
    # World-frame goal
    world_goal_x = init_pos[0] + goal_x
    world_goal_y = init_pos[1] + goal_y

    update_interval = rospy.get_param('~update_interval', 3.0)

    rospy.wait_for_service('/gazebo/get_model_state', timeout=10)
    get_model_state = rospy.ServiceProxy('/gazebo/get_model_state', GetModelState)

    nav_as = actionlib.SimpleActionClient('/move_base', MoveBaseAction)
    rospy.loginfo("Drift corrector: waiting for move_base...")
    nav_as.wait_for_server()
    rospy.loginfo("Drift corrector: ready (goal world=(%.2f,%.2f) interval=%.1fs)" %
                  (world_goal_x, world_goal_y, update_interval))

    listener = tf.TransformListener()
    rate = rospy.Rate(1.0 / update_interval)

    while not rospy.is_shutdown():
        try:
            listener.waitForTransform('odom', 'base_link', rospy.Time(0), rospy.Duration(1.0))
            (trans, _) = listener.lookupTransform('odom', 'base_link', rospy.Time(0))
            odom_x, odom_y = trans[0], trans[1]

            resp = get_model_state('jackal', 'world')
            if resp.success:
                gt_x = resp.pose.position.x
                gt_y = resp.pose.position.y
                drift_x = odom_x - gt_x
                drift_y = odom_y - gt_y
                drift_mag = math.hypot(drift_x, drift_y)

                corrected_x = world_goal_x + drift_x
                corrected_y = world_goal_y + drift_y

                goal = MoveBaseGoal()
                goal.target_pose.header.frame_id = 'odom'
                goal.target_pose.header.stamp = rospy.Time.now()
                goal.target_pose.pose.position.x = corrected_x
                goal.target_pose.pose.position.y = corrected_y
                goal.target_pose.pose.orientation = Quaternion(0, 0, 0, 1)
                nav_as.send_goal(goal)

                if drift_mag > 0.1:
                    rospy.loginfo("Drift corrector: drift=%.2fm, corrected goal=(%.2f,%.2f)" %
                                  (drift_mag, corrected_x, corrected_y))
        except Exception as e:
            rospy.logwarn_throttle(5.0, "Drift corrector: %s" % str(e))

        rate.sleep()


if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass
