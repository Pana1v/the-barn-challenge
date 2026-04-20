/*
 * A* planner for BARN challenge -- C++ port of nav_astar.py
 * Direct 1:1 port: same behavior, same params, same topics, same CSV log format.
 *
 * Fixes carried over from Python:
 *   F1: Smooth path (shortcut + gradient descent)
 *   F2: Velocity polygon (multi-candidate forward projection)
 *   F3: Angular rate limiting (prevent yaw oscillation)
 *   F4: Cornered = back up via breadcrumbs, never spin in tight spaces
 */

#include <ros/ros.h>
#include <geometry_msgs/Twist.h>
#include <geometry_msgs/PoseStamped.h>
#include <sensor_msgs/LaserScan.h>
#include <nav_msgs/Path.h>
#include <nav_msgs/OccupancyGrid.h>
#include <tf/transform_listener.h>
#include <tf/transform_broadcaster.h>
#include <robot_localization/SetPose.h>
#include <visualization_msgs/MarkerArray.h>

#include <Eigen/Dense>

#include <cmath>
#include <vector>
#include <queue>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <limits>
#include <cstdio>
#include <cstdlib>
#include <sys/stat.h>
#include <chrono>

// ---------------------------------------------------------------------------
// Utility
// ---------------------------------------------------------------------------
static inline double normalizeAngle(double a) {
    a = std::fmod(a, 2.0 * M_PI);
    if (a > M_PI)  a -= 2.0 * M_PI;
    if (a < -M_PI) a += 2.0 * M_PI;
    return a;
}

static inline double clampd(double v, double lo, double hi) {
    return (v < lo) ? lo : (v > hi) ? hi : v;
}

// ---------------------------------------------------------------------------
// AStarPlanner
// ---------------------------------------------------------------------------
class AStarPlanner {
public:
    // Grid cost constants
    static constexpr double UNKNOWN  = 0.8;
    static constexpr double FREE     = 0.0;
    static constexpr double OCCUPIED = 1.0;

    AStarPlanner() : tf_listener_() {
        ros::NodeHandle nh;
        ros::NodeHandle pnh("~");

        // Parameter helper: try private ns, then nav_astar/ ns, then default
        auto p_double = [&](const std::string& name, double def) -> double {
            double v;
            if (pnh.getParam(name, v)) return v;
            if (nh.getParam("nav_astar/" + name, v)) return v;
            return def;
        };
        auto p_int = [&](const std::string& name, int def) -> int {
            int v;
            if (pnh.getParam(name, v)) return v;
            if (nh.getParam("nav_astar/" + name, v)) return v;
            return def;
        };

        grid_res_          = p_double("grid_resolution", 0.04);
        inflation_radius_  = p_double("inflation_radius", 0.22);
        cost_radius_       = p_double("cost_radius", 0.40);
        max_lin_           = p_double("max_linear_vel", 1.2);
        max_ang_           = p_double("max_angular_vel", 1.5);
        accel_limit_       = p_double("accel_limit", 0.5);
        Kp_ang_            = p_double("Kp_ang", 3.5);
        align_thresh_      = p_double("align_threshold", 1.0);
        replan_interval_   = p_double("replan_interval", 1.0);
        goal_tolerance_    = p_double("goal_tolerance", 0.01);
        checkpoint_dist_   = p_double("checkpoint_reach_dist", 0.25);
        safe_margin_       = p_double("safe_margin", 0.02);
        stop_clearance_    = p_double("stop_clearance", 0.02);
        slow_clearance_    = p_double("slow_clearance", 0.10);
        breadcrumb_spacing_= p_double("breadcrumb_spacing", 0.3);
        stuck_check_interval_ = p_double("stuck_check_interval", 10.0);
        stuck_threshold_   = p_double("stuck_threshold", 0.3);
        recovery_max_time_ = p_double("recovery_max_time", 8.0);
        collision_stop_threshold_ = p_int("collision_stop_threshold", 12);

        {
            bool v;
            if (pnh.getParam("repeat_breadcrumb", v)) repeat_breadcrumb_ = v;
            else if (nh.getParam("nav_astar/repeat_breadcrumb", v)) repeat_breadcrumb_ = v;
            else repeat_breadcrumb_ = false;
        }

        // Hybrid A* params
        hybrid_step_       = p_double("hybrid_step_size", 0.30);
        hybrid_num_steer_  = p_int("hybrid_num_steer", 5);
        hybrid_heading_bins_ = p_int("hybrid_heading_bins", 72);
        hybrid_max_steer_  = p_double("hybrid_max_steer", 0.8);
        hybrid_analytic_dist_ = p_double("hybrid_analytic_dist", 1.5);
        analytic_heading_tol_ = p_double("analytic_heading_tol", 1.0);
        analytic_cost_limit_ = p_double("analytic_cost_limit", 0.3);
        obstacle_cost_weight_ = p_double("obstacle_cost_weight", 15.0);
        steer_penalty_     = p_double("steer_penalty", 0.05);
        h_weight_          = p_double("hybrid_h_weight", 0.5);
        spin_recovery_threshold_ = p_int("spin_recovery_threshold", 40);

        robot_length_      = p_double("robot_length", 0.42);
        robot_width_       = p_double("robot_width", 0.31);
        robot_radius_      = std::hypot(robot_length_ / 2.0, robot_width_ / 2.0);
        // Control radius = half robot width (not inflation radius).
        // Inflation keeps the planner's center away from walls;
        // control radius is the actual footprint for collision checking.
        control_radius_    = p_double("robot_width", 0.31) / 2.0;  // 0.155m
        proj_dt_           = p_double("proj_dt", 0.1);
        proj_horizon_      = p_double("proj_horizon", 0.4);
        proj_clearance_    = p_double("proj_clearance", 0.04);
        max_ang_accel_     = p_double("max_angular_accel", 4.0);
        prev_ang_vel_      = 0.0;

        // Goal from rosparam
        std::vector<double> goal_rel;
        if (!nh.getParam("goal_position", goal_rel) || goal_rel.size() < 2) {
            goal_rel = {0.0, 10.0};
        }
        goal_x_ = goal_rel[0];
        goal_y_ = goal_rel[1];

        // map→odom offset + spawn yaw for EKF correction
        std::vector<double> init_pos;
        if (!nh.getParam("init_position", init_pos) || init_pos.size() < 3) {
            init_pos = {0.0, 0.0, 0.0};
        }
        map_origin_x_  = init_pos[0];
        map_origin_y_  = init_pos[1];
        spawn_yaw_     = init_pos[2];

        // State
        robot_x_ = robot_y_ = robot_yaw_ = 0.0;
        have_scan_ = false;
        path_idx_ = 0;
        last_plan_time_ = 0.0;
        prev_lin_vel_ = 0.0;
        consecutive_stops_ = 0;
        spin_count_ = 0;

        // Recovery
        last_bc_valid_ = false;
        stuck_check_init_ = false;
        in_recovery_ = false;
        recovery_start_ = 0.0;
        recovery_trail_idx_ = 0;
        recovery_count_ = 0;
        plan_fail_count_ = 0;
        blend_ticks_ = 0;

        // Progress deadlock
        progress_time_init_ = false;
        progress_window_  = p_double("progress_window", 15.0);
        progress_min_     = p_double("progress_min", 0.3);
        deadlock_escape_  = false;
        escape_start_     = 0.0;
        escape_heading_   = 0.0;
        escape_phase_     = 0;

        // Laser TF
        laser_tf_valid_ = false;
        laser_tx_ = laser_ty_ = laser_yaw_ = 0.0;

        // Grid
        local_size_ = p_double("local_size", 12.0);
        grid_cells_ = static_cast<int>(local_size_ / grid_res_);

        // Pub/Sub
        cmd_pub_      = nh.advertise<geometry_msgs::Twist>("/cmd_vel", 1);
        path_pub_     = nh.advertise<nav_msgs::Path>("/nav_astar/path", 1, true);
        costmap_pub_  = nh.advertise<nav_msgs::OccupancyGrid>("/astar/costmap", 1, true);
        // ctrl_costmap removed — single costmap for planning + viz
        marker_pub_   = nh.advertise<visualization_msgs::MarkerArray>("/astar/debug_markers", 1, true);
        scan_sub_     = nh.subscribe("/front/scan", 1, &AStarPlanner::scanCb, this);

        // Logger
        {
            struct stat st;
            if (stat("/tmp/nav_logs", &st) != 0) {
                mkdir("/tmp/nav_logs", 0755);
            }
            world_idx_ = 0;
            nh.param("/world_idx", world_idx_, 0);
            char lp[256];
            std::snprintf(lp, sizeof(lp), "/tmp/nav_logs/astar_w%d.csv", world_idx_);
            log_file_.open(lp, std::ios::out | std::ios::trunc);
            log_file_ << "t,x,y,yaw,lv,av,obs,fwd,cl,tx,ty,ae,st,branch,pl,d2g,cstops,spins,pfails,rcnt,bc_cnt,trail\n";
        }

        // Breadcrumb trail persistence
        using_loaded_trail_ = false;
        loaded_trail_idx_ = 0;
        if (repeat_breadcrumb_) loadBreadcrumbs();

        ROS_INFO("Hybrid A* planner [C++]: goal(%.1f,%.1f) grid=%d inflation=%.3f cost=%.3f step=%.2f steer=%d bins=%d",
                 goal_x_, goal_y_, grid_cells_, inflation_radius_, cost_radius_,
                 hybrid_step_, hybrid_num_steer_, hybrid_heading_bins_);
    }

    // -----------------------------------------------------------------------
    void run() {
        ros::Rate rate(20);

        // Correct EKF yaw FIRST, before any scan/pose data is used.
        // The EKF starts at yaw=0 (Gazebo spawn) but run.py resets the
        // robot to spawn_yaw via set_model_state. The EKF has no absolute
        // yaw source, so it stays wrong. Inject the known spawn yaw now.
        {
            ros::ServiceClient set_pose_cli =
                ros::NodeHandle().serviceClient<robot_localization::SetPose>(
                    "/set_pose");
            if (set_pose_cli.waitForExistence(ros::Duration(5.0))) {
                robot_localization::SetPose srv;
                srv.request.pose.header.frame_id = "odom";
                srv.request.pose.header.stamp    = ros::Time::now();
                srv.request.pose.pose.pose.position.x = 0.0;
                srv.request.pose.pose.pose.position.y = 0.0;
                srv.request.pose.pose.pose.position.z = 0.0;
                double hy = spawn_yaw_ / 2.0;
                srv.request.pose.pose.pose.orientation.z = std::sin(hy);
                srv.request.pose.pose.pose.orientation.w = std::cos(hy);
                srv.request.pose.pose.covariance[0]  = 0.5;   // x
                srv.request.pose.pose.covariance[7]  = 0.5;   // y
                srv.request.pose.pose.covariance[35] = 0.01;  // yaw ← tight
                if (set_pose_cli.call(srv))
                    ROS_INFO("A*: EKF yaw reset to spawn_yaw=%.2f rad", spawn_yaw_);
                else
                    ROS_WARN("A*: set_pose call failed — EKF yaw may be wrong");
            } else {
                ROS_WARN("A*: /set_pose service not available — EKF yaw uncorrected");
            }
            // Let EKF settle and new scans arrive with corrected TF
            ros::Duration(1.0).sleep();
            ros::spinOnce();
        }

        // Wait for scan + pose (now with correct yaw from start)
        while (ros::ok()) {
            ros::spinOnce();
            if (have_scan_ && getPose()) break;
            rate.sleep();
        }
        ros::Duration(0.5).sleep();
        ros::spinOnce();
        getPose();

        // Wipe any stale markers from a previous run
        {
            visualization_msgs::MarkerArray clear;
            visualization_msgs::Marker m;
            m.header.frame_id = "map";
            m.header.stamp = ros::Time::now();
            m.action = visualization_msgs::Marker::DELETEALL;
            clear.markers.push_back(m);
            marker_pub_.publish(clear);
        }

        ROS_INFO("A*: ready (%.2f,%.2f) yaw=%.2f -> goal(%.2f,%.2f)",
                 robot_x_, robot_y_, robot_yaw_, goal_x_, goal_y_);

        while (ros::ok()) {
            ros::spinOnce();

            // Publish map→odom so RViz shows world-frame coordinates
            tf::Transform map_to_odom;
            map_to_odom.setOrigin(tf::Vector3(map_origin_x_, map_origin_y_, 0.0));
            map_to_odom.setRotation(tf::Quaternion(0, 0, 0, 1));
            map_tf_pub_.sendTransform(
                tf::StampedTransform(map_to_odom, ros::Time::now(), "map", "odom"));

            if (!getPose()) {
                rate.sleep();
                continue;
            }

            double d2g = std::hypot(goal_x_ - robot_x_, goal_y_ - robot_y_);
            if (d2g < goal_tolerance_) {
                ROS_INFO("A*: GOAL!");
                saveBreadcrumbs();
                geometry_msgs::Twist stop;
                cmd_pub_.publish(stop);
                break;
            }

            updateBc();
            double obs_d = minObs();
            double fwd_d = fwdMin(1.05);
            double fc    = fwd_d - control_radius_;
            double cl    = obs_d - control_radius_ - safe_margin_;

            // Deadlock escape takes priority
            if (deadlock_escape_) {
                doDeadlockEscape();
                std::string esc_br = (escape_phase_ == 0) ? "esc_rot" : "esc_drive";
                logRow(0, 0, obs_d, fc, cl, escape_heading_, 0, 0, "esc", esc_br, 0, d2g);
                pubDebugMarkers(robot_x_ + 2.0 * std::cos(escape_heading_),
                                robot_y_ + 2.0 * std::sin(escape_heading_),
                                esc_br, normalizeAngle(escape_heading_ - robot_yaw_));
                rate.sleep();
                continue;
            }

            // Check sustained lack of goal progress
            if (checkProgress(d2g)) {
                ROS_WARN("A*: DEADLOCK -- no goal progress in %.0fs, escaping", progress_window_);
                deadlock_escape_ = true;
                escape_start_ = ros::Time::now().toSec();
                escape_heading_ = findClearHeading();
                escape_phase_ = 0;
                in_recovery_ = false;
                doDeadlockEscape();
                logRow(0, 0, obs_d, fc, cl, escape_heading_, 0, 0, "esc", "esc_init", 0, d2g);
                pubDebugMarkers(robot_x_ + 2.0 * std::cos(escape_heading_),
                                robot_y_ + 2.0 * std::sin(escape_heading_),
                                "esc_init", normalizeAngle(escape_heading_ - robot_yaw_));
                rate.sleep();
                continue;
            }

            if (in_recovery_) {
                doRecovery();
                double rtx = (!recovery_trail_.empty() && recovery_trail_idx_ < static_cast<int>(recovery_trail_.size())) ? recovery_trail_[recovery_trail_idx_].first : robot_x_;
                double rty = (!recovery_trail_.empty() && recovery_trail_idx_ < static_cast<int>(recovery_trail_.size())) ? recovery_trail_[recovery_trail_idx_].second : robot_y_;
                logRow(0, 0, obs_d, fc, cl, rtx, rty, 0, "rec", "rec_cont", 0, d2g);
                pubDebugMarkers(rtx, rty, "rec_cont", 0.0);
                rate.sleep();
                continue;
            }

            // Trigger recovery: stuck, too many stops, or spinning
            bool was_stuck = checkStuck();
            if (was_stuck || spin_count_ >= spin_recovery_threshold_) {
                std::string reason;
                if (was_stuck) reason = "rec_stuck";
                else reason = "rec_spins";
                doRecovery();
                double rtx = (!recovery_trail_.empty() && recovery_trail_idx_ < static_cast<int>(recovery_trail_.size())) ? recovery_trail_[recovery_trail_idx_].first : robot_x_;
                double rty = (!recovery_trail_.empty() && recovery_trail_idx_ < static_cast<int>(recovery_trail_.size())) ? recovery_trail_[recovery_trail_idx_].second : robot_y_;
                logRow(0, 0, obs_d, fc, cl, rtx, rty, 0, "rec", reason, 0, d2g);
                pubDebugMarkers(rtx, rty, reason, 0.0);
                rate.sleep();
                continue;
            }

            // Target waypoint
            double tx, ty;
            if (d2g < 2.0) {
                tx = goal_x_;
                ty = goal_y_;
            } else if (using_loaded_trail_ && loaded_trail_idx_ < static_cast<int>(loaded_good_trail_.size())) {
                // Follow loaded breadcrumb trail
                auto& wp = loaded_good_trail_[loaded_trail_idx_];
                double wd = std::hypot(wp.first - robot_x_, wp.second - robot_y_);
                if (wd < checkpoint_dist_) {
                    loaded_trail_idx_++;
                    if (loaded_trail_idx_ >= static_cast<int>(loaded_good_trail_.size())) {
                        using_loaded_trail_ = false;
                        ROS_INFO("A*: loaded trail exhausted -- switching to A*");
                    }
                }
                if (using_loaded_trail_) {
                    tx = loaded_good_trail_[loaded_trail_idx_].first;
                    ty = loaded_good_trail_[loaded_trail_idx_].second;
                } else {
                    tx = goal_x_;
                    ty = goal_y_;
                }
            } else {
                double now = ros::Time::now().toSec();
                double iv = std::min(replan_interval_, 1.0);
                if (path_.empty() || path_idx_ >= static_cast<int>(path_.size()) - 1
                    || (now - last_plan_time_) >= iv) {

                    auto t0 = std::chrono::steady_clock::now();
                    bool ok = plan();
                    auto t1 = std::chrono::steady_clock::now();
                    double plan_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
                    ROS_INFO_THROTTLE(5.0, "A* plan time: %.1f ms", plan_ms);

                    if (ok) {
                        last_plan_time_ = now;
                        plan_fail_count_ = 0;
                        blend_ticks_ = 3;  // let momentum carry before switching waypoints
                    } else {
                        last_plan_time_ = now - iv + 0.3;
                        plan_fail_count_++;
                        if (plan_fail_count_ >= 9) {
                            ROS_WARN("A*: %d plan failures -- retreating", plan_fail_count_);
                            plan_fail_count_ = 0;
                            doRecovery();
                            double rtx = (!recovery_trail_.empty() && recovery_trail_idx_ < static_cast<int>(recovery_trail_.size())) ? recovery_trail_[recovery_trail_idx_].first : 0.0;
                            double rty = (!recovery_trail_.empty() && recovery_trail_idx_ < static_cast<int>(recovery_trail_.size())) ? recovery_trail_[recovery_trail_idx_].second : 0.0;
                            logRow(0, 0, obs_d, fc, cl, rtx, rty, 0, "rec", "rec_pfail", 0, d2g);
                            pubDebugMarkers(rtx, rty, "rec_pfail", 0.0);
                            rate.sleep();
                            continue;
                        }
                    }
                }

                if (!path_.empty() && path_idx_ < static_cast<int>(path_.size())) {
                    if (blend_ticks_ > 0) {
                        blend_ticks_--;  // hold old waypoint to match Python's implicit lag
                    } else {
                        advanceIdx();
                    }
                    tx = path_[path_idx_].first;
                    ty = path_[path_idx_].second;
                } else {
                    tx = goal_x_;
                    ty = goal_y_;
                }
            }

            // --- Path-following controller ---
            double dx = tx - robot_x_;
            double dy = ty - robot_y_;
            double tyaw = std::atan2(dy, dx);
            double ae = normalizeAngle(tyaw - robot_yaw_);
            cl = obs_d - control_radius_ - safe_margin_;

            geometry_msgs::Twist cmd;
            std::string branch = "nav";

            if (cl < stop_clearance_) {
                double rc = rearClear();
                if (rc > robot_radius_ + 0.1) {
                    cmd.linear.x = -0.25;
                    branch = "estop_back";
                } else {
                    branch = "estop_halt";
                }
                consecutive_stops_++;

            } else if (std::fabs(ae) > align_thresh_) {
                double f_d, l_d, r_d;
                sectorClearance(f_d, l_d, r_d);
                double turn_side_d = (ae > 0) ? l_d : r_d;
                double rot_margin = robot_radius_ + 0.04;

                if (cl < 0.15) {
                    spin_count_ += 3;
                    branch = "rot_tight";
                } else if (f_d < rot_margin && turn_side_d < rot_margin) {
                    ROS_WARN_THROTTLE(2.0, "A*: surrounded (f=%.2f s=%.2f) -- retreating", f_d, turn_side_d);
                    spin_count_ += 3;
                    branch = "rot_surr";
                } else if (!rotationSafe((ae > 0 ? 1.0 : -1.0) * 1.5)) {
                    double rc = rearClear();
                    if (rc > robot_radius_ + 0.1) {
                        cmd.linear.x = -0.2;
                        ROS_INFO_THROTTLE(2.0, "A*: rotation unsafe, backing up");
                        branch = "rot_backup";
                    } else {
                        spin_count_ += 3;
                        branch = "rot_trapped";
                    }
                } else {
                    double max_rot = std::min(max_ang_, 1.5);
                    double raw_av = clampd(Kp_ang_ * ae, -max_rot, max_rot);
                    double dav = clampd(raw_av - prev_ang_vel_, -max_ang_accel_, max_ang_accel_);
                    cmd.angular.z = prev_ang_vel_ + dav;
                    prev_ang_vel_ = cmd.angular.z;
                    branch = "rot";
                }
                consecutive_stops_ = 0;

            } else {
                // Drive forward
                double sp = max_lin_;
                if (cl < slow_clearance_) {
                    sp *= std::max(0.15, cl / slow_clearance_);
                }
                if (fc < 0.5) {
                    sp *= std::max(0.1, fc / 0.5);
                }
                sp *= std::max(0.3, 1.0 - std::fabs(ae) / M_PI);
                sp = std::min(sp, prev_lin_vel_ + accel_limit_);
                sp = std::max(0.05, sp);

                double raw_av = clampd(Kp_ang_ * ae, -max_ang_, max_ang_);
                double dav = clampd(raw_av - prev_ang_vel_, -max_ang_accel_, max_ang_accel_);
                double ac = prev_ang_vel_ + dav;

                // [F2] Velocity polygon: try full speed, then half, then stop
                double ss;
                bool safe = checkFwdCollision(sp, ac, ss);
                if (!safe) {
                    double ss2;
                    bool safe2 = checkFwdCollision(sp * 0.5, ac * 0.5, ss2);
                    if (safe2) {
                        sp *= 0.5;
                        ac *= 0.5;
                        branch = "drive_half";
                    } else {
                        sp = 0.0;
                        ac = 0.0;
                        last_plan_time_ = 0.0;
                        consecutive_stops_++;
                        branch = "drive_blocked";
                    }
                } else {
                    branch = "drive";
                }

                cmd.linear.x = sp;
                cmd.angular.z = ac;
                prev_lin_vel_ = sp;
                prev_ang_vel_ = ac;

                if (sp > 0)
                    consecutive_stops_ = std::max(0, consecutive_stops_ - 1);
                if (sp < 0.1 && std::fabs(ac) > 1.5)
                    spin_count_++;
                else
                    spin_count_ = std::max(0, spin_count_ - 1);
            }

            std::string st = (cmd.linear.x == 0.0 && cmd.angular.z == 0.0) ? "stp" : "nav";
            logRow(cmd.linear.x, cmd.angular.z, obs_d, fc, cl, tx, ty, ae, st, branch,
                   static_cast<int>(path_.size()), d2g);

            // Live debug: goal distance, headings, branch
            double goal_hdg = std::atan2(goal_y_ - robot_y_, goal_x_ - robot_x_);
            double goal_ae  = normalizeAngle(goal_hdg - robot_yaw_);
            ROS_INFO_THROTTLE(1.0,
                "[SNAP] d2g=%.2fm  pos=(%.2f,%.2f)  yaw=%.1f°  "
                "goal_hdg=%.1f°  goal_err=%.1f°  wp=(%.2f,%.2f)  wp_err=%.1f°  "
                "obs=%.2f  lv=%.2f  av=%.2f  branch=%s  stops=%d  spins=%d",
                d2g,
                robot_x_, robot_y_,
                robot_yaw_ * 180.0 / M_PI,
                goal_hdg  * 180.0 / M_PI,
                goal_ae   * 180.0 / M_PI,
                tx, ty,
                ae * 180.0 / M_PI,
                obs_d,
                cmd.linear.x, cmd.angular.z,
                branch.c_str(),
                consecutive_stops_, spin_count_);

            cmd_pub_.publish(cmd);
            pubDebugMarkers(tx, ty, branch, ae);
            rate.sleep();
        }

        saveBreadcrumbs();
        log_file_.close();
        geometry_msgs::Twist stop;
        cmd_pub_.publish(stop);
    }

private:
    // -----------------------------------------------------------------------
    // Parameters
    // -----------------------------------------------------------------------
    double grid_res_, inflation_radius_, cost_radius_;
    double max_lin_, max_ang_, accel_limit_;
    double Kp_ang_, align_thresh_;
    double replan_interval_, goal_tolerance_, checkpoint_dist_;
    double safe_margin_, stop_clearance_, slow_clearance_;
    double breadcrumb_spacing_, stuck_check_interval_, stuck_threshold_;
    double recovery_max_time_;
    int    collision_stop_threshold_;

    double robot_length_, robot_width_;
    double robot_radius_, control_radius_;
    double proj_dt_, proj_horizon_, proj_clearance_;
    double max_ang_accel_;
    double prev_ang_vel_;

    // Hybrid A*
    double hybrid_step_;
    int    hybrid_num_steer_;
    int    hybrid_heading_bins_;
    double hybrid_max_steer_;
    double hybrid_analytic_dist_;
    double analytic_heading_tol_;
    double analytic_cost_limit_;
    double obstacle_cost_weight_;
    double steer_penalty_;
    double h_weight_;
    int    spin_recovery_threshold_;

    double goal_x_, goal_y_;

    // State
    double robot_x_, robot_y_, robot_yaw_;
    bool   have_scan_;
    sensor_msgs::LaserScan scan_data_;
    Eigen::VectorXd scan_ranges_;
    Eigen::VectorXd scan_angles_;
    std::vector<std::pair<double,double>> path_;
    int    path_idx_;
    double last_plan_time_;
    double prev_lin_vel_;
    int    consecutive_stops_;
    int    spin_count_;

    // Recovery
    std::vector<std::pair<double,double>> breadcrumbs_;
    bool   last_bc_valid_;
    double last_bc_x_, last_bc_y_;
    bool   stuck_check_init_;
    double stuck_check_time_;
    double stuck_check_x_, stuck_check_y_;
    bool   in_recovery_;
    double recovery_start_;
    // Trail: reversed breadcrumb waypoints to follow sequentially
    std::vector<std::pair<double,double>> recovery_trail_;
    int    recovery_trail_idx_;
    int    recovery_count_;
    int    plan_fail_count_;
    int    blend_ticks_;

    // Breadcrumb trail persistence
    struct BcEntry { double x, y; bool bad; };
    std::vector<BcEntry> bc_log_;
    bool   repeat_breadcrumb_;
    int    world_idx_;
    std::vector<std::pair<double,double>> loaded_good_trail_;
    std::vector<std::pair<double,double>> loaded_bad_bcs_;
    bool   using_loaded_trail_;
    int    loaded_trail_idx_;

    // Collision check visualization: projected footprint poses (odom frame)
    struct FpPose { double x, y, th; bool collided; };
    std::vector<FpPose> collision_viz_;

    // Progress deadlock
    bool   progress_time_init_;
    double progress_time_, progress_d2g_;
    double progress_window_, progress_min_;
    bool   deadlock_escape_;
    double escape_start_, escape_heading_;
    int    escape_phase_;

    // TF
    tf::TransformListener  tf_listener_;
    tf::TransformBroadcaster map_tf_pub_;
    double map_origin_x_, map_origin_y_, spawn_yaw_;
    bool   laser_tf_valid_;
    double laser_tx_, laser_ty_, laser_yaw_;

    // Grid
    double local_size_;
    int    grid_cells_;


    // Reusable Hybrid A* buffers (avoid repeated alloc/free)
    std::vector<float> ha_gs_, ha_node_x_, ha_node_y_, ha_node_th_;
    std::vector<int>   ha_came_from_;
    std::vector<bool>  ha_closed_;
    int                ha_buf_size_ = 0;

    // Pub/Sub
    ros::Publisher  cmd_pub_;
    ros::Publisher  path_pub_;
    ros::Publisher  costmap_pub_;
    ros::Publisher  marker_pub_;
    ros::Subscriber scan_sub_;

    // Logger
    std::ofstream log_file_;

    // -----------------------------------------------------------------------
    // -----------------------------------------------------------------------
    // Sensor
    // -----------------------------------------------------------------------
    void scanCb(const sensor_msgs::LaserScan::ConstPtr& msg) {
        scan_data_ = *msg;
        int n = static_cast<int>(msg->ranges.size());
        scan_ranges_.resize(n);
        scan_angles_.resize(n);
        for (int i = 0; i < n; ++i) {
            scan_ranges_(i) = msg->ranges[i];
            scan_angles_(i) = msg->angle_min + i * msg->angle_increment;
        }
        have_scan_ = true;
    }

    bool getPose() {
        try {
            tf_listener_.waitForTransform("odom", "base_link", ros::Time(0), ros::Duration(0.5));
            tf::StampedTransform transform;
            tf_listener_.lookupTransform("odom", "base_link", ros::Time(0), transform);
            robot_x_ = transform.getOrigin().x();
            robot_y_ = transform.getOrigin().y();
            double roll, pitch, yaw;
            tf::Matrix3x3(transform.getRotation()).getRPY(roll, pitch, yaw);
            robot_yaw_ = yaw;
            return true;
        } catch (...) {
            return false;
        }
    }

    void updateLaserTf() {
        if (!have_scan_) return;
        try {
            tf_listener_.waitForTransform("odom", scan_data_.header.frame_id,
                                          ros::Time(0), ros::Duration(0.2));
            tf::StampedTransform transform;
            tf_listener_.lookupTransform("odom", scan_data_.header.frame_id,
                                         ros::Time(0), transform);
            laser_tx_  = transform.getOrigin().x();
            laser_ty_  = transform.getOrigin().y();
            double r, p, y;
            tf::Matrix3x3(transform.getRotation()).getRPY(r, p, y);
            laser_yaw_ = y;
            laser_tf_valid_ = true;
        } catch (...) {
            laser_tx_  = robot_x_;
            laser_ty_  = robot_y_;
            laser_yaw_ = robot_yaw_;
            laser_tf_valid_ = true;
        }
    }

    // Returns obstacle points in odom frame (Nx2 Eigen matrix)
    Eigen::MatrixXd scanToOdom() {
        if (!have_scan_ || scan_ranges_.size() == 0) {
            return Eigen::MatrixXd(0, 2);
        }
        int n = static_cast<int>(scan_ranges_.size());
        // Build validity mask
        std::vector<int> valid_idx;
        valid_idx.reserve(n);
        for (int i = 0; i < n; ++i) {
            double r = scan_ranges_(i);
            if (std::isfinite(r) && r > scan_data_.range_min && r < scan_data_.range_max)
                valid_idx.push_back(i);
        }
        if (valid_idx.empty()) return Eigen::MatrixXd(0, 2);

        updateLaserTf();
        double tx = laser_tf_valid_ ? laser_tx_ : robot_x_;
        double ty = laser_tf_valid_ ? laser_ty_ : robot_y_;
        double yaw = laser_tf_valid_ ? laser_yaw_ : robot_yaw_;
        double c = std::cos(yaw), s = std::sin(yaw);

        int nv = static_cast<int>(valid_idx.size());
        Eigen::MatrixXd pts(nv, 2);
        for (int k = 0; k < nv; ++k) {
            int i = valid_idx[k];
            double r = scan_ranges_(i);
            double a = scan_angles_(i);
            double lx = r * std::cos(a);
            double ly = r * std::sin(a);
            pts(k, 0) = tx + lx * c - ly * s;
            pts(k, 1) = ty + lx * s + ly * c;
        }
        return pts;
    }

    // -----------------------------------------------------------------------
    // Grid
    // -----------------------------------------------------------------------
    // Visible mask: ray-cast along LIDAR rays, mark cells as visible
    // Returns flat vector<bool> of size n*n (row-major)
    std::vector<bool> visibleMask(double gox, double goy) {
        int n = grid_cells_;
        std::vector<bool> vis(n * n, false);
        if (!have_scan_ || scan_ranges_.size() == 0) return vis;

        updateLaserTf();
        double tx = laser_tf_valid_ ? laser_tx_ : robot_x_;
        double ty = laser_tf_valid_ ? laser_ty_ : robot_y_;
        double yaw = laser_tf_valid_ ? laser_yaw_ : robot_yaw_;

        int total = static_cast<int>(scan_ranges_.size());
        int ns = static_cast<int>(scan_data_.range_max / (grid_res_ * 1.5)) + 1;

        // Subsample by stride 5
        for (int idx = 0; idx < total; idx += 5) {
            double ra = scan_angles_(idx) + yaw;
            double rr = scan_ranges_(idx);
            if (!std::isfinite(rr) || rr < scan_data_.range_min) {
                rr = scan_data_.range_max;
            }
            rr = std::min(rr, static_cast<double>(scan_data_.range_max));

            double cos_ra = std::cos(ra);
            double sin_ra = std::sin(ra);

            for (int s = 0; s <= ns; ++s) {
                double frac = (ns > 0) ? static_cast<double>(s) / ns : 0.0;
                double ad = rr * frac;
                double px = tx + ad * cos_ra;
                double py = ty + ad * sin_ra;
                int gi = static_cast<int>((px - gox) / grid_res_);
                int gj = static_cast<int>((py - goy) / grid_res_);
                if (gi >= 0 && gi < n && gj >= 0 && gj < n) {
                    vis[gi * n + gj] = true;
                }
            }
        }

        // Mark robot footprint cells as visible
        int ri = static_cast<int>((robot_x_ - gox) / grid_res_);
        int rj = static_cast<int>((robot_y_ - goy) / grid_res_);
        int rc = static_cast<int>(robot_radius_ / grid_res_) + 2;
        int i0 = std::max(0, ri - rc);
        int i1 = std::min(n, ri + rc + 1);
        int j0 = std::max(0, rj - rc);
        int j1 = std::min(n, rj + rc + 1);
        for (int i = i0; i < i1; ++i)
            for (int j = j0; j < j1; ++j)
                vis[i * n + j] = true;

        return vis;
    }

    // Build local occupancy grid, returns flat vector<float> (row-major n*n)
    // Also sets ox_, oy_ (grid origin in odom frame)
    std::vector<float> buildGrid(const Eigen::MatrixXd& obstacles, double& ox, double& oy) {
        int n = grid_cells_;
        double half = local_size_ / 2.0;
        ox = robot_x_ - half;
        oy = robot_y_ - half;

        std::vector<float> grid(n * n, static_cast<float>(UNKNOWN));

        // Apply visibility mask
        std::vector<bool> vis = visibleMask(ox, oy);
        for (int k = 0; k < n * n; ++k) {
            if (vis[k]) grid[k] = static_cast<float>(FREE);
        }

        if (obstacles.rows() == 0) return grid;

        // Place obstacle points
        struct OCell { int i, j; };
        std::vector<OCell> obs_cells;
        obs_cells.reserve(obstacles.rows());
        for (int k = 0; k < static_cast<int>(obstacles.rows()); ++k) {
            int gi = static_cast<int>((obstacles(k, 0) - ox) / grid_res_);
            int gj = static_cast<int>((obstacles(k, 1) - oy) / grid_res_);
            if (gi >= 0 && gi < n && gj >= 0 && gj < n) {
                grid[gi * n + gj] = static_cast<float>(OCCUPIED);
                obs_cells.push_back({gi, gj});
            }
        }
        if (obs_cells.empty()) return grid;

        // Multi-source BFS distance transform with true Euclidean distance.
        // Each cell tracks (nearest_obs_i, nearest_obs_j) and computes exact distance.
        {
            int max_r = static_cast<int>(std::ceil(cost_radius_ / grid_res_));
            float max_dist = static_cast<float>(max_r) * static_cast<float>(grid_res_);

            // For each cell: nearest obstacle grid coords (-1 = unvisited)
            std::vector<int16_t> near_i(n * n, -1);
            std::vector<int16_t> near_j(n * n, -1);

            std::vector<bool> seen(n * n, false);
            std::queue<int> bfs;
            for (auto& c : obs_cells) {
                int idx = c.i * n + c.j;
                if (!seen[idx]) {
                    seen[idx] = true;
                    near_i[idx] = static_cast<int16_t>(c.i);
                    near_j[idx] = static_cast<int16_t>(c.j);
                    bfs.push(idx);
                }
            }

            static const int DI8[8] = {-1, 1, 0, 0, -1, -1, 1, 1};
            static const int DJ8[8] = {0, 0, -1, 1, -1, 1, -1, 1};
            while (!bfs.empty()) {
                int idx = bfs.front(); bfs.pop();
                int ci = idx / n, cj = idx % n;
                int oi = near_i[idx], oj = near_j[idx];
                for (int d = 0; d < 8; ++d) {
                    int ni = ci + DI8[d], nj = cj + DJ8[d];
                    if (ni < 0 || ni >= n || nj < 0 || nj >= n) continue;
                    int nidx = ni * n + nj;
                    if (near_i[nidx] >= 0) continue;  // already assigned
                    // Check if within max radius (Manhattan pre-filter)
                    if (std::abs(ni - oi) + std::abs(nj - oj) > max_r * 2) continue;
                    // True Euclidean distance from this cell to nearest obstacle
                    double ed = std::hypot(ni - oi, nj - oj) * grid_res_;
                    if (ed > max_dist) continue;
                    near_i[nidx] = static_cast<int16_t>(oi);
                    near_j[nidx] = static_cast<int16_t>(oj);
                    bfs.push(nidx);
                }
            }

            // Apply cost from distance field
            double ir = inflation_radius_;
            double cr = cost_radius_;
            for (int k = 0; k < n * n; ++k) {
                if (near_i[k] < 0) continue;  // no obstacle nearby
                int ci = k / n, cj = k % n;
                if (ci == near_i[k] && cj == near_j[k]) continue;  // obstacle cell itself
                double d = std::hypot(ci - near_i[k], cj - near_j[k]) * grid_res_;
                float cost;
                if (d <= ir) {
                    cost = static_cast<float>(OCCUPIED);
                } else if (d <= cr) {
                    double t = (d - ir) / (cr - ir);
                    cost = static_cast<float>(0.95 * (1.0 - t));
                } else {
                    continue;
                }
                if (cost > grid[k]) grid[k] = cost;
            }
        }

        // Inflate bad breadcrumbs from loaded trail
        if (repeat_breadcrumb_ && !loaded_bad_bcs_.empty()) {
            int bad_r = static_cast<int>(cost_radius_ / grid_res_);
            for (auto& bc : loaded_bad_bcs_) {
                int ci = static_cast<int>((bc.first - ox) / grid_res_);
                int cj = static_cast<int>((bc.second - oy) / grid_res_);
                for (int di = -bad_r; di <= bad_r; ++di) {
                    for (int dj = -bad_r; dj <= bad_r; ++dj) {
                        int ni = ci + di, nj = cj + dj;
                        if (ni >= 0 && ni < n && nj >= 0 && nj < n) {
                            double dist = std::hypot(di, dj) * grid_res_;
                            if (dist <= cost_radius_) {
                                float cost = static_cast<float>(
                                    OCCUPIED * std::max(0.0, 1.0 - dist / cost_radius_));
                                int idx = ni * n + nj;
                                if (cost > grid[idx]) grid[idx] = cost;
                            }
                        }
                    }
                }
            }
        }

        return grid;
    }

    // -----------------------------------------------------------------------
    // Hybrid A*  — searches in (x, y, θ) continuous space
    // -----------------------------------------------------------------------
    struct HybridNode {
        double f;
        int idx;  // flat index into (i, j, k) grid
        bool operator>(const HybridNode& o) const { return f > o.f; }
    };

    // Check if an arc from (x,y,θ) with curvature κ for length step is collision-free.
    // Samples the arc at sub-step resolution and checks the costmap.
    // Returns accumulated grid cost along the arc (0 = fully free).
    double arcCost(const std::vector<float>& grid, double ox, double oy,
                   int n, double x, double y, double th, double kappa, double step,
                   double& out_x, double& out_y, double& out_th) const {
        int nsub = std::max(3, static_cast<int>(std::ceil(step / (grid_res_ * 0.7))));
        double ds = step / nsub;
        double acc_cost = 0.0;
        double cx = x, cy = y, cth = th;
        for (int s = 0; s < nsub; ++s) {
            cth += kappa * ds;
            cx += std::cos(cth) * ds;
            cy += std::sin(cth) * ds;
            int gi = static_cast<int>((cx - ox) / grid_res_);
            int gj = static_cast<int>((cy - oy) / grid_res_);
            if (gi < 0 || gi >= n || gj < 0 || gj >= n) return -1.0;
            float cell = grid[gi * n + gj];
            if (cell >= static_cast<float>(OCCUPIED)) return -1.0;
            acc_cost += static_cast<double>(cell);
        }
        out_x = cx; out_y = cy; out_th = normalizeAngle(cth);
        return acc_cost / nsub;  // average cost
    }

    // Try straight-line analytic expansion from (x,y) to (gx,gy) at heading th.
    // Returns true if the line is collision-free and heading-aligned enough.
    bool analyticExpand(const std::vector<float>& grid, double ox, double oy, int n,
                        double x, double y, double th, double gx, double gy,
                        std::vector<std::pair<double,double>>& seg) const {
        double dx = gx - x, dy = gy - y;
        double dist = std::hypot(dx, dy);
        if (dist < 0.05) { seg.push_back({gx, gy}); return true; }
        double target_th = std::atan2(dy, dx);
        if (std::fabs(normalizeAngle(target_th - th)) > analytic_heading_tol_) return false;

        int nsteps = std::max(3, static_cast<int>(std::ceil(dist / (grid_res_ * 0.7))));
        for (int s = 1; s <= nsteps; ++s) {
            double f = static_cast<double>(s) / nsteps;
            double px = x + dx * f, py = y + dy * f;
            int gi = static_cast<int>((px - ox) / grid_res_);
            int gj = static_cast<int>((py - oy) / grid_res_);
            if (gi < 0 || gi >= n || gj < 0 || gj >= n) return false;
            if (grid[gi * n + gj] >= static_cast<float>(analytic_cost_limit_)) return false;
            seg.push_back({px, py});
        }
        return true;
    }

    std::vector<std::pair<double,double>> hybridAstar(
        const std::vector<float>& grid, double ox, double oy,
        double sx, double sy, double sth,
        double gx, double gy)
    {
        int n = grid_cells_;
        int nbins = hybrid_heading_bins_;
        double bin_size = 2.0 * M_PI / nbins;
        int nsteer = hybrid_num_steer_;
        double step = hybrid_step_;
        double max_kappa = hybrid_max_steer_;

        // Ensure start is in free space
        {
            int si = static_cast<int>((sx - ox) / grid_res_);
            int sj = static_cast<int>((sy - oy) / grid_res_);
            if (si < 0 || si >= n || sj < 0 || sj >= n) return {};
            if (grid[si * n + sj] >= static_cast<float>(OCCUPIED)) {
                auto f = nearestFree(grid, si, sj, n);
                if (f.first < 0) return {};
                sx = ox + (f.first + 0.5) * grid_res_;
                sy = oy + (f.second + 0.5) * grid_res_;
            }
        }

        // 3D index: (i, j, k) → flat
        auto toIdx = [&](int i, int j, int k) -> int {
            return (i * n + j) * nbins + k;
        };
        auto thetaBin = [&](double th) -> int {
            double a = normalizeAngle(th);
            if (a < 0) a += 2.0 * M_PI;
            int k = static_cast<int>(a / bin_size) % nbins;
            return k;
        };

        int total = n * n * nbins;
        if (total > ha_buf_size_) {
            ha_gs_.resize(total);
            ha_came_from_.resize(total);
            ha_node_x_.resize(total);
            ha_node_y_.resize(total);
            ha_node_th_.resize(total);
            ha_closed_.resize(total);
            ha_buf_size_ = total;
        }
        std::fill(ha_gs_.begin(), ha_gs_.begin() + total, std::numeric_limits<float>::infinity());
        std::fill(ha_came_from_.begin(), ha_came_from_.begin() + total, -1);
        std::fill(ha_node_x_.begin(), ha_node_x_.begin() + total, 0.0f);
        std::fill(ha_node_y_.begin(), ha_node_y_.begin() + total, 0.0f);
        std::fill(ha_node_th_.begin(), ha_node_th_.begin() + total, 0.0f);
        std::fill(ha_closed_.begin(), ha_closed_.begin() + total, false);
        auto& gs = ha_gs_;
        auto& came_from = ha_came_from_;
        auto& node_x = ha_node_x_;
        auto& node_y = ha_node_y_;
        auto& node_th = ha_node_th_;
        auto& closed = ha_closed_;

        int sk = thetaBin(sth);
        int si = static_cast<int>((sx - ox) / grid_res_);
        int sj = static_cast<int>((sy - oy) / grid_res_);
        si = std::max(0, std::min(n - 1, si));
        sj = std::max(0, std::min(n - 1, sj));
        int sidx = toIdx(si, sj, sk);
        gs[sidx] = 0.0f;
        node_x[sidx] = static_cast<float>(sx);
        node_y[sidx] = static_cast<float>(sy);
        node_th[sidx] = static_cast<float>(sth);

        std::priority_queue<HybridNode, std::vector<HybridNode>, std::greater<HybridNode>> heap;
        heap.push({std::hypot(gx - sx, gy - sy), sidx});

        // Precompute steering curvatures
        std::vector<double> kappas(nsteer);
        for (int s = 0; s < nsteer; ++s) {
            kappas[s] = -max_kappa + 2.0 * max_kappa * s / std::max(1, nsteer - 1);
        }

        int goal_idx = -1;  // set when we reach the goal
        // Analytic expansion storage: if analytic succeeds, store the segment here
        std::vector<std::pair<double,double>> analytic_seg;
        int analytic_parent = -1;

        int iterations = 0;
        while (!heap.empty() && iterations < 200000) {
            ++iterations;
            HybridNode cur = heap.top();
            heap.pop();
            int cidx = cur.idx;
            if (closed[cidx]) continue;
            closed[cidx] = true;

            double cx = static_cast<double>(node_x[cidx]);
            double cy = static_cast<double>(node_y[cidx]);
            double cth = static_cast<double>(node_th[cidx]);

            double d2g = std::hypot(gx - cx, gy - cy);

            // Goal check: within one step
            if (d2g < step * 1.2) {
                goal_idx = cidx;
                break;
            }

            // Analytic expansion: try straight-line when close
            if (d2g < hybrid_analytic_dist_) {
                analytic_seg.clear();
                if (analyticExpand(grid, ox, oy, n, cx, cy, cth, gx, gy, analytic_seg)) {
                    analytic_parent = cidx;
                    goal_idx = cidx;
                    break;
                }
            }

            // Expand: try each steering angle
            for (int s = 0; s < nsteer; ++s) {
                double kappa = kappas[s];
                double nx, ny, nth;
                double arc_c = arcCost(grid, ox, oy, n, cx, cy, cth, kappa, step, nx, ny, nth);
                if (arc_c < 0.0) continue;  // collision

                int ni = static_cast<int>((nx - ox) / grid_res_);
                int nj = static_cast<int>((ny - oy) / grid_res_);
                int nk = thetaBin(nth);
                if (ni < 0 || ni >= n || nj < 0 || nj >= n) continue;
                int nidx = toIdx(ni, nj, nk);

                float tg = gs[cidx] + static_cast<float>(step * (1.0 + obstacle_cost_weight_ * arc_c));
                // Penalize steering
                tg += static_cast<float>(steer_penalty_ * std::fabs(kappa) / max_kappa);

                if (tg < gs[nidx]) {
                    gs[nidx] = tg;
                    double h = std::hypot(gx - nx, gy - ny);
                    heap.push({static_cast<double>(tg) + h * h_weight_, nidx});
                    came_from[nidx] = cidx;
                    node_x[nidx] = static_cast<float>(nx);
                    node_y[nidx] = static_cast<float>(ny);
                    node_th[nidx] = static_cast<float>(nth);
                }
            }
        }

        if (goal_idx < 0) return {};  // no path found

        // Reconstruct: walk back from goal_idx
        std::vector<std::pair<double,double>> path;
        // If analytic expansion succeeded, prepend the analytic segment
        if (analytic_parent >= 0 && !analytic_seg.empty()) {
            // First reconstruct from analytic_parent back to start
            int idx = analytic_parent;
            while (idx >= 0) {
                path.push_back({static_cast<double>(node_x[idx]),
                                static_cast<double>(node_y[idx])});
                idx = came_from[idx];
            }
            std::reverse(path.begin(), path.end());
            // Append analytic segment
            for (auto& p : analytic_seg) path.push_back(p);
        } else {
            int idx = goal_idx;
            while (idx >= 0) {
                path.push_back({static_cast<double>(node_x[idx]),
                                static_cast<double>(node_y[idx])});
                idx = came_from[idx];
            }
            std::reverse(path.begin(), path.end());
        }

        ROS_INFO_THROTTLE(5.0, "Hybrid A*: %d iterations, %zu waypoints", iterations, path.size());
        return path;
    }

    std::pair<int,int> nearestFree(const std::vector<float>& grid, int ci, int cj, int n) {
        for (int r = 1; r < 25; ++r) {
            for (int di = -r; di <= r; ++di) {
                for (int dj = -r; dj <= r; ++dj) {
                    if (std::abs(di) == r || std::abs(dj) == r) {
                        int ni = ci + di, nj = cj + dj;
                        if (ni >= 0 && ni < n && nj >= 0 && nj < n &&
                            grid[ni * n + nj] < 1.0f) {
                            return {ni, nj};
                        }
                    }
                }
            }
        }
        return {-1, -1};
    }

    void publishCostmap(const std::vector<float>& grid, const Eigen::MatrixXd& /*obstacles*/, double ox, double oy) {
        int n = grid_cells_;

        nav_msgs::OccupancyGrid msg;
        msg.header.stamp = ros::Time::now();
        msg.header.frame_id = "odom";
        msg.info.resolution = static_cast<float>(grid_res_);
        msg.info.width  = n;
        msg.info.height = n;
        msg.info.origin.position.x = ox;
        msg.info.origin.position.y = oy;
        msg.info.origin.orientation.w = 1.0;
        msg.data.resize(n * n);
        for (int xi = 0; xi < n; ++xi) {
            for (int yi = 0; yi < n; ++yi) {
                float v = grid[xi * n + yi];
                int ros_idx = yi * n + xi;
                msg.data[ros_idx] = (v <= 0.0f) ? 0 : static_cast<int8_t>(std::min(100.0f, v * 100.0f));
            }
        }
        costmap_pub_.publish(msg);
    }


    // -----------------------------------------------------------------------
    // Planning
    // -----------------------------------------------------------------------
    bool plan() {
        Eigen::MatrixXd obstacles = scanToOdom();
        double ox, oy;
        std::vector<float> grid = buildGrid(obstacles, ox, oy);
        publishCostmap(grid, obstacles, ox, oy);

        double dx = goal_x_ - robot_x_;
        double dy = goal_y_ - robot_y_;
        double dist = std::hypot(dx, dy);
        if (dist < 0.1) return false;

        // Compute planning target (clamped to grid)
        double la = std::min(dist, local_size_ * 0.48);
        double target_x = robot_x_ + (dx / dist) * la;
        double target_y = robot_y_ + (dy / dist) * la;

        auto wp = hybridAstar(grid, ox, oy,
                              robot_x_, robot_y_, robot_yaw_,
                              target_x, target_y);
        if (wp.empty()) {
            ROS_WARN_THROTTLE(2.0, "Hybrid A*: no path");
            return false;
        }

        // Light resample: Hybrid A* output is already smooth arcs,
        // just ensure uniform waypoint spacing for the controller
        path_ = resamplePath(wp, 0.15);
        path_idx_ = 0;
        advanceIdx();
        pubPath();
        return true;
    }

    // Uniform resample: interpolate waypoints at fixed spacing
    std::vector<std::pair<double,double>> resamplePath(
        const std::vector<std::pair<double,double>>& wp, double spacing)
    {
        if (wp.size() <= 1) return wp;
        std::vector<std::pair<double,double>> out;
        out.push_back(wp[0]);
        double acc = 0.0;
        for (int i = 1; i < static_cast<int>(wp.size()); ++i) {
            double ddx = wp[i].first - wp[i - 1].first;
            double ddy = wp[i].second - wp[i - 1].second;
            double seg = std::hypot(ddx, ddy);
            if (seg < 1e-6) continue;
            acc += seg;
            while (acc >= spacing) {
                acc -= spacing;
                double f = 1.0 - acc / seg;
                out.push_back({wp[i - 1].first + ddx * f, wp[i - 1].second + ddy * f});
            }
        }
        if (out.back() != wp.back()) out.push_back(wp.back());
        return out;
    }

    void advanceIdx() {
        while (path_idx_ < static_cast<int>(path_.size()) - 1) {
            double ddx = path_[path_idx_].first - robot_x_;
            double ddy = path_[path_idx_].second - robot_y_;
            double ang_err = std::fabs(normalizeAngle(std::atan2(ddy, ddx) - robot_yaw_));
            // Advance if close enough, or waypoint is clearly behind the robot (already passed)
            if (std::hypot(ddx, ddy) < checkpoint_dist_ || ang_err > (M_PI * 5.0 / 6.0)) {
                path_idx_++;
            } else {
                break;
            }
        }
    }

    // -----------------------------------------------------------------------
    // Debug markers for RViz
    // -----------------------------------------------------------------------
    void pubDebugMarkers(double tx, double ty, const std::string& branch, double ae) {
        if (marker_pub_.getNumSubscribers() == 0) return;
        visualization_msgs::MarkerArray ma;
        ros::Time now = ros::Time::now();

        // Helper: fill common marker fields
        auto mk = [&](int id, int type) {
            visualization_msgs::Marker m;
            m.header.frame_id = "odom";
            m.header.stamp = now;
            m.ns = "astar_debug";
            m.id = id;
            m.type = type;
            m.action = visualization_msgs::Marker::ADD;
            m.pose.orientation.w = 1.0;
            m.lifetime = ros::Duration(0.3);
            return m;
        };

        double d2g = std::hypot(goal_x_ - robot_x_, goal_y_ - robot_y_);

        // 0 — Goal: large red sphere (persistent so it doesn't flicker)
        {
            auto m = mk(0, visualization_msgs::Marker::SPHERE);
            m.lifetime = ros::Duration(2.0);
            m.pose.position.x = goal_x_;
            m.pose.position.y = goal_y_;
            m.scale.x = m.scale.y = m.scale.z = 0.5;
            m.color.r = 1.0f; m.color.g = 0.0f; m.color.b = 0.0f; m.color.a = 1.0f;
            ma.markers.push_back(m);
        }

        // 1 — Current waypoint target: yellow sphere
        {
            auto m = mk(1, visualization_msgs::Marker::SPHERE);
            m.pose.position.x = tx;
            m.pose.position.y = ty;
            m.scale.x = m.scale.y = m.scale.z = 0.25;
            m.color.r = 1.0f; m.color.g = 1.0f; m.color.b = 0.0f; m.color.a = 1.0f;
            ma.markers.push_back(m);
        }

        // 2 — Arrow: robot → goal (cyan)
        {
            auto m = mk(2, visualization_msgs::Marker::ARROW);
            geometry_msgs::Point p0, p1;
            p0.x = robot_x_; p0.y = robot_y_; p0.z = 0.05;
            p1.x = goal_x_;  p1.y = goal_y_;  p1.z = 0.05;
            m.points.push_back(p0); m.points.push_back(p1);
            m.scale.x = 0.04; m.scale.y = 0.10; m.scale.z = 0.0;
            m.color.r = 0.0f; m.color.g = 1.0f; m.color.b = 1.0f; m.color.a = 0.8f;
            ma.markers.push_back(m);
        }

        // 3 — Arrow: robot → waypoint target (green), only if different from goal
        if (std::hypot(tx - goal_x_, ty - goal_y_) > 0.1) {
            auto m = mk(3, visualization_msgs::Marker::ARROW);
            geometry_msgs::Point p0, p1;
            p0.x = robot_x_; p0.y = robot_y_; p0.z = 0.05;
            p1.x = tx;       p1.y = ty;       p1.z = 0.05;
            m.points.push_back(p0); m.points.push_back(p1);
            m.scale.x = 0.04; m.scale.y = 0.10; m.scale.z = 0.0;
            m.color.r = 0.0f; m.color.g = 1.0f; m.color.b = 0.0f; m.color.a = 1.0f;
            ma.markers.push_back(m);
        } else {
            // Clear old waypoint arrow when pointing at goal
            auto m = mk(3, visualization_msgs::Marker::DELETE);
            ma.markers.push_back(m);
        }

        // 4 — Robot heading arrow (blue)
        {
            auto m = mk(4, visualization_msgs::Marker::ARROW);
            geometry_msgs::Point p0, p1;
            p0.x = robot_x_; p0.y = robot_y_; p0.z = 0.05;
            p1.x = robot_x_ + 0.8 * std::cos(robot_yaw_);
            p1.y = robot_y_ + 0.8 * std::sin(robot_yaw_);
            p1.z = 0.05;
            m.points.push_back(p0); m.points.push_back(p1);
            m.scale.x = 0.05; m.scale.y = 0.12; m.scale.z = 0.0;
            m.color.r = 0.2f; m.color.g = 0.4f; m.color.b = 1.0f; m.color.a = 1.0f;
            ma.markers.push_back(m);
        }

        // 5 — Status text above robot
        {
            auto m = mk(5, visualization_msgs::Marker::TEXT_VIEW_FACING);
            m.pose.position.x = robot_x_;
            m.pose.position.y = robot_y_;
            m.pose.position.z = 0.6;
            m.scale.z = 0.25;
            m.color.r = 1.0f; m.color.g = 1.0f; m.color.b = 1.0f; m.color.a = 1.0f;
            char buf[128];
            std::snprintf(buf, sizeof(buf), "%s\nd2g=%.2fm ae=%.0f°",
                          branch.c_str(), d2g, ae * 180.0 / M_PI);
            m.text = buf;
            ma.markers.push_back(m);
        }

        // 6 — Breadcrumbs: small blue spheres
        if (!breadcrumbs_.empty()) {
            auto m = mk(6, visualization_msgs::Marker::SPHERE_LIST);
            m.scale.x = m.scale.y = m.scale.z = 0.10;
            m.color.r = 0.3f; m.color.g = 0.3f; m.color.b = 1.0f; m.color.a = 0.7f;
            for (auto& bc : breadcrumbs_) {
                geometry_msgs::Point p;
                p.x = bc.first; p.y = bc.second; p.z = 0.02;
                m.points.push_back(p);
            }
            ma.markers.push_back(m);
        }

        // 7 — Recovery trail: magenta spheres along the backtrack path
        if (in_recovery_ && !recovery_trail_.empty()) {
            auto m = mk(7, visualization_msgs::Marker::SPHERE_LIST);
            m.scale.x = m.scale.y = m.scale.z = 0.15;
            m.color.r = 1.0f; m.color.g = 0.0f; m.color.b = 1.0f; m.color.a = 0.8f;
            for (int ri = recovery_trail_idx_; ri < static_cast<int>(recovery_trail_.size()); ++ri) {
                geometry_msgs::Point p;
                p.x = recovery_trail_[ri].first;
                p.y = recovery_trail_[ri].second;
                p.z = 0.08;
                m.points.push_back(p);
            }
            ma.markers.push_back(m);
        }

        // 8 — Loaded good trail: blue squares
        if (repeat_breadcrumb_ && !loaded_good_trail_.empty()) {
            auto m = mk(8, visualization_msgs::Marker::CUBE_LIST);
            m.scale.x = m.scale.y = 0.12; m.scale.z = 0.02;
            m.color.r = 0.2f; m.color.g = 0.4f; m.color.b = 1.0f; m.color.a = 0.6f;
            m.lifetime = ros::Duration(2.0);
            for (auto& bc : loaded_good_trail_) {
                geometry_msgs::Point p;
                p.x = bc.first; p.y = bc.second; p.z = 0.01;
                m.points.push_back(p);
            }
            ma.markers.push_back(m);
        }

        // 9 — Loaded bad breadcrumbs: red squares
        if (repeat_breadcrumb_ && !loaded_bad_bcs_.empty()) {
            auto m = mk(9, visualization_msgs::Marker::CUBE_LIST);
            m.scale.x = m.scale.y = 0.12; m.scale.z = 0.02;
            m.color.r = 1.0f; m.color.g = 0.2f; m.color.b = 0.2f; m.color.a = 0.5f;
            m.lifetime = ros::Duration(2.0);
            for (auto& bc : loaded_bad_bcs_) {
                geometry_msgs::Point p;
                p.x = bc.first; p.y = bc.second; p.z = 0.01;
                m.points.push_back(p);
            }
            ma.markers.push_back(m);
        }

        // 10 — Collision check footprints: green (safe) / red (collision)
        if (!collision_viz_.empty()) {
            double hx = robot_length_ + 2.0 * proj_clearance_;
            double hy = robot_width_  + 2.0 * proj_clearance_;
            for (int ci = 0; ci < static_cast<int>(collision_viz_.size()); ++ci) {
                auto& fp = collision_viz_[ci];
                auto m = mk(100 + ci, visualization_msgs::Marker::CUBE);
                m.pose.position.x = fp.x;
                m.pose.position.y = fp.y;
                m.pose.position.z = 0.02;
                double hz = fp.th / 2.0;
                m.pose.orientation.z = std::sin(hz);
                m.pose.orientation.w = std::cos(hz);
                m.scale.x = hx;
                m.scale.y = hy;
                m.scale.z = 0.02;
                m.lifetime = ros::Duration(0.15);
                if (fp.collided) {
                    m.color.r = 1.0f; m.color.g = 0.0f; m.color.b = 0.0f; m.color.a = 0.5f;
                } else {
                    m.color.r = 0.0f; m.color.g = 1.0f; m.color.b = 0.0f; m.color.a = 0.25f;
                }
                ma.markers.push_back(m);
            }
        }

        marker_pub_.publish(ma);
    }

    void pubPath() {
        if (path_pub_.getNumSubscribers() == 0) return;
        nav_msgs::Path msg;
        msg.header.frame_id = "odom";
        msg.header.stamp = ros::Time::now();
        msg.poses.reserve(path_.size());
        for (auto& wp : path_) {
            geometry_msgs::PoseStamped ps;
            ps.header = msg.header;
            ps.pose.position.x = wp.first;
            ps.pose.position.y = wp.second;
            ps.pose.orientation.w = 1.0;
            msg.poses.push_back(ps);
        }
        path_pub_.publish(msg);
    }

    // -----------------------------------------------------------------------
    // Safety
    // -----------------------------------------------------------------------
    bool checkFwdCollision(double lv, double av, double& out_speed) {
        out_speed = lv;
        collision_viz_.clear();
        if (!have_scan_ || scan_ranges_.size() == 0) return true;

        int n = static_cast<int>(scan_ranges_.size());
        // Build valid obstacle points in base_link frame
        std::vector<double> olx, oly;
        olx.reserve(n); oly.reserve(n);
        for (int i = 0; i < n; ++i) {
            double r = scan_ranges_(i);
            if (std::isfinite(r) && r > scan_data_.range_min && r < scan_data_.range_max) {
                olx.push_back(r * std::cos(scan_angles_(i)));
                oly.push_back(r * std::sin(scan_angles_(i)));
            }
        }
        if (olx.empty()) return true;

        int nobs = static_cast<int>(olx.size());
        // Robot half-extents + clearance margin for the OBB check
        double hx = robot_length_ / 2.0 + proj_clearance_;  // front/back
        double hy = robot_width_  / 2.0 + proj_clearance_;   // left/right

        // Transform base_link obstacle points to odom for visualization storage
        double rcy = std::cos(robot_yaw_), rsy = std::sin(robot_yaw_);

        double x = 0.0, y = 0.0, th = 0.0;
        double dt = proj_dt_;
        double t = 0.0;
        while (t < proj_horizon_) {
            x += lv * std::cos(th) * dt;
            y += lv * std::sin(th) * dt;
            th += av * dt;
            t += dt;

            // Store pose in odom frame for marker visualization
            double odom_x = robot_x_ + x * rcy - y * rsy;
            double odom_y = robot_y_ + x * rsy + y * rcy;
            double odom_th = robot_yaw_ + th;

            // Transform obstacles into the projected robot frame and check OBB
            double ct = std::cos(-th), st = std::sin(-th);
            bool hit = false;
            for (int k = 0; k < nobs; ++k) {
                double dx = olx[k] - x;
                double dy = oly[k] - y;
                double lx = dx * ct - dy * st;
                double ly = dx * st + dy * ct;
                if (std::fabs(lx) < hx && std::fabs(ly) < hy) {
                    hit = true;
                    break;
                }
            }
            collision_viz_.push_back({odom_x, odom_y, odom_th, hit});
            if (hit) {
                double frac = std::max(0.0, t / proj_horizon_);
                out_speed = std::max(0.0, lv * frac * 0.4);
                return false;
            }
        }
        return true;
    }

    double minObs() {
        if (!have_scan_ || scan_ranges_.size() == 0) return std::numeric_limits<double>::infinity();
        double mn = std::numeric_limits<double>::infinity();
        int n = static_cast<int>(scan_ranges_.size());
        for (int i = 0; i < n; ++i) {
            double r = scan_ranges_(i);
            if (std::isfinite(r) && r > scan_data_.range_min && r < mn) mn = r;
        }
        return mn;
    }

    double fwdMin(double half_angle = 1.05) {
        if (!have_scan_ || scan_ranges_.size() == 0) return std::numeric_limits<double>::infinity();
        double mn = std::numeric_limits<double>::infinity();
        int n = static_cast<int>(scan_ranges_.size());
        for (int i = 0; i < n; ++i) {
            double r = scan_ranges_(i);
            double a = scan_angles_(i);
            if (std::fabs(a) < half_angle && std::isfinite(r) && r > scan_data_.range_min && r < mn)
                mn = r;
        }
        return mn;
    }

    double rearClear() {
        if (!have_scan_ || scan_ranges_.size() == 0) return std::numeric_limits<double>::infinity();
        double mn = std::numeric_limits<double>::infinity();
        int n = static_cast<int>(scan_ranges_.size());
        for (int i = 0; i < n; ++i) {
            double r = scan_ranges_(i);
            double a = scan_angles_(i);
            if (std::fabs(a) > 2.1 && std::isfinite(r) && r > scan_data_.range_min && r < mn)
                mn = r;
        }
        return mn;
    }

    void sectorClearance(double& f_d, double& l_d, double& r_d) {
        f_d = l_d = r_d = std::numeric_limits<double>::infinity();
        if (!have_scan_ || scan_ranges_.size() == 0) return;
        int n = static_cast<int>(scan_ranges_.size());
        for (int i = 0; i < n; ++i) {
            double r = scan_ranges_(i);
            double a = scan_angles_(i);
            if (!std::isfinite(r) || r <= scan_data_.range_min) continue;
            if (std::fabs(a) < 0.52) {
                if (r < f_d) f_d = r;
            }
            if (a > 0.52 && a < 2.09) {
                if (r < l_d) l_d = r;
            }
            if (a < -0.52 && a > -2.09) {
                if (r < r_d) r_d = r;
            }
        }
    }

    bool rotationSafe(double angular_vel) {
        if (!have_scan_ || scan_ranges_.size() == 0) return true;

        int n = static_cast<int>(scan_ranges_.size());
        std::vector<double> olx, oly;
        olx.reserve(n); oly.reserve(n);
        for (int i = 0; i < n; ++i) {
            double r = scan_ranges_(i);
            if (std::isfinite(r) && r > scan_data_.range_min && r < scan_data_.range_max) {
                olx.push_back(r * std::cos(scan_angles_(i)));
                oly.push_back(r * std::sin(scan_angles_(i)));
            }
        }
        if (olx.empty()) return true;

        int nobs = static_cast<int>(olx.size());

        // Robot half-dimensions (Jackal: 0.42 x 0.31)
        static const double hx = 0.21, hy = 0.155;
        static const double corners[4][2] = {{hx, hy}, {hx, -hy}, {-hx, hy}, {-hx, -hy}};

        // Project rotation over 0.5s in 5 steps
        for (int step = 1; step <= 5; ++step) {
            double theta = angular_vel * 0.1 * step;
            double ct = std::cos(theta), st = std::sin(theta);
            for (int c = 0; c < 4; ++c) {
                double rx = corners[c][0] * ct - corners[c][1] * st;
                double ry = corners[c][0] * st + corners[c][1] * ct;
                double min_d2 = std::numeric_limits<double>::infinity();
                for (int k = 0; k < nobs; ++k) {
                    double ddx = olx[k] - rx;
                    double ddy = oly[k] - ry;
                    double d2 = ddx * ddx + ddy * ddy;
                    if (d2 < min_d2) min_d2 = d2;
                }
                if (std::sqrt(min_d2) < 0.04) return false;
            }
        }
        return true;
    }

    // -----------------------------------------------------------------------
    // Recovery
    // -----------------------------------------------------------------------
    void updateBc() {
        double px = robot_x_, py = robot_y_;
        if (!last_bc_valid_) {
            breadcrumbs_.push_back({px, py});
            bc_log_.push_back({px, py, false});
            last_bc_x_ = px; last_bc_y_ = py;
            last_bc_valid_ = true;
            return;
        }
        if (std::hypot(px - last_bc_x_, py - last_bc_y_) >= breadcrumb_spacing_) {
            breadcrumbs_.push_back({px, py});
            bc_log_.push_back({px, py, false});
            last_bc_x_ = px; last_bc_y_ = py;
            if (static_cast<int>(breadcrumbs_.size()) > 50) {
                breadcrumbs_.erase(breadcrumbs_.begin(),
                                   breadcrumbs_.begin() + (static_cast<int>(breadcrumbs_.size()) - 50));
            }
        }
    }

    bool checkStuck() {
        double now = ros::Time::now().toSec();
        if (!stuck_check_init_) {
            stuck_check_init_ = true;
            stuck_check_time_ = now;
            stuck_check_x_ = robot_x_;
            stuck_check_y_ = robot_y_;
            return false;
        }
        if (now - stuck_check_time_ >= stuck_check_interval_) {
            double d = std::hypot(robot_x_ - stuck_check_x_, robot_y_ - stuck_check_y_);
            stuck_check_time_ = now;
            stuck_check_x_ = robot_x_;
            stuck_check_y_ = robot_y_;
            return d < stuck_threshold_;
        }
        return false;
    }

    void doRecovery() {
        double now = ros::Time::now().toSec();
        if (!in_recovery_) {
            in_recovery_ = true;
            recovery_start_ = now;
            recovery_count_++;
            int back_n = std::min(5 * recovery_count_, static_cast<int>(breadcrumbs_.size()));

            // Build recovery trail: the last back_n breadcrumbs in reverse order
            recovery_trail_.clear();
            recovery_trail_idx_ = 0;
            if (back_n >= 1 && static_cast<int>(breadcrumbs_.size()) >= back_n) {
                int start = static_cast<int>(breadcrumbs_.size()) - 1;
                int end = static_cast<int>(breadcrumbs_.size()) - back_n;
                for (int i = start; i >= end; --i) {
                    recovery_trail_.push_back(breadcrumbs_[i]);
                }
                // Mark removed breadcrumbs as bad in full log
                int log_sz = static_cast<int>(bc_log_.size());
                for (int i = 0; i < back_n && (log_sz - 1 - i) >= 0; ++i) {
                    bc_log_[log_sz - 1 - i].bad = true;
                }
                breadcrumbs_.resize(breadcrumbs_.size() - back_n);
            }
            ROS_WARN("A*: recovery #%d (trail %zu waypoints)", recovery_count_, recovery_trail_.size());
        }

        if (now - recovery_start_ > recovery_max_time_) {
            in_recovery_ = false;
            recovery_trail_.clear();
            consecutive_stops_ = 0;
            spin_count_ = 0;
            last_plan_time_ = 0.0;
            return;
        }

        geometry_msgs::Twist cmd;
        double obs_d = minObs();

        // Collision guard: obstacle inside footprint — reverse away
        if (obs_d < robot_radius_ - 0.02) {
            ROS_WARN_THROTTLE(1.0, "A*: recovery -- obstacle at %.2fm, reversing", obs_d);
            double rc = rearClear();
            if (rc > robot_radius_ + 0.05) {
                cmd.linear.x = -0.3;
            }
            cmd_pub_.publish(cmd);
            return;
        }

        // No trail — blind reverse (no rear LIDAR coverage, so always allow)
        if (recovery_trail_.empty() || recovery_trail_idx_ >= static_cast<int>(recovery_trail_.size())) {
            cmd.linear.x = -0.10;
            cmd_pub_.publish(cmd);
            return;
        }

        // Follow trail waypoint by waypoint
        double tx = recovery_trail_[recovery_trail_idx_].first;
        double ty = recovery_trail_[recovery_trail_idx_].second;
        double ddx = tx - robot_x_;
        double ddy = ty - robot_y_;
        double wp_dist = std::hypot(ddx, ddy);

        // Advance to next waypoint if close enough
        if (wp_dist < 0.25) {
            recovery_trail_idx_++;
            if (recovery_trail_idx_ >= static_cast<int>(recovery_trail_.size())) {
                // Finished trail
                in_recovery_ = false;
                recovery_trail_.clear();
                consecutive_stops_ = 0;
                spin_count_ = 0;
                last_plan_time_ = 0.0;
                return;
            }
            tx = recovery_trail_[recovery_trail_idx_].first;
            ty = recovery_trail_[recovery_trail_idx_].second;
            ddx = tx - robot_x_;
            ddy = ty - robot_y_;
        }

        // Reverse toward waypoint — steer while backing up, never turn around
        // No rearClear() gate: LIDAR has no rear coverage, check is meaningless
        double err = normalizeAngle(std::atan2(ddy, ddx) - robot_yaw_);
        double rear_err = normalizeAngle(err + M_PI);  // 0 = waypoint directly behind
        cmd.linear.x = -0.10;
        cmd.angular.z = clampd(1.0 * rear_err, -0.33, 0.33);
        cmd_pub_.publish(cmd);
    }

    // -----------------------------------------------------------------------
    // Breadcrumb trail persistence
    // -----------------------------------------------------------------------
    void saveBreadcrumbs() {
        char path[256];
        std::snprintf(path, sizeof(path), "/tmp/nav_logs/breadcrumbs_w%d.csv", world_idx_);
        std::ofstream f(path, std::ios::out | std::ios::trunc);
        if (!f.is_open()) return;
        f << "seq,x,y,status\n";
        for (int i = 0; i < static_cast<int>(bc_log_.size()); ++i) {
            f << i << "," << bc_log_[i].x << "," << bc_log_[i].y
              << "," << (bc_log_[i].bad ? "bad" : "good") << "\n";
        }
        f.close();
        ROS_INFO("A*: saved %zu breadcrumbs to %s", bc_log_.size(), path);
    }

    void loadBreadcrumbs() {
        char path[256];
        std::snprintf(path, sizeof(path), "/tmp/nav_logs/breadcrumbs_w%d.csv", world_idx_);
        std::ifstream f(path);
        if (!f.is_open()) {
            ROS_WARN("A*: no breadcrumb file at %s", path);
            repeat_breadcrumb_ = false;
            return;
        }
        std::string line;
        std::getline(f, line);  // skip header
        loaded_good_trail_.clear();
        loaded_bad_bcs_.clear();
        while (std::getline(f, line)) {
            int seq; double x, y; char status[16];
            if (std::sscanf(line.c_str(), "%d,%lf,%lf,%15s", &seq, &x, &y, status) == 4) {
                if (std::string(status) == "good")
                    loaded_good_trail_.push_back({x, y});
                else
                    loaded_bad_bcs_.push_back({x, y});
            }
        }
        using_loaded_trail_ = !loaded_good_trail_.empty();
        loaded_trail_idx_ = 0;
        ROS_INFO("A*: loaded %zu good + %zu bad breadcrumbs from %s",
                 loaded_good_trail_.size(), loaded_bad_bcs_.size(), path);
    }

    // -----------------------------------------------------------------------
    // Progress deadlock
    // -----------------------------------------------------------------------
    bool checkProgress(double d2g) {
        double now = ros::Time::now().toSec();
        if (!progress_time_init_) {
            progress_time_init_ = true;
            progress_time_ = now;
            progress_d2g_ = d2g;
            return false;
        }
        if (d2g < progress_d2g_ - progress_min_) {
            progress_time_ = now;
            progress_d2g_ = d2g;
            return false;
        }
        if (now - progress_time_ >= progress_window_) {
            return true;
        }
        return false;
    }

    double findClearHeading() {
        if (!have_scan_ || scan_ranges_.size() == 0) return robot_yaw_;
        int total = static_cast<int>(scan_ranges_.size());
        int n_sectors = 12;
        int sector_size = total / n_sectors;
        double best_score = -1.0;
        double best_angle = robot_yaw_;
        double goal_angle = std::atan2(goal_y_ - robot_y_, goal_x_ - robot_x_);

        for (int i = 0; i < n_sectors; ++i) {
            int start = i * sector_size;
            int end = start + sector_size;
            double clearance = std::numeric_limits<double>::infinity();
            bool has_valid = false;
            for (int k = start; k < end && k < total; ++k) {
                double r = scan_ranges_(k);
                if (std::isfinite(r) && r > 0.01) {
                    has_valid = true;
                    if (r < clearance) clearance = r;
                }
            }
            if (!has_valid) continue;
            if (clearance < robot_radius_ + 0.10) continue;
            int mid_idx = (start + end) / 2;
            if (mid_idx >= total) mid_idx = total - 1;
            double world_angle = robot_yaw_ + scan_angles_(mid_idx);
            double goal_align = std::cos(normalizeAngle(world_angle - goal_angle));
            double score = 0.6 * std::min(clearance, 3.0) / 3.0 + 0.4 * (goal_align + 1.0) / 2.0;
            if (score > best_score) {
                best_score = score;
                best_angle = world_angle;
            }
        }
        return best_angle;
    }

    void doDeadlockEscape() {
        double now = ros::Time::now().toSec();
        double elapsed = now - escape_start_;
        geometry_msgs::Twist cmd;

        if (escape_phase_ == 0) {
            double ae = normalizeAngle(escape_heading_ - robot_yaw_);
            if (std::fabs(ae) < 0.3 || elapsed > 3.0) {
                escape_phase_ = 1;
                escape_start_ = now;
                ROS_WARN("A*: deadlock escape -- drive phase");
            } else {
                double av = clampd(2.0 * ae, -1.5, 1.5);
                if (rotationSafe(av)) {
                    cmd.angular.z = av;
                }
            }
        } else {
            if (elapsed > 4.0) {
                endDeadlockEscape();
                return;
            }
            double fwd = fwdMin(1.05);
            if (fwd > robot_radius_ + 0.15) {
                cmd.linear.x = 0.3;
                double ae = normalizeAngle(escape_heading_ - robot_yaw_);
                cmd.angular.z = clampd(1.5 * ae, -1.0, 1.0);
            } else {
                endDeadlockEscape();
                return;
            }
        }
        cmd_pub_.publish(cmd);
    }

    void endDeadlockEscape() {
        deadlock_escape_ = false;
        in_recovery_ = false;
        recovery_trail_.clear();
        consecutive_stops_ = 0;
        spin_count_ = 0;
        path_.clear();
        last_plan_time_ = 0.0;
        progress_time_ = ros::Time::now().toSec();
        progress_d2g_ = std::hypot(goal_x_ - robot_x_, goal_y_ - robot_y_);
        ROS_WARN("A*: deadlock escape done -- replanning");
    }

    // -----------------------------------------------------------------------
    // Logger
    // -----------------------------------------------------------------------
    void logRow(double lv, double av, double o, double f, double cl,
                double tx, double ty, double ae,
                const std::string& st, const std::string& branch,
                int pl, double d) {
        if (!log_file_.is_open()) return;
        char buf[512];
        std::snprintf(buf, sizeof(buf),
            "%.3f,%.3f,%.3f,%.2f,%.3f,%.3f,%.3f,%.3f,%.3f,%.2f,%.2f,%.3f,%s,%s,%d,%.3f,%d,%d,%d,%d,%d,%d\n",
            ros::Time::now().toSec(), robot_x_, robot_y_, robot_yaw_,
            lv, av, o, f, cl, tx, ty, ae,
            st.c_str(), branch.c_str(), pl, d,
            consecutive_stops_, spin_count_, plan_fail_count_, recovery_count_,
            static_cast<int>(bc_log_.size()), using_loaded_trail_ ? 1 : 0);
        log_file_ << buf;
        log_file_.flush();
    }
};

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------
int main(int argc, char** argv) {
    ros::init(argc, argv, "nav_astar");
    AStarPlanner planner;
    planner.run();
    return 0;
}
