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

        robot_radius_      = std::hypot(0.42 / 2.0, 0.31 / 2.0); // 0.261m
        control_radius_    = robot_radius_;  // 0.261m — match actual robot diagonal
        proj_dt_           = 0.1;
        proj_horizon_      = 0.8;
        proj_clearance_    = 0.04;
        max_ang_accel_     = 4.0;
        prev_ang_vel_      = 0.0;

        // Goal from rosparam
        std::vector<double> goal_rel;
        if (!nh.getParam("goal_position", goal_rel) || goal_rel.size() < 2) {
            goal_rel = {0.0, 10.0};
        }
        goal_x_ = goal_rel[0];
        goal_y_ = goal_rel[1];

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
        recovery_target_valid_ = false;
        recovery_count_ = 0;
        plan_fail_count_ = 0;
        blend_ticks_ = 0;

        // Progress deadlock
        progress_time_init_ = false;
        progress_window_  = 30.0;
        progress_min_     = 0.5;
        deadlock_escape_  = false;
        escape_start_     = 0.0;
        escape_heading_   = 0.0;
        escape_phase_     = 0;

        // Laser TF
        laser_tf_valid_ = false;
        laser_tx_ = laser_ty_ = laser_yaw_ = 0.0;

        // Grid
        local_size_ = 12.0;
        grid_cells_ = static_cast<int>(local_size_ / grid_res_);

        buildKernel();

        // Pub/Sub
        cmd_pub_      = nh.advertise<geometry_msgs::Twist>("/cmd_vel", 1);
        path_pub_     = nh.advertise<nav_msgs::Path>("/nav_astar/path", 1, true);
        costmap_pub_  = nh.advertise<nav_msgs::OccupancyGrid>("/astar/costmap", 1, true);
        ctrl_costmap_pub_ = nh.advertise<nav_msgs::OccupancyGrid>("/astar/controller_costmap", 1, true);
        steer_samples_pub_ = nh.advertise<nav_msgs::Path>("/astar/steer_samples", 1);
        scan_sub_     = nh.subscribe("/front/scan", 1, &AStarPlanner::scanCb, this);

        // Logger
        {
            struct stat st;
            if (stat("/tmp/nav_logs", &st) != 0) {
                mkdir("/tmp/nav_logs", 0755);
            }
            int wid = 0;
            nh.param("/world_idx", wid, 0);
            char lp[256];
            std::snprintf(lp, sizeof(lp), "/tmp/nav_logs/astar_w%d.csv", wid);
            log_file_.open(lp, std::ios::out | std::ios::trunc);
            log_file_ << "t,x,y,yaw,lv,av,obs,fwd,cl,tx,ty,ae,st,branch,pl,d2g,cstops,spins,pfails,rcnt\n";
        }

        ROS_INFO("A* planner [C++]: goal(%.1f,%.1f) grid=%d inflation=%.3f cost=%.3f kernel=%zu",
                 goal_x_, goal_y_, grid_cells_, inflation_radius_, cost_radius_, kernel_.size());
    }

    // -----------------------------------------------------------------------
    void run() {
        ros::Rate rate(20);

        // Wait for scan + pose
        while (ros::ok()) {
            ros::spinOnce();
            if (have_scan_ && getPose()) break;
            rate.sleep();
        }
        ros::Duration(0.5).sleep();
        ros::spinOnce();
        getPose();

        ROS_INFO("A*: ready (%.2f,%.2f) yaw=%.2f -> goal(%.2f,%.2f)",
                 robot_x_, robot_y_, robot_yaw_, goal_x_, goal_y_);

        while (ros::ok()) {
            ros::spinOnce();
            if (!getPose()) {
                rate.sleep();
                continue;
            }

            double d2g = std::hypot(goal_x_ - robot_x_, goal_y_ - robot_y_);
            if (d2g < goal_tolerance_) {
                ROS_INFO("A*: GOAL!");
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
                rate.sleep();
                continue;
            }

            if (in_recovery_) {
                doRecovery();
                double rtx = recovery_target_valid_ ? recovery_target_x_ : 0.0;
                double rty = recovery_target_valid_ ? recovery_target_y_ : 0.0;
                logRow(0, 0, obs_d, fc, cl, rtx, rty, 0, "rec", "rec_cont", 0, d2g);
                rate.sleep();
                continue;
            }

            // Trigger recovery: stuck, too many stops, or spinning
            bool was_stuck = checkStuck();
            if (was_stuck || consecutive_stops_ >= collision_stop_threshold_ || spin_count_ >= 20) {
                std::string reason;
                if (was_stuck) reason = "rec_stuck";
                else if (consecutive_stops_ >= collision_stop_threshold_) reason = "rec_stops";
                else reason = "rec_spins";
                doRecovery();
                double rtx = recovery_target_valid_ ? recovery_target_x_ : 0.0;
                double rty = recovery_target_valid_ ? recovery_target_y_ : 0.0;
                logRow(0, 0, obs_d, fc, cl, rtx, rty, 0, "rec", reason, 0, d2g);
                rate.sleep();
                continue;
            }

            // Target waypoint
            double tx, ty;
            if (d2g < 2.0) {
                tx = goal_x_;
                ty = goal_y_;
            } else {
                double now = ros::Time::now().toSec();
                double iv = std::min(replan_interval_, 1.0);
                if (path_.empty() || path_idx_ >= static_cast<int>(path_.size()) - 1) {

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
                        if (plan_fail_count_ >= 3) {
                            ROS_WARN("A*: %d plan failures -- retreating", plan_fail_count_);
                            plan_fail_count_ = 0;
                            doRecovery();
                            double rtx = recovery_target_valid_ ? recovery_target_x_ : 0.0;
                            double rty = recovery_target_valid_ ? recovery_target_y_ : 0.0;
                            logRow(0, 0, obs_d, fc, cl, rtx, rty, 0, "rec", "rec_pfail", 0, d2g);
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
                    spin_count_ = 20;
                    branch = "rot_tight";
                } else if (f_d < rot_margin && turn_side_d < rot_margin) {
                    ROS_WARN_THROTTLE(2.0, "A*: surrounded (f=%.2f s=%.2f) -- retreating", f_d, turn_side_d);
                    spin_count_ = 20;
                    branch = "rot_surr";
                } else if (!rotationSafe((ae > 0 ? 1.0 : -1.0) * 1.5)) {
                    double rc = rearClear();
                    if (rc > robot_radius_ + 0.1) {
                        cmd.linear.x = -0.2;
                        ROS_INFO_THROTTLE(2.0, "A*: rotation unsafe, backing up");
                        branch = "rot_backup";
                    } else {
                        spin_count_ += 6;
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
                // Costmap clearance bias: steer toward lower-cost side
                double cost_bias = costmapSteerBias();
                raw_av += clampd(cost_bias * 0.8, -0.5, 0.5);
                raw_av = clampd(raw_av, -max_ang_, max_ang_);
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
            cmd_pub_.publish(cmd);
            pubPath();
            rate.sleep();
        }

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

    double robot_radius_, control_radius_;
    double proj_dt_, proj_horizon_, proj_clearance_;
    double max_ang_accel_;
    double prev_ang_vel_;

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
    bool   recovery_target_valid_;
    double recovery_target_x_, recovery_target_y_;
    int    recovery_count_;
    int    plan_fail_count_;
    int    blend_ticks_;

    // Progress deadlock
    bool   progress_time_init_;
    double progress_time_, progress_d2g_;
    double progress_window_, progress_min_;
    bool   deadlock_escape_;
    double escape_start_, escape_heading_;
    int    escape_phase_;

    // TF
    tf::TransformListener tf_listener_;
    bool   laser_tf_valid_;
    double laser_tx_, laser_ty_, laser_yaw_;

    // Grid
    double local_size_;
    int    grid_cells_;
    struct KernelEntry { int di, dj; double cost; };
    std::vector<KernelEntry> kernel_;

    // Cached grid for controller cost queries
    std::vector<float> cached_grid_;
    double cached_ox_, cached_oy_;
    bool   cached_grid_valid_ = false;

    // Pub/Sub
    ros::Publisher  cmd_pub_;
    ros::Publisher  path_pub_;
    ros::Publisher  costmap_pub_;
    ros::Publisher  ctrl_costmap_pub_;
    ros::Publisher  steer_samples_pub_;
    ros::Subscriber scan_sub_;

    // Logger
    std::ofstream log_file_;

    // -----------------------------------------------------------------------
    // Kernel
    // -----------------------------------------------------------------------
    void buildKernel() {
        double ir = inflation_radius_;  // hard wall for A*
        double cr = cost_radius_;       // soft gradient for controller
        int rc = std::min(static_cast<int>(std::ceil(cr / grid_res_)), 16);
        kernel_.clear();
        for (int di = -rc; di <= rc; ++di) {
            for (int dj = -rc; dj <= rc; ++dj) {
                double d = std::hypot(di, dj) * grid_res_;
                if (d <= ir) {
                    // Inside inflation radius: impassable for A*
                    kernel_.push_back({di, dj, OCCUPIED});
                } else if (d <= cr) {
                    // Between inflation and cost radius: exponential falloff for controller
                    double t = (d - ir) / (cr - ir);  // 0 at inflation edge, 1 at cost edge
                    double cost = 0.95 * std::exp(-7.0 * t);
                    kernel_.push_back({di, dj, cost});
                }
            }
        }
    }

    // -----------------------------------------------------------------------
    // Costmap-based clearance steering
    // -----------------------------------------------------------------------
    // Sample grid cost at a world point; returns 0 if out of bounds or no grid
    float sampleCost(double wx, double wy) const {
        if (!cached_grid_valid_) return 0.0f;
        int n = grid_cells_;
        int gi = static_cast<int>((wx - cached_ox_) / grid_res_);
        int gj = static_cast<int>((wy - cached_oy_) / grid_res_);
        if (gi < 0 || gi >= n || gj < 0 || gj >= n) return 0.0f;
        return cached_grid_[gi * n + gj];
    }

    // Returns angular bias: positive = steer left, negative = steer right
    // Samples 32 points in an 8x4 grid (8 lookahead distances x 4 lateral offsets per side)
    // Also publishes sample points as a Path msg for rviz visualization
    double costmapSteerBias() {
        if (!cached_grid_valid_) return 0.0;
        double left_cost = 0.0, right_cost = 0.0;
        double cy = std::cos(robot_yaw_), sy = std::sin(robot_yaw_);

        // 8 lookahead distances, 4 lateral offsets = 32 sample pairs (64 points total)
        static constexpr int N_FWD = 8;
        static constexpr int N_LAT = 4;
        static constexpr double fwd_min = 0.1, fwd_max = 0.8;
        static constexpr double lat_min = 0.10, lat_max = 0.30;

        nav_msgs::Path viz;
        viz.header.stamp = ros::Time::now();
        viz.header.frame_id = "odom";
        std::vector<geometry_msgs::PoseStamped> left_pts, right_pts;

        for (int fi = 0; fi < N_FWD; ++fi) {
            double fwd = fwd_min + (fwd_max - fwd_min) * fi / (N_FWD - 1);
            for (int li = 0; li < N_LAT; ++li) {
                double lat = lat_min + (lat_max - lat_min) * li / (N_LAT - 1);

                // Left sample
                double lx = robot_x_ + fwd * cy - lat * sy;
                double ly = robot_y_ + fwd * sy + lat * cy;
                float lc = sampleCost(lx, ly);
                left_cost += lc;

                // Right sample
                double rx = robot_x_ + fwd * cy + lat * sy;
                double ry = robot_y_ + fwd * sy - lat * cy;
                float rc = sampleCost(rx, ry);
                right_cost += rc;

                // Collect left/right separately to avoid zig-zag in Path viz
                geometry_msgs::PoseStamped ps;
                ps.header = viz.header;
                ps.pose.orientation.w = 1.0;
                ps.pose.position.x = lx;
                ps.pose.position.y = ly;
                ps.pose.position.z = lc;
                left_pts.push_back(ps);
                ps.pose.position.x = rx;
                ps.pose.position.y = ry;
                ps.pose.position.z = rc;
                right_pts.push_back(ps);
            }
        }

        viz.poses.insert(viz.poses.end(), left_pts.begin(), left_pts.end());
        viz.poses.insert(viz.poses.end(), right_pts.begin(), right_pts.end());
        steer_samples_pub_.publish(viz);

        // Steer away from the higher-cost side
        return (right_cost - left_cost);  // positive → steer left
    }

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

        // De-duplicate obstacle cells using a flat bool array
        std::vector<bool> seen(n * n, false);
        std::vector<OCell> unique_obs;
        unique_obs.reserve(obs_cells.size());
        for (auto& c : obs_cells) {
            int key = c.i * n + c.j;
            if (!seen[key]) {
                seen[key] = true;
                unique_obs.push_back(c);
            }
        }

        // Inflate using kernel
        for (auto& oc : unique_obs) {
            for (auto& ke : kernel_) {
                int ni = oc.i + ke.di;
                int nj = oc.j + ke.dj;
                if (ni >= 0 && ni < n && nj >= 0 && nj < n) {
                    int idx = ni * n + nj;
                    if (ke.cost > grid[idx]) {
                        grid[idx] = static_cast<float>(ke.cost);
                    }
                }
            }
        }

        return grid;
    }

    // -----------------------------------------------------------------------
    // A*
    // -----------------------------------------------------------------------
    struct AStarNode {
        double f;
        int i, j;
        bool operator>(const AStarNode& o) const { return f > o.f; }
    };

    std::vector<std::pair<int,int>> astar(const std::vector<float>& grid,
                                           int si, int sj, int gi, int gj) {
        int n = grid_cells_;
        si = std::max(0, std::min(n - 1, si));
        sj = std::max(0, std::min(n - 1, sj));
        gi = std::max(0, std::min(n - 1, gi));
        gj = std::max(0, std::min(n - 1, gj));

        // If start/goal occupied, find nearest free
        if (grid[si * n + sj] >= OCCUPIED) {
            auto f = nearestFree(grid, si, sj, n);
            if (f.first < 0) return {};
            si = f.first; sj = f.second;
        }
        if (grid[gi * n + gj] >= OCCUPIED) {
            auto f = nearestFree(grid, gi, gj, n);
            if (f.first < 0) return {};
            gi = f.first; gj = f.second;
        }

        static const int DI[8] = {-1, 1, 0, 0, -1, -1, 1, 1};
        static const int DJ[8] = {0, 0, -1, 1, -1, 1, -1, 1};
        static const double SC[8] = {1.0, 1.0, 1.0, 1.0, 1.414, 1.414, 1.414, 1.414};
        static const double TURN_PEN = 0.25;

        std::vector<float> gs(n * n, std::numeric_limits<float>::infinity());
        std::vector<int> came_from(n * n, -1);  // flat index of parent, -1 = none
        // Encode direction as (di+1)*3+(dj+1), range 0..8; 9 = no direction
        std::vector<uint8_t> cdir(n * n, 9);
        std::vector<bool> closed(n * n, false);

        gs[si * n + sj] = 0.0f;
        std::priority_queue<AStarNode, std::vector<AStarNode>, std::greater<AStarNode>> heap;
        heap.push({0.0, si, sj});

        int iterations = 0;
        while (!heap.empty() && iterations < 150000) {
            ++iterations;
            AStarNode cur = heap.top();
            heap.pop();
            int ci = cur.i, cj = cur.j;
            int cidx = ci * n + cj;
            if (closed[cidx]) continue;
            closed[cidx] = true;

            if (ci == gi && cj == gj) {
                // Reconstruct path
                std::vector<std::pair<int,int>> path;
                int idx = ci * n + cj;
                path.push_back({ci, cj});
                while (came_from[idx] >= 0) {
                    idx = came_from[idx];
                    path.push_back({idx / n, idx % n});
                }
                std::reverse(path.begin(), path.end());
                return path;
            }

            uint8_t pd = cdir[cidx];
            for (int d = 0; d < 8; ++d) {
                int ni = ci + DI[d];
                int nj = cj + DJ[d];
                if (ni < 0 || ni >= n || nj < 0 || nj >= n) continue;
                int nidx = ni * n + nj;
                if (closed[nidx]) continue;
                float cc = grid[nidx];
                if (cc >= OCCUPIED) continue;

                float tg = gs[cidx] + static_cast<float>(SC[d]) * (1.0f + 4.0f * cc);
                // Direction-change penalty
                uint8_t nd = static_cast<uint8_t>((DI[d] + 1) * 3 + (DJ[d] + 1));
                if (pd != 9 && nd != pd) {
                    tg += static_cast<float>(TURN_PEN);
                }
                if (tg < gs[nidx]) {
                    gs[nidx] = tg;
                    double h = std::hypot(ni - gi, nj - gj);
                    heap.push({tg + h, ni, nj});
                    came_from[nidx] = cidx;
                    cdir[nidx] = nd;
                }
            }
        }
        return {};
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

    void publishCostmap(const std::vector<float>& grid, const Eigen::MatrixXd& obstacles, double ox, double oy) {
        if (costmap_pub_.getNumSubscribers() == 0) return;
        int n = grid_cells_;
        double cr = cost_radius_;
        int kr = std::min(static_cast<int>(std::ceil(cr / grid_res_)), 8);

        // Build smooth exponential costmap for visualization (no hard inflation step)
        std::vector<float> smooth(n * n, 0.0f);
        if (obstacles.rows() > 0) {
            std::vector<bool> seen(n * n, false);
            std::vector<std::pair<int,int>> obs_cells;
            for (int k = 0; k < static_cast<int>(obstacles.rows()); ++k) {
                int gi = static_cast<int>((obstacles(k, 0) - ox) / grid_res_);
                int gj = static_cast<int>((obstacles(k, 1) - oy) / grid_res_);
                if (gi >= 0 && gi < n && gj >= 0 && gj < n) {
                    int idx = gi * n + gj;
                    if (!seen[idx]) {
                        seen[idx] = true;
                        smooth[idx] = 1.0f;
                        obs_cells.push_back({gi, gj});
                    }
                }
            }
            for (auto& oc : obs_cells) {
                for (int di = -kr; di <= kr; ++di) {
                    for (int dj = -kr; dj <= kr; ++dj) {
                        int ni = oc.first + di;
                        int nj = oc.second + dj;
                        if (ni < 0 || ni >= n || nj < 0 || nj >= n) continue;
                        double d = std::hypot(di, dj) * grid_res_;
                        if (d < 1e-6 || d > cr) continue;
                        float cost = static_cast<float>(std::exp(-3.0 * d / cr));
                        int idx = ni * n + nj;
                        if (cost > smooth[idx]) smooth[idx] = cost;
                    }
                }
            }
        }

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
                float v = smooth[xi * n + yi];
                int ros_idx = yi * n + xi;
                msg.data[ros_idx] = (v <= 0.0f) ? 0 : static_cast<int8_t>(std::min(100.0f, v * 100.0f));
            }
        }
        costmap_pub_.publish(msg);
    }

    // Publish cost-only grid (no inflation wall) — shows what the controller sees
    void publishCtrlCostmap(const Eigen::MatrixXd& obstacles, double ox, double oy) {
        if (ctrl_costmap_pub_.getNumSubscribers() == 0) return;
        int n = grid_cells_;
        double cr = cost_radius_;
        int kr = std::min(static_cast<int>(std::ceil(cr / grid_res_)), 8);

        // Build cost-only grid: smooth falloff from obstacle, no hard wall
        std::vector<float> cgrid(n * n, 0.0f);

        if (obstacles.rows() > 0) {
            // Collect unique obstacle cells
            std::vector<bool> seen(n * n, false);
            std::vector<std::pair<int,int>> obs_cells;
            for (int k = 0; k < static_cast<int>(obstacles.rows()); ++k) {
                int gi = static_cast<int>((obstacles(k, 0) - ox) / grid_res_);
                int gj = static_cast<int>((obstacles(k, 1) - oy) / grid_res_);
                if (gi >= 0 && gi < n && gj >= 0 && gj < n) {
                    int idx = gi * n + gj;
                    if (!seen[idx]) {
                        seen[idx] = true;
                        cgrid[idx] = 1.0f;
                        obs_cells.push_back({gi, gj});
                    }
                }
            }
            // Apply cost-only kernel (pure exponential, no inflation)
            for (auto& oc : obs_cells) {
                for (int di = -kr; di <= kr; ++di) {
                    for (int dj = -kr; dj <= kr; ++dj) {
                        int ni = oc.first + di;
                        int nj = oc.second + dj;
                        if (ni < 0 || ni >= n || nj < 0 || nj >= n) continue;
                        double d = std::hypot(di, dj) * grid_res_;
                        if (d < 1e-6 || d > cr) continue;
                        float cost = static_cast<float>(std::exp(-3.0 * d / cr));
                        int idx = ni * n + nj;
                        if (cost > cgrid[idx]) cgrid[idx] = cost;
                    }
                }
            }
        }

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
                float v = cgrid[xi * n + yi];
                int ros_idx = yi * n + xi;
                msg.data[ros_idx] = (v <= 0.0f) ? 0 : static_cast<int8_t>(std::min(100.0f, v * 100.0f));
            }
        }
        ctrl_costmap_pub_.publish(msg);
    }

    // -----------------------------------------------------------------------
    // Planning
    // -----------------------------------------------------------------------
    bool plan() {
        Eigen::MatrixXd obstacles = scanToOdom();
        double ox, oy;
        std::vector<float> grid = buildGrid(obstacles, ox, oy);
        publishCostmap(grid, obstacles, ox, oy);
        publishCtrlCostmap(obstacles, ox, oy);
        cached_grid_ = grid;
        cached_ox_ = ox;
        cached_oy_ = oy;
        cached_grid_valid_ = true;
        int n = grid_cells_;

        double si_d = (robot_x_ - ox) / grid_res_;
        double sj_d = (robot_y_ - oy) / grid_res_;

        double dx = goal_x_ - robot_x_;
        double dy = goal_y_ - robot_y_;
        double dist = std::hypot(dx, dy);
        if (dist < 0.1) return false;

        double la = std::min(dist, local_size_ * 0.48);
        double target_x = robot_x_ + (dx / dist) * la;
        double target_y = robot_y_ + (dy / dist) * la;
        double gi_d = std::max(0.0, std::min(static_cast<double>(n - 1), (target_x - ox) / grid_res_));
        double gj_d = std::max(0.0, std::min(static_cast<double>(n - 1), (target_y - oy) / grid_res_));

        auto gp = astar(grid, static_cast<int>(si_d), static_cast<int>(sj_d),
                         static_cast<int>(gi_d), static_cast<int>(gj_d));
        if (gp.empty()) {
            ROS_WARN_THROTTLE(2.0, "A*: no path");
            return false;
        }

        // Convert grid cells to odom coordinates
        std::vector<std::pair<double,double>> wp;
        wp.reserve(gp.size());
        for (auto& c : gp) {
            wp.push_back({ox + c.first * grid_res_, oy + c.second * grid_res_});
        }

        // [F1] Smooth
        auto sm = smooth(wp, grid, ox, oy);
        path_ = sm;
        path_idx_ = 0;
        advanceIdx();
        pubPath();
        return true;
    }

    std::vector<std::pair<double,double>> smooth(
        const std::vector<std::pair<double,double>>& wp,
        const std::vector<float>& grid, double ox, double oy)
    {
        if (wp.size() <= 2) return wp;
        int n = grid_cells_;

        // Shortcut pass
        std::vector<std::pair<double,double>> sc;
        sc.push_back(wp[0]);
        int i = 0;
        while (i < static_cast<int>(wp.size()) - 1) {
            int bj = i + 1;
            for (int j = static_cast<int>(wp.size()) - 1; j > i + 1; --j) {
                if (lineClear(wp[i], wp[j], grid, ox, oy, n)) {
                    bj = j;
                    break;
                }
            }
            sc.push_back(wp[bj]);
            i = bj;
        }

        if (sc.size() <= 2) return sc;

        // Resample at uniform spacing
        double spacing = 0.12;
        std::vector<std::pair<double,double>> rs;
        rs.push_back(sc[0]);
        double ac = 0.0;
        for (int i = 1; i < static_cast<int>(sc.size()); ++i) {
            double ddx = sc[i].first - sc[i - 1].first;
            double ddy = sc[i].second - sc[i - 1].second;
            double seg = std::hypot(ddx, ddy);
            if (seg < 1e-6) continue;
            ac += seg;
            while (ac >= spacing) {
                ac -= spacing;
                double f = 1.0 - ac / seg;
                rs.push_back({sc[i - 1].first + ddx * f, sc[i - 1].second + ddy * f});
            }
        }
        if (rs.back() != sc.back()) {
            rs.push_back(sc.back());
        }
        if (rs.size() <= 2) return sc;

        // Gradient descent: smooth curvature while staying in free space
        int np = static_cast<int>(rs.size());
        std::vector<double> ptx(np), pty(np);
        std::vector<double> origx(np), origy(np);
        for (int k = 0; k < np; ++k) {
            ptx[k] = origx[k] = rs[k].first;
            pty[k] = origy[k] = rs[k].second;
        }

        double w_smooth = 0.25;
        double w_data = 0.1;
        for (int iter = 0; iter < 40; ++iter) {
            for (int k = 1; k < np - 1; ++k) {
                double mx = (ptx[k - 1] + ptx[k + 1]) / 2.0;
                double my = (pty[k - 1] + pty[k + 1]) / 2.0;
                double nx = ptx[k] + w_smooth * (mx - ptx[k]) + w_data * (origx[k] - ptx[k]);
                double ny = pty[k] + w_smooth * (my - pty[k]) + w_data * (origy[k] - pty[k]);
                int gi = static_cast<int>((nx - ox) / grid_res_);
                int gj = static_cast<int>((ny - oy) / grid_res_);
                if (gi >= 0 && gi < n && gj >= 0 && gj < n && grid[gi * n + gj] < 0.8f) {
                    ptx[k] = nx;
                    pty[k] = ny;
                }
            }
        }

        // Subsample to ~15 waypoints
        int step = std::max(1, np / 15);
        std::vector<std::pair<double,double>> result;
        for (int k = 0; k < np; k += step) {
            result.push_back({ptx[k], pty[k]});
        }
        if (result.back().first != ptx[np - 1] || result.back().second != pty[np - 1]) {
            result.push_back({ptx[np - 1], pty[np - 1]});
        }
        return result;
    }

    bool lineClear(const std::pair<double,double>& p1,
                   const std::pair<double,double>& p2,
                   const std::vector<float>& grid,
                   double ox, double oy, int n) {
        double ddx = p2.first - p1.first;
        double ddy = p2.second - p1.second;
        double dist = std::hypot(ddx, ddy);
        if (dist < 0.01) return true;
        int steps = static_cast<int>(dist / (grid_res_ * 0.5)) + 1;
        for (int s = 0; s <= steps; ++s) {
            double f = static_cast<double>(s) / std::max(1, steps);
            int gi = static_cast<int>((p1.first + ddx * f - ox) / grid_res_);
            int gj = static_cast<int>((p1.second + ddy * f - oy) / grid_res_);
            if (gi >= 0 && gi < n && gj >= 0 && gj < n) {
                if (grid[gi * n + gj] >= OCCUPIED) return false;
            } else {
                return false;
            }
        }
        return true;
    }

    void advanceIdx() {
        while (path_idx_ < static_cast<int>(path_.size()) - 1) {
            double ddx = path_[path_idx_].first - robot_x_;
            double ddy = path_[path_idx_].second - robot_y_;
            if (std::hypot(ddx, ddy) < checkpoint_dist_) {
                path_idx_++;
            } else {
                break;
            }
        }
    }

    void pubPath() {
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
        if (!have_scan_ || scan_ranges_.size() == 0) return true;

        int n = static_cast<int>(scan_ranges_.size());
        // Build valid obstacle points in base_link
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
        double eff_radius = (std::fabs(av) > 0.5) ? robot_radius_ : control_radius_;

        double x = 0.0, y = 0.0, th = 0.0;
        double dt = proj_dt_;
        double t = 0.0;
        while (t < proj_horizon_) {
            x += lv * std::cos(th) * dt;
            y += lv * std::sin(th) * dt;
            th += av * dt;
            t += dt;

            double min_d2 = std::numeric_limits<double>::infinity();
            for (int k = 0; k < nobs; ++k) {
                double ddx = olx[k] - x;
                double ddy = oly[k] - y;
                double d2 = ddx * ddx + ddy * ddy;
                if (d2 < min_d2) min_d2 = d2;
            }
            double cl = std::sqrt(min_d2);
            if (cl - eff_radius < proj_clearance_) {
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
            last_bc_x_ = px; last_bc_y_ = py;
            last_bc_valid_ = true;
            return;
        }
        if (std::hypot(px - last_bc_x_, py - last_bc_y_) >= breadcrumb_spacing_) {
            breadcrumbs_.push_back({px, py});
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
            int back_n = std::min(3 + recovery_count_ / 3, static_cast<int>(breadcrumbs_.size()));
            if (back_n >= 1 && static_cast<int>(breadcrumbs_.size()) >= back_n) {
                int idx = static_cast<int>(breadcrumbs_.size()) - back_n;
                recovery_target_x_ = breadcrumbs_[idx].first;
                recovery_target_y_ = breadcrumbs_[idx].second;
                recovery_target_valid_ = true;
                breadcrumbs_.resize(breadcrumbs_.size() - back_n);
            } else {
                recovery_target_valid_ = false;
            }
            ROS_WARN("A*: recovery #%d (backtrack %d)", recovery_count_, back_n >= 1 ? back_n : 0);
        }

        if (now - recovery_start_ > recovery_max_time_) {
            in_recovery_ = false;
            recovery_target_valid_ = false;
            consecutive_stops_ = 0;
            spin_count_ = 0;
            last_plan_time_ = 0.0;
            return;
        }

        geometry_msgs::Twist cmd;
        double obs_d = minObs();

        // Collision guard: obstacle inside footprint
        if (obs_d < robot_radius_ - 0.02) {
            ROS_WARN_THROTTLE(1.0, "A*: recovery -- obstacle at %.2fm, escape fwd", obs_d);
            double fwd = fwdMin(1.05);
            if (fwd > robot_radius_) {
                cmd.linear.x = 0.2;
            }
            cmd_pub_.publish(cmd);
            return;
        }

        double rc = rearClear();
        bool rear_ok = rc > robot_radius_ + 0.05;

        if (!recovery_target_valid_) {
            if (rear_ok) cmd.linear.x = -0.3;
        } else {
            double ddx = recovery_target_x_ - robot_x_;
            double ddy = recovery_target_y_ - robot_y_;
            if (std::hypot(ddx, ddy) < 0.3) {
                in_recovery_ = false;
                recovery_target_valid_ = false;
                consecutive_stops_ = 0;
                spin_count_ = 0;
                last_plan_time_ = 0.0;
                return;
            }
            double err = normalizeAngle(std::atan2(ddy, ddx) - robot_yaw_);
            if (std::fabs(err) > 2.0) {
                if (rear_ok) cmd.linear.x = -0.3;
            } else {
                double fwd = fwdMin(1.05);
                if (fwd > robot_radius_ + 0.1) {
                    cmd.linear.x = 0.25;
                    cmd.angular.z = clampd(2.0 * err, -max_ang_, max_ang_);
                } else if (rear_ok) {
                    cmd.linear.x = -0.2;
                }
            }
        }
        cmd_pub_.publish(cmd);
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
        recovery_target_valid_ = false;
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
            "%.3f,%.3f,%.3f,%.2f,%.3f,%.3f,%.3f,%.3f,%.3f,%.2f,%.2f,%.3f,%s,%s,%d,%.3f,%d,%d,%d,%d\n",
            ros::Time::now().toSec(), robot_x_, robot_y_, robot_yaw_,
            lv, av, o, f, cl, tx, ty, ae,
            st.c_str(), branch.c_str(), pl, d,
            consecutive_stops_, spin_count_, plan_fail_count_, recovery_count_);
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
