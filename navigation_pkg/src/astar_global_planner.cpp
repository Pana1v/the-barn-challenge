/*
 * A* Global Planner plugin for move_base.
 * Uses move_base's costmap (obstacle layer + inflation) and runs A* search
 * with path smoothing. DWA handles local control.
 */

#include <ros/ros.h>
#include <nav_core/base_global_planner.h>
#include <costmap_2d/costmap_2d_ros.h>
#include <costmap_2d/costmap_2d.h>
#include <geometry_msgs/PoseStamped.h>
#include <nav_msgs/Path.h>
#include <pluginlib/class_list_macros.h>
#include <tf/tf.h>

#include <cmath>
#include <vector>
#include <queue>
#include <algorithm>

namespace astar_planner {

class AStarGlobalPlanner : public nav_core::BaseGlobalPlanner {
public:
    AStarGlobalPlanner() : costmap_(nullptr), initialized_(false) {}

    AStarGlobalPlanner(std::string name, costmap_2d::Costmap2DROS* costmap_ros)
        : costmap_(nullptr), initialized_(false) {
        initialize(name, costmap_ros);
    }

    void initialize(std::string name, costmap_2d::Costmap2DROS* costmap_ros) override {
        if (initialized_) return;

        ros::NodeHandle pnh("~/" + name);
        costmap_ros_ = costmap_ros;
        costmap_ = costmap_ros->getCostmap();

        // Parameters
        pnh.param("cost_weight", cost_weight_, 4.0);
        pnh.param("unknown_cost", unknown_cost_, 0.8);
        pnh.param("turn_penalty", turn_penalty_, 0.15);
        pnh.param("smooth_weight", smooth_weight_, 0.3);
        pnh.param("smooth_iters", smooth_iters_, 50);

        // Visualize the A* path separately
        plan_pub_ = pnh.advertise<nav_msgs::Path>("astar_plan", 1, true);

        ROS_INFO("AStarGlobalPlanner: initialized (cost_weight=%.1f, unknown=%.2f)",
                 cost_weight_, unknown_cost_);
        initialized_ = true;
    }

    bool makePlan(const geometry_msgs::PoseStamped& start,
                  const geometry_msgs::PoseStamped& goal,
                  std::vector<geometry_msgs::PoseStamped>& plan) override {
        if (!initialized_) {
            ROS_ERROR("AStarGlobalPlanner not initialized");
            return false;
        }

        plan.clear();
        costmap_ = costmap_ros_->getCostmap();

        unsigned int sx, sy, gx, gy;
        if (!costmap_->worldToMap(start.pose.position.x, start.pose.position.y, sx, sy)) {
            ROS_WARN("A* planner: start position outside costmap");
            return false;
        }
        if (!costmap_->worldToMap(goal.pose.position.x, goal.pose.position.y, gx, gy)) {
            ROS_WARN("A* planner: goal position outside costmap");
            return false;
        }

        int nx = costmap_->getSizeInCellsX();
        int ny = costmap_->getSizeInCellsY();

        // A* search
        auto grid_path = astar(sx, sy, gx, gy, nx, ny);
        if (grid_path.empty()) {
            ROS_WARN("A* planner: no path found");
            return false;
        }

        // Convert to world coordinates
        std::vector<std::pair<double,double>> wp;
        wp.reserve(grid_path.size());
        for (auto& c : grid_path) {
            double wx, wy;
            costmap_->mapToWorld(c.first, c.second, wx, wy);
            wp.push_back({wx, wy});
        }

        // Smooth the path
        wp = smooth(wp, nx, ny);

        // Build plan
        ros::Time now = ros::Time::now();
        plan.reserve(wp.size());
        for (size_t i = 0; i < wp.size(); ++i) {
            geometry_msgs::PoseStamped ps;
            ps.header.stamp = now;
            ps.header.frame_id = costmap_ros_->getGlobalFrameID();
            ps.pose.position.x = wp[i].first;
            ps.pose.position.y = wp[i].second;
            ps.pose.position.z = 0.0;
            // Set orientation toward next waypoint
            if (i + 1 < wp.size()) {
                double yaw = std::atan2(wp[i+1].second - wp[i].second,
                                        wp[i+1].first - wp[i].first);
                ps.pose.orientation = tf::createQuaternionMsgFromYaw(yaw);
            } else {
                ps.pose.orientation = goal.pose.orientation;
            }
            plan.push_back(ps);
        }

        // Publish for visualization
        nav_msgs::Path path_msg;
        path_msg.header.stamp = now;
        path_msg.header.frame_id = costmap_ros_->getGlobalFrameID();
        path_msg.poses = plan;
        plan_pub_.publish(path_msg);

        ROS_INFO_THROTTLE(2.0, "A* planner: path with %zu waypoints", plan.size());
        return true;
    }

private:
    // Get normalized cost for a cell (0.0=free, 1.0=lethal, unknown_cost_ for unknown)
    double getCellCost(unsigned int mx, unsigned int my) const {
        unsigned char c = costmap_->getCost(mx, my);
        if (c == costmap_2d::LETHAL_OBSTACLE || c == costmap_2d::INSCRIBED_INFLATED_OBSTACLE)
            return 1.0;
        if (c == costmap_2d::NO_INFORMATION)
            return unknown_cost_;
        return static_cast<double>(c) / 252.0;  // 0-252 → 0.0-1.0
    }

    std::vector<std::pair<int,int>> astar(int sx, int sy, int gx, int gy, int nx, int ny) {
        static const int DI[] = {-1, -1, -1, 0, 0, 1, 1, 1};
        static const int DJ[] = {-1, 0, 1, -1, 1, -1, 0, 1};
        static const double SC[] = {1.414, 1.0, 1.414, 1.0, 1.0, 1.414, 1.0, 1.414};

        int total = nx * ny;
        std::vector<float> gs(total, std::numeric_limits<float>::max());
        std::vector<int> parent(total, -1);
        std::vector<bool> closed(total, false);
        std::vector<uint8_t> cdir(total, 9);

        auto idx = [ny](int x, int y) { return x * ny + y; };

        // Check start/goal are not lethal
        if (getCellCost(sx, sy) >= 1.0) {
            auto f = nearestFree(sx, sy, nx, ny);
            if (f.first < 0) return {};
            sx = f.first; sy = f.second;
        }
        if (getCellCost(gx, gy) >= 1.0) {
            auto f = nearestFree(gx, gy, nx, ny);
            if (f.first < 0) return {};
            gx = f.first; gy = f.second;
        }

        int si = idx(sx, sy);
        int gi = idx(gx, gy);
        gs[si] = 0.0f;

        // Priority queue: (f-cost, index)
        using PQE = std::pair<float, int>;
        std::priority_queue<PQE, std::vector<PQE>, std::greater<PQE>> pq;
        float h0 = static_cast<float>(std::hypot(gx - sx, gy - sy));
        pq.push({h0, si});

        while (!pq.empty()) {
            auto [f, ci] = pq.top();
            pq.pop();
            if (closed[ci]) continue;
            closed[ci] = true;

            int cx = ci / ny;
            int cy = ci % ny;

            if (ci == gi) {
                // Reconstruct path
                std::vector<std::pair<int,int>> path;
                int cur = gi;
                while (cur != si) {
                    path.push_back({cur / ny, cur % ny});
                    cur = parent[cur];
                    if (cur < 0) return {};
                }
                path.push_back({sx, sy});
                std::reverse(path.begin(), path.end());
                return path;
            }

            uint8_t pd = cdir[ci];
            for (int d = 0; d < 8; ++d) {
                int nx2 = cx + DI[d];
                int ny2 = cy + DJ[d];
                if (nx2 < 0 || nx2 >= nx || ny2 < 0 || ny2 >= ny) continue;
                int ni = idx(nx2, ny2);
                if (closed[ni]) continue;

                double cc = getCellCost(nx2, ny2);
                if (cc >= 1.0) continue;  // lethal

                float tg = gs[ci] + static_cast<float>(SC[d]) * (1.0f + cost_weight_ * cc);

                // Turn penalty
                uint8_t nd = static_cast<uint8_t>((DI[d] + 1) * 3 + (DJ[d] + 1));
                if (pd != 9 && nd != pd) {
                    tg += static_cast<float>(turn_penalty_);
                }

                if (tg < gs[ni]) {
                    gs[ni] = tg;
                    parent[ni] = ci;
                    cdir[ni] = nd;
                    float h = static_cast<float>(std::hypot(gx - nx2, gy - ny2));
                    pq.push({tg + h, ni});
                }
            }
        }
        return {};  // no path
    }

    std::pair<int,int> nearestFree(int si, int sj, int nx, int ny) {
        for (int r = 1; r <= 10; ++r) {
            for (int di = -r; di <= r; ++di) {
                for (int dj = -r; dj <= r; ++dj) {
                    if (std::abs(di) != r && std::abs(dj) != r) continue;
                    int ni = si + di, nj = sj + dj;
                    if (ni >= 0 && ni < nx && nj >= 0 && nj < ny) {
                        if (getCellCost(ni, nj) < 1.0)
                            return {ni, nj};
                    }
                }
            }
        }
        return {-1, -1};
    }

    std::vector<std::pair<double,double>> smooth(
            const std::vector<std::pair<double,double>>& wp, int nx, int ny) {
        if (wp.size() <= 2) return wp;

        // Line-of-sight shortcut
        std::vector<std::pair<double,double>> sc = {wp[0]};
        size_t i = 0;
        while (i < wp.size() - 1) {
            size_t best = i + 1;
            for (size_t j = wp.size() - 1; j > i + 1; --j) {
                if (lineClear(wp[i], wp[j], nx, ny)) {
                    best = j;
                    break;
                }
            }
            sc.push_back(wp[best]);
            i = best;
        }

        if (sc.size() <= 2) return sc;

        // Gradient descent smoothing
        auto result = sc;
        for (int iter = 0; iter < smooth_iters_; ++iter) {
            for (size_t k = 1; k + 1 < result.size(); ++k) {
                double ox = result[k].first;
                double oy = result[k].second;
                // Pull toward midpoint of neighbors
                double mx = (result[k-1].first + result[k+1].first) / 2.0;
                double my = (result[k-1].second + result[k+1].second) / 2.0;
                double nx2 = ox + smooth_weight_ * (mx - ox);
                double ny2 = oy + smooth_weight_ * (my - oy);
                // Verify the smoothed point is not in obstacle
                unsigned int gx, gy;
                if (costmap_->worldToMap(nx2, ny2, gx, gy)) {
                    if (getCellCost(gx, gy) < 0.9) {
                        result[k] = {nx2, ny2};
                    }
                }
            }
        }
        return result;
    }

    bool lineClear(const std::pair<double,double>& a, const std::pair<double,double>& b,
                   int nx, int ny) {
        double dx = b.first - a.first;
        double dy = b.second - a.second;
        double dist = std::hypot(dx, dy);
        double step = costmap_->getResolution() * 0.5;
        int nsteps = static_cast<int>(dist / step) + 1;
        for (int i = 0; i <= nsteps; ++i) {
            double f = static_cast<double>(i) / nsteps;
            double wx = a.first + dx * f;
            double wy = a.second + dy * f;
            unsigned int mx, my;
            if (!costmap_->worldToMap(wx, wy, mx, my)) return false;
            if (getCellCost(mx, my) >= 1.0) return false;
        }
        return true;
    }

    costmap_2d::Costmap2DROS* costmap_ros_;
    costmap_2d::Costmap2D* costmap_;
    bool initialized_;

    double cost_weight_;
    double unknown_cost_;
    double turn_penalty_;
    double smooth_weight_;
    int smooth_iters_;

    ros::Publisher plan_pub_;
};

}  // namespace astar_planner

PLUGINLIB_EXPORT_CLASS(astar_planner::AStarGlobalPlanner, nav_core::BaseGlobalPlanner)
