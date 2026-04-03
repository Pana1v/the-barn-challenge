#!/usr/bin/env python3
"""
A* planner for BARN challenge — rebuilt from v15 baseline (6/12 worlds passing).
Fixes applied:
  F1: Smooth path (shortcut + gradient descent)
  F2: Velocity polygon (multi-candidate forward projection)
  F3: Angular rate limiting (prevent yaw oscillation)
  F4: Cornered = back up via breadcrumbs, never spin in tight spaces
"""

import math
import heapq
import os
import csv
import numpy as np
import rospy
import tf
from geometry_msgs.msg import Twist, PoseStamped
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Path

try:
    from scipy.ndimage import distance_transform_edt
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


class AStarPlanner:
    UNKNOWN = 0.45
    FREE = 0.0
    OCCUPIED = 1.0

    def __init__(self):
        rospy.init_node('nav_astar', anonymous=False)
        ns = 'nav_astar'
        def p(name, default):
            return rospy.get_param('~' + name, rospy.get_param(ns + '/' + name, default))

        self.grid_res       = p('grid_resolution', 0.08)
        self.inflation_radius = p('inflation_radius', 0.22)
        self.cost_radius    = p('cost_radius', 0.40)
        self.max_lin        = p('max_linear_vel', 1.2)
        self.max_ang        = p('max_angular_vel', 2.5)
        self.accel_limit    = p('accel_limit', 0.5)
        self.Kp_ang         = p('Kp_ang', 3.5)
        self.align_thresh   = p('align_threshold', 1.0)
        self.replan_interval = p('replan_interval', 1.0)
        self.goal_tolerance = p('goal_tolerance', 0.01)  # let run.py's Gazebo check handle termination
        self.checkpoint_dist = p('checkpoint_reach_dist', 0.25)
        self.safe_margin    = p('safe_margin', 0.02)
        self.stop_clearance = p('stop_clearance', 0.02)
        self.slow_clearance = p('slow_clearance', 0.10)
        self.breadcrumb_spacing = p('breadcrumb_spacing', 0.3)
        self.stuck_check_interval = p('stuck_check_interval', 10.0)
        self.stuck_threshold = p('stuck_threshold', 0.3)
        self.recovery_max_time = p('recovery_max_time', 8.0)
        self.collision_stop_threshold = int(p('collision_stop_threshold', 12))

        self.robot_radius = math.hypot(0.42 / 2.0, 0.31 / 2.0)  # 0.261m — used for PLANNING
        self.control_radius = 0.22  # match inflation — prevents turn-clipping
        self.proj_dt = 0.1
        self.proj_horizon = 0.8
        self.proj_clearance = 0.04

        # [F3] Angular rate limit
        self.max_ang_accel = 4.0  # rad/s per cycle at 20Hz
        self.prev_ang_vel = 0.0

        goal_rel = rospy.get_param('goal_position', [0, 10])
        self.goal_x = goal_rel[0]
        self.goal_y = goal_rel[1]

        # State
        self.robot_x = self.robot_y = self.robot_yaw = 0.0
        self.scan_data = self.scan_ranges = self.scan_angles = None
        self.path = []
        self.path_idx = 0
        self.last_plan_time = 0.0
        self._last_grid = None  # cached costmap for trajectory scoring
        self.prev_lin_vel = 0.0
        self.consecutive_stops = 0
        self.spin_count = 0

        # Recovery
        self.breadcrumbs = []
        self.last_bc_pos = None
        self.stuck_check_time = self.stuck_check_pos = None
        self.in_recovery = False
        self.recovery_start = 0.0
        self.recovery_target = None
        self.recovery_count = 0
        self.plan_fail_count = 0  # consecutive A* failures → trigger retreat

        # Progress-based deadlock detection
        self._progress_time = None   # when we last recorded d2g
        self._progress_d2g = None    # d2g at that time
        self._progress_window = 30.0 # seconds without 0.5m progress → deadlock
        self._progress_min = 0.5     # minimum progress in window
        self._deadlock_escape = False
        self._escape_start = 0.0
        self._escape_heading = 0.0
        self._escape_phase = 0       # 0=rotate, 1=drive

        # TF
        self.tf_listener = tf.TransformListener()
        self._laser_t = None
        self._laser_yaw = None

        # Grid
        self.local_size = 12.0  # covers full start-to-goal (10m)
        self.grid_cells = int(self.local_size / self.grid_res)
        self._build_kernel()

        # Pub/Sub
        self.cmd_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=1)
        self.path_pub = rospy.Publisher('/nav_astar/path', Path, queue_size=1)
        rospy.Subscriber('/front/scan', LaserScan, self._scan_cb)

        # Logger
        log_dir = '/tmp/nav_logs'
        os.makedirs(log_dir, exist_ok=True)
        wid = rospy.get_param('/world_idx', 'x')
        lp = os.path.join(log_dir, 'astar_w%s.csv' % wid)
        self._lf = open(lp, 'w', newline='')
        self._lw = csv.writer(self._lf)
        self._lw.writerow(['t', 'x', 'y', 'yaw', 'lv', 'av', 'obs', 'fwd', 'cl', 'tx', 'ty', 'ae', 'st', 'branch', 'pl', 'd2g', 'cstops', 'spins', 'pfails', 'rcnt'])

        rospy.loginfo("A* planner: goal(%.1f,%.1f) grid=%d scipy=%s" %
                       (self.goal_x, self.goal_y, self.grid_cells, HAS_SCIPY))

    def _build_kernel(self):
        """Precompute inflation kernel (fallback when scipy unavailable)."""
        pi = self.inflation_radius * 1.15
        pc = self.cost_radius * 1.15
        rc = min(int(math.ceil(pc / self.grid_res)), 8)
        self._kernel = []
        for di in range(-rc, rc + 1):
            for dj in range(-rc, rc + 1):
                d = math.hypot(di, dj) * self.grid_res
                if d <= pi:
                    self._kernel.append((di, dj, self.OCCUPIED))
                elif d <= pc:
                    f = (d - pi) / max(0.001, pc - pi)
                    self._kernel.append((di, dj, 0.85 * (1.0 - f)))

    # ── Sensor ──────────────────────────────────────────────────────────

    def _scan_cb(self, msg):
        self.scan_data = msg
        r = np.array(msg.ranges)
        self.scan_ranges = r
        self.scan_angles = np.linspace(
            msg.angle_min, msg.angle_min + (len(r) - 1) * msg.angle_increment, len(r))

    def _get_pose(self):
        try:
            self.tf_listener.waitForTransform('odom', 'base_link', rospy.Time(0), rospy.Duration(0.5))
            t, r = self.tf_listener.lookupTransform('odom', 'base_link', rospy.Time(0))
            self.robot_x, self.robot_y = t[0], t[1]
            self.robot_yaw = tf.transformations.euler_from_quaternion(r)[2]
            return True
        except Exception:
            return False

    def _update_laser_tf(self):
        if not self.scan_data:
            return
        try:
            self.tf_listener.waitForTransform(
                'odom', self.scan_data.header.frame_id, rospy.Time(0), rospy.Duration(0.2))
            t, rot = self.tf_listener.lookupTransform(
                'odom', self.scan_data.header.frame_id, rospy.Time(0))
            self._laser_t = t
            self._laser_yaw = tf.transformations.euler_from_quaternion(rot)[2]
        except Exception:
            self._laser_t = [self.robot_x, self.robot_y, 0]
            self._laser_yaw = self.robot_yaw

    def _scan_to_odom(self):
        """Current LIDAR scan → obstacle points in odom frame."""
        if self.scan_ranges is None:
            return np.empty((0, 2))
        rng = self.scan_ranges
        scan = self.scan_data
        v = np.isfinite(rng) & (rng > scan.range_min) & (rng < scan.range_max)
        if not np.any(v):
            return np.empty((0, 2))
        r, a = rng[v], self.scan_angles[v]
        lx, ly = r * np.cos(a), r * np.sin(a)
        self._update_laser_tf()
        t = self._laser_t or [self.robot_x, self.robot_y, 0]
        yaw = self._laser_yaw if self._laser_yaw is not None else self.robot_yaw
        c, s = math.cos(yaw), math.sin(yaw)
        return np.column_stack([t[0] + lx * c - ly * s, t[1] + lx * s + ly * c])

    # ── Grid ────────────────────────────────────────────────────────────

    def _visible_mask(self, gox, goy):
        """Vectorized ray-cast: mark cells along LIDAR rays as visible."""
        if self.scan_ranges is None:
            return None
        n = self.grid_cells
        vis = np.zeros((n, n), dtype=bool)
        self._update_laser_tf()
        t = self._laser_t or [self.robot_x, self.robot_y, 0]
        yaw = self._laser_yaw if self._laser_yaw is not None else self.robot_yaw
        rng, ang = self.scan_ranges, self.scan_angles

        idx = np.arange(0, len(rng), 5)
        ra = ang[idx] + yaw
        rr = rng[idx].copy()
        bad = ~np.isfinite(rr) | (rr < self.scan_data.range_min)
        rr[bad] = self.scan_data.range_max
        rr = np.minimum(rr, self.scan_data.range_max)

        ns = int(float(self.scan_data.range_max) / (self.grid_res * 1.5)) + 1
        ds = np.linspace(0, 1, ns)
        ad = rr[:, None] * ds[None, :]
        px = t[0] + ad * np.cos(ra)[:, None]
        py = t[1] + ad * np.sin(ra)[:, None]
        gi = ((px - gox) / self.grid_res).astype(int)
        gj = ((py - goy) / self.grid_res).astype(int)
        gf, gjf = gi.ravel(), gj.ravel()
        m = (gf >= 0) & (gf < n) & (gjf >= 0) & (gjf < n)
        vis[gf[m], gjf[m]] = True

        ri = int((self.robot_x - gox) / self.grid_res)
        rj = int((self.robot_y - goy) / self.grid_res)
        rc = int(self.robot_radius / self.grid_res) + 2
        vis[max(0, ri - rc):min(n, ri + rc + 1), max(0, rj - rc):min(n, rj + rc + 1)] = True
        return vis

    def _build_grid(self, obstacles):
        """Build local occupancy grid with visibility mask and inflation."""
        n = self.grid_cells
        half = self.local_size / 2
        ox, oy = self.robot_x - half, self.robot_y - half

        grid = np.full((n, n), self.UNKNOWN, dtype=np.float32)
        vis = self._visible_mask(ox, oy)
        if vis is not None:
            grid[vis] = self.FREE

        if len(obstacles) == 0:
            return grid, ox, oy

        gi = ((obstacles[:, 0] - ox) / self.grid_res).astype(int)
        gj = ((obstacles[:, 1] - oy) / self.grid_res).astype(int)
        m = (gi >= 0) & (gi < n) & (gj >= 0) & (gj < n)
        gi, gj = gi[m], gj[m]
        if len(gi) == 0:
            return grid, ox, oy
        grid[gi, gj] = self.OCCUPIED

        # Inflate using kernel (applies to all unique obstacle cells)
        obs_set = set(zip(gi.tolist(), gj.tolist()))
        for oi, oj in obs_set:
            for di, dj, cost in self._kernel:
                ni, nj = oi + di, oj + dj
                if 0 <= ni < n and 0 <= nj < n and cost > grid[ni, nj]:
                    grid[ni, nj] = cost

        return grid, ox, oy

    # ── A* ──────────────────────────────────────────────────────────────

    def _astar(self, grid, si, sj, gi, gj):
        """A* with direction-change penalty for straighter paths."""
        n = grid.shape[0]
        si, sj = max(0, min(n - 1, int(si))), max(0, min(n - 1, int(sj)))
        gi, gj = max(0, min(n - 1, int(gi))), max(0, min(n - 1, int(gj)))

        if grid[si, sj] >= self.OCCUPIED:
            f = self._nearest_free(grid, si, sj, n)
            if not f:
                return []
            si, sj = f
        if grid[gi, gj] >= self.OCCUPIED:
            f = self._nearest_free(grid, gi, gj, n)
            if not f:
                return []
            gi, gj = f

        NBRS = [(-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0),
                (-1, -1, 1.414), (-1, 1, 1.414), (1, -1, 1.414), (1, 1, 1.414)]
        TURN_PEN = 0.25

        heap = [(0.0, si, sj)]
        gs = np.full((n, n), np.inf, dtype=np.float32)
        gs[si, sj] = 0.0
        came = {}
        cdir = {}
        closed = np.zeros((n, n), dtype=bool)
        it = 0

        while heap and it < 150000:
            it += 1
            _, ci, cj = heapq.heappop(heap)
            if closed[ci, cj]:
                continue
            closed[ci, cj] = True
            if ci == gi and cj == gj:
                path = [(ci, cj)]
                while (ci, cj) in came:
                    ci, cj = came[(ci, cj)]
                    path.append((ci, cj))
                path.reverse()
                return path

            pd = cdir.get((ci, cj))
            for di, dj, sc in NBRS:
                ni, nj = ci + di, cj + dj
                if 0 <= ni < n and 0 <= nj < n and not closed[ni, nj]:
                    cc = grid[ni, nj]
                    if cc >= self.OCCUPIED:
                        continue
                    tg = gs[ci, cj] + sc * (1.0 + 4.0 * cc)
                    if pd and (di, dj) != pd:
                        tg += TURN_PEN
                    if tg < gs[ni, nj]:
                        gs[ni, nj] = tg
                        h = math.hypot(ni - gi, nj - gj)
                        heapq.heappush(heap, (tg + h, ni, nj))
                        came[(ni, nj)] = (ci, cj)
                        cdir[(ni, nj)] = (di, dj)
        return []

    @staticmethod
    def _nearest_free(grid, ci, cj, n):
        for r in range(1, 25):
            for di in range(-r, r + 1):
                for dj in range(-r, r + 1):
                    if abs(di) == r or abs(dj) == r:
                        ni, nj = ci + di, cj + dj
                        if 0 <= ni < n and 0 <= nj < n and grid[ni, nj] < 1.0:
                            return (ni, nj)
        return None

    # ── Planning ────────────────────────────────────────────────────────

    def _plan(self):
        obstacles = self._scan_to_odom()
        grid, ox, oy = self._build_grid(obstacles)
        self._last_grid = (grid, ox, oy)  # cache for controller trajectory scoring
        n = self.grid_cells

        si = (self.robot_x - ox) / self.grid_res
        sj = (self.robot_y - oy) / self.grid_res

        dx = self.goal_x - self.robot_x
        dy = self.goal_y - self.robot_y
        dist = math.hypot(dx, dy)
        if dist < 0.1:
            return False

        # Plan toward goal — full distance, clamped to grid bounds
        la = min(dist, self.local_size * 0.48)
        tx = self.robot_x + (dx / dist) * la
        ty = self.robot_y + (dy / dist) * la
        gi = max(0, min(n - 1, (tx - ox) / self.grid_res))
        gj = max(0, min(n - 1, (ty - oy) / self.grid_res))

        gp = self._astar(grid, si, sj, gi, gj)
        if not gp:
            rospy.logwarn_throttle(2.0, "A*: no path")
            return False

        wp = [(ox + pi * self.grid_res, oy + pj * self.grid_res) for pi, pj in gp]

        # [F1] Smooth: shortcut then gradient descent
        sm = self._smooth(wp, grid, ox, oy)
        self.path = sm
        self.path_idx = 0
        self._advance_idx()
        self._pub_path()
        return True

    def _smooth(self, wp, grid, ox, oy):
        """[F1] Line-of-sight shortcut + gradient descent smoothing."""
        if len(wp) <= 2:
            return wp
        n = grid.shape[0]

        # Shortcut
        sc = [wp[0]]
        i = 0
        while i < len(wp) - 1:
            bj = i + 1
            for j in range(len(wp) - 1, i + 1, -1):
                if self._line_clear(wp[i], wp[j], grid, ox, oy, n):
                    bj = j
                    break
            sc.append(wp[bj])
            i = bj

        if len(sc) <= 2:
            return sc

        # Resample at uniform spacing
        spacing = 0.12
        rs = [sc[0]]
        ac = 0.0
        for i in range(1, len(sc)):
            dx = sc[i][0] - sc[i - 1][0]
            dy = sc[i][1] - sc[i - 1][1]
            seg = math.hypot(dx, dy)
            if seg < 1e-6:
                continue
            ac += seg
            while ac >= spacing:
                ac -= spacing
                f = 1.0 - ac / seg
                rs.append((sc[i - 1][0] + dx * f, sc[i - 1][1] + dy * f))
        if rs[-1] != sc[-1]:
            rs.append(sc[-1])
        if len(rs) <= 2:
            return sc

        # Gradient descent: smooth curvature while staying in free space
        pt = np.array(rs, dtype=np.float64)
        orig = list(rs)
        w_smooth = 0.25
        w_data = 0.1
        for _ in range(40):
            for k in range(1, len(pt) - 1):
                mx = (pt[k - 1, 0] + pt[k + 1, 0]) / 2
                my = (pt[k - 1, 1] + pt[k + 1, 1]) / 2
                nx = pt[k, 0] + w_smooth * (mx - pt[k, 0]) + w_data * (orig[k][0] - pt[k, 0])
                ny = pt[k, 1] + w_smooth * (my - pt[k, 1]) + w_data * (orig[k][1] - pt[k, 1])
                gi = int((nx - ox) / self.grid_res)
                gj = int((ny - oy) / self.grid_res)
                if 0 <= gi < n and 0 <= gj < n and grid[gi, gj] < 0.8:
                    pt[k, 0] = nx
                    pt[k, 1] = ny

        # Subsample to ~15 waypoints
        step = max(1, len(pt) // 15)
        result = [(pt[k, 0], pt[k, 1]) for k in range(0, len(pt), step)]
        if (pt[-1, 0], pt[-1, 1]) != result[-1]:
            result.append((pt[-1, 0], pt[-1, 1]))
        return result

    def _line_clear(self, p1, p2, grid, ox, oy, n):
        dx, dy = p2[0] - p1[0], p2[1] - p1[1]
        dist = math.hypot(dx, dy)
        if dist < 0.01:
            return True
        steps = int(dist / (self.grid_res * 0.5)) + 1
        for s in range(steps + 1):
            f = s / max(1, steps)
            gi = int((p1[0] + dx * f - ox) / self.grid_res)
            gj = int((p1[1] + dy * f - oy) / self.grid_res)
            if 0 <= gi < n and 0 <= gj < n:
                if grid[gi, gj] >= self.OCCUPIED:
                    return False
            else:
                return False
        return True

    def _advance_idx(self):
        while self.path_idx < len(self.path) - 1:
            dx = self.path[self.path_idx][0] - self.robot_x
            dy = self.path[self.path_idx][1] - self.robot_y
            if math.hypot(dx, dy) < self.checkpoint_dist:
                self.path_idx += 1
            else:
                break

    def _pub_path(self):
        msg = Path()
        msg.header.frame_id = 'odom'
        msg.header.stamp = rospy.Time.now()
        for wx, wy in self.path:
            ps = PoseStamped()
            ps.header = msg.header
            ps.pose.position.x = wx
            ps.pose.position.y = wy
            ps.pose.orientation.w = 1.0
            msg.poses.append(ps)
        self.path_pub.publish(msg)

    # ── Safety ──────────────────────────────────────────────────────────

    def _check_fwd_collision(self, lv, av):
        """[F2] Forward trajectory rollout collision check.
        Uses robot_radius (full diagonal) when turning significantly,
        control_radius when driving mostly straight."""
        if self.scan_ranges is None:
            return True, lv
        sc = self.scan_data
        rng = self.scan_ranges
        ang = self.scan_angles
        v = np.isfinite(rng) & (rng > sc.range_min) & (rng < sc.range_max)
        if not np.any(v):
            return True, lv
        olx = rng[v] * np.cos(ang[v])
        oly = rng[v] * np.sin(ang[v])

        # Use full robot radius when turning hard (corners sweep wider)
        eff_radius = self.robot_radius if abs(av) > 0.5 else self.control_radius

        x = y = th = 0.0
        dt = self.proj_dt
        t = 0.0
        while t < self.proj_horizon:
            x += lv * math.cos(th) * dt
            y += lv * math.sin(th) * dt
            th += av * dt
            t += dt
            dx = olx - x
            dy = oly - y
            cl = float(np.sqrt(np.min(dx * dx + dy * dy)))
            if cl - eff_radius < self.proj_clearance:
                frac = max(0.0, t / self.proj_horizon)
                return False, max(0.0, lv * frac * 0.4)
        return True, lv

    def _min_obs(self):
        if self.scan_ranges is None:
            return float('inf')
        r = self.scan_ranges
        v = np.isfinite(r) & (r > self.scan_data.range_min)
        return float(np.min(r[v])) if np.any(v) else float('inf')

    def _fwd_min(self, half_angle=1.05):
        """Min distance in forward ±60° cone."""
        if self.scan_ranges is None:
            return float('inf')
        r = self.scan_ranges
        a = self.scan_angles
        fw = np.abs(a) < half_angle
        v = np.isfinite(r) & (r > self.scan_data.range_min) & fw
        return float(np.min(r[v])) if np.any(v) else float('inf')

    def _rear_clear(self):
        """Min distance in rear ±60° cone (for safe backup)."""
        if self.scan_ranges is None:
            return float('inf')
        r = self.scan_ranges
        a = self.scan_angles
        rr = np.abs(a) > 2.1
        v = np.isfinite(r) & (r > self.scan_data.range_min) & rr
        return float(np.min(r[v])) if np.any(v) else float('inf')

    def _sector_clearance(self):
        """Min distance in front (±30°), left (30-120°), right (-120 to -30°)."""
        if self.scan_ranges is None:
            return float('inf'), float('inf'), float('inf')
        r = self.scan_ranges
        a = self.scan_angles
        v = np.isfinite(r) & (r > self.scan_data.range_min)
        front = np.abs(a) < 0.52
        left = (a > 0.52) & (a < 2.09)
        right = (a < -0.52) & (a > -2.09)
        f_d = float(np.min(r[v & front])) if np.any(v & front) else float('inf')
        l_d = float(np.min(r[v & left])) if np.any(v & left) else float('inf')
        r_d = float(np.min(r[v & right])) if np.any(v & right) else float('inf')
        return f_d, l_d, r_d

    def _rotation_safe(self, angular_vel):
        """Check if rotating in place would clip obstacles with robot corners."""
        if self.scan_ranges is None:
            return True
        sc = self.scan_data
        rng = self.scan_ranges
        ang = self.scan_angles
        v = np.isfinite(rng) & (rng > sc.range_min) & (rng < sc.range_max)
        if not np.any(v):
            return True
        olx = rng[v] * np.cos(ang[v])
        oly = rng[v] * np.sin(ang[v])

        # Robot half-dimensions (Jackal: 0.42 x 0.31)
        hx, hy = 0.21, 0.155
        corners = [(hx, hy), (hx, -hy), (-hx, hy), (-hx, -hy)]

        # Project rotation over 0.5s in 5 steps
        for step in range(1, 6):
            theta = angular_vel * 0.1 * step
            ct, st = math.cos(theta), math.sin(theta)
            for cx, cy in corners:
                rx = cx * ct - cy * st
                ry = cx * st + cy * ct
                dx = olx - rx
                dy = oly - ry
                min_d = float(np.sqrt(np.min(dx * dx + dy * dy)))
                if min_d < 0.04:
                    return False
        return True

    # ── Recovery ────────────────────────────────────────────────────────

    def _update_bc(self):
        pos = (self.robot_x, self.robot_y)
        if self.last_bc_pos is None:
            self.breadcrumbs.append(pos)
            self.last_bc_pos = pos
            return
        if math.hypot(pos[0] - self.last_bc_pos[0], pos[1] - self.last_bc_pos[1]) >= self.breadcrumb_spacing:
            self.breadcrumbs.append(pos)
            self.last_bc_pos = pos
            if len(self.breadcrumbs) > 50:
                self.breadcrumbs = self.breadcrumbs[-50:]

    def _check_stuck(self):
        now = rospy.get_time()
        if self.stuck_check_time is None:
            self.stuck_check_time = now
            self.stuck_check_pos = (self.robot_x, self.robot_y)
            return False
        if now - self.stuck_check_time >= self.stuck_check_interval:
            d = math.hypot(self.robot_x - self.stuck_check_pos[0],
                           self.robot_y - self.stuck_check_pos[1])
            self.stuck_check_time = now
            self.stuck_check_pos = (self.robot_x, self.robot_y)
            return d < self.stuck_threshold
        return False

    def _do_recovery(self):
        """[F4] Recovery: back up along breadcrumbs. Escalate distance on repeated failures."""
        now = rospy.get_time()
        if not self.in_recovery:
            self.in_recovery = True
            self.recovery_start = now
            self.recovery_count += 1
            # Escalate aggressively: 5, 10, 15, 20... breadcrumbs back
            back_n = min(5 * self.recovery_count, len(self.breadcrumbs))
            if back_n >= 1 and len(self.breadcrumbs) >= back_n:
                self.recovery_target = self.breadcrumbs[-back_n]
                self.breadcrumbs = self.breadcrumbs[:-back_n]
            else:
                self.recovery_target = None
            rospy.logwarn("A*: recovery #%d (backtrack %d)" % (self.recovery_count, back_n if back_n >= 1 else 0))

        if now - self.recovery_start > self.recovery_max_time:
            self.in_recovery = False
            self.recovery_target = None
            self.consecutive_stops = 0
            self.spin_count = 0
            self.last_plan_time = 0.0
            return

        cmd = Twist()
        obs_d = self._min_obs()

        # Collision guard: if obstacle inside robot footprint, don't back into it
        # but DO allow forward motion to escape
        if obs_d < self.robot_radius - 0.02:
            rospy.logwarn_throttle(1.0, "A*: recovery — obstacle at %.2fm, escape fwd" % obs_d)
            fwd = self._fwd_min()
            if fwd > self.robot_radius:
                cmd.linear.x = 0.2  # crawl forward to escape
            self.cmd_pub.publish(cmd)
            return

        rc = self._rear_clear()
        rear_ok = rc > self.robot_radius + 0.05  # safe margin for backing up

        if self.recovery_target is None:
            if rear_ok:
                cmd.linear.x = -0.3
        else:
            dx = self.recovery_target[0] - self.robot_x
            dy = self.recovery_target[1] - self.robot_y
            if math.hypot(dx, dy) < 0.3:
                self.in_recovery = False
                self.recovery_target = None
                self.consecutive_stops = 0
                self.spin_count = 0
                self.last_plan_time = 0.0
                return
            err = self._na(math.atan2(dy, dx) - self.robot_yaw)
            if abs(err) > 2.0:
                # Target is behind — back up toward it
                if rear_ok:
                    cmd.linear.x = -0.3
            else:
                # Target is in front — drive toward it cautiously
                fwd = self._fwd_min()
                if fwd > self.robot_radius + 0.1:
                    cmd.linear.x = 0.25
                    cmd.angular.z = np.clip(2.0 * err, -self.max_ang, self.max_ang)
                elif rear_ok:
                    cmd.linear.x = -0.2  # can't go forward, back up more

        self.cmd_pub.publish(cmd)

    @staticmethod
    def _na(a):
        a = a % (2 * math.pi)
        if a > math.pi:
            a -= 2 * math.pi
        return a

    # ── Progress-based deadlock escape ────────────────────────────────

    def _check_progress(self, d2g):
        """Return True if robot has made no goal progress in _progress_window seconds."""
        now = rospy.get_time()
        if self._progress_time is None:
            self._progress_time = now
            self._progress_d2g = d2g
            return False
        # If we made progress, reset the window
        if d2g < self._progress_d2g - self._progress_min:
            self._progress_time = now
            self._progress_d2g = d2g
            return False
        # Check if window expired
        if now - self._progress_time >= self._progress_window:
            return True
        return False

    def _find_clear_heading(self):
        """Find best heading: clearest LIDAR direction biased toward goal."""
        if self.scan_ranges is None:
            return self.robot_yaw
        n_sectors = 12
        sector_size = len(self.scan_ranges) // n_sectors
        best_score = -1
        best_angle = self.robot_yaw
        goal_angle = math.atan2(self.goal_y - self.robot_y, self.goal_x - self.robot_x)
        for i in range(n_sectors):
            start = i * sector_size
            end = start + sector_size
            sector = self.scan_ranges[start:end]
            valid = sector[np.isfinite(sector) & (sector > 0.01)]
            if len(valid) == 0:
                continue
            clearance = float(np.min(valid))
            if clearance < self.robot_radius + 0.10:
                continue
            mid_idx = (start + end) // 2
            world_angle = self.robot_yaw + self.scan_angles[mid_idx]
            goal_align = math.cos(self._na(world_angle - goal_angle))
            score = 0.6 * min(clearance, 3.0) / 3.0 + 0.4 * (goal_align + 1.0) / 2.0
            if score > best_score:
                best_score = score
                best_angle = world_angle
        return best_angle

    def _do_deadlock_escape(self):
        """Rotate toward clear heading, then creep forward."""
        now = rospy.get_time()
        elapsed = now - self._escape_start
        cmd = Twist()

        if self._escape_phase == 0:
            # Rotate phase (max 3s)
            ae = self._na(self._escape_heading - self.robot_yaw)
            if abs(ae) < 0.3 or elapsed > 3.0:
                self._escape_phase = 1
                self._escape_start = now
                rospy.logwarn("A*: deadlock escape — drive phase")
            else:
                av = np.clip(2.0 * ae, -1.5, 1.5)
                if self._rotation_safe(av):
                    cmd.angular.z = av
        else:
            # Drive phase (max 4s)
            if elapsed > 4.0:
                self._end_deadlock_escape()
                return
            fwd = self._fwd_min()
            if fwd > self.robot_radius + 0.15:
                cmd.linear.x = 0.3
                ae = self._na(self._escape_heading - self.robot_yaw)
                cmd.angular.z = np.clip(1.5 * ae, -1.0, 1.0)
            else:
                # Blocked — end escape early
                self._end_deadlock_escape()
                return

        self.cmd_pub.publish(cmd)

    def _end_deadlock_escape(self):
        """Clean up after escape attempt."""
        self._deadlock_escape = False
        self.in_recovery = False
        self.recovery_target = None
        self.consecutive_stops = 0
        self.spin_count = 0
        self.path = []
        self.last_plan_time = 0.0
        # Reset progress window from current position
        self._progress_time = rospy.get_time()
        self._progress_d2g = math.hypot(self.goal_x - self.robot_x, self.goal_y - self.robot_y)
        rospy.logwarn("A*: deadlock escape done — replanning")

    # ── Main loop ───────────────────────────────────────────────────────

    def run(self):
        rate = rospy.Rate(20)

        while not rospy.is_shutdown():
            if self.scan_data and self._get_pose():
                break
            rate.sleep()
        rospy.sleep(0.5)
        self._get_pose()
        rospy.loginfo("A*: ready (%.2f,%.2f) yaw=%.2f -> goal(%.2f,%.2f)" %
                       (self.robot_x, self.robot_y, self.robot_yaw, self.goal_x, self.goal_y))

        while not rospy.is_shutdown():
            if not self._get_pose():
                rate.sleep()
                continue

            d2g = math.hypot(self.goal_x - self.robot_x, self.goal_y - self.robot_y)
            if d2g < self.goal_tolerance:
                rospy.loginfo("A*: GOAL!")
                self.cmd_pub.publish(Twist())
                break

            self._update_bc()
            obs_d = self._min_obs()
            fwd_d = self._fwd_min()
            fc = fwd_d - self.control_radius

            # Deadlock escape takes priority over everything
            cl = obs_d - self.control_radius - self.safe_margin
            if self._deadlock_escape:
                self._do_deadlock_escape()
                esc_br = 'esc_rot' if self._escape_phase == 0 else 'esc_drive'
                self._log(0, 0, obs_d, fc, cl, self._escape_heading, 0, 0, 'esc', esc_br, 0, d2g)
                rate.sleep()
                continue

            # Check for sustained lack of goal progress → deadlock escape
            if self._check_progress(d2g):
                rospy.logwarn("A*: DEADLOCK — no goal progress in %.0fs, escaping" % self._progress_window)
                self._deadlock_escape = True
                self._escape_start = rospy.get_time()
                self._escape_heading = self._find_clear_heading()
                self._escape_phase = 0
                self.in_recovery = False
                self._do_deadlock_escape()
                self._log(0, 0, obs_d, fc, cl, self._escape_heading, 0, 0, 'esc', 'esc_init', 0, d2g)
                rate.sleep()
                continue

            if self.in_recovery:
                self._do_recovery()
                rt = self.recovery_target
                rtx, rty = (rt[0], rt[1]) if rt else (0, 0)
                self._log(0, 0, obs_d, fc, cl, rtx, rty, 0, 'rec', 'rec_cont', 0, d2g)
                rate.sleep()
                continue

            # [F4] Trigger recovery: stuck, too many stops, or spinning
            was_stuck = self._check_stuck()
            if was_stuck or self.consecutive_stops >= self.collision_stop_threshold or self.spin_count >= 20:
                reason = 'rec_stuck' if was_stuck else ('rec_stops' if self.consecutive_stops >= self.collision_stop_threshold else 'rec_spins')
                self._do_recovery()
                rt = self.recovery_target
                rtx, rty = (rt[0], rt[1]) if rt else (0, 0)
                self._log(0, 0, obs_d, fc, cl, rtx, rty, 0, 'rec', reason, 0, d2g)
                rate.sleep()
                continue

            # Near goal: drive straight without A* (avoids plan failure oscillation)
            if d2g < 2.0:
                tx, ty = self.goal_x, self.goal_y
            else:
                # Replan
                now = rospy.get_time()
                iv = min(self.replan_interval, 1.0)
                if not self.path or now - self.last_plan_time >= iv or self.path_idx >= len(self.path) - 1:
                    if self._plan():
                        self.last_plan_time = now
                        self.plan_fail_count = 0
                    else:
                        self.last_plan_time = now - iv + 0.3
                        self.plan_fail_count += 1
                        # [F5] No valid path → retreat via breadcrumbs
                        if self.plan_fail_count >= 3:
                            rospy.logwarn("A*: %d plan failures — retreating" % self.plan_fail_count)
                            self.plan_fail_count = 0
                            self._do_recovery()
                            rt = self.recovery_target
                            rtx, rty = (rt[0], rt[1]) if rt else (0, 0)
                            self._log(0, 0, obs_d, fc, cl, rtx, rty, 0, 'rec', 'rec_pfail', 0, d2g)
                            rate.sleep()
                            continue

                # Target waypoint
                if self.path and self.path_idx < len(self.path):
                    self._advance_idx()
                    tx, ty = self.path[self.path_idx]
                else:
                    tx, ty = self.goal_x, self.goal_y

            # ── Path-following controller ──
            dx = tx - self.robot_x
            dy = ty - self.robot_y
            tyaw = math.atan2(dy, dx)
            ae = self._na(tyaw - self.robot_yaw)
            cl = obs_d - self.control_radius - self.safe_margin

            cmd = Twist()
            branch = 'nav'

            if cl < self.stop_clearance:
                # Emergency: obstacle touching — back up if rear clear, else stop
                rc = self._rear_clear()
                if rc > self.robot_radius + 0.1:
                    cmd.linear.x = -0.25
                    branch = 'estop_back'
                else:
                    branch = 'estop_halt'
                self.consecutive_stops += 1

            elif abs(ae) > self.align_thresh:
                # Need to rotate — but in tight spaces, just back up instead
                f_d, l_d, r_d = self._sector_clearance()
                turn_side_d = l_d if ae > 0 else r_d
                rot_margin = self.robot_radius + 0.04  # ~0.30m
                tight = cl < 0.20 or obs_d < self.robot_radius + 0.15

                if tight:
                    # Constrained space — don't rotate, just back up gracefully
                    rc = self._rear_clear()
                    if rc > self.robot_radius + 0.1:
                        cmd.linear.x = -0.25
                        branch = 'backup_tight'
                    else:
                        # Can't even back up — trigger full recovery
                        self.consecutive_stops = self.collision_stop_threshold
                        branch = 'trapped'
                elif f_d < rot_margin and turn_side_d < rot_margin:
                    # Surrounded — back up, don't spin
                    rc = self._rear_clear()
                    if rc > self.robot_radius + 0.1:
                        cmd.linear.x = -0.25
                        branch = 'backup_surr'
                    else:
                        self.consecutive_stops = self.collision_stop_threshold
                        branch = 'trapped'
                elif not self._rotation_safe(np.sign(ae) * 1.5):
                    # Rotation would clip — back up
                    rc = self._rear_clear()
                    if rc > self.robot_radius + 0.1:
                        cmd.linear.x = -0.2
                        branch = 'rot_backup'
                    else:
                        self.consecutive_stops = self.collision_stop_threshold
                        branch = 'rot_trapped'
                else:
                    # Open space — safe to rotate
                    max_rot = min(self.max_ang, 1.5)
                    raw_av = np.clip(self.Kp_ang * ae, -max_rot, max_rot)
                    dav = np.clip(raw_av - self.prev_ang_vel,
                                   -self.max_ang_accel, self.max_ang_accel)
                    cmd.angular.z = self.prev_ang_vel + dav
                    self.prev_ang_vel = cmd.angular.z
                    branch = 'rot'
                self.consecutive_stops = 0

            else:
                # Drive forward — sample 32 trajectories, pick best
                max_sp = self.max_lin
                if cl < self.slow_clearance:
                    max_sp *= max(0.15, cl / self.slow_clearance)
                max_sp = min(max_sp, self.prev_lin_vel + self.accel_limit)
                max_sp = max(0.10, max_sp)

                # Desired angular vel from P-controller
                raw_av = np.clip(self.Kp_ang * ae, -self.max_ang, self.max_ang)
                dav = np.clip(raw_av - self.prev_ang_vel,
                               -self.max_ang_accel, self.max_ang_accel)
                desired_av = self.prev_ang_vel + dav

                # Sample 32 trajectories: 4 speeds x 8 angular velocities
                speed_samples = [max_sp, max_sp * 0.7, max_sp * 0.4, max_sp * 0.15]
                av_spread = min(self.max_ang, 1.5)  # spread around desired
                av_samples = [desired_av + av_spread * f for f in
                              [-0.6, -0.3, -0.1, 0.0, 0.1, 0.3, 0.6, 1.0]]
                av_samples = [np.clip(a, -self.max_ang, self.max_ang) for a in av_samples]

                best_score = -1.0
                best_sp = 0.0
                best_av = 0.0
                best_branch = 'drive_blocked'

                # Get costmap for trajectory scoring
                if hasattr(self, '_last_grid') and self._last_grid is not None:
                    score_grid, sgox, sgoy = self._last_grid
                else:
                    score_grid = None

                for s_sp in speed_samples:
                    for s_av in av_samples:
                        safe, _ = self._check_fwd_collision(s_sp, s_av)
                        if not safe:
                            continue

                        # Score: speed + heading alignment + costmap clearance
                        speed_score = s_sp / self.max_lin
                        heading_err = abs(ae - (s_av * 0.3))  # projected heading error
                        heading_score = max(0.0, 1.0 - heading_err / math.pi)

                        # Costmap score: check cost at projected position
                        cost_score = 1.0
                        if score_grid is not None:
                            px = self.robot_x + s_sp * math.cos(self.robot_yaw) * 0.5
                            py = self.robot_y + s_sp * math.sin(self.robot_yaw) * 0.5
                            gi = int((px - sgox) / self.grid_res)
                            gj = int((py - sgoy) / self.grid_res)
                            n = score_grid.shape[0]
                            if 0 <= gi < n and 0 <= gj < n:
                                cost_score = max(0.0, 1.0 - score_grid[gi, gj])

                        score = 0.4 * speed_score + 0.3 * heading_score + 0.3 * cost_score
                        if score > best_score:
                            best_score = score
                            best_sp = s_sp
                            best_av = s_av
                            best_branch = 'drive'

                if best_score < 0:
                    # All 32 trajectories blocked
                    best_sp = 0.0
                    best_av = 0.0
                    self.last_plan_time = 0.0
                    self.consecutive_stops += 1
                    best_branch = 'drive_blocked'

                sp = best_sp
                ac = best_av
                branch = best_branch

                cmd.linear.x = sp
                cmd.angular.z = ac
                self.prev_lin_vel = sp
                self.prev_ang_vel = ac

                if sp > 0:
                    self.consecutive_stops = max(0, self.consecutive_stops - 1)
                if sp < 0.1 and abs(ac) > 1.5:
                    self.spin_count += 1
                else:
                    self.spin_count = max(0, self.spin_count - 1)

            st = 'stp' if cmd.linear.x == 0 and cmd.angular.z == 0 else 'nav'
            self._log(cmd.linear.x, cmd.angular.z, obs_d, fc, cl, tx, ty, ae, st, branch, len(self.path), d2g)
            self.cmd_pub.publish(cmd)
            rate.sleep()

        self._lf.close()
        self.cmd_pub.publish(Twist())

    def _log(self, lv, av, o, f, cl, tx, ty, ae, st, branch, pl, d):
        self._lw.writerow([
            '%.3f' % rospy.get_time(), '%.3f' % self.robot_x, '%.3f' % self.robot_y,
            '%.2f' % self.robot_yaw, '%.3f' % lv, '%.3f' % av,
            '%.3f' % o, '%.3f' % f, '%.3f' % cl, '%.2f' % tx, '%.2f' % ty,
            '%.3f' % ae, st, branch, '%d' % pl, '%.3f' % d,
            '%d' % self.consecutive_stops, '%d' % self.spin_count,
            '%d' % self.plan_fail_count, '%d' % self.recovery_count])


if __name__ == '__main__':
    try:
        AStarPlanner().run()
    except rospy.ROSInterruptException:
        pass
