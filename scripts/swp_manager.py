#!/usr/bin/env python3
import math
import threading
from collections import defaultdict, deque

import rospy
from geometry_msgs.msg import Point
from nav_msgs.msg import OccupancyGrid
from std_msgs.msg import ColorRGBA
from visualization_msgs.msg import Marker, MarkerArray


class SWPManager:
    def __init__(self):
        self.enabled = rospy.get_param("~swp/enabled", True)
        self.update_rate = float(rospy.get_param("~swp/update_rate", 10.0))
        self.region_z = float(rospy.get_param("~swp/region_z", 0.15))
        self.wpb_z = float(rospy.get_param("~swp/wpb_z", 0.25))
        self.region_alpha = float(rospy.get_param("~swp/region_alpha", 0.35))
        self.wpb_alpha = float(rospy.get_param("~swp/wpb_alpha", 0.9))
        self.min_cluster_size = int(rospy.get_param("~swp/min_cluster_size", 1))
        self.seed_search_radius_cells = int(rospy.get_param("~swp/seed_search_radius_cells", 2))
        self.debug = rospy.get_param("~swp/debug", True)

        self.lock = threading.Lock()
        self.map_msg = None
        self.context = None
        self.last_context_key = None
        self.last_region_marker_count = 0
        self.last_wpb_marker_count = 0

        self.map_sub = rospy.Subscriber("/map", OccupancyGrid, self.map_callback, queue_size=1)
        self.region_pub = rospy.Publisher("/swp_regions", MarkerArray, queue_size=1)
        self.wpb_pub = rospy.Publisher("/swp_wpb", MarkerArray, queue_size=1)

        period = 1.0 / self.update_rate if self.update_rate > 0.0 else 0.1
        self.timer = rospy.Timer(rospy.Duration(period), self.timer_callback)

    def map_callback(self, msg):
        with self.lock:
            self.map_msg = msg

    def set_context(self, selected_subregion, subregion_center, map_origin_resized,
                    map_size_resized, n_w, n_h, frontiers, frontier_cluster_dist):
        context = {
            "selected_subregion": selected_subregion,
            "subregion_center": list(subregion_center),
            "map_origin_resized": list(map_origin_resized),
            "map_size_resized": list(map_size_resized),
            "n_w": int(n_w),
            "n_h": int(n_h),
            "frontiers": [self._frontier_to_xy(frontier) for frontier in frontiers],
            "frontier_cluster_dist": float(frontier_cluster_dist),
        }
        with self.lock:
            self.context = context

    def clear_context(self):
        with self.lock:
            self.context = None

    def timer_callback(self, _event):
        if not self.enabled:
            self._clear_markers()
            return

        with self.lock:
            map_msg = self.map_msg
            context = self.context.copy() if self.context is not None else None

        if map_msg is None or context is None:
            if context is None and (self.last_context_key is not None or
                                    self.last_region_marker_count > 0 or
                                    self.last_wpb_marker_count > 0):
                self._clear_markers()
                self.last_context_key = None
            return

        context_key = (
            context["selected_subregion"],
            context["n_w"],
            context["n_h"],
            round(context["subregion_center"][0], 3),
            round(context["subregion_center"][1], 3),
        )
        if context_key != self.last_context_key:
            self._clear_markers()
            self.last_context_key = context_key

        result = self.updateSWP(map_msg, context)
        self._log_result(context, result)
        self.publish_result(map_msg, result)

    def updateSWP(self, map_msg, context):
        raw_clusters = self._cluster_frontiers(
            context["frontiers"],
            context["frontier_cluster_dist"],
        )
        clusters = [cluster for cluster in raw_clusters if len(cluster) >= self.min_cluster_size]

        if len(clusters) == 0:
            return {
                "regions": {},
                "wpb_cells": set(),
                "wpb_adjacent_labels": {},
                "stats": {
                    "frontiers": len(context["frontiers"]),
                    "raw_clusters": len(raw_clusters),
                    "valid_clusters": 0,
                    "seed_cells": 0,
                },
            }

        bounds = self._subregion_bounds(map_msg, context)
        labels = {}
        wpb_cells = set()
        wpb_adjacent_labels = defaultdict(set)
        active = deque()
        seed_count = 0

        for label_id, cluster in enumerate(clusters):
            seeds = self._cluster_seed_cells(map_msg, cluster, bounds)
            seed_count += len(seeds)
            for cell in seeds:
                if cell in wpb_cells:
                    continue
                owner = labels.get(cell)
                if owner is None:
                    labels[cell] = label_id
                    active.append((cell[0], cell[1], label_id))
                elif owner != label_id:
                    self._mark_wpb(cell, wpb_cells, wpb_adjacent_labels, owner, label_id)

        while active:
            layer = []
            for _ in range(len(active)):
                layer.append(active.popleft())

            claims = defaultdict(set)
            for x, y, label_id in layer:
                cell = (x, y)
                if cell in wpb_cells or labels.get(cell) != label_id:
                    continue
                if self._is_subregion_boundary_cell(cell, bounds):
                    self._mark_wpb(cell, wpb_cells, wpb_adjacent_labels, label_id)
                    continue

                for nx, ny in self._neighbors4(x, y):
                    neighbor = (nx, ny)
                    if not self._inside_bounds(neighbor, bounds):
                        self._mark_wpb(cell, wpb_cells, wpb_adjacent_labels, label_id)
                        continue
                    if not self._is_unknown(map_msg, neighbor):
                        continue
                    if neighbor in wpb_cells:
                        wpb_adjacent_labels[neighbor].add(label_id)
                        continue
                    owner = labels.get(neighbor)
                    if owner is None:
                        claims[neighbor].add(label_id)
                    elif owner != label_id:
                        self._mark_wpb(cell, wpb_cells, wpb_adjacent_labels, label_id, owner)

            for cell, claim_labels in claims.items():
                if cell in wpb_cells or cell in labels:
                    continue
                if len(claim_labels) > 1:
                    self._mark_wpb(cell, wpb_cells, wpb_adjacent_labels, *claim_labels)
                    continue
                label_id = next(iter(claim_labels))
                labels[cell] = label_id
                active.append((cell[0], cell[1], label_id))

        regions = defaultdict(list)
        for cell, label_id in labels.items():
            if cell not in wpb_cells:
                regions[label_id].append(cell)

        return {
            "regions": dict(regions),
            "wpb_cells": wpb_cells,
            "wpb_adjacent_labels": dict(wpb_adjacent_labels),
            "stats": {
                "frontiers": len(context["frontiers"]),
                "raw_clusters": len(raw_clusters),
                "valid_clusters": len(clusters),
                "seed_cells": seed_count,
            },
        }

    def publish_result(self, map_msg, result):
        stamp = rospy.Time.now()
        region_array = MarkerArray()
        wpb_array = MarkerArray()

        region_count = 0
        for label_id in sorted(result["regions"].keys()):
            cells = result["regions"][label_id]
            if not cells:
                continue
            marker = self._make_cube_list_marker(
                frame_id=map_msg.header.frame_id or "map",
                stamp=stamp,
                ns="swp_regions",
                marker_id=label_id,
                resolution=map_msg.info.resolution,
                z=self.region_z,
                color=self._region_color(label_id),
            )
            marker.points = [self._cell_to_point(map_msg, cell, self.region_z) for cell in cells]
            region_array.markers.append(marker)
            region_count += 1

        components = self._connected_components(result["wpb_cells"])
        for marker_id, component in enumerate(components):
            marker = self._make_cube_list_marker(
                frame_id=map_msg.header.frame_id or "map",
                stamp=stamp,
                ns="swp_wpb",
                marker_id=marker_id,
                resolution=map_msg.info.resolution,
                z=self.wpb_z,
                color=ColorRGBA(1.0, 0.0, 0.8, self.wpb_alpha),
            )
            marker.points = [self._cell_to_point(map_msg, cell, self.wpb_z) for cell in component]
            wpb_array.markers.append(marker)

        self.region_pub.publish(region_array)
        self.wpb_pub.publish(wpb_array)
        self._delete_stale_markers(self.region_pub, "swp_regions", region_count, self.last_region_marker_count)
        self._delete_stale_markers(self.wpb_pub, "swp_wpb", len(components), self.last_wpb_marker_count)
        self.last_region_marker_count = region_count
        self.last_wpb_marker_count = len(components)

    def _frontier_to_xy(self, frontier):
        return [float(frontier[0]), float(frontier[1])]

    def _cluster_frontiers(self, frontiers, cluster_dist):
        if len(frontiers) == 0:
            return []

        visited = [False] * len(frontiers)
        clusters = []
        for start_idx in range(len(frontiers)):
            if visited[start_idx]:
                continue
            cluster = []
            queue = deque([start_idx])
            visited[start_idx] = True
            while queue:
                idx = queue.popleft()
                cluster.append(frontiers[idx])
                for next_idx in range(len(frontiers)):
                    if visited[next_idx]:
                        continue
                    if self._distance(frontiers[idx], frontiers[next_idx]) <= cluster_dist:
                        visited[next_idx] = True
                        queue.append(next_idx)
            clusters.append(cluster)
        return clusters

    def _cluster_seed_cells(self, map_msg, cluster, bounds):
        seeds = set()
        for frontier in cluster:
            anchor = self._world_to_cell(map_msg, frontier)
            if anchor is None:
                continue
            frontier_seeds = self._unknown_cells_around_anchor(map_msg, anchor, bounds)
            seeds.update(frontier_seeds)
        return seeds

    def _unknown_cells_around_anchor(self, map_msg, anchor, bounds):
        for radius in range(1, max(1, self.seed_search_radius_cells) + 1):
            candidates = []
            for dx in range(-radius, radius + 1):
                for dy in range(-radius, radius + 1):
                    if abs(dx) + abs(dy) != radius:
                        continue
                    cell = (anchor[0] + dx, anchor[1] + dy)
                    if self._inside_bounds(cell, bounds) and self._is_unknown(map_msg, cell):
                        candidates.append(cell)
            if candidates:
                return set(candidates)
        return set()

    def _subregion_bounds(self, map_msg, context):
        subregion_width = context["map_size_resized"][0] / context["n_w"]
        subregion_height = context["map_size_resized"][1] / context["n_h"]
        center_x, center_y = context["subregion_center"]
        min_x = center_x - subregion_width / 2.0
        max_x = center_x + subregion_width / 2.0
        min_y = center_y - subregion_height / 2.0
        max_y = center_y + subregion_height / 2.0

        min_cell = self._world_to_cell(map_msg, [min_x, min_y], clamp=True)
        eps = max(map_msg.info.resolution * 1e-3, 1e-6)
        max_cell = self._world_to_cell(map_msg, [max_x - eps, max_y - eps], clamp=True)
        return {
            "min_x": min(min_cell[0], max_cell[0]),
            "max_x": max(min_cell[0], max_cell[0]),
            "min_y": min(min_cell[1], max_cell[1]),
            "max_y": max(min_cell[1], max_cell[1]),
        }

    def _world_to_cell(self, map_msg, point, clamp=False):
        resolution = map_msg.info.resolution
        origin_x = map_msg.info.origin.position.x
        origin_y = map_msg.info.origin.position.y
        x = int(math.floor((point[0] - origin_x) / resolution))
        y = int(math.floor((point[1] - origin_y) / resolution))
        if clamp:
            x = min(max(x, 0), map_msg.info.width - 1)
            y = min(max(y, 0), map_msg.info.height - 1)
            return x, y
        if x < 0 or y < 0 or x >= map_msg.info.width or y >= map_msg.info.height:
            return None
        return x, y

    def _cell_to_point(self, map_msg, cell, z):
        point = Point()
        point.x = (cell[0] + 0.5) * map_msg.info.resolution + map_msg.info.origin.position.x
        point.y = (cell[1] + 0.5) * map_msg.info.resolution + map_msg.info.origin.position.y
        point.z = z
        return point

    def _is_unknown(self, map_msg, cell):
        x, y = cell
        return map_msg.data[y * map_msg.info.width + x] == -1

    def _inside_bounds(self, cell, bounds):
        x, y = cell
        return bounds["min_x"] <= x <= bounds["max_x"] and bounds["min_y"] <= y <= bounds["max_y"]

    def _is_subregion_boundary_cell(self, cell, bounds):
        x, y = cell
        return x == bounds["min_x"] or x == bounds["max_x"] or y == bounds["min_y"] or y == bounds["max_y"]

    def _neighbors4(self, x, y):
        return ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1))

    def _mark_wpb(self, cell, wpb_cells, wpb_adjacent_labels, *labels):
        wpb_cells.add(cell)
        for label_id in labels:
            wpb_adjacent_labels[cell].add(label_id)

    def _connected_components(self, cells):
        remaining = set(cells)
        components = []
        while remaining:
            start = remaining.pop()
            component = [start]
            queue = deque([start])
            while queue:
                x, y = queue.popleft()
                for neighbor in self._neighbors4(x, y):
                    if neighbor in remaining:
                        remaining.remove(neighbor)
                        component.append(neighbor)
                        queue.append(neighbor)
            components.append(component)
        return components

    def _make_cube_list_marker(self, frame_id, stamp, ns, marker_id, resolution, z, color):
        marker = Marker()
        marker.header.frame_id = frame_id
        marker.header.stamp = stamp
        marker.ns = ns
        marker.id = marker_id
        marker.type = Marker.CUBE_LIST
        marker.action = Marker.ADD
        marker.pose.orientation.w = 1.0
        marker.scale.x = resolution
        marker.scale.y = resolution
        marker.scale.z = 0.03
        marker.color = color
        marker.lifetime = rospy.Duration()
        return marker

    def _region_color(self, label_id):
        palette = [
            (0.10, 0.45, 0.95),
            (0.95, 0.55, 0.10),
            (0.20, 0.75, 0.35),
            (0.90, 0.20, 0.30),
            (0.55, 0.35, 0.95),
            (0.00, 0.70, 0.75),
            (0.95, 0.80, 0.15),
            (0.85, 0.35, 0.70),
        ]
        r, g, b = palette[label_id % len(palette)]
        return ColorRGBA(r, g, b, self.region_alpha)

    def _clear_markers(self):
        region_clear = MarkerArray()
        region_marker = Marker()
        region_marker.action = Marker.DELETEALL
        region_clear.markers.append(region_marker)
        wpb_clear = MarkerArray()
        wpb_marker = Marker()
        wpb_marker.action = Marker.DELETEALL
        wpb_clear.markers.append(wpb_marker)
        self.region_pub.publish(region_clear)
        self.wpb_pub.publish(wpb_clear)
        self.last_region_marker_count = 0
        self.last_wpb_marker_count = 0

    def _delete_stale_markers(self, publisher, namespace, current_count, last_count):
        if current_count >= last_count:
            return
        delete_array = MarkerArray()
        for marker_id in range(current_count, last_count):
            marker = Marker()
            marker.header.frame_id = "map"
            marker.header.stamp = rospy.Time.now()
            marker.ns = namespace
            marker.id = marker_id
            marker.action = Marker.DELETE
            delete_array.markers.append(marker)
        publisher.publish(delete_array)

    def _distance(self, p1, p2):
        return math.hypot(p1[0] - p2[0], p1[1] - p2[1])

    def _log_result(self, context, result):
        if not self.debug:
            return
        stats = result.get("stats", {})
        region_cells = sum(len(cells) for cells in result["regions"].values())
        rospy.loginfo_throttle(
            2.0,
            "SWP selected_subregion=%s frontiers=%d clusters=%d/%d seeds=%d regions=%d region_cells=%d wpb_cells=%d",
            context["selected_subregion"],
            stats.get("frontiers", 0),
            stats.get("valid_clusters", 0),
            stats.get("raw_clusters", 0),
            stats.get("seed_cells", 0),
            len(result["regions"]),
            region_cells,
            len(result["wpb_cells"]),
        )
