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
        self.region_alpha = float(rospy.get_param("~swp/region_alpha", 0.75))
        self.wpb_alpha = float(rospy.get_param("~swp/wpb_alpha", 0.9))
        self.min_cluster_size = int(rospy.get_param("~swp/min_cluster_size", 1))
        self.seed_search_radius_cells = int(rospy.get_param("~swp/seed_search_radius_cells", 2))
        self.seed_fallback_radius_cells = int(rospy.get_param("~swp/seed_fallback_radius_cells", 6))
        self.publish_seed_cells = rospy.get_param("~swp/publish_seed_cells", True)
        self.seed_z = float(rospy.get_param("~swp/seed_z", 0.35))
        self.seed_alpha = float(rospy.get_param("~swp/seed_alpha", 1.0))
        self.marker_id_min_overlap_ratio = float(rospy.get_param("~swp/marker_id_min_overlap_ratio", 0.2))
        self.scope = rospy.get_param("~swp/scope", "all_subregions")
        self.debug = rospy.get_param("~swp/debug", True)

        self.lock = threading.Lock()
        self.map_msg = None
        self.context = None
        self.last_context_key = None
        self.last_region_marker_count = 0
        self.last_wpb_marker_count = 0
        self.last_region_marker_ids = set()
        self.last_wpb_marker_ids = set()
        self.last_seed_marker_ids = set()
        self.last_source_signature = None
        self.last_seed_signature = None
        self.last_region_signature = None
        self.last_wpb_signature = None
        self.region_marker_tracks = {}
        self.next_region_marker_id = 0

        self.map_sub = rospy.Subscriber("/map", OccupancyGrid, self.map_callback, queue_size=1)
        self.region_pub = rospy.Publisher("/swp_regions", MarkerArray, queue_size=1)
        self.wpb_pub = rospy.Publisher("/swp_wpb", MarkerArray, queue_size=1)
        self.seed_pub = rospy.Publisher("/swp_seed_cells", MarkerArray, queue_size=1)

        period = 1.0 / self.update_rate if self.update_rate > 0.0 else 0.1
        self.timer = rospy.Timer(rospy.Duration(period), self.timer_callback)

    def map_callback(self, msg):
        with self.lock:
            self.map_msg = msg

    def set_context(self, selected_subregion, subregion_center, map_origin_resized,
                    map_size_resized, n_w, n_h, frontiers, frontier_cluster_dist,
                    subregion_centers=None, active_subregions=None, selected_frontiers=None):
        context = {
            "selected_subregion": selected_subregion,
            "subregion_center": list(subregion_center),
            "subregion_centers": [list(center) for center in subregion_centers] if subregion_centers is not None else [list(subregion_center)],
            "active_subregions": list(active_subregions) if active_subregions is not None else [selected_subregion],
            "map_origin_resized": list(map_origin_resized),
            "map_size_resized": list(map_size_resized),
            "n_w": int(n_w),
            "n_h": int(n_h),
            "frontiers": [self._frontier_to_xy(frontier) for frontier in frontiers],
            "selected_frontiers": [self._frontier_to_xy(frontier) for frontier in selected_frontiers] if selected_frontiers is not None else [],
            "frontier_cluster_dist": float(frontier_cluster_dist),
        }
        with self.lock:
            self.context = context

    def clear_context(self):
        with self.lock:
            self.context = None

    def timer_callback(self, _event):
        if not self.enabled:
            self._clear_markers(reason="swp_disabled")
            return

        with self.lock:
            map_msg = self.map_msg
            context = self.context.copy() if self.context is not None else None

        if map_msg is None or context is None:
            if context is None and (self.last_context_key is not None or
                                    self.last_region_marker_count > 0 or
                                    self.last_wpb_marker_count > 0):
                self._clear_markers(reason="context_missing")
                self.last_context_key = None
            return

        if self.scope == "all_subregions":
            context_key = (
                self.scope,
                context["n_w"],
                context["n_h"],
                round(context["map_origin_resized"][0], 3),
                round(context["map_origin_resized"][1], 3),
                round(context["map_size_resized"][0], 3),
                round(context["map_size_resized"][1], 3),
                tuple(sorted(context["active_subregions"])),
            )
        else:
            context_key = (
                self.scope,
                context["selected_subregion"],
                context["n_w"],
                context["n_h"],
                round(context["subregion_center"][0], 3),
                round(context["subregion_center"][1], 3),
            )
        if context_key != self.last_context_key:
            rospy.logwarn(
                "SWP_CONTEXT_KEY_CHANGED old=%s new=%s last_regions=%d last_wpb=%d",
                self.last_context_key,
                context_key,
                self.last_region_marker_count,
                self.last_wpb_marker_count,
            )
            self.last_context_key = context_key

        result = self.updateSWP(map_msg, context)
        self._attach_publish_stats(result)
        self.publish_result(map_msg, result)
        self._log_result(context, result)

    def updateSWP(self, map_msg, context):
        if self.scope == "all_subregions":
            return self._update_all_subregions(map_msg, context)
        return self._update_selected_subregion(map_msg, context)

    def _update_selected_subregion(self, map_msg, context):
        raw_clusters = self._cluster_frontiers(
            context["selected_frontiers"] if context["selected_frontiers"] else context["frontiers"],
            context["frontier_cluster_dist"],
        )
        clusters = [cluster for cluster in raw_clusters if len(cluster) >= self.min_cluster_size]

        seed_stats = self._empty_seed_stats()

        if len(clusters) == 0:
            return self._empty_result(context, raw_clusters, seed_stats, mode="selected_subregion")

        bounds = self._subregion_bounds(map_msg, context)
        result = self._propagate_clusters(map_msg, clusters, bounds, seed_stats, label_offset=0)
        result["stats"].update({
            "mode": "selected_subregion",
            "frontiers": len(context["selected_frontiers"] if context["selected_frontiers"] else context["frontiers"]),
            "total_frontiers": len(context["frontiers"]),
            "raw_clusters": len(raw_clusters),
            "valid_clusters": len(clusters),
            "active_subregions": 1,
            "source_signature": self._clusters_signature(clusters),
        })
        return result

    def _update_all_subregions(self, map_msg, context):
        raw_clusters = self._cluster_frontiers(
            context["frontiers"],
            context["frontier_cluster_dist"],
        )
        clusters = [cluster for cluster in raw_clusters if len(cluster) >= self.min_cluster_size]
        seed_stats = self._empty_seed_stats()

        if len(clusters) == 0:
            return self._empty_result(context, raw_clusters, seed_stats, mode="all_subregions")

        clusters_by_subregion = defaultdict(list)
        dropped_clusters = 0
        for cluster in clusters:
            subregion_idx = self._cluster_subregion(cluster, context)
            if subregion_idx is None:
                dropped_clusters += 1
                continue
            clusters_by_subregion[subregion_idx].append(cluster)

        merged_regions = {}
        merged_wpb_cells = set()
        merged_wpb_adjacent_labels = defaultdict(set)
        merged_seed_cells = set()
        merged_cluster_seed_counts = []
        subregion_summaries = []
        label_offset = 0

        for subregion_idx in sorted(clusters_by_subregion.keys()):
            bounds = self._subregion_bounds_for_index(map_msg, context, subregion_idx)
            subregion_clusters = clusters_by_subregion[subregion_idx]
            result = self._propagate_clusters(
                map_msg,
                subregion_clusters,
                bounds,
                seed_stats,
                label_offset=label_offset,
            )
            merged_regions.update(result["regions"])
            merged_wpb_cells.update(result["wpb_cells"])
            merged_seed_cells.update(result["seed_cells"])
            for cell, labels in result["wpb_adjacent_labels"].items():
                merged_wpb_adjacent_labels[cell].update(labels)
            merged_cluster_seed_counts.extend(result["stats"]["cluster_seed_counts"])
            subregion_summaries.append(
                "%s:c%d/s%d/r%d/w%d" % (
                    subregion_idx,
                    len(subregion_clusters),
                    result["stats"]["seed_cells"],
                    len(result["regions"]),
                    len(result["wpb_cells"]),
                )
            )
            label_offset += len(subregion_clusters)

        region_cells = sum(len(cells) for cells in merged_regions.values())
        return {
            "regions": merged_regions,
            "wpb_cells": merged_wpb_cells,
            "wpb_adjacent_labels": dict(merged_wpb_adjacent_labels),
            "seed_cells": merged_seed_cells,
            "stats": {
                "mode": "all_subregions",
                "frontiers": len(context["frontiers"]),
                "total_frontiers": len(context["frontiers"]),
                "selected_frontiers": len(context["selected_frontiers"]),
                "raw_clusters": len(raw_clusters),
                "valid_clusters": len(clusters),
                "dropped_clusters": dropped_clusters,
                "active_subregions": len(clusters_by_subregion),
                "seed_cells": sum(merged_cluster_seed_counts),
                "seed_failures": seed_stats,
                "cluster_seed_counts": merged_cluster_seed_counts,
                "subregion_summaries": subregion_summaries,
                "region_cells": region_cells,
                "source_signature": self._clusters_by_subregion_signature(clusters_by_subregion),
            },
        }

    def _propagate_clusters(self, map_msg, clusters, bounds, seed_stats, label_offset):
        labels = {}
        wpb_cells = set()
        wpb_adjacent_labels = defaultdict(set)
        active = deque()
        all_seed_cells = set()
        seed_count = 0
        cluster_seed_counts = []

        for local_label_id, cluster in enumerate(clusters):
            label_id = label_offset + local_label_id
            seeds = self._cluster_seed_cells(map_msg, cluster, bounds, seed_stats)
            seed_count += len(seeds)
            all_seed_cells.update(seeds)
            cluster_seed_counts.append(len(seeds))
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
            "seed_cells": all_seed_cells,
            "stats": {
                "seed_cells": seed_count,
                "seed_failures": seed_stats,
                "cluster_seed_counts": cluster_seed_counts,
            },
        }

    def publish_result(self, map_msg, result):
        stamp = rospy.Time.now()
        region_array = MarkerArray()
        wpb_array = MarkerArray()
        seed_array = MarkerArray()
        publish_stats = result.setdefault("publish_stats", {})

        current_region_ids = set()
        region_marker_ids, region_marker_stats = self._assign_region_marker_ids(result["regions"])
        publish_stats.update(region_marker_stats)
        for label_id in sorted(result["regions"].keys()):
            cells = result["regions"][label_id]
            if not cells:
                continue
            marker_id = region_marker_ids[label_id]
            marker = self._make_cube_list_marker(
                frame_id=map_msg.header.frame_id or "map",
                stamp=stamp,
                ns="swp_regions",
                marker_id=marker_id,
                resolution=map_msg.info.resolution,
                z=self.region_z,
                color=self._region_color(marker_id),
            )
            marker.points = [self._cell_to_point(map_msg, cell, self.region_z) for cell in cells]
            region_array.markers.append(marker)
            current_region_ids.add(marker_id)

        components = self._connected_components(result["wpb_cells"])
        current_wpb_ids = set()
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
            current_wpb_ids.add(marker_id)

        self.region_pub.publish(region_array)
        self.wpb_pub.publish(wpb_array)
        current_seed_ids = self._append_seed_markers(map_msg, stamp, result, seed_array)
        publish_stats["seed_marker_points"] = sum(len(marker.points) for marker in seed_array.markers)
        publish_stats["seed_marker_count"] = len(seed_array.markers)
        publish_stats["seed_publish_enabled"] = self.publish_seed_cells
        self.seed_pub.publish(seed_array)
        self._delete_stale_markers(self.region_pub, "swp_regions", current_region_ids, self.last_region_marker_ids)
        self._delete_stale_markers(self.wpb_pub, "swp_wpb", current_wpb_ids, self.last_wpb_marker_ids)
        self._delete_stale_markers(self.seed_pub, "swp_seed_cells", current_seed_ids, self.last_seed_marker_ids)
        self.last_region_marker_ids = current_region_ids
        self.last_wpb_marker_ids = current_wpb_ids
        self.last_seed_marker_ids = current_seed_ids
        self.last_region_marker_count = len(current_region_ids)
        self.last_wpb_marker_count = len(current_wpb_ids)

    def _append_seed_markers(self, map_msg, stamp, result, seed_array):
        if not self.publish_seed_cells or not result["seed_cells"]:
            return set()
        marker = self._make_cube_list_marker(
            frame_id=map_msg.header.frame_id or "map",
            stamp=stamp,
            ns="swp_seed_cells",
            marker_id=0,
            resolution=map_msg.info.resolution,
            z=self.seed_z,
            color=ColorRGBA(0.0, 0.0, 0.0, self.seed_alpha),
        )
        marker.points = [self._cell_to_point(map_msg, cell, self.seed_z) for cell in result["seed_cells"]]
        seed_array.markers.append(marker)
        return {0}

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

    def _cluster_seed_cells(self, map_msg, cluster, bounds, seed_stats):
        seeds = set()
        for frontier in cluster:
            anchor = self._world_to_cell(map_msg, frontier)
            if anchor is None:
                seed_stats["anchor_outside_map"] += 1
                continue
            frontier_seeds = self._unknown_cells_around_anchor(map_msg, anchor, bounds, seed_stats)
            seeds.update(frontier_seeds)
        return seeds

    def _unknown_cells_around_anchor(self, map_msg, anchor, bounds, seed_stats):
        if not self._inside_bounds(anchor, bounds):
            seed_stats["anchor_outside_subregion"] += 1

        search_radius = max(1, self.seed_search_radius_cells)
        fallback_radius = max(search_radius, self.seed_fallback_radius_cells)
        candidates, inspected = self._collect_unknown_candidates(map_msg, anchor, bounds, search_radius)
        fallback_used = False
        if not candidates and fallback_radius > search_radius:
            fallback_candidates, fallback_inspected = self._collect_unknown_candidates(
                map_msg,
                anchor,
                bounds,
                fallback_radius,
            )
            candidates = fallback_candidates
            inspected = fallback_inspected
            fallback_used = bool(candidates)

        if candidates:
            min_distance = min(distance for _, distance in candidates)
            selected = {cell for cell, distance in candidates if distance == min_distance}
            seed_stats["seeded_frontiers"] += 1
            if fallback_used:
                seed_stats["seed_fallback_used"] += 1
            seed_stats["max_seed_distance"] = max(seed_stats["max_seed_distance"], min_distance)
            return selected

        seed_stats["no_unknown_near_anchor"] += 1
        for key, value in inspected.items():
            seed_stats[key] += value
        return set()

    def _collect_unknown_candidates(self, map_msg, anchor, bounds, radius):
        inspected_unknown = 0
        inspected_free = 0
        inspected_occupied = 0
        inspected_outside_subregion = 0
        inspected_outside_map = 0
        candidates = []
        for dx in range(-radius, radius + 1):
            for dy in range(-radius, radius + 1):
                distance = abs(dx) + abs(dy)
                if distance == 0 or distance > radius:
                    continue
                cell = (anchor[0] + dx, anchor[1] + dy)
                if not self._inside_map(map_msg, cell):
                    inspected_outside_map += 1
                    continue
                if not self._inside_bounds(cell, bounds):
                    inspected_outside_subregion += 1
                    continue
                value = self._cell_value(map_msg, cell)
                if value == -1:
                    inspected_unknown += 1
                    candidates.append((cell, distance))
                elif 0 <= value <= 10:
                    inspected_free += 1
                else:
                    inspected_occupied += 1
        inspected = {
            "inspected_unknown": inspected_unknown,
            "inspected_free": inspected_free,
            "inspected_occupied": inspected_occupied,
            "inspected_outside_subregion": inspected_outside_subregion,
            "inspected_outside_map": inspected_outside_map,
        }
        return candidates, inspected

    def _subregion_bounds(self, map_msg, context):
        return self._subregion_bounds_for_center(map_msg, context, context["subregion_center"])

    def _subregion_bounds_for_index(self, map_msg, context, subregion_idx):
        if 0 <= subregion_idx < len(context["subregion_centers"]):
            center = context["subregion_centers"][subregion_idx]
        else:
            subregion_width = context["map_size_resized"][0] / context["n_w"]
            subregion_height = context["map_size_resized"][1] / context["n_h"]
            center = [
                context["map_origin_resized"][0] + int(subregion_idx % context["n_w"]) * subregion_width + subregion_width / 2.0,
                context["map_origin_resized"][1] + int(subregion_idx / context["n_w"]) * subregion_height + subregion_height / 2.0,
            ]
        return self._subregion_bounds_for_center(map_msg, context, center)

    def _subregion_bounds_for_center(self, map_msg, context, center):
        subregion_width = context["map_size_resized"][0] / context["n_w"]
        subregion_height = context["map_size_resized"][1] / context["n_h"]
        center_x, center_y = center
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

    def _cluster_subregion(self, cluster, context):
        centroid = [
            sum(point[0] for point in cluster) / len(cluster),
            sum(point[1] for point in cluster) / len(cluster),
        ]
        active_subregions = set(context["active_subregions"])
        for subregion_idx in sorted(active_subregions):
            if 0 <= subregion_idx < len(context["subregion_centers"]):
                center = context["subregion_centers"][subregion_idx]
            else:
                continue
            if self._is_inside_subregion(context, center, centroid):
                return subregion_idx
        return None

    def _is_inside_subregion(self, context, center, point):
        subregion_width = context["map_size_resized"][0] / context["n_w"]
        subregion_height = context["map_size_resized"][1] / context["n_h"]
        x1 = center[0] - subregion_width / 2.0
        x2 = center[0] + subregion_width / 2.0
        y1 = center[1] - subregion_height / 2.0
        y2 = center[1] + subregion_height / 2.0
        return x1 < point[0] < x2 and y1 < point[1] < y2

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
        return self._cell_value(map_msg, cell) == -1

    def _cell_value(self, map_msg, cell):
        x, y = cell
        return map_msg.data[y * map_msg.info.width + x]

    def _inside_map(self, map_msg, cell):
        x, y = cell
        return 0 <= x < map_msg.info.width and 0 <= y < map_msg.info.height

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
            (0.00, 0.36, 1.00),
            (1.00, 0.42, 0.00),
            (0.00, 0.88, 0.18),
            (1.00, 0.00, 0.18),
            (0.62, 0.00, 1.00),
            (0.00, 0.92, 1.00),
            (1.00, 0.86, 0.00),
            (1.00, 0.00, 0.72),
            (0.50, 1.00, 0.00),
            (0.00, 0.58, 0.32),
        ]
        r, g, b = palette[label_id % len(palette)]
        return ColorRGBA(r, g, b, self.region_alpha)

    def _clear_markers(self, reason, old_context_key=None, new_context_key=None):
        rospy.logwarn(
            "SWP_CLEAR reason=%s old=%s new=%s last_regions=%d last_wpb=%d last_region_ids=%s last_wpb_ids=%s",
            reason,
            old_context_key,
            new_context_key,
            self.last_region_marker_count,
            self.last_wpb_marker_count,
            sorted(self.last_region_marker_ids)[:20],
            sorted(self.last_wpb_marker_ids)[:20],
        )
        region_clear = MarkerArray()
        region_marker = Marker()
        region_marker.action = Marker.DELETEALL
        region_clear.markers.append(region_marker)
        wpb_clear = MarkerArray()
        wpb_marker = Marker()
        wpb_marker.action = Marker.DELETEALL
        wpb_clear.markers.append(wpb_marker)
        seed_clear = MarkerArray()
        seed_marker = Marker()
        seed_marker.action = Marker.DELETEALL
        seed_clear.markers.append(seed_marker)
        self.region_pub.publish(region_clear)
        self.wpb_pub.publish(wpb_clear)
        self.seed_pub.publish(seed_clear)
        self.last_region_marker_count = 0
        self.last_wpb_marker_count = 0
        self.last_region_marker_ids = set()
        self.last_wpb_marker_ids = set()
        self.last_seed_marker_ids = set()
        self.region_marker_tracks = {}
        self.next_region_marker_id = 0

    def _assign_region_marker_ids(self, regions):
        region_cells = {
            label_id: set(cells)
            for label_id, cells in regions.items()
            if cells
        }
        label_to_marker_id = {}
        used_marker_ids = set()
        reused = 0
        created = 0

        candidates = []
        for label_id, cells in region_cells.items():
            if not cells:
                continue
            for marker_id, previous_cells in self.region_marker_tracks.items():
                overlap = len(cells & previous_cells)
                if overlap == 0:
                    continue
                overlap_ratio = float(overlap) / float(len(cells))
                candidates.append((overlap_ratio, overlap, label_id, marker_id))

        candidates.sort(reverse=True)
        assigned_labels = set()
        for overlap_ratio, _overlap, label_id, marker_id in candidates:
            if overlap_ratio < self.marker_id_min_overlap_ratio:
                continue
            if label_id in assigned_labels or marker_id in used_marker_ids:
                continue
            label_to_marker_id[label_id] = marker_id
            assigned_labels.add(label_id)
            used_marker_ids.add(marker_id)
            reused += 1

        for label_id in sorted(region_cells.keys()):
            if label_id in label_to_marker_id:
                continue
            marker_id = self.next_region_marker_id
            self.next_region_marker_id += 1
            label_to_marker_id[label_id] = marker_id
            used_marker_ids.add(marker_id)
            created += 1

        self.region_marker_tracks = {
            label_to_marker_id[label_id]: cells
            for label_id, cells in region_cells.items()
        }

        if created > 0 or (region_cells and reused == 0):
            rospy.logwarn(
                "SWP_REGION_TRACK reused=%d new=%d tracks=%d min_overlap=%.3f labels=%s marker_ids=%s",
                reused,
                created,
                len(self.region_marker_tracks),
                self.marker_id_min_overlap_ratio,
                sorted(region_cells.keys())[:20],
                sorted(label_to_marker_id.values())[:20],
            )

        return label_to_marker_id, {
            "region_marker_reused": reused,
            "region_marker_new": created,
            "region_marker_tracks": len(self.region_marker_tracks),
        }

    def _delete_stale_markers(self, publisher, namespace, current_ids, last_ids):
        stale_ids = last_ids - current_ids
        if not stale_ids:
            return
        if namespace in ("swp_regions", "swp_wpb"):
            rospy.logwarn(
                "SWP_STALE_DELETE namespace=%s stale=%d current=%d last=%d stale_ids=%s current_ids=%s",
                namespace,
                len(stale_ids),
                len(current_ids),
                len(last_ids),
                sorted(stale_ids)[:20],
                sorted(current_ids)[:20],
            )
        delete_array = MarkerArray()
        for marker_id in sorted(stale_ids):
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

    def _empty_seed_stats(self):
        return {
            "seeded_frontiers": 0,
            "anchor_outside_map": 0,
            "anchor_outside_subregion": 0,
            "no_unknown_near_anchor": 0,
            "seed_fallback_used": 0,
            "max_seed_distance": 0,
            "inspected_unknown": 0,
            "inspected_free": 0,
            "inspected_occupied": 0,
            "inspected_outside_subregion": 0,
            "inspected_outside_map": 0,
        }

    def _empty_result(self, context, raw_clusters, seed_stats, mode):
        frontier_count = len(context["frontiers"])
        if mode == "selected_subregion" and context["selected_frontiers"]:
            frontier_count = len(context["selected_frontiers"])
        return {
            "regions": {},
            "wpb_cells": set(),
            "wpb_adjacent_labels": {},
            "seed_cells": set(),
            "stats": {
                "mode": mode,
                "frontiers": frontier_count,
                "total_frontiers": len(context["frontiers"]),
                "selected_frontiers": len(context["selected_frontiers"]),
                "raw_clusters": len(raw_clusters),
                "valid_clusters": 0,
                "dropped_clusters": 0,
                "active_subregions": 0,
                "seed_cells": 0,
                "seed_failures": seed_stats,
                "cluster_seed_counts": [],
                "subregion_summaries": [],
                "source_signature": self._sequence_signature([]),
            },
        }

    def _attach_publish_stats(self, result):
        source_signature = result["stats"].get(
            "source_signature",
            self._sequence_signature(result["stats"].get("cluster_seed_counts", [])),
        )
        seed_signature = self._cell_signature(result["seed_cells"])
        region_signature = self._region_signature(result["regions"])
        wpb_signature = self._cell_signature(result["wpb_cells"])

        result["publish_stats"] = {
            "source_signature": source_signature,
            "seed_signature": seed_signature,
            "region_signature": region_signature,
            "wpb_signature": wpb_signature,
            "source_changed": self.last_source_signature is not None and source_signature != self.last_source_signature,
            "seed_changed": self.last_seed_signature is not None and seed_signature != self.last_seed_signature,
            "region_changed": self.last_region_signature is not None and region_signature != self.last_region_signature,
            "wpb_changed": self.last_wpb_signature is not None and wpb_signature != self.last_wpb_signature,
            "seed_marker_points": 0,
            "seed_marker_count": 0,
            "seed_publish_enabled": self.publish_seed_cells,
        }
        self.last_source_signature = source_signature
        self.last_seed_signature = seed_signature
        self.last_region_signature = region_signature
        self.last_wpb_signature = wpb_signature

    def _sequence_signature(self, values):
        return hash(tuple(values))

    def _clusters_signature(self, clusters):
        serialized_clusters = []
        for cluster in clusters:
            points = tuple(sorted((round(point[0], 2), round(point[1], 2)) for point in cluster))
            serialized_clusters.append(points)
        return hash(tuple(sorted(serialized_clusters)))

    def _clusters_by_subregion_signature(self, clusters_by_subregion):
        serialized_subregions = []
        for subregion_idx in sorted(clusters_by_subregion.keys()):
            serialized_subregions.append(
                (subregion_idx, self._clusters_signature(clusters_by_subregion[subregion_idx]))
            )
        return hash(tuple(serialized_subregions))

    def _cell_signature(self, cells):
        return hash(tuple(sorted(cells)))

    def _region_signature(self, regions):
        items = []
        for label_id in sorted(regions.keys()):
            items.append((label_id, self._cell_signature(regions[label_id]), len(regions[label_id])))
        return hash(tuple(items))

    def _log_result(self, context, result):
        if not self.debug:
            return
        stats = result.get("stats", {})
        publish_stats = result.get("publish_stats", {})
        seed_failures = stats.get("seed_failures", {})
        cluster_seed_counts = stats.get("cluster_seed_counts", [])
        subregion_summaries = stats.get("subregion_summaries", [])
        region_cells = sum(len(cells) for cells in result["regions"].values())
        rospy.loginfo_throttle(
            2.0,
            "SWP mode=%s selected_subregion=%s frontiers=%d selected_frontiers=%d clusters=%d/%d dropped=%d active_subregions=%d seeds=%d seed_marker_points=%d seed_marker_count=%d seed_publish_enabled=%s seed_changed=%s source_changed=%s region_changed=%s wpb_changed=%s region_marker_reused=%d region_marker_new=%d region_marker_tracks=%d seed_fallback_used=%d max_seed_distance=%d regions=%d region_cells=%d wpb_cells=%d cluster_seed_counts=%s subregions=%s seed_failures=%s",
            stats.get("mode", self.scope),
            context["selected_subregion"],
            stats.get("frontiers", 0),
            stats.get("selected_frontiers", len(context["selected_frontiers"])),
            stats.get("valid_clusters", 0),
            stats.get("raw_clusters", 0),
            stats.get("dropped_clusters", 0),
            stats.get("active_subregions", 0),
            stats.get("seed_cells", 0),
            publish_stats.get("seed_marker_points", 0),
            publish_stats.get("seed_marker_count", 0),
            publish_stats.get("seed_publish_enabled", self.publish_seed_cells),
            publish_stats.get("seed_changed", False),
            publish_stats.get("source_changed", False),
            publish_stats.get("region_changed", False),
            publish_stats.get("wpb_changed", False),
            publish_stats.get("region_marker_reused", 0),
            publish_stats.get("region_marker_new", 0),
            publish_stats.get("region_marker_tracks", 0),
            seed_failures.get("seed_fallback_used", 0),
            seed_failures.get("max_seed_distance", 0),
            len(result["regions"]),
            region_cells,
            len(result["wpb_cells"]),
            cluster_seed_counts[:10],
            subregion_summaries[:10],
            seed_failures,
        )
