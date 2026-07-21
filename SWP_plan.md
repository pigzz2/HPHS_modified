# Subregion-bounded Wavefront Partition (SWP) Plan

## Goal

Implement Subregion-bounded Wavefront Partition (SWP) in the current HPHS project as a visualization-only unknown-space partition module.

This change must not alter the original HPHS exploration decision pipeline. When SWP is disabled or unavailable, HPHS should still run with its existing flow:

```text
frontier sampling -> subregion selection -> frontier selection -> local goal
```

The wave overlap boundary is named Wavefront Partition Boundary (WPB).

## Scope

This implementation only covers unknown-region partitioning and RViz visualization.

It does not change or implement:

- subregion generation logic
- subregion ordering
- frontier selection
- waypoint/local goal selection
- exploration decision policies based on SWP regions

## Module Structure

Add a new standalone module:

```text
scripts/swp_manager.py
```

Class name:

```python
class SWPManager:
```

`Explorer` only passes the selected HPHS context to `SWPManager`. The manager owns SWP computation, result caching, and RViz publishing.

## Runtime Model

SWP should mimic the STG-style manager pattern:

- `SWPManager` subscribes to `/map` and keeps the latest `nav_msgs/OccupancyGrid` snapshot.
- `SWPManager` owns a `rospy.Timer(...)`.
- The timer calls `updateSWP()` at a configurable rate.
- The normal planning loop does not compute SWP directly.
- The normal planning loop only updates the latest selected-subregion context.

For the first version, the default update rate is 10 Hz, even if map/frontiers have not changed.

## Explorer Integration

In `scripts/explorer.py`, after the original HPHS selected subregion has been determined:

```python
self.setSubregion()
self.classflyFrontiers()
self.arrangeSubregion()
```

call a manager context update such as:

```python
self.swp_manager.set_context(
    selected_subregion=self.selected_subregion,
    subregion_center=self.subregion_center[self.selected_subregion],
    map_origin_resized=[self.map_origin_x_resized, self.map_origin_y_resized],
    map_size_resized=[self.map_width_resized, self.map_height_resized],
    n_w=self.n_w,
    n_h=self.n_h,
    frontiers=self.classflied_frontiers[self.selected_subregion],
    frontier_cluster_dist=self.total_frontier_vicinity,
)
```

`Explorer` should not call `updateSWP()` directly.

## Enable Flag

SWP is enabled by default.

Add ROS parameters:

```text
~swp/enabled = true
~swp/update_rate = 10.0
~swp/region_z = 0.15
~swp/wpb_z = 0.25
~swp/region_alpha = 0.35
~swp/wpb_alpha = 0.9
```

When `~swp/enabled` is false:

- `SWPManager` should not compute SWP.
- SWP markers should be cleared or not published.
- Original HPHS behavior must be unchanged.

## Selected Subregion Boundary

SWP only runs inside the current HPHS selected subregion.

The selected subregion uses the existing HPHS rectangular definition:

```text
subregion_width  = map_width_resized / n_w
subregion_height = map_height_resized / n_h
center           = subregion_center[selected_subregion]
```

The selected subregion rectangle is a hard boundary:

- wave propagation cannot leave the selected subregion
- unknown cells outside the selected subregion receive no SWP label
- adjacent HPHS subregions are not considered in the current SWP pass
- when wave reaches the selected subregion boundary, the last ring of unknown cells inside the boundary is marked as WPB

## Wave Sources

Use the existing HPHS frontiers assigned to the selected subregion:

```python
self.classflied_frontiers[self.selected_subregion]
```

These frontiers are sparse map-frame points, so they must be clustered before wave propagation.

Cluster rule:

- Use the existing HPHS distance threshold:

```python
self.total_frontier_vicinity
```

- Each cluster is called a frontier cluster.
- Each frontier cluster receives a unique `label_id`.
- Each frontier cluster is one wave source.

## Source Initialization

Frontier points are not treated as unknown-space interior cells.

For each frontier cluster:

1. Convert each frontier point from map coordinates to grid index.
2. Treat the frontier grid as a source anchor.
3. Start propagation from unknown 4-neighbor cells adjacent to the anchor.
4. If no adjacent unknown cell exists for the cluster, skip that cluster.

This preserves the semantic that frontiers lie on the known-free / unknown boundary, while waves propagate inside unknown space.

## Grid Semantics

SWP runs only in 2D projected `OccupancyGrid` space.

Use map values as:

```text
-1        unknown, propagable
0..10     free, blocked
>=20      occupied/obstacle, blocked
outside selected subregion, blocked
```

## Propagation Rule

SWP uses multi-source BFS wave propagation.

This is intentionally different from STG-planner's free-space wave propagation. STG's implementation is closer to iterative thinning / skeletonization for extracting an accurate medial topological graph in free space. SWP runs in unknown space and only needs a stable partition result, so multi-source BFS is sufficient and simpler to run at the manager update rate.

For each update:

1. Initialize active wavefront cells from each frontier cluster's adjacent unknown cells.
2. Each active cell carries its frontier-cluster `label_id`.
3. An unknown cell first reached by one cluster is assigned that cluster label.
4. If multiple clusters try to occupy the same unknown cell, that overlap cell is marked as WPB.
5. If a wavefront reaches a cell already owned by a different label, the contact is marked as WPB and propagation stops along that contact.
6. If a wavefront reaches the selected subregion boundary, the last ring of unknown cells inside the boundary is marked as WPB.
7. WPB cells do not enter the next active wavefront.
8. Wave propagation stops at free cells, occupied cells, WPB cells, and selected-subregion boundaries.

WPB cells may be adjacent to, or semantically shared by, multiple region labels. Internally, each WPB cell should keep the set of adjacent labels when this information is available. For visualization, WPB should be deduplicated and displayed as connected boundary segments instead of being duplicated per adjacent region.

Expected internal outputs:

```python
region_labels  # int grid, unknown cell -> frontier cluster label_id, else -1
wpb_mask       # bool grid, True for WPB cells
wpb_adjacent_labels  # optional dict/cell map: WPB cell -> set(label_id)
```

For this first implementation, these outputs are used only for visualization.

## RViz Visualization

`SWPManager` publishes all SWP/WPB visualization results.

Topics:

```text
/swp_wpb
/swp_regions
```

Recommended message types:

```text
/swp_regions  visualization_msgs/MarkerArray
/swp_wpb      visualization_msgs/MarkerArray
```

Visualization requirements:

- `/swp_regions` publishes one marker per region label.
- `/swp_wpb` publishes one marker per WPB connected component.
- WPB is red or magenta.
- Regions are shown as full cell coverage within the selected subregion.
- Cells belonging to different frontier clusters use different colors.
- WPB should be drawn above region cells using a slightly higher z value.
- Region markers should use transparent colors.
- If selected subregion changes, publish marker cleanup first so stale `/swp_regions` and `/swp_wpb` markers do not remain in RViz.

Suggested defaults:

```text
region z      = 0.15
WPB z         = 0.25
region alpha  = 0.35
WPB alpha     = 0.9
```

## Files To Modify

Expected changes:

```text
scripts/swp_manager.py      new SWPManager module
scripts/explorer.py         instantiate manager and update selected context
launch/run.launch           add ~swp/enabled, ~swp/update_rate, and visualization parameters
rviz/visualization.rviz     add /swp_wpb and /swp_regions displays
```

Possible change:

```text
CMakeLists.txt
```

Only needed if the new Python script must be added to `catkin_install_python`.

## Validation Plan

Do not compile ROS in this step.

Run Python syntax checks:

```bash
python3 -m py_compile scripts/explorer.py scripts/swp_manager.py
```

Run a small offline synthetic grid test for `SWPManager` internals:

- frontier clusters are created
- unknown cells receive labels
- WPB is generated
- WPB does not continue propagating
- cells outside the selected subregion remain unlabeled

## Future Work

Future iterations may use SWP outputs for:

- replacing or refining HPHS subregion ranking
- selecting frontier clusters instead of individual frontier points
- planning within a WPB-bounded unknown region
- reducing SWP update cost by skipping unchanged map/frontier states
- limiting visualization marker density if large maps become expensive
