import threading
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np


@dataclass
class CarlaLiveFrame:
    carla_frame: int
    timestamp: float
    image: np.ndarray
    lidar: np.ndarray
    imu: Dict
    timestamp_skew_s: float
    world_snapshot: Optional[object] = None
    world_frame_at_return: Optional[int] = None
    simulation_lag_frames: int = 0


class _SensorFrameQueue:
    """Bounded frame-indexed buffer for one CARLA sensor.

    CARLA GPU sensors, especially RGB cameras, may deliver data a small number
    of simulation frames later than CPU sensors even when the world is running
    synchronously.  The buffer therefore retains several frame-numbered
    packets so the adapter can form an exact RGB/LiDAR/IMU intersection after
    advancing the simulator enough for the delayed camera to catch up.
    """

    def __init__(self, sensor, transform_fn, maxsize: int = 32):
        self.sensor = sensor
        self.transform_fn = transform_fn
        self.maxsize = max(4, int(maxsize))
        self.dropped = 0
        self.last_frame = None
        self.closed = False
        self._items = OrderedDict()
        self._condition = threading.Condition()
        sensor.listen(self._callback)

    def _callback(self, data):
        if self.closed:
            return

        frame = int(data.frame)
        timestamp = float(data.timestamp)
        transformed = self.transform_fn(data)

        with self._condition:
            if self.closed:
                return

            self.last_frame = frame
            self._items[frame] = (timestamp, transformed)
            self._items.move_to_end(frame)

            while len(self._items) > self.maxsize:
                self._items.popitem(last=False)
                self.dropped += 1

            self._condition.notify_all()

    def available_frames(self):
        with self._condition:
            return tuple(self._items.keys())

    def has_frame(self, frame: int) -> bool:
        frame = int(frame)
        with self._condition:
            return frame in self._items

    def pop_exact(self, frame: int):
        frame = int(frame)
        with self._condition:
            item = self._items.pop(frame, None)
        if item is None:
            raise KeyError(frame)
        return item

    def discard_before(self, frame: int):
        frame = int(frame)
        with self._condition:
            stale = [f for f in self._items.keys() if f < frame]
            for f in stale:
                self._items.pop(f, None)
                self.dropped += 1

    def wait_for_update(self, previous_last_frame, timeout: float) -> bool:
        deadline = time.monotonic() + max(0.0, float(timeout))
        with self._condition:
            while not self.closed and self.last_frame == previous_last_frame:
                remaining = deadline - time.monotonic()
                if remaining <= 0.0:
                    return False
                self._condition.wait(timeout=remaining)
            return self.last_frame != previous_last_frame

    def get_exact(self, frame: int, timeout: float):
        """Compatibility helper used by tests and diagnostics.

        This method still enforces the requested exact frame.  The live
        adapter itself no longer calls it immediately after every world tick;
        instead it buffers all three sensors and selects their oldest common
        frame, which is required for delayed GPU cameras.
        """

        deadline = time.monotonic() + float(timeout)
        frame = int(frame)

        with self._condition:
            while True:
                if frame in self._items:
                    return self._items.pop(frame)

                if self.last_frame is not None and int(self.last_frame) > frame:
                    raise RuntimeError(
                        f"Sensor advanced to frame {self.last_frame} while waiting "
                        f"for frame {frame}. This indicates live sensor/world "
                        "synchronization was lost."
                    )

                remaining = deadline - time.monotonic()
                if remaining <= 0.0:
                    raise TimeoutError(
                        f"Timed out waiting for sensor frame {frame}; "
                        f"last callback frame={self.last_frame}, dropped={self.dropped}"
                    )

                self._condition.wait(timeout=remaining)

    def close(self):
        # Pure-Python shutdown only.  Do not call CARLA sensor.stop() here.
        # CARLA 0.9.13 on Windows can fast-fail during repeated/unordered
        # native unsubscribe calls.  The owning runner destroys all sensor
        # actors in one CARLA batch after callbacks have been disabled.
        with self._condition:
            self.closed = True
            self._items.clear()
            self._condition.notify_all()


def camera_bgr(data):
    arr = np.frombuffer(data.raw_data, dtype=np.uint8).reshape(data.height, data.width, 4)
    return arr[:, :, :3].copy()


def lidar_xyzi(data):
    return np.frombuffer(data.raw_data, dtype=np.float32).reshape(-1, 4).copy()


def imu_dict(data):
    return {
        "accelerometer": {
            "x": float(data.accelerometer.x),
            "y": float(data.accelerometer.y),
            "z": float(data.accelerometer.z),
        },
        "gyroscope": {
            "x": float(data.gyroscope.x),
            "y": float(data.gyroscope.y),
            "z": float(data.gyroscope.z),
        },
        "compass": float(data.compass),
        "timestamp": float(data.timestamp),
        "carla_frame": int(data.frame),
        "dataset_frame": int(data.frame),
    }


class CarlaLiveSensorAdapter:
    """Live RGB + LiDAR + IMU adapter feeding the unchanged LVIMOT core.

    The world remains synchronous and fixed-step.  Each public ``tick`` pumps
    the simulator by at least one step, then waits for the oldest CARLA frame
    that exists in all three bounded sensor buffers.  If the GPU camera is one
    or two frames late, the adapter advances additional synchronous ticks so
    that camera output can catch up while LiDAR/IMU packets remain buffered.

    Every frame returned to LVIMOT is therefore still exact-frame aligned:
    RGB, LiDAR and IMU all carry the same CARLA frame ID.  No ego pose or actor
    label is used by this class.
    """

    def __init__(
        self,
        world,
        camera,
        lidar,
        imu,
        timeout_s: float = 10.0,
        max_timestamp_skew_s: float = 1e-3,
        queue_size: int = 32,
        callback_grace_s: float = 0.15,
        max_pump_ticks: int = 8,
        snapshot_buffer_size: int = 64,
    ):
        self.world = world
        self.timeout_s = max(1.0, float(timeout_s))
        self.max_timestamp_skew_s = float(max_timestamp_skew_s)
        self.callback_grace_s = max(0.01, float(callback_grace_s))
        self.max_pump_ticks = max(2, int(max_pump_ticks))
        self.snapshot_buffer_size = max(8, int(snapshot_buffer_size))

        self.camera = _SensorFrameQueue(camera, camera_bgr, queue_size)
        self.lidar = _SensorFrameQueue(lidar, lidar_xyzi, queue_size)
        self.imu = _SensorFrameQueue(imu, imu_dict, queue_size)

        self._last_returned_frame = None
        self._latest_world_frame = None
        self._world_ticks_total = 0
        self._world_ticks_last_call = 0
        self._skipped_processed_frames = 0
        self._max_simulation_lag_frames = 0
        self._snapshots = OrderedDict()

    def _record_world_snapshot(self, tick_frame: int):
        snapshot = None
        try:
            snapshot = self.world.get_snapshot()
        except Exception:
            snapshot = None

        if snapshot is not None:
            try:
                snapshot_frame = int(snapshot.frame)
            except Exception:
                snapshot_frame = int(tick_frame)
            self._snapshots[snapshot_frame] = snapshot
            self._snapshots.move_to_end(snapshot_frame)

            while len(self._snapshots) > self.snapshot_buffer_size:
                self._snapshots.popitem(last=False)

        return snapshot

    def _advance_world_once(self) -> int:
        frame = int(self.world.tick())
        self._latest_world_frame = frame
        self._world_ticks_total += 1
        self._world_ticks_last_call += 1
        self._record_world_snapshot(frame)
        return frame

    def _common_frames(self):
        camera_frames = set(self.camera.available_frames())
        lidar_frames = set(self.lidar.available_frames())
        imu_frames = set(self.imu.available_frames())
        common = camera_frames.intersection(lidar_frames, imu_frames)

        if self._last_returned_frame is not None:
            common = {f for f in common if f > self._last_returned_frame}

        return sorted(common)

    def _wait_for_common_frame(self, timeout: float):
        deadline = time.monotonic() + max(0.0, float(timeout))
        while True:
            common = self._common_frames()
            if common:
                return common[0]

            remaining = deadline - time.monotonic()
            if remaining <= 0.0:
                return None

            # The camera is normally the delayed GPU sensor, but sleeping here
            # also allows LiDAR/IMU callback threads to finish without forcing
            # a busy spin or prematurely advancing the simulation.
            time.sleep(min(0.002, remaining))

    def _format_timeout(self, pumped_ticks: int) -> str:
        return (
            "Timed out waiting for an exact common RGB/LiDAR/IMU CARLA frame; "
            f"latest_world_frame={self._latest_world_frame}, "
            f"last_returned_frame={self._last_returned_frame}, "
            f"camera_last={self.camera.last_frame}, "
            f"lidar_last={self.lidar.last_frame}, "
            f"imu_last={self.imu.last_frame}, "
            f"camera_buffer={list(self.camera.available_frames())}, "
            f"lidar_buffer={list(self.lidar.available_frames())}, "
            f"imu_buffer={list(self.imu.available_frames())}, "
            f"pumped_ticks={pumped_ticks}, "
            f"dropped=(cam:{self.camera.dropped}, "
            f"lidar:{self.lidar.dropped}, imu:{self.imu.dropped})"
        )

    def tick(self) -> CarlaLiveFrame:
        deadline = time.monotonic() + self.timeout_s
        self._world_ticks_last_call = 0
        pumped_ticks = 0

        while True:
            if pumped_ticks >= self.max_pump_ticks:
                raise TimeoutError(self._format_timeout(pumped_ticks))

            if time.monotonic() >= deadline:
                raise TimeoutError(self._format_timeout(pumped_ticks))

            self._advance_world_once()
            pumped_ticks += 1

            remaining = deadline - time.monotonic()
            if remaining <= 0.0:
                raise TimeoutError(self._format_timeout(pumped_ticks))

            common_frame = self._wait_for_common_frame(
                min(self.callback_grace_s, remaining)
            )
            if common_frame is None:
                # No exact common frame yet.  This is expected for a delayed
                # GPU camera: advance another deterministic synchronous tick.
                continue

            ts_cam, image = self.camera.pop_exact(common_frame)
            ts_lidar, lidar = self.lidar.pop_exact(common_frame)
            ts_imu, imu = self.imu.pop_exact(common_frame)

            timestamps = np.array([ts_cam, ts_lidar, ts_imu], dtype=np.float64)
            skew = float(np.max(timestamps) - np.min(timestamps))
            if not np.isfinite(skew) or skew > self.max_timestamp_skew_s:
                raise RuntimeError(
                    "CARLA live sensor timestamp mismatch for frame "
                    f"{common_frame}: camera={ts_cam:.6f}, "
                    f"lidar={ts_lidar:.6f}, imu={ts_imu:.6f}, "
                    f"skew={skew:.6f}s"
                )

            if self._last_returned_frame is not None:
                gap = int(common_frame) - int(self._last_returned_frame) - 1
                if gap > 0:
                    self._skipped_processed_frames += gap

            self._last_returned_frame = int(common_frame)

            # Packets older than the emitted frame can never be used again.
            # Discard them only after the exact synchronized packet has been
            # formed, so delayed cameras never cause premature data loss.
            self.camera.discard_before(common_frame)
            self.lidar.discard_before(common_frame)
            self.imu.discard_before(common_frame)

            snapshot = self._snapshots.pop(int(common_frame), None)
            stale_snapshot_frames = [
                f for f in self._snapshots.keys() if f < int(common_frame)
            ]
            for f in stale_snapshot_frames:
                self._snapshots.pop(f, None)

            if self._latest_world_frame is None:
                simulation_lag = 0
            else:
                simulation_lag = max(
                    0, int(self._latest_world_frame) - int(common_frame)
                )
            self._max_simulation_lag_frames = max(
                self._max_simulation_lag_frames, simulation_lag
            )

            return CarlaLiveFrame(
                carla_frame=int(common_frame),
                timestamp=float(ts_cam),
                image=image,
                lidar=lidar,
                imu=imu,
                timestamp_skew_s=skew,
                world_snapshot=snapshot,
                world_frame_at_return=(
                    None
                    if self._latest_world_frame is None
                    else int(self._latest_world_frame)
                ),
                simulation_lag_frames=int(simulation_lag),
            )

    def diagnostics(self) -> Dict:
        current_world = self._latest_world_frame
        camera_last = self.camera.last_frame
        lidar_last = self.lidar.last_frame
        imu_last = self.imu.last_frame

        def _lag(last_frame):
            if current_world is None or last_frame is None:
                return None
            return max(0, int(current_world) - int(last_frame))

        return {
            "camera_dropped": int(self.camera.dropped),
            "lidar_dropped": int(self.lidar.dropped),
            "imu_dropped": int(self.imu.dropped),
            "camera_last_frame": camera_last,
            "lidar_last_frame": lidar_last,
            "imu_last_frame": imu_last,
            "camera_lag_frames": _lag(camera_last),
            "lidar_lag_frames": _lag(lidar_last),
            "imu_lag_frames": _lag(imu_last),
            "latest_world_frame": current_world,
            "last_processed_carla_frame": self._last_returned_frame,
            "world_ticks_total": int(self._world_ticks_total),
            "world_ticks_last_call": int(self._world_ticks_last_call),
            "skipped_processed_frames": int(self._skipped_processed_frames),
            "max_simulation_lag_frames": int(self._max_simulation_lag_frames),
        }

    def close(self):
        self.camera.close()
        self.lidar.close()
        self.imu.close()
