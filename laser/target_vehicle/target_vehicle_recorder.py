import os
import cv2
import time
import threading
import numpy as np
import carla
from laser.sensor import CameraSensor


def _safe_stop_destroy(actor):
    if actor is None:
        return
    try:
        if hasattr(actor, "is_listening") and actor.is_listening:
            actor.stop()
    except Exception:
        pass
    try:
        if hasattr(actor, "is_alive") and not actor.is_alive:
            return
        actor.destroy()
    except Exception:
        pass


def _env_flag(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() not in ("0", "false", "no", "off", "")


class TargetVehicleRecorder:
    """Episode artifacts under se_records/.

    Performance (sync-mode ticks were ~10s with 1080p + binary recorder):
      LASER_VIDEO=1           enable bird's-eye mp4 (default OFF)
      LASER_VIDEO_WIDTH/HEIGHT/FPS  (defaults 1920/1080/20 — stock LASER quality)
      LASER_CLIENT_RECORDER=1 enable client.start_recorder (default OFF)
    """

    def __init__(self, client) -> None:
        self.client = client
        self.current_time = time.localtime()
        self.directory = os.path.join('se_records', time.strftime('%m%d-%H-%M-%S', self.current_time))
        os.makedirs(self.directory, exist_ok=True)
        os.environ['SAVE_PATH'] = self.directory

        self._video_lock = threading.Lock()
        self._video_closed = False
        self.video_recorder = None
        self.camera = None
        self._collision_sensor = None
        self._video_size = (1920, 1080)
        self._recorder_started = False

        # Binary recorder is optional: additional_data=True was a major sync stall.
        if _env_flag("LASER_CLIENT_RECORDER", default=False):
            carla_recording_path_on_server = os.path.join(
                self.directory, f"{time.strftime('%m%d-%H-%M-%S', self.current_time)}recording.log"
            )
            print(f"target_vehicle_recorder: \nrecording_path: {carla_recording_path_on_server}")
            # additional_data=False — full dumps make world.tick multi-second.
            self.client.start_recorder(carla_recording_path_on_server, False)
            self._recorder_started = True
        else:
            print("target_vehicle_recorder: client recorder OFF (LASER_CLIENT_RECORDER=1 to enable)")

    def register_actor(self, carla_actor):
        self._carla_actor = carla_actor
        self._world = self._carla_actor.get_world()

        # Collision sensor only (lightweight).
        self.collision_num = 0
        blueprint = self._world.get_blueprint_library().find('sensor.other.collision')
        self._collision_sensor = self._world.spawn_actor(blueprint, carla.Transform(), attach_to=self._carla_actor)
        self._collision_sensor.listen(lambda event: self.collision_sensor_callback())

        # Bird's-eye video is opt-in. A 1080p RGB camera every sync tick was the
        # dominant cause of ~10s/tick stalls on RenderOffScreen nodes.
        if not _env_flag("LASER_VIDEO", default=False):
            print("target_vehicle_recorder: video OFF (LASER_VIDEO=1 to enable)")
            return

        video_fps = int(os.environ.get("LASER_VIDEO_FPS", "10"))
        w = int(os.environ.get("LASER_VIDEO_WIDTH", "640"))
        h = int(os.environ.get("LASER_VIDEO_HEIGHT", "360"))
        self._video_size = (w, h)

        fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
        self.video_recorder = cv2.VideoWriter()
        video_path = os.path.join(self.directory, time.strftime('%m%d-%H-%M-%S', self.current_time) + '.mp4')
        print(f"target_vehicle_recorder: \nvideo_path: {video_path} size={w}x{h} fps={video_fps}")
        self.video_recorder.open(video_path, fourcc, video_fps, (w, h), True)
        sensor_args = {
            'sensor_bp_name': 'sensor.camera.rgb',
            'sensor_type': 'Front Camera RGB',
            'sensor_bp_args': {
                'image_size_x': str(w),
                'image_size_y': str(h),
                'fov': '90',
                'sensor_tick': str(1.0 / max(video_fps, 1)),
            }
        }

        location = carla.Location(0, 0, 30.0)
        transform = carla.Transform(location, carla.Rotation(roll=90, yaw=180, pitch=-90))
        self.camera = CameraSensor(self, sensor_args, self._carla_actor, transform)
        self.camera.sensor.listen(self._on_image)

    def _on_image(self, sensor_data):
        with self._video_lock:
            if self._video_closed or self.video_recorder is None:
                return
            image = np.array(sensor_data.raw_data)
            h, w = sensor_data.height, sensor_data.width
            image = image.reshape((h, w, 4))
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
            if (w, h) != self._video_size:
                image = cv2.resize(image, self._video_size)
            self.video_recorder.write(image)

    def collision_sensor_callback(self):
        self.collision_num += 1

    def destroy(self):
        try:
            if self.camera is not None:
                _safe_stop_destroy(self.camera.sensor)
        except Exception as e:
            print(f"camera destroy failed: {e}")
        try:
            _safe_stop_destroy(self._collision_sensor)
        except Exception as e:
            print(f"collision sensor destroy failed: {e}")

        with self._video_lock:
            self._video_closed = True
            if self.video_recorder is not None:
                try:
                    self.video_recorder.release()
                except Exception as e:
                    print(f"VideoWriter.release failed: {e}")
                self.video_recorder = None

        if self._recorder_started:
            try:
                self.client.stop_recorder()
            except Exception as e:
                print(f"stop_recorder failed: {e}")
        print('released')

    def parse_CollisionEvent(self, sensor_data, agent):
        set_agent = {12, 13, 14, 15, 16, 17, 18, 19}
        if set_agent & set(sensor_data.other_actor.semantic_tags):
            with open(os.path.join(self.directory, 'collisions.txt'), 'w') as file:
                file.write(f"{sensor_data.timestamp}, {sensor_data.actor}, {sensor_data.other_actor}")
