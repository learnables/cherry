import gym
from gym import error, version
from gym.wrappers.monitoring import video_recorder as GymVideoRecorder
import os, json, numpy as np, six
from gym.utils import atomic_write, closer
from gym.utils.json_utils import json_encode_np
from cherry.envs.base import Wrapper
import time
from datetime import datetime
from six import StringIO
import subprocess
import tempfile
import os.path
import distutils.spawn, distutils.version

FILE_PREFIX = 'openaigym'
def touch(path):
    open(path, 'a').close()

class VideoRecorder(GymVideoRecorder.VideoRecorder):
    """VideoRecorder renders a nice movie of a rollout, frame by frame. It
    comes with an `enabled` option so you can still use the same code
    on episodes where you don't want to record video.

    Note:
        You are responsible for calling `close` on a created
        VideoRecorder, or else you may leak an encoder process.

    Args:
        env (Env): Environment to take video of.
        path (Optional[str]): Path to the video file; will be randomly chosen if omitted.
        base_path (Optional[str]): Alternatively, path to the video file without extension, which will be added.
        metadata (Optional[dict]): Contents to save to the metadata file.
        enabled (bool): Whether to actually record video, or just no-op (for convenience)
        format: Format of the output video, choose between 'gif' and 'mp4'
    """

    def __init__(self, env, format='gif', path=None, metadata=None, enabled=True, base_path=None):
        """Overrides original constructor to add support for generating gifs."""
        self.format = format

        modes = env.metadata.get('render.modes', [])
        self._async = env.metadata.get('semantics.async')
        self.enabled = enabled

        # Don't bother setting anything else if not enabled
        if not self.enabled:
            return

        self.ansi_mode = False
        if 'rgb_array' not in modes:
            if 'ansi' in modes:
                self.ansi_mode = True
            else:
                # Whoops, turns out we shouldn't be enabled after all
                self.enabled = False
                return

        if path is not None and base_path is not None:
            raise error.Error("You can pass at most one of `path` or `base_path`.")

        self.last_frame = None
        self.env = env

        required_ext = '.json' if self.ansi_mode else '.'+format
        if path is None:
            if base_path is not None:
                # Base path given, append ext
                path = base_path + required_ext
            else:
                # Otherwise, just generate a unique filename
                with tempfile.NamedTemporaryFile(suffix=required_ext, delete=False) as f:
                    path = f.name
        self.path = path

        path_base, actual_ext = os.path.splitext(self.path)

        if actual_ext != required_ext:
            hint = " HINT: The environment is text-only, therefore we're recording its text output in a structured JSON format." if self.ansi_mode else ''
            raise error.Error("Invalid path given: {} -- must have file extension {}.{}".format(self.path, required_ext, hint))
        # Touch the file in any case, so we know it's present. (This
        # corrects for platform platform differences. Using ffmpeg on
        # OS X, the file is precreated, but not on Linux.
        touch(path)

        self.frames_per_sec = env.metadata.get('video.frames_per_second', 30)
        self.encoder = None # lazily start the process
        self.broken = False

        # Dump metadata
        self.metadata = metadata or {}
        self.metadata['content_type'] = 'video/vnd.openai.ansivid' if self.ansi_mode else 'video/mp4'
        self.metadata_path = '{}.meta.json'.format(path_base)
        self.write_metadata()

        self.empty = True

    def write_metadata(self):
        """Override original method to disable it."""
        pass

    def _encode_image_frame(self, frame):
        if not self.encoder:
            self.encoder = ImageEncoderWithGif(self.path, frame.shape, self.frames_per_sec,self.format)
            self.metadata['encoder_version'] = self.encoder.version_info

        try:
            self.encoder.capture_frame(frame)
        except error.InvalidFrame as e:
            self.broken = True
        else:
            self.empty = False

class ImageEncoderWithGif(GymVideoRecorder.ImageEncoder):
    def __init__(self, output_path, frame_shape, frames_per_sec, format):
        self.format = format
        super(ImageEncoderWithGif, self).__init__(output_path, frame_shape, frames_per_sec)

    def start(self):
        self.cmdline = (self.backend,
                     '-nostats',
                     '-loglevel', 'error', # suppress warnings
                     '-y',
                     '-r', '%d' % self.frames_per_sec,

                     # input
                     '-f', 'rawvideo',
                     '-s:v', '{}x{}'.format(*self.wh),
                     '-pix_fmt',('rgb32' if self.includes_alpha else 'rgb24'),
                     '-i', '-', # this used to be /dev/stdin, which is not Windows-friendly

                     # output
                     '-vf', 'scale=trunc(iw/2)*2:trunc(ih/2)*2')
        if self.format == "mp4":
            self.cmdline += ('-vcodec', 'libx264')
        self.cmdline += ('-pix_fmt', 'yuv420p', self.output_path)

        if hasattr(os,'setsid'): #setsid not present on Windows
            self.proc = subprocess.Popen(self.cmdline, stdin=subprocess.PIPE, preexec_fn=os.setsid)
        else:
            self.proc = subprocess.Popen(self.cmdline, stdin=subprocess.PIPE)

class Recorder(Wrapper):
    '''

    **Description**
    
    Create training videos for arbitrary environment.

    **Arguments**
    
    * **env** (gym Environment or gym Environment wrapped in any Cherry wrappers, *required*) - Training environment.
    * **directory** (string, *required*) - Relative path to where videos will be saved.
    * **format** (choose in ['gif', 'mp4'], *optional*, default=None) - Format of the output videos. If it's text environment, the format will be json.
    * **video_callable** (Method, *optional*, default=None) - A method that decides whether to generate a video given an episode id. If set to None, it generates a video for every episode.
    * **force** (bool, *optional*, default=False): Clear out existing training data from this directory (by deleting every file prefixed with "openaigym.").
    * **uid** (string, *optional*, defulat=None): A unique id used as part of the suffix for the file. By default, uses os.getpid().
    
    **References**

    1. openai/gym
    
    **Example**

    ~~~python
    import gym
    import cherry.envs as envs

    env = gym.make("[gym_environment_name]")
    env = envs.Recorder(record_env, './videos/', format='gif') # format will be gif if it's not set. Choose between gif and mp4
    env = envs.Logger(env, interval)
    env = envs.Torch(env)
    env = envs.Runner(env)
    
    # During training
    env.run(get_action, episodes=3, render=True) # get_action is a function that generates an action from the policy when given a state.
    

    '''
    def __init__(self, env, directory, format="gif", video_callable=None, force=False, uid=None):
        super(Recorder, self).__init__(env)

        env_name = env.spec.id
        ts = time.time()
        date = datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S.%f')
        directory = os.path.join(directory, env_name)
        directory = os.path.join(directory, date)

        self.format = format
        self.video_recorder = None
        self.enabled = False
        self.episode_id = 0
        self._monitor_id = None
        self.env_semantics_autoreset = env.metadata.get('semantics.autoreset')
        self.output_files = []
        self._start(directory, video_callable, force, uid)

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        done = self._after_step(observation, reward, done, info)

        return observation, reward, done, info

    def reset(self, **kwargs):
        observation = self.env.reset(**kwargs)
        self._after_reset(observation)

        return observation

    def _start(self, directory, video_callable=None, force=False,
               uid=None, mode=None):
        """Start recording.
        Args:
            directory (str): A per-training run directory where to record stats.
            video_callable (Optional[function, False]): function that takes in the index of the episode and outputs a boolean, indicating whether we should record a video on this episode. The default (for video_callable is None) is to take perfect cubes, capped at 1000. False disables video recording.
            force (bool): Clear out existing training data from this directory (by deleting every file prefixed with "openaigym.").
            uid (Optional[str]): A unique id used as part of the suffix for the file. By default, uses os.getpid().
        """
        if self.env.spec is None:
            env_id = '(unknown)'
        else:
            env_id = self.env.spec.id

        if not os.path.exists(directory):
            if six.PY3:
                os.makedirs(directory, exist_ok=True)
            else:
                os.makedirs(directory)

        if video_callable is None:
            video_callable = gen_video_every_episode;
        elif video_callable == False:
            video_callable = disable_videos
        elif not callable(video_callable):
            raise error.Error('You must provide a function, None, or False for video_callable, not {}: {}'.format(type(video_callable), video_callable))
        self.video_callable = video_callable

        # Check on whether we need to clear anything
        if force:
            clear_recorder_files(directory)

        self._monitor_id = recorder_closer.register(self)

        self.enabled = True
        self.directory = os.path.abspath(directory)
        # We use the 'openai-gym' prefix to determine if a file is
        # ours
        self.file_prefix = FILE_PREFIX
        self.file_infix = '{}.{}'.format(self._monitor_id, uid if uid else os.getpid())

        if not os.path.exists(directory): os.mkdir(directory)

    def close(self):
        """Flush all monitor data to disk and close any open rending windows."""
        super(Recorder, self).close()

        if not self.enabled:
            return
        if self.video_recorder is not None:
            self._close_video_recorder()

        # Stop tracking this for autoclose
        recorder_closer.unregister(self._monitor_id)
        self.enabled = False

    def _after_step(self, observation, reward, done, info):
        if not self.enabled: return done

        if done and self.env_semantics_autoreset:
            # For envs with BlockingReset wrapping VNCEnv, this observation will be the first one of the new episode
            self.reset_video_recorder()
            self.episode_id += 1

        # Record video
        self.video_recorder.capture_frame()
        return done

    def _after_reset(self, observation):
        if not self.enabled: return

        self.reset_video_recorder()

        # Bump *after* all reset activity has finished
        self.episode_id += 1

    def reset_video_recorder(self):
        # Close any existing video recorder
        if self.video_recorder:
            self._close_video_recorder()

        # Start recording the next video.
        #
        # TODO: calculate a more correct 'episode_id' upon merge

        self.video_recorder = VideoRecorder(
            env=self.env,
            base_path=os.path.join(self.directory, '{}.video.{}.video{:06}'.format(self.file_prefix, self.file_infix, self.episode_id)),
            metadata={'episode_id': self.episode_id},
            enabled=self._video_enabled(),
            format = self.format
        )
        self.output_files.append(os.path.relpath(self.video_recorder.path))
        self.video_recorder.capture_frame()

    def _close_video_recorder(self):
        self.video_recorder.close()

    def _video_enabled(self):
        return self.video_callable(self.episode_id)

    def __del__(self):
        # Make sure we've closed up shop when garbage collecting
        self.close()

    def get_video_paths(self):
        return self.output_files

def gen_video_every_episode(episode_id):
    return True

recorder_closer = closer.Closer()

def clear_recorder_files(training_dir):
    files = detect_recorder_files(training_dir)
    if len(files) == 0:
        return

    for file in files:
        os.unlink(file)

def detect_recorder_files(training_dir):
    return [os.path.join(training_dir, f) for f in os.listdir(training_dir) if f.startswith(FILE_PREFIX + '.')]
