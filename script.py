from magnum import Vector3
import os
import quaternion
import habitat_sim.sim
import numpy as np
from scipy.io import wavfile

class Simulation():
  def __init__(self, landscape_path): 
    self.backend_cfg = habitat_sim.SimulatorConfiguration()
    self.backend_cfg.scene_id = landscape_path
    self.backend_cfg.scene_dataset_config_file = "/workspace/data/custom_data/hm3d_annotated_basis.scene_dataset_config.json"
    self.backend_cfg.load_semantic_mesh = True
    self.backend_cfg.enable_physics = False

    agent_config = habitat_sim.AgentConfiguration()
    cfg = habitat_sim.Configuration(self.backend_cfg, [agent_config])
    self.sim = habitat_sim.Simulator(cfg)
    # create the acoustic configs
    acoustics_config = habitat_sim.sensor.RLRAudioPropagationConfiguration()
    acoustics_config.enableMaterials = True
    # create channel layout
    channel_layout = habitat_sim.sensor.RLRAudioPropagationChannelLayout()
    channel_layout.channelType = (
        habitat_sim.sensor.RLRAudioPropagationChannelLayoutType.Mono
    )
    channel_layout.channelCount = 2
    # create the Audio sensor specs, assign the acoustics_config and the channel_layout.
    # note that the outputDirectory should already exist for each iteration step.
    # for the example below, folders /home/AudioSimulation0, /home/AudioSimulation1 ... should
    # exist based on the number of iterations
    audio_sensor_spec = habitat_sim.AudioSensorSpec()
    audio_sensor_spec.uuid = "audio_sensor"
    audio_sensor_spec.outputDirectory = "/tmp/AudioSimulation"
    audio_sensor_spec.acousticsConfig = acoustics_config
    audio_sensor_spec.channelLayout = channel_layout
    audio_sensor_spec.position = [0.0, 1.5, 0.0]  # audio sensor has a height of 1.5m
    audio_sensor_spec.acousticsConfig.sampleRate = 48000
    # whether indrect (reverberation) is present in the rendered IR
    audio_sensor_spec.acousticsConfig.indirect = True
    # add the audio sensor
    self.sim.add_sensor(audio_sensor_spec)
    # Get the audio sensor object
    audio_sensor = self.sim.get_agent(0)._sensors["audio_sensor"]
    # set audio source location, no need to set the agent location, will be set implicitly
    audio_sensor.setAudioSourceTransform(np.array([3.1035, 1.57245, -4.15972]))
    # optionally, set the audio materials json
    audio_sensor.setAudioMaterialsJSON("/workspace/data/mp3d_material_config.json")

  def get_IR(self):
    ir = np.array(self.sim.get_sensor_observations()["audio_sensor"])