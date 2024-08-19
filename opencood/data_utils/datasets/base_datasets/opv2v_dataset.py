import os
import random
from typing import List, Dict

from torch.utils.data import Dataset

from logger import get_logger
from opencood.data_utils.datasets.dataset_models import Scenario, SensorsData, Timestamp, Agent
from opencood.data_utils.datasets.utils import load_yaml_data, load_lidar_data, load_camera_data, load_depth_data
from opencood.utils.globals import VISUALIZE, MODEL_MODE
from opencood.utils.file_utils import load_yaml

logger = get_logger()


def contractuser(path):
    """
    与 os.path.expanduser 的功能正好相反, 这个函数的作用是将类似 `/home/zfq` 这类路径替换为 `~`.
    这样输出到控制台的信息就不包含个人信息, 发到网上调试时可以更好地保护隐私.
    :param path:
    :return:
    """
    home_dir = os.path.expanduser("~")
    if path.startswith(home_dir):
        return path.replace(home_dir, "~", 1)
    return path


def get_sensor_files(agent_path: str, timestamp: str, sensor_type: str) -> List[str]:
    """
    根据给出的 agent 文件夹路径，提取出 agent 文件夹下所有包含 timestamp 的文件
    :param agent_path: 智能体基础路径 (包含场景 + agent_id)
    :param timestamp: 时间戳
    :param sensor_type: 需要提取的传感器的名称
    :return:
    """
    camera_files = []
    for i in range(4):  # 相机数量, 这里默认为 4, 因为在 OPV2V 数据集中就只有 4 个相机
        camera_files.append(os.path.join(agent_path, timestamp, f"_{sensor_type}{i}.png"))
    return camera_files


def extract_timestamps(yaml_files: List[str]) -> List[str]:
    """
    根据给出的 yaml 文件的文件路径，提取出 yaml 文件名中包含的时间戳+
    :param yaml_files:
    :return:
    """
    timestamps = []
    for file in yaml_files:
        file_name = file.split("/")[-1]
        timestamp = file_name.replace(".yaml", "")
        timestamps.append(timestamp)
    return timestamps


class OPV2VDataset(Dataset):
    def __init__(self, configs: Dict, visualize: bool, train=True):
        self.train = train
        self.scenario_database: List[Scenario] = []  # 存储场景数据 (场景数据库)
        self.len_record: List[int] = []  # 记录数据长度, 采用了一种类似于前缀和的形式.
        self.max_agent = configs.get("train_params", {}).get("max_cav", 5)  # 支持的最大智能体的数量
        self.label_type = configs["label_type"]

        """需要加载哪些文件"""
        self.load_lidar_file = True if "lidar" in configs["input_source"] or visualize else False
        self.load_camera_file = True if "camera" in configs["input_source"] else False
        self.load_depth_file = True if "depth" in configs["input_source"] else False

        # 加载数据
        train_dir = configs["train_dir"] if self.train else configs["validate_dir"]
        logger.important(f"Lode data from {train_dir}")  # 保护隐私, 先输出再对 `~` 进行解析
        train_dir = os.path.expanduser(train_dir)  # 单独命名成一个变量,
        self.scenario_directories = sorted(dir.path for dir in os.scandir(train_dir) if dir.is_dir())

        # TODO: add_data_extension 没有加上去

    def initialize(self, heterogeneous=False, adaptor=None):
        """
        初始化数据集, 包括 (是一个三重循环):
        1. 加载全部场景数据
        2. 加载全部 agent 数据
        3. 加载全部时间戳下的数据
        """
        """加载全部场景的数据"""
        for scenario_directory in self.scenario_directories:
            agents_in_scenario = [agent_id.name for agent_id in os.scandir(scenario_directory) if agent_id.is_dir()]
            if self.train:
                random.shuffle(agents_in_scenario)  # 随机打乱 agent 的 id 号
            else:
                agents_in_scenario = sorted(agents_in_scenario)

            if len(agents_in_scenario) <= 0:
                raise ValueError(f"No agents in scenario {scenario_directory}")

            scenario_name = scenario_directory.split("/")[-1]
            # TODO: 这里有一段没抄过来

            if heterogeneous:  # TODO: 如果是异质的情况, 需要进行一些处理, 具体含义还不是很清楚
                agents_in_scenario = adaptor.reorder_agents(agents_in_scenario, scenario_name)

            scenario = Scenario(scenario_name)

            """加载全部 agent 的数据"""
            for j, agent_id in enumerate(agents_in_scenario):
                # 如果某个场景中 agent 大于 max_agent 个, 就不再读入多余的 agent 数据了, 并且在命令行给出提示
                if j > self.max_agent - 1:
                    logger.warning(f"There are too many agents in {contractuser(scenario_directory)}, max_agent: {self.max_agent}")  # fmt: skip
                    break
                agent = Agent(agent_id)
                agent_path = os.path.join(scenario_directory, agent_id)
                # TODO: 这里加一个判断的意义是什么?

                yaml_files = sorted(entry.path for entry in os.scandir(agent_path)
                                    if entry.is_file() and entry.name.endswith(".yaml") and "additional" not in entry.name)  # fmt: skip
                # 过滤掉一个不合法的场景. TODO: 这里先去掉这一部分, 如果后续训练中有需要则加上
                # yaml_files = [file for file in yaml_files if not ("2021_08_20_21_10_24" in file and "000265" in file)]

                """提取并加载全部时间戳下的数据"""
                timestamps = extract_timestamps(yaml_files)
                for timestamp in timestamps:
                    timestamp_data = Timestamp(
                        timestamp=timestamp,
                        yaml=os.path.join(agent_path, f"{timestamp}.yaml"),
                        lidar=os.path.join(agent_path, f"{timestamp}.pcd"),
                        cameras=get_sensor_files(agent_path, timestamp, "camera"),
                        depths=get_sensor_files(agent_path, timestamp, "depth"),
                    )

                    if heterogeneous:
                        agent_modality = adaptor.reassign_agent_modality(scenario_name, agent_id, j)
                        timestamp_data.agent_modality = agent_modality
                        timestamp_data.lidar = adaptor.switch_lidar_channels(agent_modality, timestamp_data.lidar)

                    # TODO: 这里有一部分没抄, 加载多余数据

                    agent.add_timestamp_data(timestamp_data)

                """
                这里我们假定所有的 agent 都有着相同的时间戳数量. 因此只记录第一个 agent 时间戳列表的长度
                """
                if j == 0:
                    agent.ego = True
                    if len(self.len_record) == 0:
                        self.len_record.append(len(timestamps))
                    else:
                        prefix_len = self.len_record[-1]  # 前面那些记录的总长度
                        self.len_record.append(prefix_len + len(timestamps))
                else:
                    agent.ego = False
                scenario.add_agent(agent)
            self.scenario_database.append(scenario)

    def __getitem__(self, idx: int):
        return self.retrieve_basedata(idx)

    def retrieve_basedata(self, idx: int):
        """
        根据 idx 的值, 获取对应的场景数据
        :param idx:
        :return: 某一场景下, 全部车辆对应的某一时间戳的数据 (这里的 "某一" 全部可以根据 `idx` 计算出来)
        """
        scenario_index = 0
        for i, v in enumerate(self.len_record):
            if idx < v:
                scenario_index = i
                break
        scenario = self.scenario_database[scenario_index]
        timestamp_index = idx if scenario_index == 0 else idx - self.len_record[scenario_index - 1]
        return_data: Dict[str, SensorsData] = {}

        """加载 agent 里面的数据 (从一个文件路径变为真正的数据)"""
        for agent in scenario.agents:
            timestamp = agent.timestamps[timestamp_index]
            ego = agent.ego
            yaml_data = load_yaml_data(timestamp.yaml)
            lidar_data = load_lidar_data(timestamp.lidar) if self.load_lidar_file else None
            camera_data = load_camera_data(timestamp.cameras) if self.load_camera_file else None
            depth_data = load_depth_data(timestamp.depths) if self.load_depth_file else None

            # TODO: file_extension 没抄
            # fmt: off
            return_data[agent.id] = SensorsData(id=agent.id, ego=ego,
                yaml_data=yaml_data, lidar_data=lidar_data, camera_data=camera_data, depth_data=depth_data,
                agent_modality=timestamp.agent_modality,  # 这里没有对是否异质作出判断
            )
            # fmt: on

        return return_data

    def __len__(self):
        return self.len_record[-1]


"""测试函数 (路径均为绝对路径)"""
if __name__ == "__main__":
    from icecream import ic

    config_path = "~/PycharmProjects/ClosedHEAL/opencood/configs/OPV2V/MoreModality/HEAL/stage1/m1_pyramid.yaml"
    config_path = os.path.expanduser(config_path)
    configs = load_yaml(config_path)
    opv2v_dataset = OPV2VDataset(configs, False)
    opv2v_dataset.initialize()
    ic(opv2v_dataset.len_record)
    data = opv2v_dataset.retrieve_basedata(0)
    ic(len(data))
    ic(data.keys())
