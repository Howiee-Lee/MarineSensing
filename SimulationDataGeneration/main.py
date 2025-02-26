import numpy as np
import pandas as pd
import geopy.distance
import geopandas as gpd
import matplotlib.pyplot as plt
from shapely.geometry import Point
from sqlalchemy import create_engine
from shapely import wkt
import random


def calculate_bearing(lat1, lon1, lat2, lon2):
    """
    计算从第一个地理坐标点 (lat1, lon1) 到第二个地理坐标点 (lat2, lon2) 的方位角。
    方位角是从正北方向顺时针测量的角度，范围为 0 到 360 度。
    参数:
    lat1, lon1: 第一个点的纬度和经度
    lat2, lon2: 第二个点的纬度和经度
    返回:
    方位角（度数）
    """
    # 将纬度和经度从度数转换为弧度
    lat1 = np.radians(lat1)
    lon1 = np.radians(lon1)
    lat2 = np.radians(lat2)
    lon2 = np.radians(lon2)
    # 计算经度差
    diffLong = lon2 - lon1
    # 计算方位角
    x = np.sin(diffLong) * np.cos(lat2)
    y = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(diffLong)
    initial_bearing = np.arctan2(x, y)
    # 将方位角从弧度转换为度数
    initial_bearing = np.degrees(initial_bearing)
    # 将方位角调整到 0 到 360 度之间
    bearing = (initial_bearing + 360) % 360
    return bearing


class DataGenerator:
    def __init__(self):
        self.engine_sensor = create_engine('postgresql://postgres:123456@localhost:5432/record_sensor')
        self.engine_ais = create_engine('postgresql://postgres:123456@localhost:5432/record_ais')
        self.engine_target = create_engine('postgresql://postgres:123456@localhost:5432/record_target')  # 数据库链接
        np.random.seed(35)  # 设置随机种子以便结果可重复
        self.crs = 'EPSG:4326'  # 数据的坐标系
        self.sensor_ship_list = []  # 所有感知船舶的仿真信息
        self.other_ships_list = []  # 所有船只的仿真信息
        self.initial_lon = 120
        self.initial_lat = 35
        self.sensor_num = 25  # 感知船只数量
        self.ship_num = 50  # 周围船只数量
        self.system_distance_error_factor = 0.78  # 距离误差因子，例如0.98表示探测到的距离会变小2%
        self.system_speed_error = 5  # 速度误差抖动范围
        self.system_direction_error = 90  # 航向误差抖动范围
        self.system_azimuth_error_deg = 1  # 方位角误差，例如偏移1度
        self.start_timestamp = 0
        self.stop_timestamp = 1200
        # 定义AIS参数
        self.ais_range = 0.1  # AIS接收范围，
        # 定义雷达参数
        self.radar_range = 0.1  # 雷达探测范围
        self.radar_precision_m = 25  # 雷达测量精度，单位为米
        self.radar_scan_interval = 10  # 雷达扫描间隔，单位为秒
        # 定义AIS参数
        self.ais_update_interval = 60  # AIS数据更新间隔，单位为秒
        self.ais_position_precision_m = 25  # AIS定位精度，单位为米
        self.fake = False  # 是否模拟假航迹
        self.fake_target_num = 5  # 假航迹出现的次数
        self.fake_length = 10  # 假航迹的连续点数量
        self.clutter = True  # 是否模拟环境杂波
        self.clutter_random_generator = random.Random()  # 创建一个新的随机数生成器对象
        self.clutter_random_generator.seed(42)  # 为新的随机数生成器设置一个新的种子
        # self.matching_pair = self.get_matching_pair()
        self.drop_out = 0.3  # 缺失数据概率

    # @staticmethod
    # def get_matching_pair():
    #     return {0: 41200000, 1: 41200001, 2: 41200002, 3: 41200003, 4: 41200004, 5: 41200005, 6: 41200006, 7: 41200007,
    #             8: 41200008, 9: 41200009, 10: 41200010, 11: 41200011, 12: 41200012, 13: 41200013, 14: 41200014,
    #             15: 41200015, 16: 41200016, 17: 41200017, 18: 41200018, 19: 41200019, 20: 41200020, 21: 41200021,
    #             22: 41200022, 23: 41200023, 24: 41200024, 25: 41200025, 26: 41200026, 27: 41200027, 28: 41200028,
    #             29: 41200029}

    def ship_data_generation(self, sensor_num, ship_num):
        self.sensor_num = sensor_num  # 感知船只数量
        self.ship_num = ship_num  # 周围船只数量
        for i in range(sensor_num):
            # 模拟感知船只的初始位置（经度和纬度）速度和方向
            sensor_ship = {
                "SendShipNO": i,
                "NO": i,
                "MMSI": 41200000 + i,
                "Lon": self.initial_lon + np.random.uniform(-0.2, 0.2),  # 在一定范围内生成随机经纬度
                "Lat": self.initial_lat + np.random.uniform(-0.2, 0.2),
                "SOG": np.random.uniform(2, 20),  # 速度范围,单位是节
                "COG": np.random.uniform(0, 360),  # 航行方向,正北为0°
            }
            self.sensor_ship_list.append(sensor_ship)

        # 随机生成周围船只的MMSI号、初始位置、速度和方向
        for j in range(ship_num):
            ship_data = {
                "SendShipNO": None,
                "NO": len(self.sensor_ship_list) + j,
                "MMSI": 41200000 + len(self.sensor_ship_list) + j,
                "Lat": self.initial_lat + np.random.uniform(-0.3, 0.3),  # 在一定范围内生成随机经纬度
                "Lon": self.initial_lon + np.random.uniform(-0.3, 0.3),
                "SOG": np.random.uniform(2, 20),  # 速度范围,单位是节
                "COG": np.random.uniform(0, 360)  # 航行方向,正北为0°
            }
            self.other_ships_list.append(ship_data)

    @staticmethod
    # 根据更新时间，更新船舶状态
    def update_ship_state(ship, time_delta_seconds):
        # ------根据感知船舶更新时间更新船舶自身运动数据-------
        time_delta_hours = time_delta_seconds / 3600
        # 根据时间间隔更新位置，以及方向和速度（小幅变化）
        ship_distance_traveled = ship["SOG"] * 1.852 * time_delta_hours
        destination = geopy.distance.distance(kilometers=ship_distance_traveled).destination(
            (ship["Lat"], ship["Lon"]), ship["COG"])
        ship["Lat"], ship["Lon"] = destination.latitude, destination.longitude
        ship["COG"] = (ship["COG"] + np.random.uniform(-5, 5)) % 360
        ship["SOG"] = max(5, min(25, ship["SOG"] + np.random.uniform(-0.5, 0.5)))

    def sensing_based_ship_data_generation(self):
        # 用于存储基于感知船舶数据生成的周围船只AIS与雷达数据
        sensory_ship_data = []
        around_ship_ais_data = []
        around_ship_radar_data = []
        all_ship_data = []
        # ————————————————基于时间开始循环生成数据——————————————————
        current_time = self.start_timestamp
        while current_time < self.stop_timestamp:
            #  按照每个感知船舶更新数据
            for sensor_ship in self.sensor_ship_list:
                # Ⅰ-----------感知船舶自身记录数据生成-----------
                sensory_ship_data.append({
                    "SendShipNO": sensor_ship['SendShipNO'],
                    "SendTime": current_time,
                    "Lon": sensor_ship['Lon'],
                    "Lat": sensor_ship['Lat'],
                    "SOG": sensor_ship['SOG'],
                    "COG": sensor_ship['COG']
                })

                # Ⅱ-----------感知船舶周围船只AIS数据生成-----------
                ais_ship = []  # 在ais范围内的船只
                for ship in self.other_ships_list:
                    if sensor_ship["Lat"] - self.ais_range < ship['Lat'] < sensor_ship["Lat"] + self.ais_range \
                            and sensor_ship["Lon"] - self.ais_range < ship['Lon'] < sensor_ship["Lon"] + self.ais_range:
                        ais_ship.append(ship)
                for ship in self.sensor_ship_list:
                    if sensor_ship["Lat"] - self.ais_range < ship['Lat'] < sensor_ship["Lat"] + self.ais_range \
                            and sensor_ship["Lon"] - self.ais_range < ship['Lon'] < sensor_ship["Lon"] + self.ais_range \
                            and ship['SendShipNO'] != sensor_ship['SendShipNO']:  # 排除自己
                        ais_ship.append(ship)
                for ship in ais_ship:
                    noisy_ship_ais_lat = ship["Lat"] + np.random.normal(0, self.ais_position_precision_m / 111000)
                    noisy_ship_ais_lon = ship["Lon"] + np.random.normal(0, self.ais_position_precision_m / (
                                111000 * np.cos(np.deg2rad(ship["Lat"]))))
                    # 根据上一次状态的速度和航向，随机生成本次数据的速度和航向，这里的随机性较小
                    noisy_ship_ais_speed = max(5, min(20, ship["SOG"] + np.random.uniform(-1, 1)))
                    noisy_ship_ais_direction = (ship["COG"] + np.random.uniform(-10, 10)) % 360
                    # 生成AIS数据
                    around_ship_ais_data.append({
                        "SendShipNO": sensor_ship['SendShipNO'],
                        "MMSI": ship["MMSI"],
                        "SendTime": current_time,
                        "Lon": noisy_ship_ais_lon,
                        "Lat": noisy_ship_ais_lat,
                        "SOG": noisy_ship_ais_speed,
                        "COG": noisy_ship_ais_direction
                    })

                # Ⅲ-----------感知船舶周围船只雷达数据生成-----------
                radar_ship = []  # 在雷达范围内的船只
                for ship in self.other_ships_list:  # 在感知范围内的船只而且不是自己
                    if sensor_ship["Lat"] - self.radar_range < ship['Lat'] < sensor_ship["Lat"] + self.radar_range \
                            and sensor_ship["Lon"] - self.radar_range < ship['Lon'] < sensor_ship[
                        "Lon"] + self.radar_range:
                        radar_ship.append(ship)
                for ship in self.sensor_ship_list:  # 在感知范围内的船只而且不是自己
                    if sensor_ship["Lat"] - self.radar_range < ship['Lat'] < sensor_ship["Lat"] + self.radar_range \
                            and sensor_ship["Lon"] - self.radar_range < ship['Lon'] < sensor_ship[
                        "Lon"] + self.radar_range \
                            and ship['SendShipNO'] != sensor_ship['SendShipNO']:
                        radar_ship.append(ship)
                for ship in radar_ship:
                    if np.random.uniform(0, 1) < self.drop_out:
                        continue  # 随机屏蔽目标
                    # 计算当前感知船只与目标船只之间的实际距离和方位角
                    radar_distance = geopy.distance.distance((sensor_ship["Lat"], sensor_ship["Lon"]),
                                                             (ship["Lat"], ship["Lon"]))
                    radar_bearing = calculate_bearing(sensor_ship["Lat"], sensor_ship["Lon"], ship["Lat"], ship["Lon"])
                    # 添加系统误差，得到雷达探测距离和方位角
                    distance_to_sensor = radar_distance.km * self.system_distance_error_factor
                    bearing_to_sensor = (radar_bearing + self.system_azimuth_error_deg) % 360
                    # 根据调整后的距离和方位角重新计算经纬度，并更新位置
                    adjusted_destination = geopy.distance.distance(kilometers=distance_to_sensor).destination(
                        (sensor_ship["Lat"], sensor_ship["Lon"]), bearing_to_sensor)
                    noisy_ship_lat, noisy_ship_lon = adjusted_destination.latitude, adjusted_destination.longitude
                    # 给位置添加测量误差
                    noisy_ship_radar_lat = noisy_ship_lat + np.random.normal(0,
                                                                             self.radar_precision_m / 111000)
                    noisy_ship_radar_lon = noisy_ship_lon + np.random.normal(0,
                                                                             self.radar_precision_m / (
                                                                                     111000 * np.cos(
                                                                                 np.deg2rad(noisy_ship_lat))))
                    # 根据上一次状态的速度和航向，随机生成本次数据的速度和航向，这里由于双多普勒效应，随机性很大
                    noisy_ship_radar_speed = max(2, min(20, ship["SOG"] + np.random.uniform(
                        -self.system_speed_error,
                        self.system_speed_error)))
                    noisy_ship_radar_direction = (ship["COG"] + np.random.uniform(-self.system_direction_error,
                                                                                  self.system_direction_error)) % 360
                    # 生成雷达数据
                    around_ship_radar_data.append({
                        "SendShipNO": sensor_ship['SendShipNO'],
                        "NO": ship["NO"],
                        "SendTime": current_time,
                        "Lon": noisy_ship_radar_lon,
                        "Lat": noisy_ship_radar_lat,
                        "SOG": noisy_ship_radar_speed,
                        "COG": noisy_ship_radar_direction
                    })

                # -----------雷达环境杂波数据生成-----------
                if self.clutter is True:  # 如果开启环境杂波
                    clutter_num = self.clutter_random_generator.randint(5, 15)  # 在这一帧的杂波点数量
                    NO_list = np.arange(self.ship_num + self.sensor_num,
                                        self.ship_num + self.sensor_num + clutter_num)
                    for i in range(clutter_num):  # 生成雷达杂波数据
                        around_ship_radar_data.append({
                            "SendShipNO": sensor_ship['SendShipNO'],
                            "NO": NO_list[i],
                            "SendTime": current_time,
                            "Lon": sensor_ship['Lon'] + self.clutter_random_generator.uniform(-0.1, 0.1),
                            "Lat": sensor_ship['Lat'] + self.clutter_random_generator.uniform(-0.1, 0.1),
                            "SOG": self.clutter_random_generator.uniform(2, 20),
                            "COG": self.clutter_random_generator.uniform(0, 360)
                        })
            # 更新时间与所有船舶的状态
            time_delta_seconds = np.random.randint(4, 6)
            for ship in self.other_ships_list:
                all_ship_data.append({
                    "SendShipNO": None,
                    "MMSI": ship["MMSI"],
                    "Time": current_time,
                    "Lon": ship['Lon'],
                    "Lat": ship['Lat'],
                    "SOG": ship['SOG'],
                    "COG": ship['COG']
                })
                self.update_ship_state(ship, time_delta_seconds)
            for ship in self.sensor_ship_list:
                all_ship_data.append({
                    "SendShipNO": ship['SendShipNO'],
                    "MMSI": ship["MMSI"],
                    "Time": current_time,
                    "Lon": ship['Lon'],
                    "Lat": ship['Lat'],
                    "SOG": ship['SOG'],
                    "COG": ship['COG']
                })
                self.update_ship_state(ship, time_delta_seconds)
            current_time += time_delta_seconds  # 更新时间

        # 转换数据为DataFrame
        sensory_ship_df = pd.DataFrame(sensory_ship_data)
        around_ship_ais_df = pd.DataFrame(around_ship_ais_data)
        around_ship_radar_df = pd.DataFrame(around_ship_radar_data)
        all_ship_df = pd.DataFrame(all_ship_data)

        # 创建 GeoDataFrame
        sensory_ship_gdf = gpd.GeoDataFrame(
            sensory_ship_df,
            geometry=gpd.points_from_xy(sensory_ship_df["Lon"], sensory_ship_df['Lat'])
        )
        around_ship_ais_gdf = gpd.GeoDataFrame(
            around_ship_ais_df,
            geometry=gpd.points_from_xy(around_ship_ais_df['Lon'], around_ship_ais_df['Lat'])
        )
        around_ship_radar_gdf = gpd.GeoDataFrame(
            around_ship_radar_df,
            geometry=gpd.points_from_xy(around_ship_radar_df['Lon'], around_ship_radar_df['Lat'])
        )
        all_ship_gdf = gpd.GeoDataFrame(
            all_ship_df,
            geometry=gpd.points_from_xy(all_ship_df['Lon'], all_ship_df['Lat'])
        )
        return sensory_ship_gdf, around_ship_ais_gdf, around_ship_radar_gdf, all_ship_gdf

    """
    def sensing_based_ship_data_generation_demo(self, num):
        self.ship_num = num
        # 模拟感知船只的初始位置（经度和纬度）速度和方向
        sensor_ship = {
            "SendShipNO": 1,
            "SendTime": self.start_timestamp,
            "Lon": self.initial_lon,  # 在一定范围内生成随机经纬度
            "Lat": self.initial_lat,
            "SOG": np.random.uniform(2, 20),  # 速度范围
            "COG": np.random.uniform(0, 360)  # 航行方向
        }
        # 用于存储生成的感知船舶自身的数据
        sensory_ship_data = [sensor_ship, ]

        # 随机生成周围船只的雷达探测批号、MMSI号、初始位置、速度和方向
        other_ships = []
        for i in range(self.ship_num):
            ship_data = {
                "NO": i,
                "MMSI": 41200000 + i,
                "Lat": self.initial_lat + np.random.uniform(-0.05, 0.05),  # 在一定范围内生成随机经纬度
                "Lon": self.initial_lon + np.random.uniform(-0.05, 0.05),
                "SOG": np.random.uniform(2, 20),  # 速度范围
                "COG": np.random.uniform(0, 360)  # 航行方向
            }
            other_ships.append(ship_data)
        # 随机生成雷达的假航迹目标和杂波点目标
        fake_target = []
        for i in range(self.ship_num, self.ship_num + self.fake_target_num):
            target_data = {
                "StartTime": np.random.randint(self.start_timestamp, self.stop_timestamp),  # 随机出现时间
                "NO": i,
                "Lat": self.initial_lat + np.random.uniform(-0.05, 0.05),  # 在一定范围内生成随机经纬度
                "Lon": self.initial_lon + np.random.uniform(-0.05, 0.05),
                "SOG": np.random.uniform(2, 20),  # 速度范围
                "COG": np.random.uniform(0, 360),  # 航行方向
                "cnt": 0  # 次数
            }
            fake_target.append(target_data)
        # 用于存储基于感知船舶数据生成的周围船只AIS与雷达数据
        around_ship_ais_data = []
        around_ship_radar_data = []

        current_time = self.start_timestamp
        while current_time < self.stop_timestamp:
            # Ⅰ-----------感知船舶自身记录数据生成-----------
            # 随机生成2到10秒之间的时间间隔
            time_delta_seconds_sensor = np.random.randint(2, 10)
            current_time += time_delta_seconds_sensor
            # 随机生成雷达的扫描时间间隔，所有的雷达数据均在同一时间间隔下产生
            time_delta_seconds_radar = np.random.randint(2, 10)
            if current_time >= self.stop_timestamp:
                break
            # 随机生成速度变化，模拟速度逐步变化
            current_speed = max(2, min(20, sensory_ship_data[-1]["SOG"] + np.random.uniform(-1,
                                                                                            1)))  # 限制速度在2到20海里/小时之间,速度变化在-1到1海里/小时之间
            # 随机生成一个较小的方向变化，模拟方向逐步变化
            current_direction = (sensory_ship_data[-1]["COG"] + np.random.uniform(-10,
                                                                                  10)) % 360  # 确保方向在0-360度之间,方向变化在-10到10度之间
            # 计算位置变化
            delta_latitude = sensory_ship_data[-1]["SOG"] * np.cos(
                np.deg2rad(sensory_ship_data[-1]["COG"])) * 0.01 / 3600 * time_delta_seconds_sensor
            delta_longitude = sensory_ship_data[-1]["SOG"] * np.sin(
                np.deg2rad(sensory_ship_data[-1]["COG"])) * 0.01 / 3600 * time_delta_seconds_sensor
            # 更新位置
            new_longitude = sensory_ship_data[-1]["Lon"] + delta_longitude
            new_latitude = sensory_ship_data[-1]["Lat"] + delta_latitude
            # 保存数据
            sensory_ship_data.append({
                "SendShipNO": 1,
                "SendTime": current_time,
                "Lon": new_longitude,
                "Lat": new_latitude,
                "SOG": current_speed,
                "COG": current_direction
            })

            # Ⅱ-----------感知船舶周围船只探测数据生成-----------
            for ship in other_ships:
                # 一、-------更新AIS数据-------
                # ①-------针对AIS的数据生成，随机生成2到10秒之间的时间间隔-------
                time_delta_seconds_ais = np.random.randint(2, 10)
                time_delta_hours = time_delta_seconds_ais / 3600
                # ②-------更新船舶的实际运动数据-------
                # 根据时间间隔更新位置，以及方向和速度（小幅变化）
                ship_distance_traveled = ship["SOG"] * 1.852 * time_delta_hours
                destination = geopy.distance.distance(kilometers=ship_distance_traveled).destination(
                    (ship["Lat"], ship["Lon"]), ship["COG"])
                # 给位置添加噪声
                noisy_ship_ais_lat = destination.latitude + np.random.normal(0, self.ais_position_precision_m / 111000)
                noisy_ship_ais_lon = destination.longitude + np.random.normal(0, self.ais_position_precision_m / (
                        111000 * np.cos(np.deg2rad(ship["Lat"]))))
                # 根据上一次状态的速度和航向，随机生成本次数据的速度和航向，这里的随机性较小
                noisy_ship_ais_speed = max(2, min(20, ship["SOG"] + np.random.uniform(-1, 1)))
                noisy_ship_ais_direction = (ship["COG"] + np.random.uniform(-10, 10)) % 360
                # 生成AIS数据
                around_ship_ais_data.append({
                    "SendShipNO": 1,
                    "MMSI": ship["MMSI"],
                    "SendTime": current_time - time_delta_seconds_sensor + time_delta_seconds_ais,
                    "Lon": noisy_ship_ais_lon,
                    "Lat": noisy_ship_ais_lat,
                    "SOG": noisy_ship_ais_speed,
                    "COG": noisy_ship_ais_direction
                })

                # 二、-------更新雷达数据-------
                # ①-------针对雷达探测一批的数据生成，根据随机生成2到10秒之间的时间间隔-------
                # time_delta_seconds_radar = np.random.randint(2, 10)
                time_delta_hours = time_delta_seconds_radar / 3600
                # ②-------根据时间间隔更新周围船舶实际运动数据-------
                ship_distance_traveled = ship["SOG"] * 1.852 * time_delta_hours
                destination = geopy.distance.distance(kilometers=ship_distance_traveled).destination(
                    (ship["Lat"], ship["Lon"]), ship["COG"])
                current_radar_lat, current_radar_lon = destination.latitude, destination.longitude
                # ③-------根据时间间隔计算当前感知船舶的位置，并以此生成雷达数据-------
                sensor_ship_distance_traveled = sensory_ship_data[-2]["SOG"] * 1.852 * time_delta_hours
                sensor_destination = geopy.distance.distance(kilometers=sensor_ship_distance_traveled).destination(
                    (sensory_ship_data[-2]["Lat"], sensory_ship_data[-2]["Lon"]), sensory_ship_data[-2]["COG"])
                sensor_lat, sensor_lon = sensor_destination.latitude, sensor_destination.longitude
                # 计算当前感知船只与目标船只之间的实际距离和方位角
                radar_distance = geopy.distance.distance((sensor_lat, sensor_lon),
                                                         (current_radar_lat, current_radar_lon))
                radar_bearing = calculate_bearing(sensor_lat, sensor_lon, current_radar_lat, current_radar_lon)
                # 添加系统误差，得到雷达探测距离和方位角
                distance_to_sensor = radar_distance.km * self.system_distance_error_factor
                bearing_to_sensor = (radar_bearing + self.system_azimuth_error_deg) % 360
                # 根据调整后的距离和方位角重新计算经纬度，并更新位置
                adjusted_destination = geopy.distance.distance(kilometers=distance_to_sensor).destination(
                    (sensor_lat, sensor_lon), bearing_to_sensor)
                noisy_ship_lat, noisy_ship_lon = adjusted_destination.latitude, adjusted_destination.longitude
                # 给位置添加测量误差
                noisy_ship_radar_lat = noisy_ship_lat + np.random.normal(0, self.ais_position_precision_m / 111000)
                noisy_ship_radar_lon = noisy_ship_lon + np.random.normal(0, self.ais_position_precision_m / (
                        111000 * np.cos(np.deg2rad(noisy_ship_lat))))
                # 根据上一次状态的速度和航向，随机生成本次数据的速度和航向，这里由于双多普勒效应，随机性很大
                noisy_ship_radar_speed = max(2, min(20, ship["SOG"] + np.random.uniform(-self.system_speed_error,
                                                                                        self.system_speed_error)))
                noisy_ship_radar_direction = (ship["COG"] + np.random.uniform(-self.system_direction_error,
                                                                              self.system_direction_error)) % 360
                # 生成雷达数据
                around_ship_radar_data.append({
                    "SendShipNO": 1,
                    "NO": ship["NO"],
                    "SendTime": current_time - time_delta_seconds_sensor + time_delta_seconds_radar,
                    "Lon": noisy_ship_radar_lon,
                    "Lat": noisy_ship_radar_lat,
                    "SOG": noisy_ship_radar_speed,
                    "COG": noisy_ship_radar_direction
                })

                # 三、-------根据感知船舶更新时间更新船舶自身运动数据-------
                time_delta_hours = time_delta_seconds_sensor / 3600
                # 根据时间间隔更新位置，以及方向和速度（小幅变化）
                ship_distance_traveled = ship["SOG"] * 1.852 * time_delta_hours
                destination = geopy.distance.distance(kilometers=ship_distance_traveled).destination(
                    (ship["Lat"], ship["Lon"]), ship["COG"])
                ship["Lat"], ship["Lon"] = destination.latitude, destination.longitude
                ship["COG"] = (ship["COG"] + np.random.uniform(-5, 5)) % 360
                ship["SOG"] = max(5, min(25, ship["SOG"] + np.random.uniform(-0.5, 0.5)))

            # Ⅲ-----------雷达假目标探测数据生成-----------
            if self.fake is True:
                for target in fake_target:
                    if (current_time - time_delta_seconds_sensor + time_delta_seconds_radar) >= target["StartTime"] and \
                            target["cnt"] < self.fake_length:  # 假轨迹开始
                        # ①-------针对雷达探测的数据生成，随机生成2到10秒之间的时间间隔-------
                        # time_delta_seconds_radar = np.random.randint(2, 10)
                        time_delta_hours = time_delta_seconds_radar / 3600
                        # ②-------根据时间间隔更新周围船舶实际运动数据-------
                        ship_distance_traveled = target["SOG"] * 1.852 * time_delta_hours
                        destination = geopy.distance.distance(kilometers=ship_distance_traveled).destination(
                            (target["Lat"], target["Lon"]), target["COG"])
                        current_radar_lat, current_radar_lon = destination.latitude, destination.longitude
                        # 给位置添加测量误差
                        noisy_ship_radar_lat = current_radar_lat + np.random.normal(0,
                                                                                    self.ais_position_precision_m / 111000)
                        noisy_ship_radar_lon = current_radar_lon + np.random.normal(0, self.ais_position_precision_m / (
                                111000 * np.cos(np.deg2rad(current_radar_lat))))
                        # 根据上一次状态的速度和航向，随机生成本次数据的速度和航向，这里由于双多普勒效应，随机性很大
                        noisy_ship_radar_speed = max(2,
                                                     min(20, target["SOG"] + np.random.uniform(-self.system_speed_error,
                                                                                               self.system_speed_error)))
                        noisy_ship_radar_direction = (target["COG"] + np.random.uniform(-self.system_direction_error,
                                                                                        self.system_direction_error)) % 360
                        target["cnt"] += 1  # 出现次数
                        around_ship_radar_data.append({
                            "SendShipNO": 1,
                            "NO": target["NO"],
                            "SendTime": current_time - time_delta_seconds_sensor + time_delta_seconds_radar,
                            "Lon": noisy_ship_radar_lon,
                            "Lat": noisy_ship_radar_lat,
                            "SOG": noisy_ship_radar_speed,
                            "COG": noisy_ship_radar_direction
                        })

            # Ⅳ-----------雷达环境杂波数据生成-----------
            # time_delta_seconds_radar = np.random.randint(2, 10)
            if self.clutter is True:  # 如果开启环境杂波
                clutter_num = np.random.randint(2, 10)  # 在这一帧的杂波点数量
                NO_list = np.arange(self.ship_num + self.fake_target_num,
                                    self.ship_num + self.fake_target_num + clutter_num)
                for i in range(clutter_num):
                    around_ship_radar_data.append({
                        "SendShipNO": 1,
                        "NO": NO_list[i],
                        "SendTime": current_time - time_delta_seconds_sensor + time_delta_seconds_radar,
                        "Lon": self.initial_lon + np.random.uniform(-0.1, 0.1),
                        "Lat": self.initial_lat + np.random.uniform(-0.1, 0.1),
                        "SOG": np.random.uniform(2, 20),
                        "COG": np.random.uniform(0, 360)
                    })

        # 转换数据为DataFrame
        sensory_ship_df = pd.DataFrame(sensory_ship_data)
        around_ship_ais_df = pd.DataFrame(around_ship_ais_data)
        around_ship_radar_df = pd.DataFrame(around_ship_radar_data)

        # 创建 GeoDataFrame
        sensory_ship_gdf = gpd.GeoDataFrame(
            sensory_ship_df,
            geometry=gpd.points_from_xy(sensory_ship_df["Lon"], sensory_ship_df['Lat'])
        )
        around_ship_ais_gdf = gpd.GeoDataFrame(
            around_ship_ais_df,
            geometry=gpd.points_from_xy(around_ship_ais_df['Lon'], around_ship_ais_df['Lat'])
        )
        around_ship_radar_gdf = gpd.GeoDataFrame(
            around_ship_radar_df,
            geometry=gpd.points_from_xy(around_ship_radar_df['Lon'], around_ship_radar_df['Lat'])
        )
        return sensory_ship_gdf, around_ship_ais_gdf, around_ship_radar_gdf
    """

    @staticmethod
    def vision_data(sensory_data, around_ais_data, around_radar_data, all_ship_df):
        fig, ax = plt.subplots(figsize=(10, 10))
        # 可视化 GeoDataFrame 中的散点数据
        sensory_data.plot(ax=ax, marker='o', color='red', markersize=30, label='sensor')
        around_ais_data.plot(ax=ax, marker='o', color='blue', markersize=20, label='ais')
        around_radar_data.plot(ax=ax, marker='o', color='green', markersize=10, label='radar')
        all_ship_df.plot(ax=ax, marker='o', color='black', markersize=10, label='real')
        # 添加标题和显示图形
        plt.title("Geographic Scatter Plot")
        plt.xlabel("Longitude")
        plt.ylabel("Latitude")
        plt.legend()
        plt.show()
        return

    def data_import(self, sensory_data, around_ais_data, around_radar_data):
        sensory_data.to_postgis(name='simulation_ship_sensor', con=self.engine_sensor, if_exists='replace')
        around_ais_data.to_postgis(name='simulation_ship_ais', con=self.engine_ais, if_exists='replace')
        around_radar_data.to_postgis(name='simulation_ship_target', con=self.engine_target, if_exists='replace')

    @staticmethod
    def geo_to_csv(sensory_data, around_ais_data, around_radar_data):
        sensory_data['geometry'] = sensory_data['geometry'].apply(lambda x: x.wkt)
        around_ais_data['geometry'] = around_ais_data['geometry'].apply(lambda x: x.wkt)
        around_radar_data['geometry'] = around_radar_data['geometry'].apply(lambda x: x.wkt)
        sensory_data.to_csv('simulation_ship_sensor.csv', index=False)
        around_ais_data.to_csv('simulation_ship_ais.csv', index=False)
        around_radar_data.to_csv('simulation_ship_noisy_target.csv', index=False)


if __name__ == '__main__':
    dgr = DataGenerator()
    dgr.ship_data_generation(sensor_num=10, ship_num=100)
    sensory_ship_gdf, around_ship_ais_gdf, around_ship_radar_gdf, all_ship_gdf = dgr.sensing_based_ship_data_generation()
    dgr.vision_data(sensory_ship_gdf, around_ship_ais_gdf, around_ship_radar_gdf, all_ship_gdf)
    dgr.geo_to_csv(sensory_ship_gdf, around_ship_ais_gdf, around_ship_radar_gdf)  # 保存
    # sensory_ship_gdf, around_ship_ais_gdf, around_ship_radar_gdf = dgr.sensing_based_ship_data_generation_demo(25)
    # dgr.vision_data(sensory_ship_gdf, around_ship_ais_gdf, around_ship_radar_gdf)
    # dgr.data_import(sensory_ship_gdf, around_ship_ais_gdf, around_ship_radar_gdf)
