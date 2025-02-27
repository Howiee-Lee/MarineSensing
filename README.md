# MarineSensing
Marine Sensing point trace data fusion involves processing the sensing data in the unit of geographical points from multiple ship platforms. This data includes the navigation information recorded by the ship's own sensors and navigation equipment, the point traces measured by the navigation radar installed on the ship, and the Automatic Identification System (AIS) data actively transmitted by other vessels and received by the AIS equipment equipped on the ship. The system identifies and matches the multi-source detection data of the same target but from different sources, so as to achieve the correction of the radar system error and the discovery of abnormal targets.<br>
The simulation data graph without clutter is as follows. The red points represent the sensing ship nodes, the blue points represent the point traces of other ships received by the sensing ships, the green points represent the target point traces detected by the radars installed on the sensing ships, and the black point traces represent the actual point traces of the ships. There are 10 sensing ships and 100 ships to be sensed, and the time range of the data is 20 minutes.<br>
It should be noted that the generation of radar data has systematic errors, which include the radial error, azimuth error, and effective detection range of radar detection. The radial error factor is uniformly set to 0.78 for all sensing ships. That is to say, for the detection of a target, the distance between the obtained radar point trace and the ship itself is 0.78 times the actual distance between the target and the ship itself. The azimuth error is symbolically set as a positive offset of 1 degree. The effective detection range usually varies according to different sea conditions and the density of targets, and there is no default value for it. <br>
![SimulationDataFigue](https://github.com/user-attachments/assets/b719cd59-7efc-42c0-a7c7-d3f2ed8e85f2)
The simulation data graph with radar clutter points is as follows. The clutter points are generated in such a way that there are randomly between 5 and 15 points appearing in each frame of the radar.  <br>
![SimulationNoisyDataFigue](https://github.com/user-attachments/assets/5198b7ab-bed6-4a29-abef-55b904f46614)
The processing effect of the Marine Sensing Point Trace Data Fusion System is to jointly process the Automatic Identification System (AIS) data of all ships, yielding unified AIS detection results. The point traces detected by the radar undergo complex preprocessing and filtering to obtain the trajectories of the detected targets. Then, the detected targets are matched with the AIS detection results. Meanwhile, the systematic errors of the radar are corrected to achieve the consistency between the radar-detected targets and the AIS-detected targets. This system can be applied to scenarios such as marine safety for wide-area perception and ship monitoring. <br>
As shown in the following figure for a certain sensing ship, the system calculates the systematic errors of this sensing ship. The first item is the radial error, the second item is the azimuth error, and the third item is the effective detection range. <br>
![SendShipNO-0-ISDBT_visualization](https://github.com/user-attachments/assets/27d8f57b-cfdd-46d3-a0bd-c743b2f73187)
# Using
