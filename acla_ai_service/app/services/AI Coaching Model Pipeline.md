+---------------------+
|  Raw Telemetry Data |  (e.g., Speed, RPM, GPS, Throttle %, Brake Pressure, Steering Angle)
+---------------------+
           |
           v
+---------------------------------------+
|        Data Cleaning & Sync           |  (Handle missing values, outliers, synchronize timestamps)
+---------------------------------------+
           |
           v
+---------------------------------------+
|     Basic Feature Engineering         |  (Calculate: Acceleration, Jerk, Slip Angle)
+---------------------------------------+
           |
           +------------------------------+
           |                              |
           v                              v
+-----------------------+       +-----------------------+
|   Digital Track Map   |       | Vehicle Dynamics Model| (Physics or ML)
|  (Computational       |       |                       | (Estimates: Tire Grip, Load)
|   Geometry)           |       |                       |
+-----------------------+       +-----------------------+
           |                              |
           v                              v
+-----------------------+       +-----------------------+
|  Label Track Section  |       |  Estimate Grip & Load |
|  Calculate Distances  |       |                       |
|  (e.g., "Turn 5 Apex")|       |                       |
+-----------------------+       +-----------------------+
           |                              |
           +-------------+----------------+
                         |
                         v
+----------------------------------------------------+
|        Enriched Contextual Time-Series Data        |  (The final input feature vector for each timestep)
| [Speed, Throttle, Track_Section, Distance_to_Apex, |
|   Lateral_G, Estimated_Tire_Grip, ...]             |
+----------------------------------------------------+
                         |
                         v
+----------------------------------------------------+
|             Seq2Seq Model w/ Attention             |
|  (e.g., Transformer Architecture)                  |
+----------------------------------------------------+
    |                                  |
    v                                  v
+-------------+             +-----------------------------+
|   Encoder   |             |           Decoder           | --+
| (Understands|             | (Generates Action Sequence) |   |
|  Situation) |             +-----------------------------+   |
+-------------+                               |               |
    |                                         v               |
    +--------------------------------> [Attention Mechanism]  |
                                              |               |
                                              v               |
+-----------------------------+       +-----------------------------+
|   Optimal Action Labels     |       |  Predicted Action Sequence |
| (From Reference Laps)       |       |  (e.g., [Throttle, Brake,  |
| [Throttle, Brake, Steering] | ----> |          Steering])        |
+-----------------------------+       +-----------------------------+