# Human Surveillance in Maritime Environments Using Drone Aerial Footage for Search and Rescue

Search and Rescue (SAR) operations in oceans are highly challenging due to vast search regions, unpredictable sea conditions, and the critical need for rapid response. Existing AI models can detect ocean objects such as swimmers and boats, but they often lack the precision and localization capabilities required for effective real-time surveillance and rescue coordination.

Critical Gap:
Current systems fail to accurately:

- Detect and track multiple ocean objects simultaneously.
- Provide precise geo-localization for real-time rescue coordination.

Proposed Solution:
A drone-based AI system designed to:

- Detect and track swimmers, swimmers with life jackets, and boats in real-time.
- Perform geo-localization of detected objects for accurate position mapping.
- Trigger distress alerts when swimmers or swimmers with life jackets are detected and tracked, including:
  - Number of detected objects
  - Snapshot of detected objects
  - Object IDs

## Objectives

- Develop an AI-powered object detection and tracking system for ocean surveillance drones.
- Detect and track swimmers, swimmers with life jackets, and boats in real time.
- Perform geo-localization of detected objects using available sensor and GPS data.
- Trigger automated alerts when swimmers or swimmers with life jackets are detected and tracked, including:
  - Object type and ID
  - Number of detected objects
  - Snapshots of detections
  - Location coordinates
  - Transmit alert data to the control unit to support timely Search and Rescue (SAR) operations.

The following scope diagram illustrates our objectes.

## Scope Diagram

![scope diagram](.\imgs\scope_diagram.png)

## Working Model Flow Diagram

![working model flow diagram](.\imgs\working_model_flow_chart.png)

## Data-set Used

Two datasets were used

1. SeaDronesSee (SDS) (Primary)
   - UAV dataset for search & rescue Humans (swimmers) + boats, buoys, life-jacket
   - ~54k annotated images extracted from 22 videos drone videos
   - Useful for detection and tracking
2. MOBDrone (Secondary)
   - 66 Full HD UAV videos (10–60 m altitude)
   - ~126k frames, ~113k human annotations
   - Real sea conditions: waves, reflections, occlusions
   - Ideal for human-only maritime tracking

## Dataset EDA and Preprocessing

Exploratory Data Analysis (EDA) and dataset sampling for the SeaDronesSee dataset were performed in the following notebook:

📄 EDA Notebook:
[seadronessee_dataset_eda.ipynb](.\detection\notebooks\seadronessee_dataset_eda.ipynb)

The final sampled dataset used for training and evaluation has been uploaded to Kaggle and can be accessed here:

🔗 Kaggle Dataset Link:
https://www.kaggle.com/datasets/rameezakther/seadronessee-mot-sampled-dataset

The MOBDrone dataset is used for inference purpose only so no preprocessing is done.

## Enhanced RT-DETR Model Training and Inferencing

Details about the proposed architecture can be found in the project documentation.

![enhanced_rt_detr_architecture](.\imgs\architecture.png)

🔧 Model Training

The model was trained using the following Kaggle notebook:

👉 Training Notebook:
https://www.kaggle.com/code/rameezakther/rt-detr-pytorch

The base code is taken from this [repo.](https://github.com/lyuwenyu/RT-DETR)

During training, the Enhanced RT-DETR source code is cloned from this repository.
The source code is located in:

```bash
./detection/src_code
```

All training metrics and comparison results are stored in the detection directory of this repository.

The trained model is publicly available on Kaggle:

👉 Trained Model:
https://www.kaggle.com/models/rameezakther/rt-detr-trained-on-seadronessee-dataset

🖼 Model Inferencing

The trained model was used to run inference on images and video using the below notebook:

👉 Inference Notebook:
https://www.kaggle.com/code/rameezakther/rt-detr-seadronessee-model-inference

Sample images inferred by the trained model are shown below:

![inference_img_1](.\detection\inferenced_images\inferenced_4.jpg) ![inference_img_2](.\detection\inferenced_images\inferenced_852.jpg) ![inference_img_3](.\detection\inferenced_images\inferenced_5042.jpg)

## Visually Augmented Kalman Tracker (VAKT)

![vakt_architecture](.\imgs\vakt.png)

The proposed tracking code is present inside the directory:

```bash
.\tracking\src_code
```

The evaluation notebook is present in kaggle:

https://www.kaggle.com/code/suryaks27/tracking

The results and comparative analysis are present in the documentation.

A batch of inferenced images are
![inference_img_1](.\tracking\inferenced_images\1.jpg)
![inference_img_2](.\tracking\inferenced_images\2.jpg)
![inference_img_3](.\tracking\inferenced_images\3.jpg)
![inference_img_4](.\tracking\inferenced_images\4.jpg)

## Localization

Two methods are implemented for localization. One is when meta data such as GPS, altitude, camera intrinsics and extrinsics values of the drone are known and another method is when these meta data are present or given. Details about how this two methods are implemented are present in the documentation.

![geo_loc_arch_1](.\imgs\geo_loc_with_data.png)
Architecture of Localization when meta data is available

![geo_loc_arch_2](.\imgs\geo_loc_without_data.png)
Architecture of Localization when meta data is not available

The first method of implementation is done for seadronessee dataset because meta is available for that dataset. This [notebook](.\localization\notebook\localization.ipynb) is the implementation of the method1. The method 2 is implemented on video which is from mobdrone dataset since it does not contain any meta data about the drone, and this method is implemented on the final integrated system.

Some of the inferenced images by method are

![inference_img_1](.\localization\inferenced_images\batch1\0.jpg)
![inference_img_2](.\localization\inferenced_images\batch2\5041.jpg)
![inference_img_3](.\localization\inferenced_images\batch3\18218.jpg)
![inference_img_4](.\localization\inferenced_images\batch4\20442.jpg)

## Integrated system

The source code present in the `.\integrated_code` directory is an implementation of the full Detection + Tracking + Localization implementation on real time aerial drone footage. In this integrated system method 2 of localization is used, and the final inferenced video can be found in the drive link in the appropriate directory.

## Other links

Drive folder for project [link](https://drive.google.com/drive/folders/1NUXLQBruQClfPWLTbQxhZzHq5ewm8xaV?usp=sharing)
