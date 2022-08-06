# ONNX HAWP Line Detection
 Python scripts for performing line detection using the HAWP model in ONNX.


![!ONNX HAWP Line Detection](https://github.com/ibaiGorordo/ONNX-HAWP-Line-Detection/blob/main/doc/img/output.jpg)

*Original image: https://commons.wikimedia.org/wiki/File:Bedroom_Mitcham.jpg*

# Requirements

 * Check the **requirements.txt** file. 
 * For ONNX, if you have a NVIDIA GPU, then install the **onnxruntime-gpu**, otherwise use the **onnxruntime** library.
 * Additionally, **pafy** and **youtube-dl** are required for youtube video inference.
 
# Installation
```
git clone https://github.com/ibaiGorordo/ONNX-HAWP-Line-Detection.git --recursive
cd ONNX-HAWP-Line-Detection
pip install -r requirements.txt
```
### ONNX Runtime
For Nvidia GPU computers:
`pip install onnxruntime-gpu`

Otherwise:
`pip install onnxruntime`

### For youtube video inference
```
pip install youtube_dl
pip install git+https://github.com/zizo-pro/pafy@b8976f22c19e4ab5515cacbfae0a3970370c102b
```

# ONNX model 
The original model was converted to different formats (including .onnx) by [PINTO0309](https://github.com/PINTO0309). 
Download the model from **[his repository](https://github.com/PINTO0309/PINTO_model_zoo/tree/main/108_HAWP)** using the download script and save them into the **[models](https://github.com/ibaiGorordo/ONNX-HAWP-Line-Detection/tree/main/models)** folder"

# Examples

 * **Image Line Detection**:
 ```
 python image_line_detection.py
 ```

 * **Webcam Line Detection**:
 ```
 python webcam_line_detection.py
 ``` 
 
 * **Video Line Detection: https://youtu.be/AKdwQwBCaTk**
 ```
 python video_line_detection.py
 ``` 
 ![!HAWP Video Line Detection](https://github.com/ibaiGorordo/ONNX-HAWP-Line-Detection/blob/main/doc/img/hawp_video.gif)
 
 *Original video: https://youtu.be/om6s2jmDJ2c*
  
# References:
* HAWP: https://github.com/cherubicXN/hawp
* PINTO0309's model zoo: https://github.com/PINTO0309/PINTO_model_zoo
* PINTO0309's model conversion tool: https://github.com/PINTO0309/openvino2tensorflow
* HAWP Original paper: https://arxiv.org/abs/2003.01663
