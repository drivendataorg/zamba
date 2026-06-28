# MegaDetector

<a id='megadetectorlite'></a>

## MegadetectorLite

Frame selection for video models is critical as it would be infeasible to train neural networks on all the frames in a video. For all the species detection models that ship with `zamba`, the default frame selection method is an efficient object detection model called MegadetectorLite that determines the likelihood that each frame contains an animal. Then, only the frames with the highest probability of detection are passed to the model.

The image models use the related [MegaDetector](https://github.com/agentmorris/MegaDetector) model to find bounding boxes around individual animals so that the classifier runs on the cropped region of interest.

MegadetectorLite combines two open-source models:

* [Megadetector](https://github.com/microsoft/CameraTraps/blob/master/megadetector.md) is a pretrained image model designed to detect animals, people, and vehicles in camera trap videos.
* [YOLOX](https://github.com/Megvii-BaseDetection/YOLOX) is a high-performance, lightweight object detection model that is much less computationally intensive than Megadetector.

While highly accurate, Megadetector is too computationally intensive to run on every frame. MegadetectorLite was created by training a YOLOX model using the predictions of the Megadetector as ground truth - this method is called [student-teacher training](https://towardsdatascience.com/knowledge-distillation-simplified-dd4973dbc764).

MegadetectorLite can be imported into Python code and used directly since it has convenient methods for `detect_image` and `detect_video`. See [the API documentation for more details](../api-reference/object-detection-megadetector_lite_yolox.md#zamba.object_detection.yolox.megadetector_lite_yolox.MegadetectorLiteYoloX).
