# ROS 2 SAM

This package is what the name suggests: Meta's `segment-anything` wrapped in a ROS 2 node. In this wrapper we offer...

 - ROS 2 services for segmenting images using point and box queries.
 - An RQT interface for specifying point queries interactively.
 - A Python client which handles the serialization of queries.


## Installation

Installation is easy: 
 1. Start by cloning this package into your ROS 2 environment and build via:

```shell
colcon build --symlink-install
```

 2. Install SAM by running `pip install git+https://github.com/facebookresearch/segment-anything.git`.


## Using ROS 2 SAM standalone

Run the SAM ROS 2 node using:

```bash
ros2 launch ros2_sam server.launch.py # will download SAM models if not not already downloaded
```

The node has three parameters:
 - `checkpoint_dir` directory containing SAM model checkpoints. If empty, models will be downloaded automatically.
 - `model_type` SAM model to use, defaults to `vit_h`. Check SAM documentation for options.
 - `device` whether to use CUDA and which device, defaults to `cuda`. Use `cpu` if you have no CUDA. If you want to use a specific GPU, set someting like `cuda:1`.

The node currently offers a single service `~/segment`, which can be called to segment an image.

You can test SAM by starting the node and then running `ros2 run ros2_sam sam_client_node`. This should yield the following result:

<img src="doc/figures/segmentation-example.png" width=50% height=50%>


### ROS 2 Services

`ros2_sam` offers a single service `~/segment` of the type `ros2_sam_msgs/srv/Segmentation`. The service definition is

```
sensor_msgs/Image        image            # Image to segment
geometry_msgs/Point[]    query_points     # Points to start segmentation from
int32[]                  query_labels     # Mark points as positive or negative samples
std_msgs/Int32MultiArray boxes            # Boxes can only be positive samples
bool                     multimask        # Generate multiple masks
bool                     logits           # Send back logits

---

sensor_msgs/Image[]   masks            # Masks generated for the query
float32[]             scores           # Scores for the masks
sensor_msgs/Image[]   logits           # Logit activations of the masks
```

The service request takes input image, input point prompts, corresponding labels and the box prompt. The service response contains the segmentation masks, confidence scores and the logit activations of the masks.

To learn more about the types and use of different queries, please refer to the [original SAM tutorial](https://github.com/facebookresearch/segment-anything/blob/main/notebooks/predictor_example.ipynb)

The service calls are wrapped up conveniently in the ROS 2 SAM client.


## Using ROS 2 SAM Client

Alternatively, if you don't feel like assembling the service calls yourself, one can use the ROS 2 SAM client instead of the service calls.

Initialize the client with the service name of the SAM segmentation service
```python
from ros2_sam import SAMClient
sam_client = SAMClient("sam_client", service_name="sam_server/segment")
```

Call the segment method with the input image, input prompt points and corresponding labels. This returns 3 segmentation masks for the object and their corresponding confidence scores
```python
img = cv2.imread('path/to/image.png')
points = np.array([[100, 100], [200, 200], [300, 300]])
labels = [1, 1, 0]
masks, scores = sam_client.sync_segment_request(img, points, labels)
```

Additional utilities for visualizing segmentation masks and input prompts
```python
from ros2_sam import show_mask, show_points
show_mask(masks[0], plt.gca())
show_points(points, np.asarray(labels), plt.gca())
```
