## Generating Spatio-Temporal Knowledge Graphs (STKGs) from Video
*********************************************************************************************************************************

We derive STKGs by converting each video into a graph whose nodes represent detected objects and whose edges encode both spatial proximity and temporal continuity. This structured representation enables downstream GNN-based reasoning to capture complex object–action dynamics. You can see the details of STKGs creation process in this work [1].

Datasets: Any annotated video collection (e.g., HMDB-51 [3], UCF-101 [4], Kinetics-400 [5], Something-Something [2]).
Object Detector: A pretrained model (e.g., Faster-R-CNN) for per-frame object bounding boxes.
Libraries: PyTorch, Torch-Geometric (for graph data structures), NumPy, etc.

Core Steps:
- Frame Sampling
 Sample at a reduced frame rate (e.g., 5 FPS) to avoid redundant frames.
 Only process frames where meaningful changes occur.


- Node Creation
Run an object detector on each sampled frame.
Create one node per detected object; use its bounding-box coordinates (and optionally appearance features) as node attributes.
Assign each node the class label of its source video.


- Edge Creation
Temporal criterion: Link nodes across consecutive frames only if the time gap ≤ 0.5 s.
Spatial criterion: Connect nodes whose bounding-box centroids lie within a spatial threshold (e.g., ≤ 20 pixels).
Store both the time interval and spatial distance as edge attributes.


- Graph Assembly
For each video, merge all per-frame nodes and valid edges into a subgraph.
Union multiple subgraphs (across videos or classes) to form a complete STKG for that source.


- Dataset Construction
Repeat for each chosen source dataset to produce single-source STKGs.
Optionally, combine samples from multiple sources into mixed STKGs of varying sizes for robustness testing.


## Usage of runnable main notebook file:
***************************************************************************************************************************************************
Although all datasets are video-based, the structure of Something-something dataset differs from HMDB, Kinetics and UCF datasets. For this reason, we updated some hyperparameters according to the data so that our model can work with the Something-something dataset without disrupting the structure of the model. 

- Python 3.9+
- PyTorch, PyTorch Geometric, NumPy, scikit‑learn, matplotlib (see `environment.yml` to import all required libraries)
- Prepared input files (relative paths expected by `load_data_from_files()`):  
  `node_features.txt`, `edges.txt`, `edge_features.txt`, `node_labels.txt`

Create the environment:
```bash
conda env create -f environment.yml
conda activate <env-name>
```

## References
***************
[1] Tataroğlu Özbulak, G. A., Shrestha, Y. R., & Calbimonte, J.-P. (in press). STKGNN: Scalable spatio-temporal knowledge graph reasoning for activity recognition. In Proceedings of the 34th ACM International Conference on Information and Knowledge Management (CIKM ’25). https://doi.org/10.1145/3746252.3761147

[2] Goyal, Raghav, et al. "The" something something" video database for learning and evaluating visual common sense." Proceedings of the IEEE international conference on computer vision. 2017.

[3] Kuehne, Hildegard, et al. "HMDB: a large video database for human motion recognition." 2011 International conference on computer vision. IEEE, 2011.

[4] Soomro, Khurram, Amir Roshan Zamir, and Mubarak Shah. "UCF101: A dataset of 101 human actions classes from videos in the wild." arXiv preprint arXiv:1212.0402 (2012).

[5] Kay, Will, et al. "The kinetics human action video dataset." arXiv preprint arXiv:1705.06950 (2017).

