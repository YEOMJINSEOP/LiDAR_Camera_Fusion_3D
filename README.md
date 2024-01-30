### Training
- 40 epochs
- 0.00707 base LR with sqrt_k scaling rule (equals to original 0.001 at batchsize = 2, equals 0.00489 at batchsize = 24)
- AdamW with Weight Decay 0.001
- CosineDecay Schedule
- Batch Size 24 (Better result possible with lower batch size, batch size chosen for economical reasons.)

## Data Preparation

### SemanticKITTI
```
./
├── 
├── ...
└── path_to_data_shown_in_config/
    ├──sequences
        ├── 00/           
        │   ├── velodyne/	
        |   |	├── 000000.bin
        |   |	├── 000001.bin
        |   |	└── ...
        │   └── labels/ 
        |       ├── 000000.label
        |       ├── 000001.label
        |       └── ...
        ├── 08/ # for validation
        ├── 11/ # 11-21 for testing
        └── 21/
	    └── ...
```

## Training
1. modify the config/semantickitti.yaml with your custom settings. We provide a sample yaml for SemanticKITTI
2. train the network by running "sh train.sh"

Code is based on https://github.com/L-Reichardt/Cylinder3D-updated-CUDA
