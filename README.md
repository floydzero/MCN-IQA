# MCN
This is the code for the paper submitted to TCSVT (A Mixture Capsule Network Towards Interpretable Blind Image Quality Assessment).
### Operating Environments
#### Hardware Environment
Our code is running on a server with GeForce RTX 3090 GPUs and a CPU model Intel(R) Core(TM) i7-9800X @ 3.80GHz. The batchsize of a single graphics card is set to 64, which can be increased as the number of graphics cards increases.
#### Software Environment
* python = 3.6.8
* pytorch = 1.7.1
* torchvision
* pyyaml
* opencv-python
* matplotlib

### Train Model
When you need to train the model, set the parameters and corresponding paths completely and then run the run.py file.

### Checkpoints
* The checkpoints for SPAQ [link](https://drive.google.com/file/d/1keZBtWr9O8gfeqBB9qHNbZ-96Eh6LggB/view?usp=sharing)
* The checkpoints for KonIQ-10k [link](https://drive.google.com/file/d/1VNYC0QbHxw1aaBTFLj4DMIC2D36B1-ng/view?usp=sharing)
* The checkpoints for KADID-10k [link](https://drive.google.com/file/d/1VNYC0QbHxw1aaBTFLj4DMIC2D36B1-ng/view?usp=sharing)
* The checkpoints for TID2013 [link](https://drive.google.com/file/d/1VNYC0QbHxw1aaBTFLj4DMIC2D36B1-ng/view?usp=sharing)

### Inference
When inferring the MOS of an image, run the infer.py file to load the weights (Checkpoints) saved after training the model, perform MOS inference on the selected single image and save the tensor of the capsule group generated in QLR during the inference process.

