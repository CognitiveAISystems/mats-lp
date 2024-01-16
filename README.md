# MATS-LP

This study addresses the challenging problem of decentralized lifelong multi-agent pathfinding. The proposed **MATS-LP** 
approach utilizes a combination of Monte Carlo Tree Search and reinforcement learning for resolving conflicts.

**Paper:** [Decentralized Monte Carlo Tree Search for Partially Observable Multi-agent Pathfinding](https://arxiv.org/abs/2312.15908)

## Installation:

To run MATS-LP, your system needs C++ Build Tools (CMake, g++, build-essential), python3 and ONNX runtime. 

Installation of Python packages:
```bash
pip3 install -r docker/requirements.txt
```

Installation of ONNX runtime:
```bash
wget https://github.com/microsoft/onnxruntime/releases/download/v1.14.1/onnxruntime-linux-x64-1.14.1.tgz \
    && tar -xf onnxruntime-linux-x64-1.14.1.tgz \
    && cp onnxruntime-linux-x64-1.14.1/lib/* /usr/lib/ && cp onnxruntime-linux-x64-1.14.1/include/* /usr/include/
```

Optionally, you could use the Dockerfile to build the image:
```bash
cd docker && sh build.sh
```

## Running MATS-LP:

To execute the **MATS-LP** algorithm and produce an animation using pre-trained weights of CostTracer, use the following command:

```bash
python3 main.py
```

The animation will be stored in the ```renders``` folder.

Using docker: 
```bash
docker run --rm -ti -v $(pwd):/code -w /code mats-lp python3 main.py
```

The hyperparameters of **MATS-LP** can be adjusted via ```MCTSConfig``` in ```main.py``` file.
The defaults of the parameters are set to the values used in the paper.

## Citation:

If you find this code helpful in your research, please cite our paper as follows:
```bibtex
@article{skrynnik2023decentralized,
  title={Decentralized Monte Carlo Tree Search for Partially Observable Multi-agent Pathfinding},
  author={Skrynnik, Alexey and Andreychuk, Anton and Yakovlev, Konstantin and Panov, Aleksandr},
  journal={arXiv preprint arXiv:2312.15908},
  year={2023}
}
```
