# 1. Setup


<pre>
!conda create -n openmmlab python=3.7 pytorch==1.6.0 cudatoolkit=10.1 torchvision -c pytorch -y
!conda activate openmmlab
!pip install openmim
!mim install mmcv-full
!git clone https://github.com/open-mmlab/mmdetection.git
!cd mmdetection
!pip install -r requirements/build.txt
!pip install -v -e .
</pre>


