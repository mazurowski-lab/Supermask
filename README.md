# Supermask

Here is the official repository for our MIDL paper [SuperMask: Generating High-resolution object masks from multi-view, unaligned low-resolution MRIs](https://arxiv.org/abs/2303.07517), where we are tryting to reconsturct high-resolution 3D objects from multi-view 2D MRIs.
<img width="1150" alt="3D-bone-new" src="https://github.com/mazurowski-lab/Supermask/assets/39239103/fd697d99-407d-43b3-9375-d72b6bb2ace3">

## How to use our code /reproduce our results
### (0) requirements.
```
pip3 install -r requirements.txt
pip3 install torch torchvision visdom
```

### (1) Data preparation
Please see the details in folder /data_preparation [to do].

### (2) Training and testing results.
Please update the train.sh for your specific setting. 
-- dataroot (the path to your patient list, for each folder, having multi-views available from 2 to n volumes).
-- labelroot (the path to your gt mask list, as the same structure of the images).
-- gpu_ids (the gpus you use, supports single or multi gpu setting).
-- model_type (**seg_align** for supermask, training segmentation and registration both; 'onlyseg' for only 3D segmentation; 'onlyalign' for only 3D registration.)

For testing, you could refer to ./test_on_heart and ./test_on_brain for the testing on different settings for brain and heart dataset.
for test_on_brain, 
- the notebooks contain keywords 'onlyseg' represents only 3D segmentation;
- 'hr' represents train and test on HR 3D mris; 128 represents volumes size;
- 8 and 16 represents slice distances;
- and 'stk' represents traditional image registration method based on iterative methods.

for test_on_heart, it contains the same naming system.

### citation
if you find our work helps and if you use or reference it in your work; please cite it as
```bib
@misc{gu2023supermask,
      title={SuperMask: Generating High-resolution object masks from multi-view, unaligned low-resolution MRIs}, 
      author={Hanxue Gu and Hongyu He and Roy Colglazier and Jordan Axelrod and Robert French and Maciej A Mazurowski},
      year={2023},
      eprint={2303.07517},
      archivePrefix={arXiv},
      primaryClass={eess.IV}
}
```


 
