# AutoCLIP: Auto-tuning Zero-Shot Classifiers for Vision-Language Models

Classifiers built upon vision-language models such as CLIP have shown remarkable zero-shot performance across a broad range of image classification tasks. Prior work has studied different ways of automatically creating descriptor sets for every class based on prompt templates, ranging from manually engineered templates over templates obtained from a large language model to templates built from random words and characters. Up until now, deriving zero-shot classifiers from the respective encoded class descriptors has remained nearly unchanged, i.e., classify to the class that maximizes cosine similarity between its averaged encoded class descriptors and the image encoding. However, weighing all class descriptors equally can be suboptimal when certain descriptors match visual clues on a given image better than others. In this work, we propose AutoCLIP, a method for auto-tuning zero-shot classifiers. AutoCLIP tunes per-image weights to each prompt template at inference time, based on statistics of class descriptor-image similarities. AutoCLIP is fully unsupervised, has very low computational overhead, and can be easily implemented in few lines of code. We show that AutoCLIP outperforms baselines across a broad range of vision-language models, datasets, and prompt templates consistently and by up to 3 percent point accuracy.

For more information about this work, please read our [TMLR paper](https://openreview.net/forum?id=gVNyEVKjqf):

> Metzen, J., Saranrittichai, P., Mummadi, C. (2024). AutoCLIP: Auto-tuning Zero-Shot Classifiers for Vision-Language Models. Transactions on Machine Learning Research (TMLR).

## Table of Contents
- [Experiment](#experiment)
- [Reference](#reference)

## Experiment

This repository provides a script to evaluate AutoCLIP on simulated CLIP data corresponding to Section 5 of [the paper](https://openreview.net/forum?id=gVNyEVKjqf). The experiment can be performed by running the command below:

```
python auto_clip_simulated.py
```

## Reference
Please refer to this citation, if this repository benefits your research:
```
@InProceedings{metzen2024autoclip,
author = {Hendrik Metzen, Jan and Saranrittichai, Piyapat and Mummadi, Chaithanya Kumar},
title = {
AutoCLIP: Auto-tuning Zero-Shot Classifiers for Vision-Language Models},
booktitle = {Proceedings of the European Conference on Computer Vision (ECCV)},
journal={Transactions on machine learning research},
year={2024}
}
```