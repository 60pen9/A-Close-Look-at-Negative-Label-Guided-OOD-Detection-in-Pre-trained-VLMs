# A Closer Look at Negative Label Guided Out-of-distribution Detection with Pre-trained Vision-Language Models
This is the code for the paper "A Closer Look at Negative Label Guided Out-of-distribution Detection with Pre-trained Vision-Language Models" (ICML 2026). 

Abstract: *Advances in pre-trained vision-language models have enabled zero-shot out-of-distribution (OOD) detection using only in-distribution (ID) labels. 
Recent methods in this direction expand the label space with negative labels to enhance the discrimination between ID and OOD inputs. 
Despite their promising progress, there remains a limited understanding of their empirical effectiveness in open-world scenarios, where negative labels can arbitrarily diverge from real OOD ones.
This paper bridges this research gap with the helm of a novel energy-based framework, where the energy function is built upon the margin between the similarity of an input to ID labels and that to negative labels.
Guided by this framework, we prove that the inherent tolerance of such methods to the sampling bias essentially stems from estimating the worst-case energy function over a KL-constrained set of potential distributions centered on the negative label distribution.
Furthermore, our theoretical analysis reveals that existing methods suffer from over-pessimism and consequently high sensitivity to outliers. 
Provably, we can alleviate these problems by leveraging Rényi divergence to refine potential distributions.*

# Dataset Preparation

## In-distribution dataset

Please download [ImageNet-1k](http://www.image-net.org/challenges/LSVRC/2012/index) 

## Out-of-distribution dataset

Following [KNN](https://arxiv.org/abs/2204.06507), we use the following 4 OOD datasets for evaluation: [iNaturalist](https://arxiv.org/pdf/1707.06642.pdf), [SUN](https://vision.princeton.edu/projects/2010/SUN/paper.pdf), [Places](http://places2.csail.mit.edu/PAMI_places.pdf), and [Textures](https://arxiv.org/pdf/1311.3618.pdf).

Please refer to [KNN](https://github.com/deeplearning-wisc/knn-ood), download OOD datasets.

# Usage
We first need to compute the image embedding, ID class embedding, filter negative label embedding with the CLIP model by running

> python eval_ood_detection.py

and perform OOD scoring by running

> python main_online.py



