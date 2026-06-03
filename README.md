# [ICML 2026] SOTAlign: Semi-Supervised Alignment of Unimodal Vision and Language Models via Optimal Transport

Authors: 
[Simon Roschmann*](https://www.eml-munich.de/people/simon-roschmann), 
[Paul Krzakala*](https://krzakalapaul.github.io),
[Sonia Mazelet](https://smazelet.github.io),
[Quentin Bouniot](https://qbouniot.github.io/),
[Zeynep Akata](https://www.eml-munich.de/people/zeynep-akata)

[![preprint](https://img.shields.io/static/v1?label=arXiv&message=2506.08641&color=B31B1B&logo=arXiv)](https://arxiv.org/abs/2602.23353)

## Abstract

The Platonic Representation Hypothesis posits that neural networks trained on different modalities converge toward a shared statistical model of the world. Recent work exploits this convergence by aligning frozen pretrained vision and language models with lightweight alignment layers, but typically relies on contrastive losses and millions of paired samples. In this work, we ask whether meaningful alignment can be achieved with substantially less supervision. We introduce a semi-supervised setting in which pretrained unimodal encoders are aligned using a small number of image–text pairs together with large amounts of unpaired data. To address this challenge, we propose SOTAlign, a two-stage framework that first recovers a coarse shared geometry from limited paired data using a linear teacher, then refines the alignment on unpaired samples via an optimal-transport-based divergence that transfers relational structure without overconstraining the target space. Unlike existing semi-supervised methods, SOTAlign effectively leverages unpaired images and text, learning robust joint embeddings across datasets and encoder pairs, and significantly outperforming supervised and semi-supervised baselines.

## Methodology

![SOTAlign architecture](assets/methodology.svg)

SOTAlign is a two-step method for the alignment of pretrained unimodal image and text encoders. First, we fit a linear alignment model only using the limited amount of available image-text pairs. Then, we use this linear model as a teacher to regularize the training of alignment layers $f$ and $g$ for a joint embedding space leveraging unimodal (unpaired) data.

## Code

We provide an implementation of the [KLOT](src/klot.py) divergence. SOTAlign leverages KLOT to compare the optimal transport plans induced by the learned space affinity matrix $K$ and the teacher space affinity matrix $K^*$. Our implementation follows the explicit gradient derived in Theorem 5.1 of the paper. This formulation avoids backpropagation through Sinkhorn iterations and thus makes the OT-based divergence more memory and runtime efficient.

The full code will be released soon.

## Citation
If you find SOTAlign useful, please star this repository and cite our work:

```bibtex
@article{roschmann2026sotalign,
  title={SOTAlign: Semi-Supervised Alignment of Unimodal Vision and Language Models via Optimal Transport},
  author={Simon Roschmann and Paul Krzakala and Sonia Mazelet and Quentin Bouniot and Zeynep Akata},
  journal={arXiv preprint arXiv:2602.23353},
  year={2026}
}
```

## Contact

If you have any questions, feel free to contact us: simon.roschmann@tum.de