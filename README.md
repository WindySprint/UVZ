# UVZ
This is the project of paper "Underwater Variable Zoom: Depth-Guided Perception Network for Underwater Image Enhancement".

# 1. Abstract 
Underwater scenes intrinsically involve degradation problems owing to heterogeneous ocean elements. Prevailing underwater image enhancement (UIE) methods stick to straightforward feature modeling to learn the mapping function, which leads to limited vision gain as it lacks more explicit physical cues (e.g., depth). In this work, we investigate injecting the depth prior into the deep UIE model for more precise scene enhancement capability. To this end, we present a novel depth-guided perception UIE framework, dubbed underwater variable zoom (UVZ). Specifically, UVZ resorts to a two-stage pipeline. First, a depth estimation network is designed to generate critical depth maps, combined with an auxiliary supervision network introduced to suppress estimation differences during training. Second, UVZ parses near-far scenarios by harnessing the predicted depth maps, enabling local and non-local perceiving in different regions. Extensive experiments on five benchmark datasets demonstrate that UVZ achieves superior visual gain and delivers promising quantitative metrics. Besides, UVZ is confirmed to exhibit good generalization in some visual tasks, especially in unusual lighting conditions.

# 2. Comparison Figures
This is the enhancement effects of some UIE methods on near-far scenarios:
<div align=center>
<img src="Figs/Near-Far View.jpg" width="55%">  
</div>

This is the enhancement comparison of all methods on the UFO dataset:
<div align=center>
<img src="Figs/Compare_UFO.jpg" width="70%">  
</div>

This is the effect comparison of different tasks before and after enhancement: 
<div align=center>
<img src="Figs/Application.jpg" width="60%"> 
</div>

# 3. Discussion
The complete code and models will be made public soon.

# 4. Citation
<pre><code>
  @article{huang2024underwater,
  title={Underwater Variable Zoom-Depth-Guided Perception Network for Underwater Image Enhancement},
  author={Huang, Zhixiong and Wang, Xinying and Li, Jinjiang and Liu, Shenglan and Feng, Lin},
  journal={arXiv preprint arXiv:2404.17883},
  year={2024}}
</code></pre>
