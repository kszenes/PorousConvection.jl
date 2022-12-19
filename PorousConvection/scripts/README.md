# 2D diffusion solver @parallel vs @parallel_indices

Both implementations perform similarly, however in a small region of problem sizes `@parallel_indices` has a slight edge over `@parallel`.

![](../docs/parallel_vs_indices.png)