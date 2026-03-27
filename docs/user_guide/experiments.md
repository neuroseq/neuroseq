# Experiments

## Experiment workflow

1. Create experiment definition.
2. Attach model and stimulus arrays.
3. Run and record outputs.
4. Export and compare results.

## Managing experiment sets

Use experiment manager dialogs to keep runs organized and reproducible.

To generate synthetic Cellpose test data (TIFF image + matching `.npy` segmentation):

```bash
neuroseq --generate-cellpose-test-data ./cellpose_test_grid.tif
```
