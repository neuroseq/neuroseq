# Multiphoton Holographic Stimulation

# Generate test .tif file

neuroseq --generate-cellpose-test-data ./cellpose_512.tif \
 --cellpose-grid-rows 10 \
 --cellpose-grid-cols 10 \
 --cellpose-radius 11 \
 --cellpose-gap 30

# Generate square test .tif file with auto-fitted radius/gap

neuroseq --generate-cellpose-test-data ./cellpose_1300.tif \
 --cellpose-grid-rows 10 \
 --cellpose-grid-cols 10 \
 --cellpose-square-size 1300
