A quick read me on data types:

Suite2p: Generated data on neuron behavior
Application use: Generally imported as F.npy / Fneu.npy / Spkys.npy (typically together)
Andreys description: F.npy, Fnew.npy and spk.npy, fluorescene should be ploted as F-0.7Fneu, Spikes should be plotted in the same way as fluorescence, 900 time points, 30 Hz sampling rate, 67 neurons
These IDs match the cellpose IDs for cellpose data, if it was generated artificially, and not from a TIF file of real neuron

Cellpose data: The segmented pixel locations of cells from an image. The image can be either real or generated (in which case it will be compatible with and generated during the same flow as the seurat "artificial experiemnt" flow)
Typical sizes: 512x512 or 140x140. TIF file of the field of view of Neurons. We use the mean intensity.

Patchclamp data: Trace data from 1 or possibly more neurons. Spikes, as is always the case, can be derived. The neuron ID in the filename corresponds to the ROI (cellpose ID)
in the masks file (note that we process the cellpose ID and shift it down by one, so an ID of "10" in the patchclamp file should be inserted to cellpose ID number 9)
Andreys description: 10 kHz Sämling rate, the neuronID10 corresponds to ROI index from masks file.
