# Examples

The examples provide a list of brian2 modeles. The first set of models is tailored to the real neurobiological experiments, and the description of the models contains addtional information on the relation of the model architecture and properties with real experiments. The parameters of the models in the examples and initial conditions can be deifned and modified in "File -> Edit Model".

## Hodgkin-Huxley Neuron

<img width="1441" height="433" alt="image" src="https://github.com/user-attachments/assets/cb9b8c37-934f-4631-bb93-139b9c910a65" />

### Stimulaiton protocol

In this example, a experimental protocol of electrophysiological and optogenetic interogation of single neuron is presented. The protocol consists of a current injections, ramps and steps, of varying amplitude and duration: two postive current 100 ms ramps with 300 pA and 600 pA maximum amplitude, three 100 ms steps of 200, 400 and 600 pA, two ramps (100 ms, 50% and 100% of light power, respectively) and three optogenetic stimulaiton steps for excitatory and inhibitory opsins (25%, 50% and 100% of light power).

### Computational model

A single Hodgkin-Huxley neuron is simulated. The model supports direct current injeciton into the neuron (virtual current clamp) and optogenetic control via a pair of opsins (for excitation and inhibition). The injected current into the neuron can be controled via _I_external_ stimulation array that supports both positive and negative current values with units expressed in _pA_. The neuron is complemented by two additional conductences that can be modulated by light that represent optogenetic tools (opsins). As no intrinsic kinetics of opsins is taken into account, the opsins are considered to be ideal and have an immediate activation and deactivation. _Exc_stimulation_ and _Inh_stimulaiton_ are corresponding arrays with units beeing a fraction of predefined maximum conductances of opsin.


**Reversal potintials and conductances defined in the model:**

$$E_{rev}^{exc} = 0   mV$$ \
$$E_{rev}^{inh} = - 80   mV$$ \
$$g_{exc} = 0.1 \ mS \ cm^{-2}$$ \
$$g_{inh} = 0.1 \ mS \ cm^{-2}$$

No any other expternal imputs (e.g. gausian noise) was implemented in the model. The model can be configured for non-neuronal cell by setting up conductances for $$Na$$ and $$K$$ voltage gated channels to zero.

### Correspoding experiment

This example is a simplified version of the experiment when a whole-cell patch clamp recording of a single neuron is conducted, and any synaptic transmission is absenend (e.g. when synaptic blockes are added).

## Leaky-Integrate-and-Fire Neuron

A simplified, Leaky-Integrate-and-Fire instead of Hodgkin-Huxley is used. The units for stimulation arrays are "$$mW/\mu m^2$$".
