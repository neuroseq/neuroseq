# Examples

The examples provide a list of brian2 modeles. The first set of models is tailored to the real neurobiological experiments, and the description of the models contains addtional information on the relation of the model architecture and properties with real experiments. The parameters of the models in the examples and initial conditions can be deifned and modified in "File -> Edit Model".

## Single Hodgkin-Huxley Neuron (Singe HH Neuron)

### Stimulaiton protocol

In this example, a experimental protocol of electrophysiological and optogenetic interogation of single neuron is presented. The protocol consists of a current injections, ramps and steps, of varying amplitude and duration. Optogenetic activation and silencing. 

### Computational model

A single Hodgkin-Huxley neuron is simulated. The model supports direct current injeciton into the neuron (virtual current clamp) and optogenetic control via a pair of opsins (for excitation and inhibition). The injected current into the neuron can be controled via *I_external* stimulation array that supports both positive and negative current values with units expressed in *pA*. The neuron is complemented by two additional conductences that can be modulated by light that represent optogenetic tools (opsins). As no intrinsic kinetics of opsins is taken into account, the opsins are considered to be ideal and have an immediate activation and deactivation. *Exc_stimulation* and *Inh_stimulaiton* are corresponding arrays with units beeing a fraction of predefined maximum conductances of opsin.

**Reversal potintials and conductances defined in the model:**

$$E_{rev}^{exc} = 0   mV$$ \
$$E_{rev}^{inh} = - 80   mV$$ \
$$g_{exc} = 2 \ mS \ cm^{-2}$$ \
$$g_{inh} = 2 \ mS \ cm^{-2}$$

No any other expternal imputs (e.g. gausian noise) was implemented in the model. The model can be configured for non-neuronal cell by setting up conductances for $$Na$$ and $$K$$ voltage gated channels to zero.

### Correspoding experiment

This example is a simplified version of the experiment when a whole-cell patch clamp recording of a single neuron is conducted, and any synaptic transmission is absenend (e.g. when synaptic blockes are added). 

## Single Leaky-Integrate-and-Fire Neuron (Singe LIF Neuron)

A simplified version of a *Singe HH Neuron* model with Leaky-Integrate-and-Fire model used instead of Hodgkin-Huxley.


