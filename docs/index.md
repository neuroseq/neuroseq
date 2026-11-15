# NeuroSeq

NeuroSeq is a neuronal sequencer: it controls the input to a neural network and reads out its output, whether the network is simulated, biological or neuromorphic. The same protocol and analysis run on all three. Source: [github.com/neuroseq/neuroseq](https://github.com/neuroseq/neuroseq).

![NeuroSeq graphical user interface](https://raw.githubusercontent.com/neuroseq/neuroseq/main/docs/images/screenshot.png)

## Concept

For any network, NeuroSeq splits the **input** into a controlled part (set by you) and an uncontrolled part, and the **output** into an observed part (read out) and an unobserved part. An experiment is a loop:

1. Design an input array (one row per neuron).
2. Deliver it to a network.
3. Read out the response and match it to the same neuron IDs.
4. Analyse, and optionally choose the next input from the result.

Because each step is substrate independent, a protocol can be developed in simulation and then run unchanged on a microscope or a neuromorphic chip.

## Where to start

| I want to...                     | Read                                                                                         |
| -------------------------------- | -------------------------------------------------------------------------------------------- |
| Install and try it               | [Installation](getting_started/installation.md), [Quickstart](getting_started/quickstart.md) |
| Learn the main window            | [GUI overview](getting_started/gui_overview.md)                                              |
| See which networks are supported | [Networks](networks/overview.md)                                                             |
| Design input                     | [Input design](input_design/overview.md)                                                     |
| Import recordings                | [Readout](readout/overview.md)                                                               |
| Analyse data                     | [Data analysis](data_analysis/analysis.md)                                                   |
| Open a ready-made project        | [Examples](examples.md)                                                                      |
| Extend NeuroSeq                  | [Developer guide](developer/developer_installation.md)                                       |
| Fix a problem                    | [Troubleshooting](common_issues.md)                                                          |

## License

[MIT](https://github.com/neuroseq/neuroseq/blob/main/LICENSE). Copyright © 2024-2026 Andrey Formozov. Contributors: [AUTHORS](https://github.com/neuroseq/neuroseq/blob/main/AUTHORS.md).
