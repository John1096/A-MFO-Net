# Selected paper figures

These six PNGs were copied unchanged from the final submitted manuscript's
`figures/` directory. They document the completed A-MFO-Net experiment reported
in that manuscript. They do not show results from the later full-window training
campaign. The PNGs contain rendered diagrams, plots, and visualizations, not CFD
field arrays or prediction files.

| File | What it shows | Manuscript location |
| --- | --- | --- |
| `room_domain_setup_annotated.png` | Room-scale and extended-room data-center geometries, including racks, air-conditioning units, hot aisles, and false ceilings. | Experimental protocol, domain figure |
| `attention_notation_v2.png` | Learned node-edge attention and degree-mean message aggregation for one target mesh cell. | Method, attention figure |
| `pipeline_notation_v2.png` | A-MFO-Net graph encoder, regular-grid 3-D FNO, and graph decoder, with a legend. | Method, architecture figure |
| `extended_room_flow_pressure_primary.png` | Ground truth and seed-42 A-MFO-Net rollouts at steps 20, 200, and 400, alongside the attention-disabled MFO-Net, FNO, and compact GINO controls. | Results, qualitative rollout figure |
| `extended_room_error_maps_multiview.png` | A-MFO-Net velocity, pressure, and temperature absolute-error maps on two mesh slices at forecast step 400. Color limits are clipped at each field's 99th percentile for display. | Results, error-map figure |
| `extended_room_step400_baseline_comparison.png` | Final-step relative-L2 error means and sample standard deviations over seeds 42–44 for A-MFO-Net and the manuscript's comparison models. The FNO pressure lower whisker is clipped at the 1% plot boundary. | Appendix, final-step comparison figure |

The submitted study used a held-out extended-room test trajectory with 90% ACU
cooling and 100% server workload, evaluated for 400 forecast steps at a
0.05-second cadence. A-MFO-Net was trained for 100 epochs with 200 selected
three-step training windows per epoch. Its final-step statistics in the
comparison plot summarize seeds 42, 43, and 44. The rollout panels use seed 42.
See [the metric CSVs](../metrics/README.md) for A-MFO-Net's per-step values.

Only paper-ready PNGs are included here. The comparison images show other
models for scientific context; no other model's source, weights, or metric CSVs
are included.
