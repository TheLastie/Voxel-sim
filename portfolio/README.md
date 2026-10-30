# Physical portfolio capture: friction and heat

The GIF shows the measured motion and thermal ledger for an oak 3×3×3 cm REV
rolling on a concrete track. It is a visualization of the saved solver output,
not a hand-authored trajectory.

- GIF: [`friction_and_heat.gif`](friction_and_heat.gif)
- Source states: [`friction_heat_road.csv`](friction_heat_road.csv), 151 states over 1.5 s of simulated time
- Rebuild the GIF with `python portfolio/render_friction_heat.py`
- GIF playback is presentation-paced at 8× slower than the physical time axis; each frame uses one saved row, without interpolated states.

The cube slows from 0.300 m/s to rest. The run ends with about 0.573 mJ of
measured friction heat, a mean temperature rise of about 0.0226 mK, and no
broken bonds. The small temperature change is the measured result and is shown
without amplification in the numeric readout.
