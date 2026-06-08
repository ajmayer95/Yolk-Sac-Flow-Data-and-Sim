# Somites15 demo bundle

Stage-15 chick yolk-sac flow demonstration, 5 tiles around the
canonical arterial and venous reference vessels.

See `LAUNCH.txt` for setup and launch instructions.

This bundle ships with the PerTileFlow viewer including the new SNR
metrics (`snr_harm_fit_db`, `snr_ac_fit_db`) and the corrected f0
source priority (PIV over kymograph).  The graph here has been through
the `repair_f0_trajectory` correction so every measurement is at the
right cardiac fundamental, not on a 2× or 3× harmonic.
