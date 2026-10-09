"""The lower bound on reconstruction error (sec:entropy, prop:coverage).

  blocks.py       orthogonal rotations U (Hadamard factors; eigenbasis or random signs) and the k-blocks
  plugin.py       the Goldfeld et al. plug-in entropy estimator and its margins t_MC, t_MD
  grid.py         the confidence-bound entropy h_LB(k, nu^2) over the grid G  -> entropy_bound.json
  lpips_phi.py    the LPIPS feature map Phi (||Phi(x) - Phi(x')||^2 = LPIPS)
  lpips_grid.py   the same grid for Phi(x)
  rho.py          rho* from gamma and the grid; the read-out for a DP-SGD schedule
  reference_scale.py  confidence bounds on the reference scale tr Sigma
"""
