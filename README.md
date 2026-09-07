# AP-PEN

The convenience yield, the implied benefit of holding physical commodities, is a
latent state inferred only from the term structure of futures prices.
Traditional methods include the Kalman Filter (KF), which assumes
linear-Gaussian dynamics and never verifies the recovered states against the
governing partial differential equation (PDE). We introduce the
Affine-Partitioned, Physics-Enforced Network (AP-PEN), which inverts the
two-factor Gibson-Schwartz model for the instantaneous convenience yield of West
Texas Intermediate (WTI) crude oil futures. Rather than penalising the system
for breaching the governing PDE, AP-PEN imposes the model's affine coefficients
analytically, so a single network need only parameterise the latent convenience
yield path. The resulting architecture satisfies the PDE by design, leaving no
residual loss to learn. Physics instead enters through a transition likelihood
on the recovered path under the physical measure, alongside no-arbitrage and
economic inequality constraints balanced by a gradient-norm scheme adapted from
Hoshisashi et al.'s Whack-a-mole Online Learning (WamOL). Validated on simulated
and real WTI data (2015-2026), AP-PEN's no-arbitrage variant uniquely avoids
collapsing into implausible estimates during the April 2020 market dislocation,
tracking the KF three times closer than a naive constant-average baseline, but
at the cost of degenerate structural parameters. On clean synthetic data, these
constraints prove counterproductive, with the closed-form least squares estimate
outperforming every network variant. This work extends WamOL and
derivative-constrained physics-informed neural networks to commodities,
contributing an identifiability analysis and a hard-constrained estimator rather
than a residual-based architecture.
