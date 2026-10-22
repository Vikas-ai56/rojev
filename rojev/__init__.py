"""rojev: System-1 / System-2 robot control SDK (docs/CODEBASE.md).

Packages, bottom to top (each depends only on those above it in this list):
  protocol       shared contracts: observations, commands, decisions
  robot          robot layer: Robot, limits, ActionSpec; robot.sim = MuJoCo backend
  control        real-time loop, execution runtime, guard
  system1        visuomotor policy contract, remote client, chunk runner
  system2        decider contract, decision worker, OneJev / MJev clients
  owner          state owner: the single writer of intent
  eval           baseline deciders (evaluation only)
  observability  tracing
  legacy_v2      quarantined v2 chunk-synchronous supervisor (deleted after roadmap T8)
"""

__version__ = "0.0.1"
