"""Ablations of ClosersRoles2L_compact288 (each one-line change, A=12 unless noted):
  compact288_A8            A=8                      -> 1.00
  compact288_A3p5          A=3.5                    -> 0.285
  ablate_no_answer_gate    ans=[] (no answer-region penalty on day-closer unit) -> 0.825
  ablate_no_B_flag         W_o of head 4 zeroed (B flag off)                    -> 0.93
"""
