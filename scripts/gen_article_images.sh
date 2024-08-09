declare -a arr=(
    # "jitter_vs_no_jitter"
    # "simple_retry_vs_backoff_closed_loop"
    # "retry_cb_vs_budget"
    "circuit_breaker_vs_retry_budget/cb_threshold=50%/fail_rate=20%"
    "circuit_breaker_vs_retry_budget/cb_threshold=10%/fail_rate=20%"
    # "retry_cb_or_budget_vs_exp_backoff_partial_failure"
    # "retry_cb_or_budget_vs_exp_backoff"
    # "any_retries_are_slowing_down_recovery"
    # "jitter_vs_no_jitter_cpu_half"
    # "dp_vs_retry_budget"
    # "exponential_backoff_just_delays_problem_good_closed_loop"
    # "exponential_backoff_just_delays_problem_bad_closed_loop"
    # "simple_retry_vs_backoff"
    # "exponential_backoff_just_delays_problem_open_loop"
    # "simple_retry_only"
)

# rm .images/*.png
for sim_name in "${arr[@]}"
do
   LANG=en ./scripts/run_pypy.sh ${sim_name} 300000
done


