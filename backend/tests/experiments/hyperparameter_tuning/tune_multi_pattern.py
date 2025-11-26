from multi_pattern_tuning import MultiPatternTuner, generate_all_patterns

# 1. Generate all 6 test patterns
patterns = generate_all_patterns(n_days=300)
# Creates: trend, mean_reversion, momentum, complex, sideways, volatile

# 2. Create multi-pattern tuner
tuner = MultiPatternTuner(
    fixed_config_path='fixed_config.json',
    tunable_config_path='tunable_config.json',
    results_dir='multi_pattern_results'
)

# 3. Find config that maximizes AVERAGE across all patterns
best_config = tuner.random_search(
    patterns_data=patterns,
    n_iter=20,
    metric='direction_accuracy',
    optimization_target='average'  # Maximizes average score
)

print(f"Best average score: {tuner.best_avg_score:.2f}%")