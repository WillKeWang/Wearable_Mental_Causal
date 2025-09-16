from data_utils import ProcessingConfig, run_preprocessing

# Standard backward-looking analysis (30 days before survey)
config = ProcessingConfig(
    window_start_offset=-30,  # 30 days before survey
    window_end_offset=0,      # up to survey date
    baseline_enabled=True
)

main_file, baseline_file = run_preprocessing(config)
print(f"Results saved to: {main_file}")