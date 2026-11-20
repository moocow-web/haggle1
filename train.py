import pandas as pd

train_path = "train.csv"
train_data = pd.read_csv(train_path)
train_data.columns
print(train_data.hours_spent)

train_data = train_data.dropna()

y = train_data.hours_spent

features = ["num_commits", "active_commit_days", "project_span_days", "commit_cadence", "pct_commits_weekend", "pct_commits_late_night", "lines_added_total", "lines_deleted_total", "churn_ratio", "median_lines_per_commit", "max_lines_per_commit", "pct_code_in_first_commit", "num_commit_authors", "lines_of_code", "num_source_files", "num_directories", "max_dir_depth", "num_dependencies", "uses_game_engine", "num_test_files", "has_backend", "has_database", "num_image_files", "num_audio_files", "num_3d_model_files", "num_font_files", "asset_size_mb", "has_custom_art", "readme_word_count", "has_docs_folder", "url_is_live", "is_mobile_responsive", "is_clone_or_tutorial", "num_releases", "author_prior_projects"]