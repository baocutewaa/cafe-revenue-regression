"""
Package: src
Cafe Revenue Forecasting Project
"""
from .step1_collect_prepare import (
    rmsle,
    load_raw_data,
    build_data_dictionary,
    check_merge_keys,
    prepare_holidays_table,
    safe_merge_tables,
    run_step1
)

from .step2_explore import (
    detect_duplicates,
    detect_s12_1000x_anomaly,
    detect_decimal_errors,
    detect_store_type_inconsistency,
    detect_district_inconsistency,
    detect_open_date_formats,
    detect_stores_missing_area,
    detect_fahrenheit_outliers,
    detect_weather_missing_dates_and_rain_nans,
    build_ten_anomalies_table,
    analyze_business_patterns,
    generate_eda_plots,
    run_step2
)

from .step3_preprocess import (
    clean_stores,
    clean_weather,
    clean_holidays,
    clean_train_outliers,
    compute_ramp_up_factor,
    build_features,
    DeSeasonalizedStoreStats,
    run_step3
)

from .step4_split import (
    get_starter_holdout_split,
    get_expanding_window_folds,
    TimeSeriesExpandingCV,
    compute_weekday_median_baseline,
    evaluate_baseline_across_splits,
    run_step4
)

from .step5_scale import (
    transform_target,
    inverse_transform_target,
    LogTargetRegressorWrapper,
    get_feature_column_types,
    build_feature_preprocessor,
    build_leakage_free_pipeline,
    run_step5
)

from .step6_model_train import (
    WeightedBlendEnsemble,
    get_candidate_models,
    evaluate_predictions,
    train_and_evaluate_all_models,
    run_time_series_cv_benchmark,
    run_step6
)

from .step7_evaluate import (
    compute_ablation_study_table,
    compute_store_type_error_breakdown,
    compute_residual_diagnostics,
    compute_feature_importances,
    generate_evaluation_plots,
    run_step7
)

from .step8_deploy import (
    DEFAULT_ENSEMBLE_WEIGHTS,
    retrain_ensemble_on_full_data,
    generate_kaggle_submission,
    package_artifact_bundle,
    predict_single_instance,
    predict_batch_dataframe,
    run_step8
)

__all__ = [
    # Step 1
    "rmsle",
    "load_raw_data",
    "build_data_dictionary",
    "check_merge_keys",
    "prepare_holidays_table",
    "safe_merge_tables",
    "run_step1",
    # Step 2
    "detect_duplicates",
    "detect_s12_1000x_anomaly",
    "detect_decimal_errors",
    "detect_store_type_inconsistency",
    "detect_district_inconsistency",
    "detect_open_date_formats",
    "detect_stores_missing_area",
    "detect_fahrenheit_outliers",
    "detect_weather_missing_dates_and_rain_nans",
    "build_ten_anomalies_table",
    "analyze_business_patterns",
    "generate_eda_plots",
    "run_step2",
    # Step 3
    "clean_stores",
    "clean_weather",
    "clean_holidays",
    "clean_train_outliers",
    "compute_ramp_up_factor",
    "build_features",
    "DeSeasonalizedStoreStats",
    "run_step3",
    # Step 4
    "get_starter_holdout_split",
    "get_expanding_window_folds",
    "TimeSeriesExpandingCV",
    "compute_weekday_median_baseline",
    "evaluate_baseline_across_splits",
    "run_step4",
    # Step 5
    "transform_target",
    "inverse_transform_target",
    "LogTargetRegressorWrapper",
    "get_feature_column_types",
    "build_feature_preprocessor",
    "build_leakage_free_pipeline",
    "run_step5",
    # Step 6
    "WeightedBlendEnsemble",
    "get_candidate_models",
    "evaluate_predictions",
    "train_and_evaluate_all_models",
    "run_time_series_cv_benchmark",
    "run_step6",
    # Step 7
    "compute_ablation_study_table",
    "compute_store_type_error_breakdown",
    "compute_residual_diagnostics",
    "compute_feature_importances",
    "generate_evaluation_plots",
    "run_step7",
    # Step 8
    "DEFAULT_ENSEMBLE_WEIGHTS",
    "retrain_ensemble_on_full_data",
    "generate_kaggle_submission",
    "package_artifact_bundle",
    "predict_single_instance",
    "predict_batch_dataframe",
    "run_step8"
]
