import polars as pl
import time

def run_quality_control(data: pl.LazyFrame) -> dict:
    """
    Executes all quality control algorithms and data grouping using pure Polars.
    Accepts a LazyFrame to optimize memory performance on large datasets.
    """
    qc_results = {}
    schema = data.collect_schema()
    datetime_columns = [
        column for column in (
            'DEPLOY_UTC', 'RETRIEVE_UTC', 'VALID_DATA_START',
            'VALID_DATA_END', 'TIMESTAMP'
        )
        if schema[column] == pl.String
    ]
    if datetime_columns:
        data = data.with_columns(
            pl.col(datetime_columns).str.to_datetime(time_zone='UTC')
        )
    
    # =========================================================================
    # PHASE 1: COUNTS, NULL CHECKS, & GLOBAL RANGES (UNIFIED PASS 1)
    # =========================================================================
    print("\n" + "="*60)
    print(" PHASE 1/4: Baseline Stats, Null Analysis, & Global Data Ranges")
    print("="*60)
    print("-> Designing unified execution plan for phase 1 metrics...")
    
    phase1_start = time.time()

    # 1. Base counts expressions
    base_exprs = [
        pl.len().alias("total_records"),
        pl.col('INSTFIELDSUMMARYID').n_unique().alias("unique_series_count")
    ]

    # 2. Null detection expressions
    null_cols = [
        'OCC_SITEID', 'LATITUDE', 'LONGITUDE', 'DEPTH_M', 'REGION', 'LOCATION',
        'INSTRUMENT_TYPE', 'DEPLOY_CRUISE', 'DEPLOY_UTC', 'RETRIEVE_CRUISE',
        'RETRIEVE_UTC', 'VALID_DATA_START', 'VALID_DATA_END'
    ]
    null_exprs = [pl.col(col).is_null().any().alias(f"null_{col}") for col in null_cols]

    # 3. Data coordinate range bounds expressions
    vars_list = ["Latitude", "Longitude", "Depth", "DeployUTC", "RetrieveUTC", "ValidDataStart", "ValidDataEnd", "TimeStamp", "Temperature"]
    cols_list = ['LATITUDE', 'LONGITUDE', 'DEPTH_M', 'DEPLOY_UTC', 'RETRIEVE_UTC', 'VALID_DATA_START', 'VALID_DATA_END', 'TIMESTAMP', 'TEMP_C']
    
    # Apply type casting safely up front
    data_clean = data.with_columns([
        pl.col('LATITUDE').cast(pl.Float64, strict=False),
        pl.col('LONGITUDE').cast(pl.Float64, strict=False),
        pl.col('DEPTH_M').cast(pl.Float64, strict=False),
    ])

    range_exprs = []
    for c in cols_list:
        range_exprs.append(pl.col(c).min().alias(f"{c}_min"))
        range_exprs.append(pl.col(c).max().alias(f"{c}_max"))

    print("-> Streaming 9 GB file for Phase 1 targets (Estimated: 15-45s)...")
    p1_stream_start = time.time()
    
    master_metrics_df = data_clean.select(
        base_exprs + null_exprs + range_exprs
    ).collect(streaming=True)
    
    print(f"-> Phase 1 stream completed in {time.time() - p1_stream_start:.2f} seconds!")
    print("-> Structuring output metadata frames...")

    # Extract Counts
    qc_results['total_records'] = master_metrics_df["total_records"][0]
    qc_results['unique_series_count'] = master_metrics_df["unique_series_count"][0]

    # Extract Null Check Dictionary
    null_checks_dict = {}
    for col in null_cols:
        is_bad = master_metrics_df[f"null_{col}"][0]
        null_checks_dict[col] = "Not OK" if is_bad else "OK"
        
    qc_results['null_checks'] = pl.DataFrame({
        "": list(null_checks_dict.keys()),
        "OK?": list(null_checks_dict.values())
    })

    # Extract Bound Ranges Dataframe
    mins = []
    maxs = []
    
    for c in cols_list:
        min_val = master_metrics_df[f"{c}_min"][0]
        max_val = master_metrics_df[f"{c}_max"][0]
        mins.append(min_val.strftime('%Y-%m-%d %H:%M:%S') if hasattr(min_val, 'strftime') else str(min_val) if min_val is not None else "")
        maxs.append(max_val.strftime('%Y-%m-%d %H:%M:%S') if hasattr(max_val, 'strftime') else str(max_val) if max_val is not None else "")

    qc_results['bounds'] = pl.DataFrame({
        "": pl.Series(vars_list, dtype=pl.String),
        "Min": pl.Series(mins, dtype=pl.String),
        "Max": pl.Series(maxs, dtype=pl.String)
    })
    
    print(f"   * Metadata mapped: {qc_results['total_records']} rows, {qc_results['unique_series_count']} unique series found.")
    print(f"✔ Completed Phase 1 processing in {time.time() - phase1_start:.2f}s total.")


    # =========================================================================
    # PHASE 2: CARDINALITY MAPPINGS & CRUISE DATES (UNIFIED PASS 2)
    # =========================================================================
    print("\n" + "="*60)
    print(" PHASE 2/4: Instrument Cardinality & Cruise Range Mappings")
    print("="*60)
    print("-> Designing unified aggregation plan for device grouping...")
    
    phase2_start = time.time()

    # Create distinct grouped plans that Polars can optimize concurrently
    summary_deploys_lazy = data.group_by('INSTFIELDSUMMARYID').agg([
        pl.col('INSTRUMENTSN').unique().cast(pl.String).str.join(", ").alias('Instruments'),
        pl.col('INSTRUMENTSN').n_unique().alias('Number')
    ])

    summary_sns_lazy = data.group_by('INSTRUMENTSN').agg([
        pl.col('INSTFIELDSUMMARYID').unique().cast(pl.String).str.join(", ").alias('IDs'),
        pl.col('INSTFIELDSUMMARYID').n_unique().alias('Number')
    ])

    instrument_type_lazy = data.select(pl.col('INSTRUMENT_TYPE').drop_nulls().unique())
    deploy_cruise_lazy = data.group_by('DEPLOY_CRUISE').agg(range=pl.format("{} to {}", pl.col('DEPLOY_UTC').min(), pl.col('DEPLOY_UTC').max()))
    recover_cruise_lazy = data.group_by('RETRIEVE_CRUISE').agg(range=pl.format("{} to {}", pl.col('RETRIEVE_UTC').min(), pl.col('RETRIEVE_UTC').max()))

    print("-> Streaming 9 GB file for Phase 2 groupings...")
    p2_stream_start = time.time()

    # Collect them smoothly using parallelized streaming structures
    summary_deploys = summary_deploys_lazy.collect(streaming=True)
    summary_sns = summary_sns_lazy.collect(streaming=True)


    qc_results['instrument_type_unique'] = instrument_type_lazy.collect()['INSTRUMENT_TYPE'].to_list()
    qc_results['summary_deployUTC'] = deploy_cruise_lazy.collect(streaming=True)
    qc_results['summary_recoverUTC'] = recover_cruise_lazy.collect(streaming=True)

    print(f"-> Phase 2 stream completed in {time.time() - p2_stream_start:.2f} seconds!")

    # Post-process the small resulting frames
    problems_id = summary_deploys.filter(pl.col('Number') > 1)
    qc_results['problems_id'] = problems_id
    qc_results['problems_id_display'] = problems_id.rename({"INSTFIELDSUMMARYID": "MISSIONINSTRUMENTID"})

    problems_sn = summary_sns.filter(pl.col('Number') > 1)
    qc_results['problems_sn'] = problems_sn
    qc_results['problems_sn_display'] = problems_sn.rename({"IDs": "MISSIONINSTRUMENTIDs"})

    print(f"   * Checked mapping overlaps: Found {len(problems_id)} ID mismatches, {len(problems_sn)} SN mismatches.")
    print(f"✔ Completed Phase 2 processing in {time.time() - phase2_start:.2f}s total.")


    # =========================================================================
    # PHASE 3: TIMELINE COMPLIANCE SCAN (UNIFIED PASS 3)
    # =========================================================================
    print("\n" + "="*60)
    print(" PHASE 3/4: Logical Timeline Violation Scanning")
    print("="*60)
    print("-> Constructing row conditional rules for streaming pass...")
    
    phase3_start = time.time()
    
    timeline_lazy = data.select([
        pl.col('INSTRUMENTSN'),
        (pl.col('VALID_DATA_START') >= pl.col('DEPLOY_UTC')).alias('VALIDDATASTART_QC'),
        (pl.col('VALID_DATA_END') <= pl.col('RETRIEVE_UTC')).alias('VALIDDATAEND_QC'),
        (pl.col('TIMESTAMP') >= pl.col('VALID_DATA_START')).alias('TIMESTAMP_START_QC'),
        (pl.col('TIMESTAMP') <= pl.col('VALID_DATA_END')).alias('TIMESTAMP_END_QC'),
        ((pl.col('TEMP_C') < 15.0) | (pl.col('TEMP_C') > 40.0)).alias('TEMP_OUT_BOUNDS')
    
    ])

    print("-> Streaming 9 GB file for Phase 3 checks...")
    p3_stream_start = time.time()

    errors_df = (
        timeline_lazy.group_by('INSTRUMENTSN')
        .agg([
            pl.col('INSTRUMENTSN').filter(pl.col('VALIDDATASTART_QC') == False).unique().alias('start_err'),
            pl.col('INSTRUMENTSN').filter(pl.col('VALIDDATAEND_QC') == False).unique().alias('end_err'),
            pl.col('INSTRUMENTSN').filter(pl.col('TIMESTAMP_START_QC') == False).unique().alias('t_start_err'),
            pl.col('INSTRUMENTSN').filter(pl.col('TIMESTAMP_END_QC') == False).unique().alias('t_end_err'),
            pl.col('INSTRUMENTSN').filter(pl.col('TEMP_OUT_BOUNDS') == True).unique().alias('temp_err')
        ])
        .collect(streaming=True)
    )



    print(f"-> Phase 3 stream completed in {time.time() - p3_stream_start:.2f} seconds!")
    print("-> Extracting precise list instances...")

    t0 = time.time()
    qc_results['validstart_errors'] = errors_df['start_err'].explode().drop_nulls().unique().to_list()
    print(f"   [Done] Extracted Valid Start errors ({time.time() - t0:.4f}s) -> Found {len(qc_results['validstart_errors'])}")

    t0 = time.time()
    qc_results['validend_errors'] = errors_df['end_err'].explode().drop_nulls().unique().to_list()
    print(f"   [Done] Extracted Valid End errors ({time.time() - t0:.4f}s) -> Found {len(qc_results['validend_errors'])}")

    t0 = time.time()
    qc_results['time_start_errors'] = errors_df['t_start_err'].explode().drop_nulls().unique().to_list()
    print(f"   [Done] Extracted Timestamp Start errors ({time.time() - t0:.4f}s) -> Found {len(qc_results['time_start_errors'])}")

    t0 = time.time()
    qc_results['time_end_errors'] = errors_df['t_end_err'].explode().drop_nulls().unique().to_list()
    print(f"   [Done] Extracted Timestamp End errors ({time.time() - t0:.4f}s) -> Found {len(qc_results['time_end_errors'])}")

    qc_results['temp_range_errors'] = errors_df['temp_err'].explode().drop_nulls().unique().to_list()
    print(f"   [Done] Extracted Temperature Range errors -> Found {len(qc_results['temp_range_errors'])} anomalies")
    
    qc_results['duplicates'] = []
    print(f"✔ Completed Phase 3 processing in {time.time() - phase3_start:.2f}s total.")


    # =========================================================================
    # PHASE 4: INTERVAL STEP OFFSETS (UNIFIED PASS 4)
    # =========================================================================
    print("\n" + "="*60)
    print(" PHASE 4/4: Sub-Group Observation Step & Interval Scanning")
    print("="*60)
    print("-> Setting up intra-group time offsets...")
    
    phase4_start = time.time()

    intervals_lazy = (
        data.group_by(['INSTFIELDSUMMARYID', 'INSTRUMENTSN', 'LOCATION'])
        .agg([
            pl.col('VALID_DATA_START').first().alias('VALID_DATA_START'),
            pl.col('VALID_DATA_END').first().alias('VALID_DATA_END'),
            pl.col('TIMESTAMP').min().alias('TIME1'),
            pl.col('TIMESTAMP').sort().slice(1, 1).first().alias('TIME2'),
            pl.len().alias('OBS_ACTUAL')
        ])
    )

    print("-> Streaming 9 GB file for Phase 4 interval scans...")
    p4_stream_start = time.time()

    intervals = intervals_lazy.collect(streaming=True)
    print(f"-> Phase 4 stream completed in {time.time() - p4_stream_start:.2f} seconds!")

    # Calculate final downstream delta steps on the aggregated small layout
    intervals = intervals.with_columns([
        ((pl.col('TIME2') - pl.col('TIME1')).dt.total_seconds() / 60.0).round(0).alias('INTERVAL'),
        ((pl.col('VALID_DATA_END') - pl.col('VALID_DATA_START')).dt.total_seconds() / 60.0).ceil().alias('TIMELOGGING')
    ]).with_columns([
        # 🛠️ CHANGE HERE: Safe division check to prevent dividing by zero
        pl.when(pl.col('INTERVAL') == 0)
        .then(pl.lit(float('nan')))
        .otherwise((pl.col('TIMELOGGING') / pl.col('INTERVAL')).ceil())
        .alias('OBS_EXPECTED')
    ]).with_columns([
        (-1.0 * (pl.col('OBS_EXPECTED') - pl.col('OBS_ACTUAL'))).alias('OBS_DIFF')
    ])

    intervals_problems = intervals.filter(
        (pl.col('OBS_DIFF').abs() >= 3) | pl.col('OBS_DIFF').is_nan()
    )
    
    qc_results['intervals_problems_table'] = intervals_problems.select(["INSTFIELDSUMMARYID", "INSTRUMENTSN", "LOCATION", "OBS_DIFF"])
    qc_results['intervals_table'] = intervals.select([
        pl.col("INSTFIELDSUMMARYID").alias("MISSIONINSTRUMENTID"),
        pl.col("INSTRUMENTSN"),
        pl.col("LOCATION").alias("ISLAND"),
        pl.col("VALID_DATA_START").alias("VALIDDATASTART"),
        pl.col("VALID_DATA_END").alias("VALIDDATAEND"),
        pl.col("INTERVAL")
    ])

    print(f"   * Completed sampling checks: Found {len(intervals_problems)} anomalous drop streams.")
    print(f"✔ Completed Phase 4 processing in {time.time() - phase4_start:.2f}s total.")
    print("\n" + "="*60 + "\n QC ROUTINE COMPLETED SUCCESSFULLY \n" + "="*60)

    return qc_results