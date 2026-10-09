import html
import io
import os
import re
from datetime import datetime
import polars as pl
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from reportlab.platypus import Paragraph, Spacer, Image, PageBreak


def _describe_source_name(str_file):
    """Return a human-friendly source label for PDF output."""
    if not str_file:
        return "Unknown source"

    raw = str(str_file).strip()
    if raw.upper().startswith("SELECT"):
        match = re.search(r"\bFROM\s+([A-Za-z0-9_$.]+)", raw, flags=re.IGNORECASE)
        if match:
            source = match.group(1)
            return source.split(".")[-1]
        return "Oracle source"

    return os.path.basename(raw)


def build_pdf_story(story, qc, STR_lazy, styles, str_file):
    """
    Appends all QC text, tables, and generated time-series charts to the ReportLab story.
    """
    # Unpack styles for readability
    title_style = styles['title_style']
    h2_style = styles['h2_style']
    normal_style = styles['normal_style']
    code_style = styles['code_style']
    df_to_pdf_table = styles['df_to_pdf_table']

    # --- Document Header Elements ---
    source_label = _describe_source_name(str_file)
    story.append(Paragraph("STR QC Report Details", title_style))
    story.append(Paragraph("The source used in this QC routine is:", normal_style))
    story.append(Paragraph(f"{html.escape(source_label)}", code_style))
    story.append(Paragraph(f"QC run at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} HST", normal_style))
    story.append(Paragraph(f"Number of records in data file: {qc['total_records']}", normal_style))
    story.append(Paragraph(f"Number of STR time series in data file: {qc['unique_series_count']}", normal_style))
    story.append(Spacer(1, 15))

    # --- Get Data Ranges ---
    story.append(Paragraph("Get data ranges", h2_style))
    story.append(df_to_pdf_table(qc['bounds']))
    story.append(Spacer(1, 15))

    # --- Check for NULLs ---
    story.append(Paragraph("Check for NULLs in all metadata fields", h2_style))
    story.append(df_to_pdf_table(qc['null_checks']))
    story.append(Spacer(1, 15))

    # --- Unique Instrument Check (Check 1) ---
    story.append(Paragraph("Is there more than one STR per MISSIONINSTRUMENTID?", h2_style))
    if qc['problems_id'].is_empty():
        story.append(Paragraph("Only one STR per MISSIONINSTRUMENTID", normal_style))
    else:
        story.append(df_to_pdf_table(qc['problems_id_display']))
    story.append(Spacer(1, 15))

    # --- Unique Serial Number Check (Check 2) ---
    story.append(Paragraph("Is there more than one MISSIONINSTRUMENTID per SN?", h2_style))
    if qc['problems_sn'].is_empty():
        story.append(Paragraph("Only one INSTFIELDSUMMARYID per SN.", normal_style))
    else:
        story.append(Paragraph("The following Instrument SN(s) have more than one INSTFIELDSUMMARYID:", normal_style))
        story.append(df_to_pdf_table(qc['problems_sn_display']))
    story.append(Spacer(1, 15))

    # --- Unique Instrument Type ---
    story.append(Paragraph("Is there more than one instrument type?", h2_style))
    if len(qc['instrument_type_unique']) == 1 and qc['instrument_type_unique'][0] == 'STR':
        story.append(Paragraph("Only STRs in data", normal_style))
    else:
        story.append(Paragraph("These instrument types present", normal_style))
        story.append(Paragraph(f"{', '.join([str(x) for x in qc['instrument_type_unique']])}", normal_style))
    story.append(Spacer(1, 15))

    # --- Cruise Ranges ---
    story.append(Paragraph("Do the deploy and recover dates match the cruises?", h2_style))
    story.append(df_to_pdf_table(qc['summary_deployUTC'].rename({"DEPLOY_CRUISE": "DEPLOYCRUISE", "range": "DEPLOYUTC.range"})))
    story.append(Spacer(1, 5))
    story.append(df_to_pdf_table(qc['summary_recoverUTC'].rename({"RETRIEVE_CRUISE": "RETRIEVECRUISE", "range": "RETRIEVEUTC.range"})))
    story.append(Spacer(1, 15))

    # --- Duplicates ---
    story.append(Paragraph("Are any STRs uploaded twice?", h2_style))
    if len(qc['duplicates']) > 0:
        story.append(Paragraph("The following MISSIONINSTRUMENTID(s) have duplicate values", normal_style))
    else:
        story.append(Paragraph("No duplicates present", normal_style))
    story.append(Spacer(1, 15))

    # --- Timeline Checks ---
    story.append(Paragraph("Are all VALIDDATASTART after DEPLOYUTC?", h2_style))
    if len(qc['validstart_errors']) > 0:
        story.append(Paragraph("STR SNs with VALIDDATASTART before DEPLOYUTC are:", normal_style))
        story.append(Paragraph(f"{qc['validstart_errors']}", normal_style))
    else:
        story.append(Paragraph("All VALIDDATASTART are OK", normal_style))

    story.append(Paragraph("Are all VALIDDATAEND before RETRIEVEUTC?", h2_style))
    if len(qc['validend_errors']) > 0:
        story.append(Paragraph("STR SNs with VALIDDATAEND after RETRIEVEUTC are:", normal_style))
        story.append(Paragraph(f"{qc['validend_errors']}", normal_style))
    else:
        story.append(Paragraph("All VALIDDATAEND are OK", normal_style))

    story.append(Paragraph("Are all TIMESTAMP after VALIDDATASTART?", h2_style))
    if len(qc['time_start_errors']) > 0:
        story.append(Paragraph("STR SNs with TIMESTAMP before VALIDDATASTART are:", normal_style))
        story.append(Paragraph(f"{qc['time_start_errors']}", normal_style))
    else:
        story.append(Paragraph("All TIMESTAMP are OK", normal_style))

    story.append(Paragraph("Are all TIMESTAMP before VALIDDATAEND?", h2_style))
    if len(qc['time_end_errors']) > 0:
        story.append(Paragraph("STR SNs with TIMESTAMP after VALIDDATAEND are:", normal_style))
        story.append(Paragraph(f"{qc['time_end_errors']}", normal_style))
    else:
        story.append(Paragraph("All TIMESTAMP are OK", normal_style))
    story.append(Spacer(1, 15))

    # --- Temperature Range Checks ---
    story.append(Paragraph("Are all recorded temperatures within expected parameters (15°C - 40°C)?", h2_style))
    if len(qc['temp_range_errors']) > 0:
        story.append(Paragraph("<b>WARNING:</b> Anomalous temperature thresholds breached. The following instrument serial numbers recorded observations outside 15°C - 40°C:", normal_style))
        story.append(Paragraph(f"{qc['temp_range_errors']}", code_style))
    else:
        story.append(Paragraph("All sensor readings fall cleanly within normal parameters (15°C - 40°C).", normal_style))
    story.append(Spacer(1, 15))

    # --- Sampling Intervals ---
    story.append(Paragraph("Do all STRs have the expected number of observations?", h2_style))
    if qc['intervals_problems_table'].is_empty():
        story.append(Paragraph("All STRs have expected number of observations", normal_style))
    else:
        story.append(Paragraph("The following STRs do not have the expected number of observations:", normal_style))
        story.append(Paragraph("(negative = observations are missing, positive = too many observations)", normal_style))
        story.append(Spacer(1, 5))
        story.append(df_to_pdf_table(qc['intervals_problems_table']))

    # --- Page Break: Summary Table ---
    story.append(PageBreak())
    story.append(Paragraph("Summary of STRs", h2_style))
    story.append(df_to_pdf_table(qc['intervals_table']))

    # --- Page Break: Time Series Plots ---
    print("Generating localized sensor time-series charts directly from the data stream...")
    story.append(PageBreak())
    story.append(Paragraph("Individual STR plots by SN:", normal_style))
    story.append(Spacer(1, 10))

    print("Downsampling data for localized sensor charts...")

    # Oracle-backed data comes in as ISO strings; normalize the timestamp before
    # the charting pass so Polars can safely truncate it by hour.
    STR_plot_lazy = STR_lazy.with_columns([
        pl.col('TIMESTAMP').str.to_datetime(time_zone='UTC', strict=False),
        pl.col('TEMP_C').cast(pl.Float64, strict=False),
    ])

    # 1. Group by instrument and hour/day to massively compress the data in-stream
    plot_data_lazy = (
        STR_plot_lazy
        .group_by([
            pl.col('INSTFIELDSUMMARYID'),
            pl.col('INSTRUMENTSN'),
            pl.col('LOCATION'),
            pl.col('TIMESTAMP').dt.truncate("1h").alias("ROUNDED_TIME")
        ])
        .agg(pl.col('TEMP_C').mean().alias('TEMP_C'))
        .sort("ROUNDED_TIME")
    )

    # 2. Collect the entire aggregated plotting dataset ONCE (takes seconds)
    all_plots_df = plot_data_lazy.collect(streaming=True)

    # 3. Now efficiently iterate through your small, in-memory plotting dataframe
    for inst_id, data_i in all_plots_df.partition_by("INSTFIELDSUMMARYID", as_dict=True).items():
        sn_i = data_i['INSTRUMENTSN'][0]
        isl_i = data_i['LOCATION'][0]
        
        fig, ax = plt.subplots(figsize=(6.5, 3))
        ax.plot(data_i['ROUNDED_TIME'], data_i['TEMP_C'], color='dodgerblue', linewidth=1)
        
        ax.set_title(f"SN={sn_i}, MIID={inst_id}, {isl_i}", fontsize=10, fontweight='bold')
        ax.set_ylabel("Temperature (°C)", fontsize=9)
        ax.grid(True, linestyle='--', alpha=0.5)
        
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=4))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%y'))
        ax.tick_params(axis='both', which='major', labelsize=8)
        
        plt.tight_layout()
        
        img_buf = io.BytesIO()
        plt.savefig(img_buf, format='png', dpi=200)
        img_buf.seek(0)
        plt.close(fig)
        
        story.append(Image(img_buf, width=450, height=200))
        story.append(Spacer(1, 15))
        
    return all_plots_df