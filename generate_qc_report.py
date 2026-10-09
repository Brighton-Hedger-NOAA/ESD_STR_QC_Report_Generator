import os
import io
import html
import math
import time
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv
import polars as pl
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

try:
    import oracledb
except ImportError:  # pragma: no cover - installed as needed for Oracle mode
    oracledb = None

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, PageBreak

# Custom report utilities and logic layout imports
from scripts.pdf_report_utils import title_style, h2_style, normal_style, code_style, df_to_pdf_table, ParagraphStyle
from scripts.qc_validation import run_quality_control

# Import the extracted presentation layer function
from scripts.pdf_content_builder import build_pdf_story

# Utility function to generate a manifest file for production output
from scripts.export_utils import generate_readme, zip_production_artifacts


load_dotenv()


def get_database_config():
    """Read Oracle connection settings from environment variables.

    A local .env file can hold credentials without hardcoding secrets into the repo.
    """
    return {
        "user": os.getenv("ORACLE_USER"),
        "password": os.getenv("ORACLE_PASSWORD"),
        "dsn": os.getenv("ORACLE_DSN"),
    }


def load_oracle_select(sql_query, config=None):
    """Load STR data by executing a SQL SELECT against Oracle."""
    if oracledb is None:
        raise ImportError(
            "The 'oracledb' package is required for Oracle database inputs. "
            "Install it with: pip install oracledb"
        )

    instant_client_dir = r"C:\Oracle\InstantClient\instantclient_23_7"
    if os.path.exists(instant_client_dir):
        oracledb.init_oracle_client(lib_dir=instant_client_dir)

    if config is None:
        config = get_database_config()

    missing = [key for key, value in config.items() if not value]
    if missing:
        raise ValueError(
            "Missing Oracle database settings. Please set the required values in your .env file: "
            + ", ".join(missing)
        )

    print("Loading STR source from Oracle SELECT query...")
    rows = []
    with oracledb.connect(
        user=config["user"],
        password=config["password"],
        dsn=config["dsn"],
    ) as connection:
        with connection.cursor() as cursor:
            cursor.execute(sql_query)
            columns = [col[0] for col in cursor.description]

            batch_size = 1_000_000
            rows_fetched = 0
            while True:
                batch = cursor.fetchmany(batch_size)
                if not batch:
                    break

                rows.extend(batch)
                rows_fetched += len(batch)

                if rows_fetched % batch_size == 0:
                    print(f"Fetched {rows_fetched:,} rows from Oracle query...")

    if not rows:
        return pl.DataFrame(schema=columns).lazy()

    return pl.DataFrame(rows, schema=columns).lazy()


def load_data_source(data_source):
    """Load the STR dataset from either a CSV path or a database query source."""
    if isinstance(data_source, str) and os.path.exists(data_source):
        print(f"Loading STR source from CSV file: {data_source}")
        return pl.scan_csv(
            data_source,
            schema_overrides={"INSTRUMENTSN": pl.String},
            try_parse_dates=True,
        )

    if isinstance(data_source, str) and data_source.strip().upper().startswith("SELECT"):
        return load_oracle_select(data_source, get_database_config())

    raise FileNotFoundError(
        f"Please provide a valid STR source path or SQL SELECT. Current: {data_source}"
    )


def generate_qc_report(str_file, file_prefix, final=False): 
    # Start the global script timer immediately
    script_start_time = time.time()

    print("="*60)
    print(f"STARTING QC REPORT GENERATION: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60)

    # Make sure the data source is present before proceeding; SQL SELECT strings
    # bypass the filesystem check because they are executed directly against Oracle.
    if isinstance(str_file, str) and not str_file.strip().upper().startswith("SELECT"):
        if not os.path.exists(str_file):
            raise FileNotFoundError(f"Please provide a valid path for str_file. Current: {str_file}")
    
    # Setup the output PDF filename based on the file prefix
    pdf_filename = f"{file_prefix}_QC_Report.pdf"

    # Creating output Paths based on the execution mode (final vs. development)

    if final:
        # Production mode: Save the PDF report to the designated output directory and generate split CSV files
        output_dir = Path(file_prefix)
        output_pdf = output_dir / pdf_filename

        # handles if the directory already exists or not, creating it if necessary
        output_dir.mkdir(parents=True, exist_ok=True)


    else:
        # Development mode: Save the PDF report to the current working directory and do not generate split CSV files
        output_pdf = Path(pdf_filename)

        print(f"Dev mode active. Saving report directly to current working directory.")


    # 1. Read everything purely as a streaming source (LazyFrame)
    STR_lazy = load_data_source(str_file)


    # 2. Run QC metrics using the streaming architecture
    print("Running Quality Control metrics...")
    qc = run_quality_control(STR_lazy) 


    # 3. Initialize PDF layout template
    doc = SimpleDocTemplate(str(output_pdf), pagesize=letter, leftMargin=54, rightMargin=54, topMargin=54, bottomMargin=54)
    story = []


    # --- Evaluation Banner Block ---
    all_checks_passed = (
        qc['problems_id'].is_empty() and 
        qc['problems_sn'].is_empty() and 
        len(qc['duplicates']) == 0 and
        len(qc['validstart_errors']) == 0 and 
        len(qc['validend_errors']) == 0 and
        len(qc['time_start_errors']) == 0 and 
        len(qc['time_end_errors']) == 0 and
        len(qc['temp_range_errors']) == 0 and
        qc['intervals_problems_table'].is_empty()
    )


    if all_checks_passed:
        success_banner_style = ParagraphStyle(
            'SuccessBanner', parent=normal_style, fontName='Times-Bold', fontSize=11,
            textColor=colors.HexColor("#2F855A"), alignment=TA_CENTER, spaceAfter=15
        )
        story.append(Paragraph("✔ STATUS: All Quality Control Checks Passed Successfully", success_banner_style))
        story.append(Spacer(1, 5))


    # Pack UI formatting layout references to pass to external builder
    report_styles = {
        'title_style': title_style,
        'h2_style': h2_style,
        'normal_style': normal_style,
        'code_style': code_style,
        'df_to_pdf_table': df_to_pdf_table
    }


    # Call external file layout code to construct headers, tables, and charts
    build_pdf_story(story, qc, STR_lazy, report_styles, str_file)

    # 4. Save and compile document structure
    print("All plots rendered. Compiling final PDF document...")
    doc.build(story)
    print(f"Report successfully compiled: {output_pdf}")

    # --- Memory-Efficient Row-Count Slice Chunking (Production Only) ---
    if final:
        print("\n--- Generating Sequential CSV Subsections (Production Architecture) ---")
        
        total_rows = qc['total_records']
        chunk_size = 1_000_000  # 1 Million row limit per file partition
        total_chunks = math.ceil(total_rows / chunk_size)

        # 1. Create the deeper subfolder cleanly tucked away
        split_sub_dir = output_dir / "UNZIPPED_SPLIT"
        split_sub_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"Streaming {total_rows:,} rows across {total_chunks} sequential files into {output_dir}...")

        for i in range(total_chunks):
            offset = i * chunk_size
            chunk_filename = split_sub_dir / f"{file_prefix}_part_{i + 1}.csv"
            
            print(f" Writing chunk {i+1}/{total_chunks}: {chunk_filename.name}")
            
            (
                STR_lazy
                .slice(offset, chunk_size)
                .sink_csv(str(chunk_filename))
            )

        generate_readme(output_pdf, total_rows, file_prefix)
        zip_production_artifacts(output_dir, split_sub_dir, file_prefix, str_file)

    total_execution_time = time.time() - script_start_time

    # Convert seconds to a clean minutes:seconds format if it takes longer
    minutes = int(total_execution_time // 60)
    seconds = total_execution_time % 60

    print("\n" + "="*60)
    if minutes > 0:
        print(f"✔ PROCESS COMPLETED TOTAL RUNTIME: {minutes} min {seconds:.2f} seconds")
    else:
        print(f"✔ PROCESS COMPLETED TOTAL RUNTIME: {total_execution_time:.2f} seconds")
    print("="*60 + "\n")



# -------------------------------------------------------------------------------------------------


if __name__ == "__main__":

    # UPDATE THIS BASEd ON THE REGION AND YEAR YOU ARE PROCESSING
    initiative = "NCRMP"
    organization = "ESD"
    data_type = "TEMPERATURE"
    region = "SAMOA"
    year = "2026"

    file_prefix = f"{initiative}_{organization}_{data_type}_{region}_{year}"

    # -------------------------------------------------------------------------
    # DATA SOURCE BELOW:
    # Use the Oracle view directly for the database-backed workflow.
    # If you want to switch back to a local CSV file, replace this SQL with a file path.
    # -------------------------------------------------------------------------

    oracle_view_sql = """
        SELECT *
        FROM GISDAT.V_NCEI_OCC_STR_PRIA_ERRDAP_2026
    """

    # -------------------------------------------------------------------------
    # TOGGLE EXECUTION MODE HERE
    # Change final=False to final=True when you are ready to archive.
    # -------------------------------------------------------------------------
    generate_qc_report(oracle_view_sql, file_prefix, final=False)
