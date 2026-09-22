import os
import tempfile
from pathlib import Path

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from cytogenetics import process_pdf, COLUMNS

load_dotenv()

st.set_page_config(

    page_title="Cytogenetics Report Extractor",

    page_icon="🧬",

    layout="wide",

)

st.title("🧬 Cytogenetics Report Extractor")

st.caption(

    "Extract patient/report fields locally, send karyotype + interpretation "

    "to Gemini for structured events, then send a second Gemini call to "

    "verify that output against the originals."

)

def get_secret(name):

    if name in st.secrets:

        return st.secrets[name]

    return os.getenv(name)

configured_keys = [

    key.strip()

    for key in [

        get_secret("GEMINI_API_KEY_1"),

        get_secret("GEMINI_API_KEY_2"),

        get_secret("GEMINI_API_KEY"),

    ]

    if key and key.strip()

]

api_configured = len(configured_keys) > 0

with st.sidebar:
    st.markdown(
        """
### Privacy boundary

The application extracts patient/report fields locally.

Gemini is called twice. Both calls receive only:

- Original karyotype
- Original interpretation
- (second call) LLM #1 structured output

Patient name, age, sex, IDs, hospital, dates and other report fields are
not included in the Gemini request.

The second call compares those originals with LLM #1 output and flags
whether human intervention is needed.
"""
    )

uploaded_files = st.file_uploader(
    "Upload cytogenetic PDF reports",
    type=["pdf"],
    accept_multiple_files=True,
    help="You can upload one or multiple reports.",
)

if uploaded_files:
    st.info(f"{len(uploaded_files)} PDF report(s) selected.")

    if not api_configured:
        st.warning(
            "⚠️ **API Key Required:** The **Process Reports** button is disabled because no Gemini API key is configured. "
            "Please configure your key in the `.env` file."
        )

    if st.button(
        "Process Reports",
        type="primary",
        use_container_width=True,
        disabled=not api_configured,
    ):
        all_rows = []

        progress = st.progress(0)
        status = st.empty()

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            for index, uploaded_file in enumerate(uploaded_files, start=1):
                status.write(
                    f"Processing {index}/{len(uploaded_files)}: "
                    f"`{uploaded_file.name}`"
                )

                pdf_path = temp_path / uploaded_file.name
                pdf_path.write_bytes(uploaded_file.getvalue())

                patient_id = f"P{index:04d}"

                try:
                    rows = process_pdf(
                        pdf_path=pdf_path,
                        patient_id=patient_id,
                    )


                    for row in rows:
                        row["Source_PDF"] = uploaded_file.name

                    all_rows.extend(rows)

                except Exception as exc:
                    all_rows.append({
                        "Patient_ID": patient_id,
                        "Patient_Name": "",
                        "Age": "",
                        "Sex": "",
                        "Sample_ID": "",
                        "Karyotype_Raw": "",
                        "Cytogenetic_Abnormality": "",
                        "Event_Type": "",
                        "Chromosome_1": "",
                        "Chromosome_2": "",
                        "Arm_1": "",
                        "Arm_2": "",
                        "Breakpoint_1": "",
                        "Breakpoint_2": "",
                        "Copy_Change": "",
                        "Event_Raw": "",
                        "Interpretation": "",
                        "Conflict_Flag": True,
                        "Conflict_Reason": str(exc),
                        "Confidence": "low",
                        "Extraction_Status": "PDF_PROCESSING_ERROR",
                        "Source_PDF": uploaded_file.name,
                        "Human_Intervention_Needed": "Yes",
                        "Verification_Reason": f"PDF processing failed: {exc}",
                    })

                progress.progress(index / len(uploaded_files))

        df = pd.DataFrame(all_rows, columns=COLUMNS)



        st.session_state["result_df"] = df

        status.success(
            f"Finished processing {len(uploaded_files)} report(s)."
        )

if "result_df" in st.session_state:
    df = st.session_state["result_df"]

    st.divider()
    st.subheader("Extraction Results")

    total_rows = len(df)
    successful = int(
        (df["Extraction_Status"] == "OK").sum()
    )
    conflicts = int(
        (df["Conflict_Flag"] == True).sum()
    ) if "Conflict_Flag" in df else 0
    human_review = int(
        (df["Human_Intervention_Needed"] == "Yes").sum()
    ) if "Human_Intervention_Needed" in df else 0

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Rows / Events", total_rows)
    col2.metric("Successful Events", successful)
    col3.metric("Conflicts", conflicts)
    col4.metric("Human Intervention", human_review)

    st.dataframe(
        df,
        use_container_width=True,
        height=600,
        hide_index=True,
    )


    excel_path = Path(
        tempfile.gettempdir()
    ) / "cytogenetics_output.xlsx"

    with pd.ExcelWriter(
        excel_path,
        engine="openpyxl"
    ) as writer:
        df.to_excel(
            writer,
            index=False,
            sheet_name="Cytogenetics",
        )

        ws = writer.book["Cytogenetics"]
        ws.freeze_panes = "A2"
        ws.auto_filter.ref = ws.dimensions

        widths = {
            "A": 12,  # Patient_ID
            "B": 22,  # Patient_Name
            "C": 8,   # Age
            "D": 10,  # Sex
            "E": 16,  # Sample_ID
            "F": 35,  # Karyotype_Raw
            "G": 40,  # Cytogenetic_Abnormality
            "H": 16,  # Event_Type
            "I": 14,  # Chromosome_1
            "J": 14,  # Chromosome_2
            "K": 10,  # Arm_1
            "L": 10,  # Arm_2
            "M": 14,  # Breakpoint_1
            "N": 14,  # Breakpoint_2
            "O": 14,  # Copy_Change
            "P": 25,  # Event_Raw
            "Q": 60,  # Interpretation
            "R": 14,  # Conflict_Flag
            "S": 35,  # Conflict_Reason
            "T": 12,  # Confidence
            "U": 18,  # Extraction_Status
            "V": 25,  # Source_PDF
            "W": 26,  # Human_Intervention_Needed
            "X": 50,  # Verification_Reason
        }

        for column, width in widths.items():
            ws.column_dimensions[column].width = width

    st.download_button(
        label="Download Excel",
        data=excel_path.read_bytes(),
        file_name="cytogenetics_output.xlsx",
        mime=(
            "application/vnd.openxmlformats-officedocument."
            "spreadsheetml.sheet"
        ),
        type="primary",
        use_container_width=True,
    )


    review_mask = (
        df["Extraction_Status"].ne("OK")
        | df["Conflict_Flag"].eq(True)
    )
    if "Human_Intervention_Needed" in df.columns:
        review_mask = review_mask | df["Human_Intervention_Needed"].eq("Yes")

    review_df = df[review_mask]

    if not review_df.empty:
        st.warning(
            f"{len(review_df)} row(s) require review."
        )

        with st.expander("Show rows requiring review"):
            st.dataframe(
                review_df,
                use_container_width=True,
                hide_index=True,
            )

else:
    st.markdown(
        """
### Workflow

**PDF → local extraction → Gemini #1 (structure) → Gemini #2 (verify) → Excel**

Upload the reports above and click **Process Reports**.

For complex karyotypes, multiple cytogenetic events are represented as
multiple rows in the same Excel worksheet.
"""
    )
