import os
import re
import time
from pathlib import Path
from typing import List, Dict, Any

import cv2
import fitz
import numpy as np
import pandas as pd
import pytesseract
from dotenv import load_dotenv
from google import genai
from pydantic import BaseModel, Field

load_dotenv()

MODEL_NAME = os.getenv("GEMINI_MODEL", "gemini-3.6-flash")



class CytogeneticEvent(BaseModel):
    event_type: str = Field(
        description=(
            "deletion, duplication, inversion, translocation, derivative, "
            "trisomy, monosomy, insertion, marker, isochromosome, ring, "
            "complex, normal, culture_failure, or other"
        )
    )
    raw: str = Field(
        description="Raw ISCN abnormality token, e.g. inv(16)(p13q22), del(7)(q22), t(8;21)(q22;q22), +21, -7, normal"
    )
    chromosome_1: str = ""
    chromosome_2: str = ""
    arm_1: str = ""
    arm_2: str = ""
    breakpoint_1: str = ""
    breakpoint_2: str = ""
    copy_change: str = ""


class CytogeneticResult(BaseModel):
    karyotype: str
    cytogenetic_abnormality: str = ""
    events: List[CytogeneticEvent] = []
    interpretation: str = ""
    conflict_flag: bool = False
    conflict_reason: str = ""
    confidence: str = "high"


class VerificationResult(BaseModel):
    karyotype_consistent: bool = Field(
        description=(
            "True if LLM #1 preserved the original karyotype and did not "
            "drop, invent, or mis-parse abnormalities."
        )
    )
    interpretation_consistent: bool = Field(
        description=(
            "True if LLM #1 structured output matches the clinical meaning "
            "of the original interpretation."
        )
    )
    human_intervention_needed: bool = Field(
        description=(
            "True if a human cytogeneticist should review this case before "
            "the structured output is trusted."
        )
    )
    verification_reason: str = Field(
        description=(
            "Short explanation of agreement or of why human review is needed."
        )
    )


VERIFICATION_INSTRUCTION = """
You are a cytogenetics quality-control reviewer.

You receive:
1. ORIGINAL KARYOTYPE from the report (source of truth for ISCN)
2. ORIGINAL INTERPRETATION from the report (source of truth for meaning)
3. LLM #1 STRUCTURED OUTPUT (parsed karyotype, events, summary)

Do not request, infer, or generate patient identity, age, sex, hospital,
dates, diagnosis, or other patient information.

Compare the original karyotype and original interpretation with LLM #1.

Set human_intervention_needed=true if ANY of the following apply:
- LLM #1 karyotype differs materially from the original karyotype
- Events miss, invent, split incorrectly, or misclassify abnormalities
- Chromosomes, arms, breakpoints, or copy_change are wrong
- Structured interpretation / cytogenetic_abnormality changes clinical meaning
- Original karyotype and original interpretation disagree, and LLM #1 did not
  flag that disagreement clearly
- Output is incomplete, internally inconsistent, or low-confidence

Set human_intervention_needed=false only when LLM #1 is a faithful, complete
structured representation of the original karyotype and interpretation.

Keep verification_reason concise and specific.
"""


SYSTEM_INSTRUCTION = """
You are a cytogenetics data-extraction engine.

You receive ONLY:
1. an ISCN karyotype string (the full karyogenetic value)
2. the interpretation text from the report

Do not request, infer, or generate patient identity, age, sex, hospital,
dates, diagnosis, or other patient information.

Tasks:
- Preserve the exact supplied karyotype in the output.
- Parse EVERY distinct cytogenetic abnormality into a separate event in the 'events' list.
- Use standard ISCN rules:
  * For del(7)(q22) or del(5)(q13q33): event_type=deletion, chromosome_1=7, arm_1=q, breakpoint_1=q22, copy_change=-1 (or deletion).
  * For inv(16)(p13q22): event_type=inversion, chromosome_1=16, chromosome_2="", arm_1=p, arm_2=q, breakpoint_1=p13, breakpoint_2=q22.
  * For translocations like t(8;21)(q22;q22) or t(1;19)(q23;p13.3): event_type=translocation, chromosome_1=8, chromosome_2=21, arm_1=q, arm_2=q, breakpoint_1=q22, breakpoint_2=q22.
  * For derivatives like der(19)t(1;19)(q23;p13.3): event_type=derivative, chromosome_1=19, chromosome_2=1, arm_1=p, arm_2=q, breakpoint_1=p13.3, breakpoint_2=q23.
  * For +21: event_type=trisomy, chromosome_1=21, copy_change=+1.
  * For -7: event_type=monosomy, chromosome_1=7, copy_change=-1.
  * For loss of sex chromosome like -Y: event_type=monosomy, chromosome_1=Y, copy_change=-1.
  * For normal karyotypes (46,XX or 46,XY with no abnormalities): return one event with event_type=normal, raw=normal.
  * For culture failures: event_type=culture_failure, raw=Culture Failure.
- cytogenetic_abnormality: Provide a concise, clear clinical cytogenetic summary (e.g. "Pericentric inversion of chromosome 16: inv(16)(p13q22)", "Deletion of 7q: del(7)(q22)").
- Compare the karyotype and interpretation. If materially inconsistent, set conflict_flag=true and explain in conflict_reason.
- confidence must be 'high', 'medium', or 'low'.
"""




COLUMNS = [
    "Patient_ID",
    "Patient_Name",
    "Age",
    "Sex",
    "Sample_ID",
    "Karyotype_Raw",
    "Cytogenetic_Abnormality",
    "Event_Type",
    "Chromosome_1",
    "Chromosome_2",
    "Arm_1",
    "Arm_2",
    "Breakpoint_1",
    "Breakpoint_2",
    "Copy_Change",
    "Event_Raw",
    "Interpretation",
    "Conflict_Flag",
    "Conflict_Reason",
    "Confidence",
    "Extraction_Status",
    "Source_PDF",
    "Human_Intervention_Needed",
    "Verification_Reason",
]


def normalize_spaces(text: str) -> str:
    return re.sub(r"[ \t]+", " ", text.replace("\x00", " ")).strip()


def extract_pdf_pages(pdf_path: Path):
    doc = fitz.open(pdf_path)
    pages = []
    for i, page in enumerate(doc):
        pages.append({
            "page_number": i + 1,
            "text": page.get_text("text") or "",
        })
    doc.close()
    return pages


def render_page(pdf_path: Path, page_number: int, dpi: int = 250):
    doc = fitz.open(pdf_path)
    page = doc[page_number - 1]
    scale = dpi / 72
    pix = page.get_pixmap(matrix=fitz.Matrix(scale, scale), alpha=False)
    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
    if pix.n == 4:
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
    else:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    doc.close()
    return img


def ocr_image(img, psm: int = 6) -> str:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_CUBIC)
    text = pytesseract.image_to_string(gray, config=f"--psm {psm}")
    return normalize_spaces(text)


def extract_patient_fields(pdf_path: Path, doc: fitz.Document) -> Dict[str, str]:
    info = {
        "Patient_Name": "",
        "Age": "",
        "Sex": "",
        "Sample_ID": "",
    }


    for b in doc[0].get_text("blocks"):
        lines = [l.strip() for l in b[4].splitlines() if l.strip()]
        for i, l in enumerate(lines):
            l_lower = l.lower()

            if l_lower == "patient name" and i + 1 < len(lines):
                val = lines[i + 1]
                if not any(k in val.lower() for k in ["requesting", "date of birth", "hospital", "sample"]):

                    val = re.sub(r"\s+\d{6,}$", "", val).strip()
                    info["Patient_Name"] = val


            elif ("age" in l_lower and any(k in l_lower for k in ["gender", "sex", "birth", "dob"])) and i + 1 < len(lines):
                val = lines[i + 1]
                m_age = re.search(r"(\d+)\s*(?:Years?|Yrs?|Y|Months?|M)?", val, re.I)
                if m_age:
                    info["Age"] = m_age.group(1)
                m_sex = re.search(r"\b(Male|Female|M|F)\b", val, re.I)
                if m_sex:
                    s = m_sex.group(1).upper()
                    info["Sex"] = "Male" if s in ["M", "MALE"] else "Female"


            elif "sample id" in l_lower and i + 1 < len(lines):
                val = lines[i + 1]
                m_id = re.search(r"(\d{6,})", val)
                if m_id:
                    info["Sample_ID"] = m_id.group(1)


    if not info["Patient_Name"]:
        clean_name = re.sub(r"\s*\(\d+\)$", "", pdf_path.stem).strip()
        info["Patient_Name"] = clean_name

    return info




def clean_karyotype_string(value: str) -> str:

    val = normalize_spaces(value)

    val = re.sub(r"\s*([,;()\[\]])\s*", r"\1", val)
    return val.strip()


def extract_karyotype(pdf_path: Path, doc: fitz.Document) -> str:
    text = doc[0].get_text("text")


    if re.search(r"\bCulture\s+Failure\b", text, re.I):
        return "Culture Failure"


    m_block = re.search(
        r"(?:Abnormal|Normal)\s*\n(.*?)(?=\n\s*(?:Interpretation|Metaphase|CHROMOSOMAL|DETAILED|\Z))",
        text,
        re.S | re.I,
    )
    if m_block:
        raw_block = m_block.group(1)

        candidate = clean_karyotype_string("".join(raw_block.splitlines()))
        if re.search(r"\d{2}(?:[~\-]\d{2})?,\s*(?:XX|XY|X|Y)", candidate, re.I):
            return candidate


    lines = [l.strip() for l in text.splitlines() if l.strip()]
    for i, line in enumerate(lines):
        if re.search(r"^\d{2}(?:[~\-]\d{2})?,\s*(?:XX|XY|X|Y)", line, re.I):
            karyo_accum = [line]

            curr = line
            idx = i + 1
            while curr.endswith(",") and idx < len(lines):
                next_l = lines[idx]
                if any(k in next_l.lower() for k in ["interpretation", "metaphase", "detailed"]):
                    break
                karyo_accum.append(next_l)
                curr = next_l
                idx += 1
            return clean_karyotype_string("".join(karyo_accum))


    pix = doc[0].get_pixmap(dpi=300)
    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    ocr_txt = pytesseract.image_to_string(gray)

    if re.search(r"\bCulture\s+Failure\b", ocr_txt, re.I):
        return "Culture Failure"


    ocr_clean = re.sub(r"\b[A-Za-z]6,\s*(XX|XY|X|Y)", r"46,\1", ocr_txt)


    m_ocr = re.search(
        r"(\b\d{2}(?:[~\-]\d{2})?,\s*(?:XX|XY|X|Y)[^\[\n\r]*(?:\[[^\]\n\r]+\])?)",
        ocr_clean,
        re.I,
    )
    if m_ocr:
        return clean_karyotype_string(m_ocr.group(1))

    return ""



def extract_interpretation(pdf_path: Path, doc: fitz.Document) -> str:
    text = doc[0].get_text("text")


    m = re.search(
        r"Interpretation\s*:\s*\n?(.*?)(?=\n\s*(?:Kindly correlate|Metaphase|CHROMOSOMAL|DETAILED|DISCLAIMER|\Z))",
        text,
        re.S | re.I,
    )
    if m:
        interp = normalize_spaces(" ".join(m.group(1).split()))
        if interp and len(interp) > 10 and not interp.startswith("$NN"):
            return interp


    img = render_page(pdf_path, 1, dpi=250)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    red1 = cv2.inRange(hsv, np.array([0, 45, 70]), np.array([12, 255, 255]))
    red2 = cv2.inRange(hsv, np.array([165, 45, 70]), np.array([179, 255, 255]))
    red = cv2.bitwise_or(red1, red2)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
    mask = cv2.morphologyEx(red, cv2.MORPH_CLOSE, kernel, iterations=2)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    H, W = img.shape[:2]
    page_area = H * W

    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        area = w * h
        if 0.03 * page_area < area < 0.60 * page_area:
            crop = img[y:y+h, x:x+w]
            crop_txt = ocr_image(crop)
            if "interpretation" in crop_txt.lower() or "chromosome" in crop_txt.lower() or "abnormal" in crop_txt.lower():
                crop_clean = re.sub(r"^.*?interpretation\s*:\s*", "", crop_txt, flags=re.I).strip()
                return crop_clean or crop_txt

    # Full page OCR fallback
    page_ocr = ocr_image(img)
    m_ocr = re.search(
        r"Interpretation\s*:\s*(.*?)(?=(?:Kindly correlate|Metaphase|CHROMOSOMAL|\Z))",
        page_ocr,
        re.S | re.I,
    )
    if m_ocr:
        return normalize_spaces(m_ocr.group(1))

    return ""




def get_gemini_api_keys() -> List[str]:
    keys = []
    for var in ["GEMINI_API_KEY_1", "GEMINI_API_KEY_2", "GEMINI_API_KEY"]:
        val = (os.getenv(var) or "").strip()
        if val and val not in keys:
            keys.append(val)
    return keys


def _gemini_candidate_models() -> List[str]:
    primary_model = os.getenv("GEMINI_MODEL", MODEL_NAME).strip()
    candidate_models = [primary_model]
    for fallback_model in [
        "gemini-3.6-flash",
        "gemini-flash-lite-latest",
        "gemini-flash-latest",
        "gemini-3.5-flash-lite",
    ]:
        if fallback_model not in candidate_models:
            candidate_models.append(fallback_model)
    return candidate_models


def _call_gemini_schema(
    prompt: str,
    system_instruction: str,
    schema,
):
    keys = get_gemini_api_keys()

    if not keys:
        raise RuntimeError(
            "GEMINI_API_KEY is not configured in .env."
        )

    last_error = None

    for model_to_use in _gemini_candidate_models():
        for api_key in keys:
            for retry in range(2):
                try:
                    client = genai.Client(api_key=api_key)

                    response = client.models.generate_content(
                        model=model_to_use,
                        contents=prompt,
                        config={
                            "system_instruction": system_instruction,
                            "response_mime_type": "application/json",
                            "response_schema": schema,
                            "temperature": 0,
                        },
                    )

                    if response.parsed is not None:
                        return response.parsed

                    return schema.model_validate_json(response.text)

                except Exception as exc:
                    last_error = exc
                    err_str = str(exc)

                    if "RESOURCE_EXHAUSTED" in err_str:
                        break

                    if "503" in err_str or "UNAVAILABLE" in err_str:
                        time.sleep(1.5 * (retry + 1))
                        continue

                    if "404" in err_str or "NOT_FOUND" in err_str:
                        break

                    break

    if last_error:
        raise last_error

    raise RuntimeError("Failed to obtain response from Gemini.")


def call_gemini(
    karyotype: str,
    interpretation: str,
) -> CytogeneticResult:
    prompt = f"""
KARYOTYPE:
{karyotype}

INTERPRETATION:
{interpretation}
"""
    return _call_gemini_schema(
        prompt=prompt,
        system_instruction=SYSTEM_INSTRUCTION,
        schema=CytogeneticResult,
    )


def call_gemini_verify(
    original_karyotype: str,
    original_interpretation: str,
    llm1_result: CytogeneticResult,
) -> VerificationResult:
    prompt = f"""
ORIGINAL KARYOTYPE:
{original_karyotype}

ORIGINAL INTERPRETATION:
{original_interpretation}

LLM #1 STRUCTURED OUTPUT:
{llm1_result.model_dump_json(indent=2)}
"""
    return _call_gemini_schema(
        prompt=prompt,
        system_instruction=VERIFICATION_INSTRUCTION,
        schema=VerificationResult,
    )



def process_pdf(
    pdf_path: Path,
    patient_id: str,
) -> List[Dict[str, Any]]:
    doc = fitz.open(pdf_path)


    patient = extract_patient_fields(pdf_path, doc)


    karyotype = extract_karyotype(pdf_path, doc)
    interpretation = extract_interpretation(pdf_path, doc)

    doc.close()


    if karyotype == "Culture Failure":
        return [{
            "Patient_ID": patient_id,
            "Patient_Name": patient["Patient_Name"],
            "Age": patient["Age"],
            "Sex": patient["Sex"],
            "Sample_ID": patient["Sample_ID"],
            "Karyotype_Raw": "Culture Failure",
            "Cytogenetic_Abnormality": "Culture Failure",
            "Event_Type": "culture_failure",
            "Chromosome_1": "",
            "Chromosome_2": "",
            "Arm_1": "",
            "Arm_2": "",
            "Breakpoint_1": "",
            "Breakpoint_2": "",
            "Copy_Change": "",
            "Event_Raw": "Culture Failure",
            "Interpretation": interpretation,
            "Conflict_Flag": False,
            "Conflict_Reason": "",
            "Confidence": "high",
            "Extraction_Status": "CULTURE_FAILURE",
            "Source_PDF": pdf_path.name,
            "Human_Intervention_Needed": "Yes",
            "Verification_Reason": (
                "Culture failure was extracted locally; LLM verification skipped."
            ),
        }]


    if not karyotype:
        return [{
            "Patient_ID": patient_id,
            "Patient_Name": patient["Patient_Name"],
            "Age": patient["Age"],
            "Sex": patient["Sex"],
            "Sample_ID": patient["Sample_ID"],
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
            "Interpretation": interpretation,
            "Conflict_Flag": True,
            "Conflict_Reason": "Karyotype string could not be extracted from report",
            "Confidence": "low",
            "Extraction_Status": "KARYOTYPE_NOT_FOUND",
            "Source_PDF": pdf_path.name,
            "Human_Intervention_Needed": "Yes",
            "Verification_Reason": (
                "Karyotype could not be extracted; LLM verification skipped."
            ),
        }]


    interpretation_for_llm = (
        interpretation
        if interpretation
        else "(No interpretation text was recovered locally.)"
    )

    result = None
    last_error = None

    for attempt in range(3):
        try:
            result = call_gemini(
                karyotype,
                interpretation_for_llm,
            )
            break
        except Exception as exc:
            last_error = exc
            if attempt < 2:
                time.sleep(2 * (attempt + 1))


    if result is None:
        return [{
            "Patient_ID": patient_id,
            "Patient_Name": patient["Patient_Name"],
            "Age": patient["Age"],
            "Sex": patient["Sex"],
            "Sample_ID": patient["Sample_ID"],
            "Karyotype_Raw": karyotype,
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
            "Interpretation": interpretation,
            "Conflict_Flag": True,
            "Conflict_Reason": f"LLM_ERROR: {last_error}",
            "Confidence": "low",
            "Extraction_Status": "LLM_FAILED",
            "Source_PDF": pdf_path.name,
            "Human_Intervention_Needed": "Yes",
            "Verification_Reason": (
                "LLM #1 failed; verification was not run."
            ),
        }]


    events = result.events or []
    if not events:
        events = [
            CytogeneticEvent(
                event_type="normal" if "normal" in karyotype.lower() else "other",
                raw=karyotype,
            )
        ]

    verification = None
    verification_error = None
    for attempt in range(3):
        try:
            verification = call_gemini_verify(
                original_karyotype=karyotype,
                original_interpretation=interpretation_for_llm,
                llm1_result=result,
            )
            break
        except Exception as exc:
            verification_error = exc
            if attempt < 2:
                time.sleep(2 * (attempt + 1))

    if verification is not None:
        human_needed = "Yes" if verification.human_intervention_needed else "No"
        verification_reason = verification.verification_reason
        if result.conflict_flag and human_needed == "No":
            human_needed = "Yes"
            verification_reason = (
                f"{verification_reason} LLM #1 also set conflict_flag."
            ).strip()
    else:
        human_needed = "Yes"
        verification_reason = f"LLM #2 verification failed: {verification_error}"

    output = []
    for event in events:
        output.append({
            "Patient_ID": patient_id,
            "Patient_Name": patient["Patient_Name"],
            "Age": patient["Age"],
            "Sex": patient["Sex"],
            "Sample_ID": patient["Sample_ID"],
            "Karyotype_Raw": karyotype,
            "Cytogenetic_Abnormality": result.cytogenetic_abnormality,
            "Event_Type": event.event_type,
            "Chromosome_1": event.chromosome_1,
            "Chromosome_2": event.chromosome_2,
            "Arm_1": event.arm_1,
            "Arm_2": event.arm_2,
            "Breakpoint_1": event.breakpoint_1,
            "Breakpoint_2": event.breakpoint_2,
            "Copy_Change": event.copy_change,
            "Event_Raw": event.raw,
            "Interpretation": interpretation,
            "Conflict_Flag": result.conflict_flag,
            "Conflict_Reason": result.conflict_reason,
            "Confidence": result.confidence,
            "Extraction_Status": "OK",
            "Source_PDF": pdf_path.name,
            "Human_Intervention_Needed": human_needed,
            "Verification_Reason": verification_reason,
        })

    return output
