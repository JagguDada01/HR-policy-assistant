"""Streamlit interface for the HR manual question-answering system."""
from importlib.util import find_spec
from io import BytesIO
from pathlib import Path

import streamlit as st
from pypdf import PdfReader, PdfWriter
from streamlit.errors import StreamlitAPIException

from src.hr_policy.config import PDF_PATH
from src.hr_policy.qa import HRPolicyQA

st.set_page_config(
    page_title="HR Policy Assistant",
    page_icon=":material/menu_book:",
    layout="centered",
)


@st.cache_resource(max_entries=1)
def get_qa(source_version):
    """Reload the QA pipeline when its implementation changes."""
    return HRPolicyQA()


def get_source_version():
    module_dir = Path(__file__).parent / "src" / "hr_policy"
    files = (module_dir / "qa.py", module_dir / "ingest.py")
    return tuple(path.stat().st_mtime_ns for path in files)


@st.cache_data(show_spinner=False, max_entries=8)
def get_pdf_page(page_number: int) -> bytes:
    """Return a one-page PDF containing the requested physical page."""
    reader = PdfReader(str(PDF_PATH))
    if page_number < 1 or page_number > len(reader.pages):
        raise ValueError(f"PDF page {page_number} is outside the manual.")
    writer = PdfWriter()
    writer.add_page(reader.pages[page_number - 1])
    output = BytesIO()
    writer.write(output)
    return output.getvalue()


st.title("HR policy assistant", icon=":material/menu_book:")
st.caption("Search the HR manual and open the cited page to check the answer.")
st.badge("Enso Group HR manual", icon=":material/description:", color="blue")
st.space("small")

with st.container(border=True):
    st.subheader("Ask a question")
    st.caption("Ask about leave, probation, benefits, or another HR policy.")
    with st.form("question_form", clear_on_submit=False):
        question = st.text_input(
            "Question",
            placeholder="Type a question about the manual",
            key="question_input",
        )
        submitted = st.form_submit_button(
            "Search the manual",
            type="primary",
            icon=":material/search:",
        )

if submitted and question.strip():
    st.session_state["last_question"] = question.strip()
    st.session_state["last_result"] = None
    st.session_state["last_error"] = None
    st.session_state["open_source_page"] = None
    try:
        with st.spinner("Searching the manual…"):
            result = get_qa(get_source_version()).ask(question.strip())
        st.session_state["last_result"] = {
            "answer": result.answer,
            "sources": result.sources,
        }
    except Exception as exc:
        st.session_state["last_error"] = str(exc)

if st.session_state.get("last_error"):
    st.error(f"The HR manual index could not be loaded: {st.session_state['last_error']}")
    st.info("Install the packages in requirements.txt, then run: python -m src.hr_policy.ingest")

last_result = st.session_state.get("last_result")
if last_result:
    st.space("medium")
    st.caption(f"Question: {st.session_state.get('last_question', '')}")

    with st.container(border=True):
        st.subheader("Answer")
        if last_result["answer"] == "Not found":
            st.badge("No supporting passage", icon=":material/search_off:", color="orange")
        else:
            st.badge("Based on the manual", icon=":material/verified:", color="green")
        st.write(last_result["answer"])

    st.subheader("Source references")
    sources = last_result.get("sources", [])
    if sources:
        for index, source in enumerate(sources):
            annexure = source.get("annexure", "")
            policy = source.get("policy", "")
            page_number = int(source["page_number"])
            section = ", ".join(part for part in (annexure, policy) if part) or "HR Manual"
            with st.container(border=True):
                source_col, action_col = st.columns([3, 1], vertical_alignment="center")
                source_col.markdown(f"**{section}**")
                source_col.caption(f"PDF page {page_number}")
                if action_col.button(
                    f"View page {page_number}",
                    key=f"view_source_{index}_{page_number}",
                    icon=":material/picture_as_pdf:",
                    width="stretch",
                ):
                    st.session_state["open_source_page"] = page_number
    else:
        st.caption("No supporting passage was found in the manual.")

page_to_view = st.session_state.get("open_source_page")
if page_to_view:
    st.space("small")
    with st.container(border=True):
        heading_col, close_col = st.columns([4, 1], vertical_alignment="center")
        heading_col.subheader(f"Source page {page_to_view}")
        if close_col.button("Close", key="close_source_page", icon=":material/close:"):
            st.session_state["open_source_page"] = None
        else:
            try:
                page_pdf = get_pdf_page(int(page_to_view))
            except Exception as exc:
                st.error(f"Could not open PDF page {page_to_view}: {exc}")
            else:
                viewer_available = hasattr(st, "pdf") and find_spec("streamlit_pdf") is not None
                viewer_error = None
                if viewer_available:
                    try:
                        st.pdf(
                            page_pdf,
                            height=720,
                            key=f"source_pdf_{page_to_view}",
                            alt=f"HR manual PDF page {page_to_view}",
                        )
                    except StreamlitAPIException as exc:
                        # A server that was running before streamlit-pdf was installed
                        # can have stale component registration until it is restarted.
                        viewer_error = exc
                else:
                    viewer_error = "The optional streamlit-pdf viewer is not installed."

                if viewer_error:
                    st.info("The inline PDF viewer is unavailable. Restart Streamlit after installing dependencies to enable it.")
                    st.download_button(
                        "Download this PDF page",
                        data=page_pdf,
                        file_name=f"hr_manual_page_{page_to_view}.pdf",
                        mime="application/pdf",
                        icon=":material/download:",
                    )
