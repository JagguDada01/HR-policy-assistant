"""Streamlit interface for the HR manual question-answering system."""
from pathlib import Path

import streamlit as st

from src.hr_policy.qa import HRPolicyQA

st.set_page_config(page_title="HR Policy Assistant", page_icon="📘", layout="centered")
st.title("HR Policy Assistant")
st.caption("Answers are based only on the supplied HR manual.")

@st.cache_resource(max_entries=1)
def get_qa(source_version):
    """Reload the QA pipeline when its implementation changes."""
    return HRPolicyQA()


def get_source_version():
    module_dir = Path(__file__).parent / "src" / "hr_policy"
    files = (module_dir / "qa.py", module_dir / "ingest.py")
    return tuple(path.stat().st_mtime_ns for path in files)

question = st.text_input("Ask a question about the HR manual", placeholder="For example: How many days of Privilege Leave are allowed?")
if st.button("Ask", type="primary", disabled=not question.strip()):
    try:
        with st.spinner("Searching the HR manual…"):
            result = get_qa(get_source_version()).ask(question.strip())
        st.subheader("Answer")
        st.write(result.answer)
        st.subheader("Source references")
        if result.sources:
            for source in result.sources:
                labels = [x for x in (source.get("annexure"), source.get("policy")) if x]
                ref = ", ".join(labels) if labels else "HR Manual"
                st.write(f"{ref} — PDF page {source['page_number']}")
        else:
            st.write("No supporting passage found in the manual.")
    except Exception as exc:
        st.error(f"The HR manual index could not be loaded: {exc}")
        st.info("Install the packages in requirements.txt, then run: python -m src.hr_policy.ingest")
