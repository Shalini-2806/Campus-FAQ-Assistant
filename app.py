from pathlib import Path

import pandas as pd
import streamlit as st

from faq_bot import FAQMatcher, FAQValidationError, load_faqs


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_FAQ_PATH = BASE_DIR / "faqs.csv"

st.set_page_config(
    page_title="Campus FAQ Assistant",
    page_icon="🎓",
    layout="centered",
    initial_sidebar_state="expanded",
)


@st.cache_data(show_spinner=False)
def read_default_faqs(path: str, modified_time: float) -> pd.DataFrame:
    return load_faqs(path)


@st.cache_resource(show_spinner=False)
def build_matcher(faqs: tuple[tuple[str, str, str], ...]) -> FAQMatcher:
    return FAQMatcher(
        [
            {
                "question": question,
                "answer": answer,
                "alternate_questions": alternates,
            }
            for question, answer, alternates in faqs
        ]
    )


st.title("🎓 Campus FAQ Assistant")
st.markdown(
    "Ask a question about college life. The assistant finds the closest "
    "matching FAQ and shows you how closely it matches."
)

with st.sidebar:
    st.header("Chat settings")
    confidence_threshold = st.slider(
        "Answer confidence threshold",
        min_value=0.10,
        max_value=0.80,
        value=0.28,
        step=0.02,
        help="Raise this to reject more uncertain matches; lower it to show more suggestions.",
    )
    uploaded_file = st.file_uploader(
        "Use your own FAQ CSV",
        type=["csv"],
        help=(
            "Upload a CSV with question and answer columns. An optional "
            "alternate_questions column accepts alternatives separated by |."
        ),
    )
    st.caption(
        "CSV columns: `question`, `answer`, and optionally `alternate_questions`."
    )
    st.download_button(
        "Download the FAQ template",
        data=DEFAULT_FAQ_PATH.read_bytes(),
        file_name="faqs.csv",
        mime="text/csv",
        use_container_width=True,
    )

try:
    if uploaded_file is None:
        faq_df = read_default_faqs(
            str(DEFAULT_FAQ_PATH), DEFAULT_FAQ_PATH.stat().st_mtime
        )
    else:
        faq_df = load_faqs(uploaded_file)
    matcher = build_matcher(
        tuple(
            (
                row["question"],
                row["answer"],
                row["alternate_questions"],
            )
            for row in faq_df.to_dict(orient="records")
        )
    )
except (FAQValidationError, OSError, pd.errors.ParserError, UnicodeDecodeError) as error:
    st.error(f"Could not load the FAQ data: {error}")
    st.info("Check that the CSV has non-empty question and answer columns.")
    st.stop()

st.caption(
    f"{len(faq_df)} FAQs ready · Ask a question or try one of the examples below."
)
st.caption(
    "Course lists are common examples, not a confirmed catalog for every college. "
    "Check the institution's official prospectus."
)

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("match"):
            with st.expander("How this answer was matched"):
                st.write(f"Closest FAQ: **{message['match']}**")
                st.progress(message["score"], text=f"Similarity: {message['score']:.0%}")

with st.expander("Browse example questions"):
    example_columns = st.columns(2)
    for index, question in enumerate(faq_df["question"]):
        with example_columns[index % 2]:
            if st.button(question, key=f"example-{index}", use_container_width=True):
                st.session_state.pending_question = question
                st.rerun()

prompt = st.chat_input("Ask a college-related question...")
if not prompt:
    prompt = st.session_state.pop("pending_question", None)

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    result = matcher.match(prompt, threshold=confidence_threshold)
    with st.chat_message("assistant"):
        st.markdown(result.answer)
        if result.matched:
            with st.expander("How this answer was matched"):
                st.write(f"Closest FAQ: **{result.question}**")
                st.progress(
                    result.score,
                    text=f"Similarity: {result.score:.0%}",
                )
        else:
            st.caption("Try rephrasing your question or browse the example questions.")

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": result.answer,
            "match": result.question if result.matched else None,
            "score": result.score,
        }
    )

if st.session_state.messages:
    if st.button("Clear conversation", type="secondary"):
        st.session_state.messages = []
        st.rerun()

st.divider()
st.caption("Private by design: questions are matched locally and are not sent to an AI service.")
