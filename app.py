import streamlit as st

from llm import ask_chatbot

# Page configuration
st.set_page_config(
    page_title="Document AI Assistant",
    page_icon="📚",
    layout="wide"
)

# Page heading
st.title("📚 Document AI Assistant")

st.write(
    "Ask questions about documents stored "
    "on your computer."
)

# Initialise conversation history
if "messages" not in st.session_state:
    st.session_state.messages = []


# Display existing conversation
for message in st.session_state.messages:

    with st.chat_message(message["role"]):

        st.markdown(message["content"])

        if message.get("sources"):

            with st.expander("View source documents"):

                for source in message["sources"]:

                    st.write(
                        f"**Document:** {source['source']}"
                    )

                    st.write(
                        f"**Chunk:** {source['chunk']}"
                    )


# Clear conversation button
if st.sidebar.button("Clear conversation"):

    st.session_state.messages = []
    st.rerun()


# Chat input
question = st.chat_input(
    "Ask something about your documents..."
)


if question:

    # Display user question
    st.session_state.messages.append({
        "role": "user",
        "content": question
    })

    with st.chat_message("user"):
        st.markdown(question)

    # Generate assistant response
    with st.chat_message("assistant"):

        with st.spinner(
            "Searching documents and generating answer..."
        ):

            try:

                # Retrieve conversation history before the current question
                chat_history = st.session_state.messages[:-1]

                answer, sources = ask_chatbot(
                    question,
                    chat_history=chat_history
                )

                st.markdown(answer)

                # Display document sources
                if sources:

                    with st.expander("View source documents"):

                        for source in sources:

                            st.write(
                                f"**Document:** {source['source']}"
                            )

                            st.write(
                                f"**Chunk:** {source['chunk']}"
                            )

                # Save assistant response
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "sources": sources
                })

            except Exception as error:

                st.error(
                    f"An error occurred: {error}"
                )
