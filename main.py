import time
import streamlit as st
from rag import process_urls, generate_answer

st.title("PageBrain")
st.caption("Paste any URL or PDF path — ask questions — get answers with sources.")

# Each user gets their own session state
if "processed" not in st.session_state:
    st.session_state.processed = False

if "messages" not in st.session_state:
    st.session_state.messages = []

# Sidebar
st.sidebar.header("📎 Add Your Sources")
st.sidebar.caption("Paste URLs or local PDF paths (e.g. C:/Users/you/file.pdf)")

url1 = st.sidebar.text_input("URL / PDF 1")
url2 = st.sidebar.text_input("URL / PDF 2")
url3 = st.sidebar.text_input("URL / PDF 3")

st.sidebar.markdown("---")
st.sidebar.caption("💡 Tips for best results:")
st.sidebar.caption("✅ Use specific article pages, not homepages")
st.sidebar.caption("✅ Use the same words as the document in your question")
st.sidebar.caption("✅ Wikipedia, ArXiv, Reuters work great")
st.sidebar.caption("❌ Avoid JavaScript-heavy pages like CNN Markets")

placeholder = st.empty()

process_url_button = st.sidebar.button("Process URLs", use_container_width=True)

if process_url_button:
    urls = [url for url in (url1, url2, url3) if url != '']
    if len(urls) == 0:
        st.sidebar.error("Please enter at least one URL or PDF path.")
    else:
        st.session_state.processed = False
        st.session_state.messages = []
        with st.spinner("Processing your sources..."):
            for status in process_urls(urls):
                placeholder.text(status)
        st.session_state.processed = True
        placeholder.empty()
        st.sidebar.success("✅ Sources processed! Ask your question below.")

# Question input
if st.session_state.processed:
    st.markdown("### 💬 Ask a Question")
    query = st.text_input(
        "Type your question here",
        placeholder="e.g. What are the key findings of this paper?"
    )

    if query:
        with st.spinner("Thinking..."):
            # Retry logic — tries 3 times before giving up
            answer = None
            sources = None
            for attempt in range(3):
                try:
                    answer, sources = generate_answer(query)
                    break
                except Exception as e:
                    error_msg = str(e).lower()
                    if "rate_limit" in error_msg or "429" in error_msg:
                        if attempt < 2:
                            wait = (attempt + 1) * 20
                            st.toast(f"⏳ Rate limit hit. Retrying in {wait} seconds...")
                            time.sleep(wait)
                        else:
                            st.warning("⏳ Too many requests right now. Please wait 60 seconds and try again.")
                    elif "vector database" in error_msg:
                        st.warning("⚠️ Please process URLs first before asking questions!")
                        break
                    else:
                        st.error(f"Something went wrong: {str(e)}")
                        break

        if answer:
            # Save to session history
            st.session_state.messages.append({
                "question": query,
                "answer": answer,
                "sources": sources
            })

            st.markdown("### 📝 Answer")
            st.write(answer)

            if sources:
                st.markdown("### 🔗 Sources")
                for source in sources.split("\n"):
                    if source.strip():
                        st.write(source)

    # Show previous questions in this session
    if len(st.session_state.messages) > 1:
        st.markdown("---")
        st.markdown("### 🕘 Previous Questions This Session")
        for i, msg in enumerate(reversed(st.session_state.messages[:-1])):
            with st.expander(f"Q: {msg['question']}"):
                st.write(msg['answer'])
                if msg['sources']:
                    st.caption(f"Sources: {msg['sources']}")

else:
    st.info("👈 Paste your URLs or PDF paths in the sidebar and click **Process URLs** to get started.")
    st.markdown("### What can you ask?")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**📚 Research Papers**")
        st.caption("What dataset did this paper use?")
        st.caption("What problem does this paper solve?")
        st.markdown("**💰 Finance & Markets**")
        st.caption("What were the key revenue drivers?")
        st.caption("What risks did management highlight?")
    with col2:
        st.markdown("**🌐 News Articles**")
        st.caption("What is the key takeaway?")
        st.caption("What are the main points?")
        st.markdown("**📄 Any Document**")
        st.caption("Summarise this document")
        st.caption("What does this say about X?")
