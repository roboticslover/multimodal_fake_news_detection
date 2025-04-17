import streamlit as st
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_community.tools import WikipediaQueryRun
from langchain.agents import initialize_agent, AgentType
from langchain.callbacks import StreamlitCallbackHandler
from langchain_core.messages import HumanMessage, SystemMessage
from PIL import Image
from io import BytesIO
import base64

# Load environment variables
load_dotenv()

# Set page configuration
st.set_page_config(
    page_title="Fake News Detector",
    page_icon="🔍",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E3A8A;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #4B5563;
        text-align: center;
        margin-bottom: 2rem;
    }
    .result-header {
        font-size: 1.5rem;
        font-weight: bold;
        margin-top: 1rem;
    }
    .fake-tag {
        background-color: #FEE2E2;
        color: #B91C1C;
        padding: 0.5rem;
        border-radius: 0.5rem;
        font-weight: bold;
    }
    .real-tag {
        background-color: #D1FAE5;
        color: #065F46;
        padding: 0.5rem;
        border-radius: 0.5rem;
        font-weight: bold;
    }
    .uncertain-tag {
        background-color: #FEF3C7;
        color: #92400E;
        padding: 0.5rem;
        border-radius: 0.5rem;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# App Header
st.markdown("<h1 class='main-header'>🔍 Multimodal Fake News Detector</h1>", unsafe_allow_html=True)
st.markdown("<p class='sub-header'>Analyze news content using AI to determine authenticity</p>", unsafe_allow_html=True)

# Check for required packages
try:
    from langchain_community.tools import DuckDuckGoSearchRun
    search_tool_available = True
except ImportError:
    search_tool_available = False
    st.warning("DuckDuckGo search package is not installed. Run 'pip install -U duckduckgo-search' to enable web search functionality.")

# Sidebar for API keys
with st.sidebar:
    st.title("⚙️ Settings")
    
    # Get API key from environment variable
    default_api_key = os.getenv("OPENAI_API_KEY", "")
    
    # Only show input field if no environment variable is set
    if not default_api_key:
        st.warning("No API key found in environment variables. For local development, enter your OpenAI API key below.")
        openai_api_key = st.text_input("OpenAI API Key", 
                                      value="",
                                      type="password", 
                                      help="Required for analysis")
    else:
        st.success("OpenAI API key detected from environment variables! ✅")
        # Create a hidden variable to store the key
        openai_api_key = default_api_key
    
    model_choice = st.selectbox(
        "Select OpenAI Model",
        ["gpt-4o", "gpt-4-turbo", "gpt-3.5-turbo"],
        index=0
    )
    
    if not search_tool_available:
        st.error("DuckDuckGo search is not available. Some functionality will be limited.")
        st.markdown("""
        To fix, open your terminal and run:
        ```
        pip install -U duckduckgo-search
        ```
        Then restart the app.
        """)
    
    st.divider()
    st.markdown("### How it works")
    st.markdown("""
    1. Enter news text or upload an image
    2. The system searches multiple sources
    3. AI analyzes the information and provides a verdict
    4. Results show why content might be fake or confirmed as real
    """)

# Initialize search tools
if search_tool_available:
    from langchain_community.tools import DuckDuckGoSearchRun
    search = DuckDuckGoSearchRun(name="Search")
else:
    search = None

wiki_wrapper = WikipediaAPIWrapper(top_k_results=2, doc_content_chars_max=500)
wiki = WikipediaQueryRun(api_wrapper=wiki_wrapper)

# Function to analyze image content
def analyze_image(image_bytes):
    if not openai_api_key:
        st.error("Please provide an OpenAI API key in the sidebar or set it as an environment variable")
        return None
    
    try:
        llm = ChatOpenAI(
            model="gpt-4o",
            api_key=openai_api_key,
            temperature=0
        )
        
        response = llm.invoke([
            SystemMessage(content="You are an expert at analyzing images. Describe the key details of this image that would be relevant for fact-checking."),
            HumanMessage(content=[
                {"type": "text", "text": "What is shown in this image? Provide key details that would be useful for verifying if this is related to a real news event."},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_bytes}"}}
            ])
        ])
        
        return response.content
    except Exception as e:
        st.error(f"Error analyzing image: {str(e)}")
        return None

# Function to verify news with search results
def verify_news(content, image_analysis=None):
    if not openai_api_key:
        st.error("Please provide an OpenAI API key in the sidebar or set it as an environment variable")
        return None, None
    
    llm = ChatOpenAI(
        model=model_choice,
        api_key=openai_api_key,
        temperature=0,
        streaming=True
    )
    
    # Add available tools
    tools = []
    if search is not None:
        tools.append(search)
    tools.append(wiki)
    
    search_agent = initialize_agent(
        tools,
        llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        handle_parsing_errors=True,
        verbose=True
    )
    
    # Prepare prompt based on input type
    if image_analysis:
        prompt = f"""
        Verify if the following content is genuine or fake news:
        
        NEWS CONTENT: {content}
        
        IMAGE ANALYSIS: {image_analysis}
        
        Steps:
        1. Search for at least 3 credible sources related to this news.
        2. Compare the news content with what you find in these sources.
        3. Look for inconsistencies, exaggerations, or fabricated elements.
        4. Provide a clear verdict: REAL, FAKE, or UNCERTAIN.
        5. Explain your reasoning in detail.
        
        Format your response with:
        - A verdict (REAL/FAKE/UNCERTAIN)
        - Summary of findings from your search
        - Explanation of your verdict
        - List of red flags if any were found
        """
    else:
        prompt = f"""
        Verify if the following content is genuine or fake news:
        
        NEWS CONTENT: {content}
        
        Steps:
        1. Search for at least 3 credible sources related to this news.
        2. Compare the news content with what you find in these sources.
        3. Look for inconsistencies, exaggerations, or fabricated elements.
        4. Provide a clear verdict: REAL, FAKE, or UNCERTAIN.
        5. Explain your reasoning in detail.
        
        Format your response with:
        - A verdict (REAL/FAKE/UNCERTAIN)
        - Summary of findings from your search
        - Explanation of your verdict
        - List of red flags if any were found
        """
    
    return search_agent, prompt

# Create tabs for text and image input
text_tab, image_tab = st.tabs(["Text Analysis", "Image Analysis"])

with text_tab:
    news_text = st.text_area(
        "Enter the news text or claim to verify:",
        height=150,
        placeholder="Paste the news content or claim that you want to fact-check..."
    )
    
    verify_text_button = st.button("Verify Text", type="primary", use_container_width=True)
    
    if verify_text_button and news_text:
        # Check if we have required components
        if not search_tool_available:
            st.warning("Search functionality is limited. Install duckduckgo-search package for better results.")
            
        search_agent, prompt = verify_news(news_text)
        
        if search_agent and prompt:
            with st.spinner("Analyzing the news content..."):
                st.subheader("Analysis Results")
                with st.container():
                    st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=True)
                    response = search_agent.run(prompt, callbacks=[st_cb])
                    
                    # Extract verdict for styling
                    if "FAKE" in response.upper().split():
                        st.markdown("<p class='fake-tag'>VERDICT: FAKE</p>", unsafe_allow_html=True)
                    elif "REAL" in response.upper().split():
                        st.markdown("<p class='real-tag'>VERDICT: REAL</p>", unsafe_allow_html=True)
                    else:
                        st.markdown("<p class='uncertain-tag'>VERDICT: UNCERTAIN</p>", unsafe_allow_html=True)
                    
                    st.write(response)

with image_tab:
    col1, col2 = st.columns([1, 1])
    
    with col1:
        uploaded_image = st.file_uploader("Upload an image to analyze:", type=["jpg", "jpeg", "png"])
        if uploaded_image:
            image = Image.open(uploaded_image)
            st.image(image, caption="Uploaded Image", use_column_width=True)
    
    with col2:
        image_context = st.text_area(
            "Add context about the image (optional):",
            height=100,
            placeholder="Add any text claims associated with this image..."
        )
    
    verify_image_button = st.button("Analyze Image", type="primary", use_container_width=True)
    
    if verify_image_button and uploaded_image:
        with st.spinner("Processing image..."):
            # Convert image to base64
            image_bytes = BytesIO()
            Image.open(uploaded_image).save(image_bytes, format="JPEG")
            image_base64 = base64.b64encode(image_bytes.getvalue()).decode("utf-8")
            
            # Analyze image content
            image_analysis = analyze_image(image_base64)
            
            if image_analysis:
                st.subheader("Image Analysis")
                with st.expander("Image Content Detection", expanded=True):
                    st.write(image_analysis)
                
                # Combine image analysis with any provided context
                content_to_verify = f"Image context: {image_context}\n\nImage appears to show: {image_analysis}"
                
                search_agent, prompt = verify_news(content_to_verify, image_analysis)
                
                if search_agent and prompt:
                    st.subheader("Verification Results")
                    with st.container():
                        st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=True)
                        response = search_agent.run(prompt, callbacks=[st_cb])
                        
                        # Extract verdict for styling
                        if "FAKE" in response.upper().split():
                            st.markdown("<p class='fake-tag'>VERDICT: FAKE</p>", unsafe_allow_html=True)
                        elif "REAL" in response.upper().split():
                            st.markdown("<p class='real-tag'>VERDICT: REAL</p>", unsafe_allow_html=True)
                        else:
                            st.markdown("<p class='uncertain-tag'>VERDICT: UNCERTAIN</p>", unsafe_allow_html=True)
                        
                        st.write(response)

# App footer
st.divider()
st.markdown("""
<div style="text-align: center; color: #6B7280; font-size: 0.8rem;">
    Powered by OpenAI + LangChain | For educational purposes only
</div>
""", unsafe_allow_html=True)