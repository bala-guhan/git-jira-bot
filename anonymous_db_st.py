import streamlit as st
from anonymous_db_bot import AnonymousDBBot
from Employee_db_bot import EmployeeDBBot
import time

# Set page config
st.set_page_config(
    page_title="DB Bots Dashboard",
    page_icon="🤖",
    layout="wide"
)

# Initialize session state for bots
if 'anonymous_bot' not in st.session_state:
    with st.spinner('Initializing Anonymous Bot and loading models...'):
        st.session_state.anonymous_bot = AnonymousDBBot()

if 'employee_bot' not in st.session_state:
    with st.spinner('Initializing Employee Bot and loading models...'):
        st.session_state.employee_bot = EmployeeDBBot()

# Title and description
st.title("🤖 DB Bots Dashboard")

# Bot selector in sidebar
with st.sidebar:
    st.header("Select Bot")
    selected_bot = st.radio(
        "Choose a bot:",
        ["Anonymous DB Bot", "Employee DB Bot"],
        index=0
    )
    
    st.header("About")
    if selected_bot == "Anonymous DB Bot":
        st.markdown("""
        This bot:
        - Analyzes project history
        - Explains technical solutions
        - Maintains anonymity
        - Focuses on code changes
        """)
    else:
        st.markdown("""
        This bot:
        - Analyzes employee performance
        - Provides HR insights
        - Tracks collaboration patterns
        - Evaluates skills and contributions
        """)
    
    st.header("How to use")
    st.markdown("""
    1. Type your question in the input box
    2. The bot will search through relevant data
    3. Get a detailed analysis of your query
    """)

# Main content area
st.subheader(f"Ask {selected_bot}")

# Create a text input for the query
query = st.text_input(
    "What would you like to know?",
    placeholder="e.g., How was the login issue fixed? or How is Alice's performance?" if selected_bot == "Anonymous DB Bot" else "e.g., How is Alice's performance? or What are the collaboration patterns?"
)

# Add a submit button
if st.button("Get Answer", type="primary"):
    if query:
        # Create a placeholder for the response
        response_placeholder = st.empty()
        
        # Show a spinner while processing
        with st.spinner('Searching through data and generating response...'):
            # Get the response from the selected bot
            if selected_bot == "Anonymous DB Bot":
                response = st.session_state.anonymous_bot.invoke(query)
            else:
                response = st.session_state.employee_bot.invoke(query)
            
            # Display the response in a nice format
            with response_placeholder.container():
                st.markdown("### Response")
                st.markdown(response['choices'][0]['message']['content'])
                
                # Show token usage in an expander
                with st.expander("Token Usage Details"):
                    st.json(response['usage'])
                
                # Show relevant chunks for Employee Bot
                if selected_bot == "Employee DB Bot":
                    with st.expander("Relevant Data Chunks"):
                        for chunk in st.session_state.employee_bot.relevant_chunks:
                            st.text(chunk)
                            st.markdown("---")
    else:
        st.warning("Please enter a question!")

# Add some spacing
st.markdown("---")

# Add a footer
st.markdown("""
<div style='text-align: center'>
    <p>DB Bots Dashboard</p>
</div>
""", unsafe_allow_html=True) 
