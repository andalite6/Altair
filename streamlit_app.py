# Import libraries
import streamlit as st
from datetime import date

# App title and sidebar
st.set_page_config(page_title="Mockup App", layout="wide")
st.title("Mockup Interface")

# Sidebar for navigation
st.sidebar.title("Navigation")
section = st.sidebar.radio("Go to", ["Home", "Form", "Data Display", "Settings"])

# Home Section
if section == "Home":
    st.header("Welcome to the Mockup Interface")
    st.write("Use this area to display the main app introduction or information.")

# Form Section
elif section == "Form":
    st.header("User Input Form")
    st.write("This form allows users to enter various types of data for testing purposes.")

    # Input fields
    user_name = st.text_input("Enter your name")
    age = st.slider("Select your age", 0, 100, 25)
    birth_date = st.date_input("Select your birthdate", date(2000, 1, 1))
    occupation = st.selectbox("Select your occupation", ["Developer", "Designer", "Manager", "Other"])
    description = st.text_area("Describe your project", "Enter details here...")

    # File upload
    file = st.file_uploader("Upload a file", type=["jpg", "png", "pdf", "txt"])

    # Submit button
    if st.button("Submit"):
        st.success("Form submitted successfully!")
        st.write("Name:", user_name)
        st.write("Age:", age)
        st.write("Birthdate:", birth_date)
        st.write("Occupation:", occupation)
        st.write("Project Description:", description)
        if file:
            st.write("Uploaded File:", file.name)

# Data Display Section
elif section == "Data Display":
    st.header("Data Display Area")
    st.write("Use this area to show mock data or test output displays.")
    
    # Sample data display
    st.write("### Example Table")
    data = {
        "Name": ["Alice", "Bob", "Charlie", "David"],
        "Age": [23, 35, 45, 29],
        "Occupation": ["Engineer", "Doctor", "Artist", "Lawyer"]
    }
    st.table(data)

    # Dynamic display
    st.write("### Dynamic Text")
    st.write(f"Hello {user_name if user_name else 'user'}, welcome to the mockup display.")

# Settings Section
elif section == "Settings":
    st.header("Settings")
    st.write("Settings for testing configuration options.")
    
    # Toggle options
    theme = st.radio("Choose a theme", ["Light", "Dark"])
    notifications = st.checkbox("Enable notifications", value=True)
    
    # Display selections
    st.write("Settings Summary:")
    st.write("Selected Theme:", theme)
    st.write("Notifications Enabled:", notifications)
