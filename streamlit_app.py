import streamlit as st
import pinecone
import fitz  # PyMuPDF
import re

# Initialize Pinecone
pinecone.init(api_key='YOUR_PINECONE_API_KEY', environment='YOUR_ENVIRONMENT')
index = pinecone.Index('your-index-name')

# Function to extract text and its properties from PDF
def extract_text_with_properties(pdf_file):
  text_segments = []
  with fitz.open(pdf_file) as doc:
      for page in doc:
          for block in page.get_text("dict")["blocks"]:
              if "lines" in block:  # Check if the block contains text
                  for line in block["lines"]:
                      for span in line["spans"]:
                          text = span["text"].strip()
                          font_size = span["size"]
                          font_flags = span["flags"]  # Check for bold text
                          is_bold = font_flags & 2  # Bold flag is usually 2
                          text_segments.append((text, is_bold, font_size))
  return text_segments

# Function to identify news segments based on text properties
def identify_news_segments(text_segments):
  headlines = []
  current_content = ""

  for text, is_bold, font_size in text_segments:
      if is_bold and text:  # If the text is bold, consider it a headline
          if current_content:  # If there's existing content, save it
              headlines.append({'headline': current_content, 'content': current_content})
              current_content = ""  # Reset for the next headline
          current_content = text  # Start a new headline
      elif current_content:  # If there's a current headline, append content
          current_content += ' ' + text

  # Add the last segment if it exists
  if current_content:
      headlines.append({'headline': current_content, 'content': current_content})

  return headlines

# Streamlit app layout
st.title("Gujarati Newspaper Segment Identifier")

# File uploader
uploaded_file = st.file_uploader("Upload a PDF of a Gujarati newspaper", type="pdf")

if uploaded_file is not None:
  # Extract text with properties from the PDF
  text_segments = extract_text_with_properties(uploaded_file)

  # Identify news segments
  news_segments = identify_news_segments(text_segments)

  # Store headlines in Pinecone
  for segment in news_segments:
      index.upsert([(segment['headline'], {'content': segment['content']})])  # Assuming segment has 'headline' and 'content'

  st.success("Headlines stored in Pinecone!")

  # User input for tag
  tag = st.text_input("Enter a tag to filter news segments (e.g., water)")

  if tag:
      # Query Pinecone for relevant news segments
      results = index.query(filter={"tag": tag}, top_k=10)  # Adjust the query as per your index schema
      st.write("Relevant News Segments:")
      for result in results['matches']:
          st.write(result['metadata']['content'])  # Display the content of the matched segments