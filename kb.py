import streamlit as st
import boto3
import PyPDF2
from io import BytesIO

def init_bedrock_client():
    try:
        bedrock = boto3.client(
            service_name='bedrock-agent-runtime',
            region_name='us-west-2'  # Replace with your region
        )
        return bedrock
    except Exception as e:
        st.error(f"Error connecting to AWS Bedrock: {str(e)}")
        return None

def process_file_upload(uploaded_file):
    if uploaded_file is not None:
        try:
            # Read file content
            content = uploaded_file.read()
            if uploaded_file.type == "application/pdf":
                # Handle PDF files using PyPDF2
                
                pdf_file = BytesIO(content)
                pdf_reader = PyPDF2.PdfReader(pdf_file)
                
                # Extract text from all pages
                text_content = []
                for page in pdf_reader.pages:
                    text_content.append(page.extract_text())
                
                return "\n".join(text_content)
            else:
                # Handle text files
                return content.decode('utf-8')
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")
            return None

def query_knowledge_base(bedrock_client, jd_count, context=None):
    try:
        response = bedrock_client.retrieve(
            knowledgeBaseId='2FATY2MKWF',
            retrievalQuery={
                'text': "Return the job descriptions that match the user's resume. <resume>" + context + "</resume>. If there is no match, return 'No matches found.'",
            },
            retrievalConfiguration={
                'vectorSearchConfiguration': {
                    'numberOfResults': jd_count,
                    # Optional: Specify search type if using OpenSearch Serverless
                    # 'overrideSearchType': 'HYBRID'  # or 'SEMANTIC'
                }
            }
        )
        
        # Parse response properly
        if 'retrievalResults' in response:
            results = []
            for result in response['retrievalResults']:
                # Extract text content and metadata
                content = result.get('content', {}).get('text', '')
                score = result.get('score', 0.0)
                location = result.get('location', {})
                
                results.append({
                    'content': content,
                    'score': score,
                    'location': location
                })
            
            return results
        return None
        
    except Exception as e:
        st.error(f"Error querying Bedrock: {str(e)}")
        return None

def main():
    st.title("Job Search")
    
    # Initialize Bedrock client
    bedrock_client = init_bedrock_client()
    if not bedrock_client:
        st.stop()
    
    # File upload section
    st.subheader("Upload Document")
    uploaded_file = st.file_uploader("Choose a file", type=['txt', 'pdf'])
    
    # Process uploaded file
    context = None
    if uploaded_file:
        context = process_file_upload(uploaded_file)
        if context:
            st.success("File uploaded successfully!")
    
    # Query section
    query = st.number_input("How many job descriptions you want to see:", min_value=1, max_value=10, value=3, step=1)
    
    if st.button("Retrive JDs"):
        if query:
            with st.spinner("Processing your question..."):
                answer = query_knowledge_base(bedrock_client, query, context)
                if answer:
                    st.write("Answer:")
                    for idx, result in enumerate(answer, 1):
                        st.write(f"Result {idx}:")
                        st.write(f"Score: {result['score']}")
                        st.write(f"Location: {result['location'].get('s3Location', 'No S3 URL available')}")
                        with st.expander("Show Content"):
                            st.write(result['content'])
                        st.write("---")
        else:
            st.warning("Please enter a question.")

if __name__ == "__main__":
    main()