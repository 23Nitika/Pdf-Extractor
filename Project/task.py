import os
import csv
import re
import fitz  # Import PyMuPDF for PDF text extraction
from pdfminer.converter import HOCRConverter
from pdfminer.high_level import extract_text
from datasets import load_dataset
from transformers import DistilBertTokenizer, DistilBertModel
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import torch

import tempfile
# Create a temporary directory for the output file
temp_dir = tempfile.gettempdir()

# Specify the full path to the output CSV file in the temporary directory
output_csv_file = os.path.join(temp_dir, "output_details.csv")

# Initialize DistilBERT tokenizer and model
tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
model = DistilBertModel.from_pretrained("distilbert-base-uncased")

# Function to load data from a CSV file
def load_data_from_csv(file_path):
    data = []
    with open(file_path, mode="r") as csv_file:
        csv_reader = csv.DictReader(csv_file)
        for row in csv_reader:
            data.append(row)
    return data

# Function to convert text to embeddings
def get_embeddings(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).numpy()

   # Access the train split of the dataset
def fetch_and_save_job_descriptions(num_descriptions_to_extract, csv_filename):
    dataset = load_dataset("jacob-hugging-face/job-descriptions")
    train_dataset = dataset["train"]

    # Extract job descriptions
    job_descriptions = []
    for i in range(num_descriptions_to_extract):
        job_description = train_dataset[i]["job_description"]
        job_descriptions.append(job_description)

    # Save job descriptions to a CSV file
    with open(csv_filename, mode="w", newline="", encoding="utf-8") as csv_file:
        csv_writer = csv.writer(csv_file)

        # Write the header row
        csv_writer.writerow(["Job Description"])

        # Write job descriptions to the CSV file
        for idx, description in enumerate(job_descriptions):
            csv_writer.writerow([description])

    print(f"Job descriptions saved to {csv_filename}")
  

# Function to extract text from a PDF using PyMuPDF
def extract_text_with_pymupdf(pdf_file_path):
    doc = fitz.open(pdf_file_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text

# Function to extract details from text
def extract_details_from_text(text, category):
    # Initialize variables to store extracted details
    skills = ""
    education = ""

    # Search for relevant information using regular expressions
    # Extracting Skills
    skills_match = re.search(r"Skills\s+([^:]+)", text)
    if skills_match:
        skills = skills_match.group(1).strip()

    # Extracting Education
    education_match = re.search(r"Education\s+([^:]+)", text)
    if education_match:
        education = education_match.group(1).strip()

    return category, skills, education


def extract_text_with_pdfminer(pdf_file_path):
    return extract_text(pdf_file_path)

def get_top_5_cvs_with_recommendations_and_save_results_to_txt(output_txt_file):
    # Load job descriptions and CV details from CSV files
    job_descriptions_file = r"C:\Users\HP\Downloads\Project1111\Project\job_descriptions.csv"  
    cv_details_file = r"C:\Users\HP\Downloads\Project1111\Project\output_details.csv"  

    job_descriptions = [row["Job Description"] for row in load_data_from_csv(job_descriptions_file)]
    cv_details = load_data_from_csv(cv_details_file)
    
    with open(output_txt_file, mode='w', encoding='utf-8') as txt_file:
        for job_description in job_descriptions:
            
            # Calculate embeddings for the job description
            job_desc_embedding = get_embeddings(job_description)

            # Calculate cosine similarity for each CV
            similarity_scores = []
            for cv in cv_details:
                cv_embedding = get_embeddings(cv["Skills"] + " " + cv["Education"])
                similarity = cosine_similarity(job_desc_embedding, cv_embedding)[0][0]
                similarity_scores.append((cv["File"], similarity, cv["Skills"], cv["Education"]))

            # Sort CVs by similarity score in descending order
            similarity_scores.sort(key=lambda x: x[1], reverse=True)

            # Select the top 5 CVs
            top_5_cvs = similarity_scores[:5]

            # Write the results to the text file
            txt_file.write(f"Job Description: {job_description}\n")
            for idx, (cv, similarity, skills, education) in enumerate(top_5_cvs, start=1):
                txt_file.write(f"Top Candidate {idx}:\n")
                txt_file.write(f"CV File: {cv}\n")
                txt_file.write(f"Similarity Score: {similarity}\n")
                txt_file.write(f"Skills: {skills}\n")
                txt_file.write(f"Education: {education}\n")
                txt_file.write("\n")
                
            txt_file.write("\n\n")    
            txt_file.write(50*"#") 
            txt_file.write("\n\n")
            
    
    print(f"Results saved to {output_txt_file}")

if __name__ == "__main__":
    
    # Specify the number of job descriptions to extract and the CSV file name
    num_descriptions_to_extract = 15
    csv_filename = "job_descriptions.csv"
    fetch_and_save_job_descriptions(num_descriptions_to_extract, csv_filename)
    
    print(100 * "#")
    
    pdf_directory = r"C:\Users\HP\Downloads\Project1111\Project\archive1"  # directory path
    output_csv_file = "output_details.csv"
    
    with open(output_csv_file, mode='w', newline='', encoding='utf-8') as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(["File", "Category", "Skills", "Education"])

        for root, dirs, files in os.walk(pdf_directory):
            for category in dirs:
                category_path = os.path.join(root, category)
                for file in os.listdir(category_path):
                    if file.endswith(".pdf"):
                        pdf_file_path = os.path.join(category_path, file)
                        text = extract_text_with_pdfminer(pdf_file_path)  

                        # Extract key details from the text
                        category, skills, education = extract_details_from_text(text, category)

                        # Create a CSV string and write it to the file
                        csv_row = f"{file},{category},{skills},{education}\n"
                        csv_file.write(csv_row)
    output_txt_file = "results.txt" 
    get_top_5_cvs_with_recommendations_and_save_results_to_txt(output_txt_file)

                
    
    print(100 * "#")