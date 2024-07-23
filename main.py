import os
import csv
import concurrent.futures
import fitz
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string
from collections import Counter
from docx import Document
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
nltk.download('punkt')
nltk.download('stopwords')
folder_path = "D:\\MyDownloads\\ats_cv_folder\\"
csv_report_path = "D:\\MyDownloads\\ats_csv_report\\report.csv"


def calculate_similarity(resume_text, job_description_text):
    documents = [resume_text, job_description_text]
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix = tfidf_vectorizer.fit_transform(documents)
    similarity_matrix = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])
    return similarity_matrix[0][0] * 100  # Convert to percentage


def keyword_matching(resume_words, job_description_words):
    resume_counter = Counter(resume_words)
    job_description_counter = Counter(job_description_words)
    common_words = set(resume_counter.keys()).intersection(set(job_description_counter.keys()))
    print(f"Common Words: {common_words}")
    missing_words = set(job_description_counter.keys()).difference(set(resume_counter.keys()))
    print(f"Missing Words in CV compare to JD: {missing_words}")
    match_score = sum([resume_counter[word] for word in common_words])
    total_keywords = len(job_description_counter)

    return match_score / total_keywords * 100  # Percentage match


def read_and_parse_pdf(pdf_path):
    document = fitz.open(pdf_path)
    text = ""
    for page_num in range(len(document)):
        page = document.load_page(page_num)
        text += page.get_text()
    return text


def preprocess_text(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    words = word_tokenize(text)
    words = [word for word in words if word not in stopwords.words('english')]
    return words


def read_and_parse_docx(file_path):
    doc = Document(file_path)
    text = "\n".join([para.text for para in doc.paragraphs])
    return text


def read_and_parse_txt(file_path):
    with open(file_path, 'r') as file:
        data = file.read()
    return data


def read_and_parse_file(file_path):
    jd_text = '''
    Job Description
● Solid experience implementing high availability, scalable cloud solutions
● Strong understanding of data structures and algorithms
● Knowledge of functional programing languages and techniques
● Knowledge of object-oriented programming languages and techniques
● Solid understanding of concurrency and concurrent programming techniques
● Solid understanding of distributed computing techniques
● Knowledge and understanding of operating with data in large scale
 
years experience operating and deploying solutions using AWS Service including S3, Lake Formation,
SQS, SNS, Lambdas, Athena, Glue, Kinesis and Kafka/MSK.
● years experience with one or more of the following programming languages: Python, Java and Scala.
● years experience with big data technologies such as Redshift, Spark, MongoDB, Parquet, Pandas,
SQL, etc.
● years experience with scripting programming languages such as Python and NodeJS.
● years experience with developing serverless data engineering infrastructure using Terraform and
managing data pipeline deployments, automation and building data observability management solutions
to manage high SLA and reliability requirements.
● Basic understanding of Machine Learning and Data Science concepts is a plus.
Keyskills - Must Have
 AWS Lambda
 
 AWS SQS
 
 AWS SNS
 
 AWS Glue
 
 AWS Athena
 
 AWS Kinesis
 
 Python
 
 Java
 
 Redshift
 
 Spark
 

 MongoDB
 
 AWS Redshift
 
 NodeJS
 
 Terraform
 
 Agile
 
 Gitlab
 
 JIRA
 
 Test driven development
Keyskills - Nice to Have
 Data Science
 
 Machine Learning
    '''
    with open(file_path, 'r') as file:
        cv_file_name = os.path.basename(file_path)
        ext = os.path.splitext(file_path)[1].lower()
        if ext == '.txt':
            print(f"processing file {cv_file_name}{ext}")
            resume_text = read_and_parse_txt(file_path)
        elif ext == '.pdf':
            print(f"processing file {cv_file_name}{ext}")
            resume_text = read_and_parse_pdf(file_path)
        elif ext == '.docx':
            print(f"processing file {cv_file_name}{ext}")
            resume_text = read_and_parse_docx(file_path)
        else:
            resume_text = "Unsupported file type"

        resume_words = preprocess_text(resume_text)
        job_description_words = preprocess_text(jd_text)
        match_percentage = keyword_matching(resume_words, job_description_words)
        return cv_file_name, f"{match_percentage:.2f}%"


def process_files_in_parallel(folder_path):
    files = [os.path.join(folder_path, file) for file in os.listdir(folder_path) if
             os.path.isfile(os.path.join(folder_path, file))]

    with concurrent.futures.ThreadPoolExecutor() as executor:
        results = list(executor.map(read_and_parse_file, files))

    return results


def save_dict_to_csv(data_dict):
    with open(csv_report_path, 'w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(['FileName', 'ParsedData'])  # Writing header
        for key, value in data_dict.items():
            writer.writerow([key, value])


def main():
    result_dict = {}
    parsed_files = process_files_in_parallel(folder_path)
    for parsed_file in parsed_files:
        print(parsed_file)
        result_dict[parsed_file[0]] = parsed_file[1]

    save_dict_to_csv(result_dict, )


if __name__ == "__main__":
    main()
