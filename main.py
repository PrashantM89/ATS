import fitz
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
nltk.download('punkt')
nltk.download('stopwords')


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


def extract_text_from_pdf(pdf_path):
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


def main():
    file_path = "D:\\MyDownloads\\DataScience_CV.pdf"
    # Uncomment and Pass your CV pdf file path else directly pass CV as string below.
    # resume_text = extract_text_from_pdf(file_path)
    resume_text = '''
    Copy Paste your CV string here
    '''
    job_description_text = '''
    Copy Paste your JD string here
    '''
    resume_words = preprocess_text(resume_text)
    job_description_words = preprocess_text(job_description_text)
    match_percentage = keyword_matching(resume_words, job_description_words)
    print(f"Match Percentage: {match_percentage:.2f}%")
    # similarity_score = calculate_similarity(resume_text, job_description_text)
    # print(f"Similarity Score: {similarity_score:.2f}%")


if __name__ == "__main__":
    main()
