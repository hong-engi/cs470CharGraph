import requests
import re
import nltk
from nltk.tokenize import sent_tokenize

# nltk에서 punkt tokenizer가 필요합니다.
nltk.download('punkt')         # 일반적으로 필요한 문장 분리기
nltk.download('punkt_tab')     # 일부 환경에서 추가로 요구됨


def download_gutenberg_text(book_id: int) -> str:
    """
    Project Gutenberg 책의 텍스트를 다운로드합니다.
    book_id: Project Gutenberg 책 ID (예: 1342 for Pride and Prejudice)
    """
    url = f"https://www.gutenberg.org/files/{book_id}/{book_id}-0.txt"  # '-0.txt' 버전
    resp = requests.get(url)
    if resp.status_code != 200:
        # '-0.txt'가 없으면 기본 txt 시도
        url = f"https://www.gutenberg.org/files/{book_id}/{book_id}.txt"
        resp = requests.get(url)
        if resp.status_code != 200:
            raise ValueError(f"책 ID {book_id}에 해당하는 텍스트를 찾을 수 없습니다.")
    return resp.text

def clean_gutenberg_text(text: str) -> str:
    """
    Project Gutenberg 텍스트에서 메타데이터(앞뒤 라이센스 및 책 소개)를 제거합니다.
    보통 시작부분과 끝부분의 특수 문구 기준으로 제거합니다.
    """
    # 시작부분
    start_match = re.search(r"\*\*\* START OF (THIS|THE) PROJECT GUTENBERG EBOOK.*\*\*\*", text, re.IGNORECASE)
    if start_match:
        text = text[start_match.end():]
    # 끝부분
    end_match = re.search(r"\*\*\* END OF (THIS|THE) PROJECT GUTENBERG EBOOK.*\*\*\*", text, re.IGNORECASE)
    if end_match:
        text = text[:end_match.start()]
    return text.strip()

def parse_text_into_sentences(text: str):
    """
    텍스트를 문장 단위로 분할하여 리스트로 반환합니다.
    """
    sentences = sent_tokenize(text)
    return sentences


def save_sentences_to_file(sentences, book_id: int, output_dir="novel"):
    """
    문장 리스트를 파일로 저장합니다.
    각 문장은 한 줄에 하나씩 저장됩니다.
    """
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, f"{book_id}.txt")
    with open(filepath, 'w', encoding='utf-8') as f:
        for sentence in sentences:
            f.write(sentence.strip() + "\n")
    print(f">>> {filepath} 저장 완료 ({len(sentences)} 문장)")

if __name__ == "__main__":
    book_id = 11051  # 예: Pride and Prejudice
    raw_text = download_gutenberg_text(book_id)
    clean_text = clean_gutenberg_text(raw_text)
    sentences = parse_text_into_sentences(clean_text)
    save_sentences_to_file(sentences, book_id)

    print(f"총 문장 수: {len(sentences)}")
    print("처음 5문장 예시:")
    for s in sentences[:5]:
        print("-", s)
