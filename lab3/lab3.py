import pandas as pd
import re
import csv
from bs4 import BeautifulSoup
import matplotlib.pyplot as plt
import seaborn as sns
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk
import ssl
import gdown

# Отключаем проверку SSL для NLTK
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

# Загружаем данные NLTK
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()


# Функция для очистки текста
def clean_text(text):
    if pd.isna(text):
        return ""
    text = BeautifulSoup(text, "html.parser").get_text()
    text = re.sub(r'<code>.*?</code>', '', text, flags=re.DOTALL)
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    words = text.split()
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    return ' '.join(words)


# Правильные URL для загрузки с Google Drive
questions_url = "https://drive.google.com/uc?id=1DP6yJRTjbklI7uM4VCQAQavffIqpf-YY"
answers_url = "https://drive.google.com/uc?id=1SskYtkXLLPIUIDT9kYL8VAYr2AXCCxMC"
tags_url = "https://drive.google.com/uc?id=1QDXBI5NsLryLfdQKTTDbSqlCty3nX4ZP"

# Локальные пути для сохранения загруженных файлов
questions_path = "Questions.csv"
answers_path = "Answers.csv"
tags_path = "Tags.csv"

try:
    print("Загрузка Questions.csv...")
    gdown.download(questions_url, questions_path, quiet=False)
    print("Загрузка Answers.csv...")
    gdown.download(answers_url, answers_path, quiet=False)
    print("Загрузка Tags.csv...")
    gdown.download(tags_url, tags_path, quiet=False)

    # Чтение данных
    questions = pd.read_csv(questions_path, encoding='latin1', nrows=100000, quoting=csv.QUOTE_ALL, escapechar='\\',
                            on_bad_lines='skip')
    answers = pd.read_csv(answers_path, encoding='latin1', nrows=100000, quoting=csv.QUOTE_ALL, escapechar='\\',
                          on_bad_lines='skip')
    tags = pd.read_csv(tags_path, encoding='latin1', nrows=100000, quoting=csv.QUOTE_ALL, escapechar='\\',
                       on_bad_lines='skip')
except Exception as e:
    print(f"Ошибка при загрузке данных: {e}")
    raise

print(f"Количество строк в Questions: {len(questions)}")
print(f"Количество строк в Answers: {len(answers)}")
print(f"Количество строк в Tags: {len(tags)}")

# Проверяем наличие необходимых столбцов
print("\nСтолбцы в Questions:", questions.columns.tolist())
print("Столбцы в Answers:", answers.columns.tolist())
print("Столбцы в Tags:", tags.columns.tolist())

# Объединение данных
try:
    merged_data = pd.merge(questions, answers, how='left', left_on='Id', right_on='ParentId',
                           suffixes=('_question', '_answer'))
    tags_grouped = tags.groupby('Id')['Tag'].apply(list).reset_index()
    tags_grouped['Id'] = tags_grouped['Id'].astype(str)
    merged_data = pd.merge(merged_data, tags_grouped, how='left', left_on='Id_question', right_on='Id')

    # Очистка текста
    merged_data['Body_question'] = merged_data['Body_question'].apply(clean_text)
    merged_data['Body_answer'] = merged_data['Body_answer'].apply(clean_text)

    # Анализ данных
    merged_data['Score_question'] = pd.to_numeric(merged_data['Score_question'], errors='coerce')
    num_questions = merged_data['Id_question'].nunique()
    print(f"Количество уникальных вопросов: {num_questions}")

    answers_per_question = merged_data.groupby('Id_question').size()
    avg_answers = answers_per_question.mean()
    print(f"Среднее число ответов на вопрос: {avg_answers:.2f}")

    print("\nРаспределение рейтингов вопросов:")
    print(merged_data['Score_question'].describe())

    all_tags = merged_data['Tag'].explode().value_counts().head(10)
    print("\nТоп-10 самых популярных тегов:")
    print(all_tags)

    programming_languages = ['python', 'java', 'javascript', 'c#', 'c++', 'ruby', 'php', 'sql', 'r', 'go']
    language_counts = merged_data['Tag'].explode()[
        merged_data['Tag'].explode().isin(programming_languages)].value_counts()
    print("\nРаспределение вопросов по языкам программирования:")
    print(language_counts)

    # Ограничиваем значения рейтинга максимальным значением 1000
    merged_data['Score_question_clipped'] = merged_data['Score_question'].clip(upper=1000)

    # Анализ данных
    merged_data['Score_question'] = pd.to_numeric(merged_data['Score_question'], errors='coerce')
    merged_data['Score_question_clipped'] = merged_data['Score_question'].clip(upper=1000)

    # Визуализация распределения рейтингов
    plt.figure(figsize=(15, 5))

    # График 1: Boxplot основного распределения
    plt.subplot(1, 3, 1)
    sns.boxplot(x=merged_data['Score_question_clipped'].clip(upper=50), whis=1.5)
    plt.title('Boxplot рейтингов (основное распределение)\nОграничено до 50 для наглядности')
    plt.xlabel('Рейтинг')

    # График 2: Гистограмма основного распределения
    plt.subplot(1, 3, 2)
    sns.histplot(merged_data['Score_question_clipped'].clip(upper=50), bins=30, kde=True)
    plt.title('Распределение рейтингов (≤50)')
    plt.xlabel('Рейтинг')
    plt.ylabel('Количество вопросов')

    # График 3: Логарифмическое распределение
    plt.subplot(1, 3, 3)
    sns.histplot(merged_data['Score_question_clipped'] + 8, bins=50, kde=True, log_scale=True)
    plt.title('Логарифмическое распределение\n(значения сдвинуты на +8)')
    plt.xlabel('Рейтинг (log scale)')
    plt.ylabel('Частота (log scale)')

    plt.tight_layout()
    plt.show()

    # Дополнительная информация о выбросах
    print("\nИнформация о выбросах:")
    print(f"Максимальный рейтинг: {merged_data['Score_question'].max()}")
    print(f"Количество вопросов с рейтингом > 1000: {len(merged_data[merged_data['Score_question'] > 1000])}")
    print(
        f"Процент вопросов с рейтингом > 1000: {len(merged_data[merged_data['Score_question'] > 1000]) / len(merged_data) * 100:.2f}%")

    # Остальные визуализации (сохранены из оригинального кода)
    plt.figure(figsize=(10, 6))
    all_tags.plot(kind='bar')
    plt.title('Топ-10 самых популярных тегов')
    plt.xlabel('Тег')
    plt.ylabel('Количество вопросов')
    plt.show()

    merged_data['word_count_question'] = merged_data['Body_question'].apply(lambda x: len(x.split()))
    plt.figure(figsize=(10, 6))
    sns.histplot(merged_data['word_count_question'], bins=50, kde=True)
    plt.title('Распределение длины вопросов (по количеству слов)')
    plt.xlabel('Количество слов')
    plt.ylabel('Частота')
    plt.show()

    plt.figure(figsize=(10, 6))
    language_counts.plot(kind='bar')
    plt.title('Распределение вопросов по языкам программирования')
    plt.xlabel('Язык')
    plt.ylabel('Количество вопросов')
    plt.show()

    print("\nПропуски в данных:")
    print(merged_data.isnull().sum())

except KeyError as ke:
    print(f"\nОшибка: отсутствует необходимый столбец - {ke}")
    print("Проверьте структуру загруженных данных.")
except Exception as e:
    print(f"\nПроизошла ошибка при обработке данных: {e}")
