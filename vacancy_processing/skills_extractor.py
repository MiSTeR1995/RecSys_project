import torch
from transformers import AutoTokenizer, AutoModel
from scipy.spatial.distance import cosine
import os
import json
import time
from tqdm import tqdm
import pandas as pd
import numpy as np
from collections import Counter


def parse_embedding_from_str(str_emb):
    emb = []
    for part in str_emb.strip()[1:-1].split(' '):
        fixed_part = part.strip()
        if len(fixed_part) > 0:
            emb.append(float(fixed_part))

    return emb


def load_embeddings(folder='data/embeddings'):
    all_embeddings = []

    for emb_file in tqdm(os.listdir(folder)):
        emb_df = pd.read_csv(f'{folder}/{emb_file}')
        all_embeddings.append(emb_df)

    merged_embeddings = pd.concat(all_embeddings)

    emb_dict = {}
    for _, row in tqdm(merged_embeddings.iterrows()):
        emb_dict[row['title']] = np.array(
            parse_embedding_from_str(row['embeddings'])
        ).astype(np.float32)

    return emb_dict


class EmbeddingExtractor:
    def __init__(self, model, tokenizer, similarity_metric=cosine, initial_embeddings={}):
        self.embeddings = initial_embeddings
        self.model = model
        self.tokenizer = tokenizer
        self.similarity_metric = similarity_metric

    def extract(self, text):
        if text in self.embeddings:
            return self.embeddings[text]

        encoded_input = self.tokenizer(
            [text], padding=True, truncation=True, max_length=64, return_tensors='pt'
        )
        with torch.no_grad():
            model_output = self.model(**encoded_input)
        embedding = model_output.pooler_output[0].numpy().astype(np.float32)

        self.embeddings[text] = embedding

        return embedding

    def extract_batch(self, texts):
        encoded_input = self.tokenizer(
            texts, padding=True, truncation=True, max_length=64, return_tensors='pt'
        )
        with torch.no_grad():
            model_output = self.model(**encoded_input)

        for i in range(len(texts)):
            text = texts[i]
            embedding = model_output.pooler_output[i]

            self.embeddings[text] = embedding

    def similarity(self, emb1, emb2):
        return 1 - self.similarity_metric(emb1, emb2)


class VacancyFinder:
    def __init__(
            self,
            embedding_extractor,
            vacancies_path='data/vacancies'
    ):
        self.embedding_extractor = embedding_extractor
        self.vacancies_path = vacancies_path
        self.titles = {}
        self.vacancy_stats_by_title = {}

        self._load_vacancies()

    def _load_vacancies(self):
        titles = os.listdir(self.vacancies_path)

        for title in titles:
            print(title)
            self.titles[title] = self.embedding_extractor.extract(title)
            self.vacancy_stats_by_title[title] = []
            for vacancy in os.listdir(f'{self.vacancies_path}/{title}'):
                vacancy_path = f'{self.vacancies_path}/{title}/{vacancy}'
                with open(vacancy_path, 'r') as f:
                    vacancy_config = json.load(f)

                name = vacancy_config['name']
                key_skills = vacancy_config['key_skills']
                id_ = vacancy[:-5]

                self.vacancy_stats_by_title[title].append([
                    name, key_skills, id_, self.embedding_extractor.extract(name)
                ])

    def _select_best_titles(self, emb, amount):
        stats = []
        for title_name, title_emb in self.titles.items():
            stats.append([title_name, title_emb, self.embedding_extractor.similarity(emb, title_emb)])

        return sorted(stats, key=lambda x: -x[-1])[:amount]

    def _select_best_vacancies(self, emb, titles, amount):
        stats = []
        for title in titles:
            for vacancy_name, key_skills, vacancy_id, vacancy_emb in self.vacancy_stats_by_title[title]:
                stats.append([
                    vacancy_name, key_skills, vacancy_id,
                    vacancy_emb, title,
                    self.embedding_extractor.similarity(emb, vacancy_emb)
                ])

        return sorted(stats, key=lambda x: -x[-1])[:amount]

    def get_best_vacancies(self, vacancy_name, nearest_titles=3, amount=20):
        emb = self.embedding_extractor.extract(vacancy_name)
        titles = self._select_best_titles(emb, nearest_titles)
        title_names = [title_name for title_name, _, _ in titles]

        best_vacancies = self._select_best_vacancies(emb, title_names, amount)
        return best_vacancies


def aggregate_skills(all_skills):
    skills_list = []
    for key_skills in all_skills:
        for skill_map in key_skills:
            skills_list.append(skill_map['name'])

    return Counter(skills_list)


def key_skills_for_professions(vacancy_finder, professions):
    final_skills = []

    for vacancy in professions:
        all_key_skills = []
        for vacancy_name, key_skills, vacancy_id, _, title, similarity in vacancy_finder.get_best_vacancies(vacancy,
                                                                                                            amount=50):
            all_key_skills.append(key_skills)

        skills = [name for name, amount in aggregate_skills(all_key_skills).most_common(10)]
        final_skills.append(skills)

    return pd.DataFrame.from_dict({'vacancy': professions, 'skills': final_skills})


def solve():
    tokenizer = AutoTokenizer.from_pretrained("cointegrated/LaBSE-en-ru")
    model = AutoModel.from_pretrained("cointegrated/LaBSE-en-ru")
    emb_dict = load_embeddings()
    embedding_extractor = EmbeddingExtractor(model, tokenizer, initial_embeddings=emb_dict)
    vacancy_finder = VacancyFinder(embedding_extractor)

    skills_df = key_skills_for_professions(vacancy_finder, ['Python developer', 'C++ программист', 'Дизайнер'])
    return skills_df


if __name__ == '__main__':
    solve()




