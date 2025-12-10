import openai
from openai import OpenAI
import pandas as pd
import json
import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="EMPTY"
)
model_name = "Qwen/Qwen3-30B-A3B-Instruct-2507-FP8"

def preprocess_prompt(article):
    tag_prompt = f"""
你是一名专业新闻信息抽取助手。请根据以下新闻内容提取关键事实要素，并对新闻生成一句话的摘要：

要素说明：
- "who": 主要人物或组织
- "when": 事件发生时间（尽量具体）
- "where": 事件发生地点
- "what": 事件核心内容（1句简述）
- "why": 事件原因或背景
- "how": 事件过程或方式（若不明确可省略简化）

输入新闻：
<news>
标题：{article["title"]}
内容：{article["content"]}
</news>

请只返回新闻的摘要
"""
    return tag_prompt

def build_llamafactory_sample(article, summary):
    summary_str = str(summary).strip()
    sample = {
        "instruction": "你是一名人民日报的新闻写作助手，请根据给定的新闻摘要撰写一篇完整的新闻报道：",
        "input": summary_json_str,
        "output": f"标题：{article['title']}\n内容：{article['content']}",
        "system": "You are a helpful assistant in writing news articles for People's Daily.",
        "history": []
    }
    return sample

def build_sample_for_article(article):
    resp = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": "You are a helpful assistant in tagging news articles."},
            {"role": "user", "content": preprocess_prompt(article)},
        ],
        n=1,
        max_tokens=512,
    )
    summary_str = resp.choices[0].message.content
    sample = build_llamafactory_sample(article, summary_str)
    return sample

sourse_path = "./data_path"
years = [2021, 2022, 2023, 2024, 2025]
ds_list = {}
for year in years:
    ds_year = pd.read_csv(f"{sourse_path}/RenMin_Daily_{year}.csv")
    ds_year['length'] = ds_year['content'].str.len()
    ds_year = ds_year[ds_year['length'].between(200, 5000)].reset_index(drop=True)
    articles = ds_year.to_dict(orient="records")
    ds_list[year] = articles

all_samples = {
    "samples2021": [],
    "samples2022": [],
    "samples2023": [],
    "samples2024": [],
    "samples2025": [],
}

MAX_WORKERS = 2048

def run(article, year):
    try:
        return build_sample_for_article(article), year
    except Exception as e:
        print(f"Error processing article id {article.get('id', 'unknown')}: {e}")
        return None

with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
    futures = [executor.submit(run, article, year) for year, articles in ds_list.items() for article in articles]

    for future in tqdm.tqdm(as_completed(futures), total=len(futures)):
        sample = future.result()
        if sample is not None:
            sample_data, year = sample
            all_samples[f"samples{year}"].append(sample_data)

for year in years:
    print(f"Year {year}: {len(all_samples[f'samples{year}'])} samples")
    with open(f"news_sft_{year}_200_5000.json", "w", encoding="utf-8") as f:
        json.dump(all_samples[f"samples{year}"], f, ensure_ascii=False, indent=2)
