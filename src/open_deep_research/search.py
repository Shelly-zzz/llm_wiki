import os
import asyncio
import json
import datetime
import requests
import random
import concurrent
import hashlib
import aiohttp
import httpx
import re
import time
from typing import List, Optional, Dict, Any, Union, Literal, Annotated, cast
from urllib.parse import unquote
from collections import defaultdict
import itertools
from pathlib import Path
import wikipedia
import pandas as pd
from bs4 import BeautifulSoup
from langsmith import traceable


def clean_all_newlines(text):
    # 1. 替换所有 Unicode 换行符 + \r\n + \r + \n 为统一分隔符
    text = re.sub(r'[\r\n\u2028\u2029]+', '\n', text)
    # 2. 合并连续分隔符
    text = re.sub(r'\n+', '\n', text)
    # 3. 删除开头/结尾的换行符
    return text.strip()

def extract_social_media_content(json_file_path: Path, person_name: str) -> dict:
    """
    从JSON文件中提取社交媒体内容

    Args:
        json_file_path: JSON文件路径
        person_name: 人物姓名

    Returns:
        包含社交媒体内容的字典
    """
    result = {
        'social_media_content': '',
        'post_count': 0,
        'recent_posts': [],
        'raw_content_addition': '',
        'platforms': set(),
        'total_engagement': {'likes': 0, 'comments': 0, 'forwards': 0}
    }

    try:
        with open(json_file_path, 'r', encoding='utf-8') as f:
            social_data = json.load(f)

        # 提取基本信息
        position = social_data.get('position', '')

        # 提取社交媒体发言
        if 'postlist' in social_data and social_data['postlist']:
            posts = social_data['postlist']
            result['post_count'] = len(posts)
            # 获取最近的发言（最多10条）
            result['recent_posts'] = posts[:20]

            # 构建社交媒体内容摘要
            platforms = set()
            total_likes = 0
            total_comments = 0
            total_forwards = 0

            content_snippets = []
            for post in result['recent_posts']:
                platform = post.get('platform', 'unknown')
                platforms.add(platform)

                # 统计互动数据
                likes = post.get('like_count', 0)
                comments = post.get('comment_count', 0)
                forwards = post.get('forward_count', 0)

                total_likes += likes
                total_comments += comments
                total_forwards += forwards

                # 提取发言内容片段
                full_text = post.get('full_text', '')
                if full_text:
                    # 截取前200个字符作为片段
                    snippet = full_text[:200] + "..." if len(full_text) > 200 else full_text
                    content_snippets.append({
                        'platform': platform,
                        'date': post.get('created_time', ''),
                        'text': snippet,
                        'likes': likes,
                        'comments': comments,
                        'forwards': forwards
                    })

            result['platforms'] = platforms
            result['total_engagement'] = {
                'likes': total_likes,
                'comments': total_comments,
                'forwards': total_forwards
            }

            # 构建内容摘要
            social_media_content = f"""
Social Media Activity Summary for {person_name}:
- Position: {position}
- Total Post Count: {result['post_count']}
- Platforms: {', '.join(platforms)}
- Interaction Statistics: {total_likes} likes, {total_comments} comments, {total_forwards} forwards

Recent Post Excerpt:"""
            for i, snippet in enumerate(content_snippets[:10], 1):
                social_media_content += f"""
{i}. [{snippet['platform']}] {snippet['date']}
   "{snippet['text']}"
   """

            result['social_media_content'] = social_media_content

            # 构建完整的原始内容
            if result['recent_posts']:
                raw_content_addition = f", Social Media Posts: {result['post_count']} posts"
                raw_content_addition += f"\n\n=== {person_name} 社交媒体发言详情 ==="
                if position:
                    raw_content_addition += f"\nPosition: {position}"
                raw_content_addition += f"\nTotal Post Count: {result['post_count']}"
                raw_content_addition += f"\nPlatforms: {', '.join(platforms)}"
                raw_content_addition += f"\nInteraction Statistics: {total_likes} likes, {total_comments} comments, {total_forwards} forwards\n"

                for i, post in enumerate(result['recent_posts'], 1):
                    raw_content_addition += f"""
--- post {i} ---
platform: {post.get('platform', 'N/A')}
created_time: {post.get('created_time', 'N/A')}
user_name: {post.get('user_name', 'N/A')}
full_text: {post.get('full_text', 'N/A')}
interaction statistics: {post.get('like_count', 0)} 点赞, {post.get('comment_count', 0)} 评论, {post.get('forward_count', 0)} 转发
"""
                result['raw_content_addition'] = raw_content_addition

    except Exception as e:
        print(f"Error reading JSON file {json_file_path}: {e}")
        result['social_media_content'] = f"无法读取 {person_name} 的社交媒体数据: {str(e)}"

    return result

def deduplicate_and_format_sources(
        search_response,
        max_tokens_per_source=5000,
        include_raw_content=True,
        deduplication_strategy: Literal["keep_first", "keep_last"] = "keep_first"
):
    """
    Takes a list of search responses and formats them into a readable string.
    Limits the raw_content to approximately max_tokens_per_source tokens.

    Args:
        search_responses: List of search response dicts, each containing:
            - query: str
            - results: List of dicts with fields:
                - title: str
                - url: str
                - content: str
                - score: float
                - raw_content: str|None
        max_tokens_per_source: int
        include_raw_content: bool
        deduplication_strategy: Whether to keep the first or last search result for each unique URL
    Returns:
        str: Formatted string with deduplicated sources
    """
    # Collect all results
    sources_list = []
    for response in search_response:
        sources_list.extend(response['results'])

    # Deduplicate by URL
    if deduplication_strategy == "keep_first":
        unique_sources = {}
        for source in sources_list:
            if source['url'] not in unique_sources:
                unique_sources[source['url']] = source
    elif deduplication_strategy == "keep_last":
        unique_sources = {source['url']: source for source in sources_list}
    else:
        raise ValueError(f"Invalid deduplication strategy: {deduplication_strategy}")

    # Format output
    formatted_text = "Content from sources:\n"
    for i, source in enumerate(unique_sources.values(), 1):
        formatted_text += f"{'=' * 80}\n"  # Clear section separator
        formatted_text += f"Source: {source['title']}\n"
        formatted_text += f"{'-' * 80}\n"  # Subsection separator
        formatted_text += f"URL: {source['url']}\n===\n"
        formatted_text += f"Most relevant content from source: {source['content']}\n===\n"
        if include_raw_content:
            # Using rough estimate of 4 characters per token
            char_limit = max_tokens_per_source * 4
            # Handle None raw_content
            raw_content = source.get('raw_content', '')
            if raw_content is None:
                raw_content = ''
                print(f"Warning: No raw_content found for source {source['url']}")
            if len(raw_content) > char_limit:
                raw_content = raw_content[:char_limit] + "... [truncated]"
            formatted_text += f"Full source content limited to {max_tokens_per_source} tokens: {raw_content}\n\n"
        formatted_text += f"{'=' * 80}\n\n"  # End section separator

    return formatted_text.strip()



@traceable
async def local_json_search(search_queries: list[str], max_results: int = 3) -> list[dict]:
    import asyncio

    def search_localx_sync(query: str, name_column:list):
        results = []
        try:
            json_folder_path= "postinfo"

            for idx, person_name in enumerate(name_column):
                if pd.isna(person_name):  # 跳过空值
                    continue
                person_name_str = str(person_name).strip()
                if (person_name_str.lower() == query.lower() or
                        person_name_str.lower() in query.lower()):

                    json_file_path = Path(json_folder_path) / f"{person_name_str}.json"
                    person_result = {
                        'title': person_name_str,
                        'url': str(json_file_path).strip(),
                        'content': None,
                        'score': 1.0,
                        'raw_content': None
                    }
                    social_data = extract_social_media_content(json_file_path, person_name_str)
                    person_result['content'] = social_data['social_media_content']
                    person_result['raw_content'] = social_data['raw_content_addition']

                    results.append(person_result)
            return {'query': query, 'results': results}
        except:
            results = []
            return {'query': query, 'results': results}


    excel_file_path = "postinfo/wiki_results.xlsx"
    excel_path = Path(excel_file_path)
    if not excel_path.exists():
        print(f"Warning: Excel file {excel_file_path} not found")
        return []
    # 读取Excel文件，假设第二列是姓名，第三列是wiki链接
    df = pd.read_excel(excel_path)
    # 获取第二列（姓名）和第三列（wiki链接）
    name_column = df.iloc[:, 1]  # 第二列

    # 使用线程池执行同步的Wikipedia搜索
    loop = asyncio.get_event_loop()
    tasks = [loop.run_in_executor(None, search_localx_sync, query,name_column) for query in search_queries]
    return await asyncio.gather(*tasks)







@traceable
async def google_search_async(search_queries: Union[str, List[str]], max_results: int = 5,
                              include_raw_content: bool = True):
    """
    Performs concurrent web searches using Google.
    Uses Google Custom Search API if environment variables are set, otherwise falls back to web scraping.

    Args:
        search_queries (List[str]): List of search queries to process
        max_results (int): Maximum number of results to return per query
        include_raw_content (bool): Whether to fetch full page content

    Returns:
        List[dict]: List of search responses from Google, one per query
    """

    # Check for API credentials from environment variables
    api_key = os.environ.get("GOOGLE_API_KEY")
    api_key = 'AIzaSyDT4uoSsRbXaGajCNOll7gFDNEuxNS6uPw'
    cx = os.environ.get("GOOGLE_CX")
    cx = '4390b4fa7a03744d2'
    use_api = bool(api_key and cx)
    print('use_api:',use_api)

    # Handle case where search_queries is a single string
    if isinstance(search_queries, str):
        search_queries = [search_queries]

    # Define user agent generator
    def get_useragent():
        """Generates a random user agent string."""
        lynx_version = f"Lynx/{random.randint(2, 3)}.{random.randint(8, 9)}.{random.randint(0, 2)}"
        libwww_version = f"libwww-FM/{random.randint(2, 3)}.{random.randint(13, 15)}"
        ssl_mm_version = f"SSL-MM/{random.randint(1, 2)}.{random.randint(3, 5)}"
        openssl_version = f"OpenSSL/{random.randint(1, 3)}.{random.randint(0, 4)}.{random.randint(0, 9)}"
        return f"{lynx_version} {libwww_version} {ssl_mm_version} {openssl_version}"

    # Create executor for running synchronous operations
    executor = None if use_api else concurrent.futures.ThreadPoolExecutor(max_workers=5)

    # Use a semaphore to limit concurrent requests
    semaphore = asyncio.Semaphore(5 if use_api else 2)

    async def search_single_query(query):
        async with semaphore:
            try:
                results = []

                # API-based search
                if use_api:
                    # The API returns up to 10 results per request
                    for start_index in range(1, max_results + 1, 10):
                        # Calculate how many results to request in this batch
                        num = min(10, max_results - (start_index - 1))

                        # Make request to Google Custom Search API
                        params = {
                            'q': query,
                            'key': api_key,
                            'cx': cx,
                            'start': start_index,
                            'num': num
                        }
                        url = 'https://www.googleapis.com/customsearch/v1'
                        proxy = os.environ.get("http_proxy")
                        print(f"Requesting {num} results for '{query}' from Google API...")

                        async with aiohttp.ClientSession() as session:
                            print("准备发送请求...")
                            async with session.get(url,params=params,proxy = proxy,timeout=30) as response:
                                #print(f"状态码: {response.status}")
                                if response.status != 200:
                                    error_text = await response.text()
                                    print(f"API error: {response.status}, {error_text}")
                                    break

                                data = await response.json()

                                # Process search results
                                for item in data.get('items', []):
                                    raw_content = item.get('snippet', '')
                                    #print('raw_content:',repr(raw_content))

                                    result = {
                                        "title": item.get('title', ''),
                                        "url": item.get('link', ''),
                                        "content": item.get('snippet', ''),
                                        "score": None,
                                        "raw_content": raw_content
                                    }
                                    #print(json.dumps(result, indent=2, ensure_ascii=False))
                                    results.append(result)

                        # Respect API quota with a small delay
                        await asyncio.sleep(0.2)

                        # If we didn't get a full page of results, no need to request more
                        if not data.get('items') or len(data.get('items', [])) < num:
                            break

                # Web scraping based search
                else:
                    # Add delay between requests
                    await asyncio.sleep(0.5 + random.random() * 1.5)
                    print(f"Scraping Google for '{query}'...")

                    # Define scraping function
                    def google_search(query, max_results):
                        try:
                            lang = "en"
                            safe = "active"
                            start = 0
                            fetched_results = 0
                            fetched_links = set()
                            search_results = []

                            while fetched_results < max_results:
                                # Send request to Google
                                resp = requests.get(
                                    url="https://www.google.com/search",
                                    headers={
                                        "User-Agent": get_useragent(),
                                        "Accept": "*/*"
                                    },
                                    params={
                                        "q": query,
                                        "num": max_results + 2,
                                        "hl": lang,
                                        "start": start,
                                        "safe": safe,
                                    },
                                    cookies={
                                        'CONSENT': 'PENDING+987',  # Bypasses the consent page
                                        'SOCS': 'CAESHAgBEhIaAB',
                                    }
                                )
                                resp.raise_for_status()

                                # Parse results
                                soup = BeautifulSoup(resp.text, "html.parser")
                                result_block = soup.find_all("div", class_="ezO2md")
                                new_results = 0

                                for result in result_block:
                                    link_tag = result.find("a", href=True)
                                    title_tag = link_tag.find("span", class_="CVA68e") if link_tag else None
                                    description_tag = result.find("span", class_="FrIlee")

                                    if link_tag and title_tag and description_tag:
                                        link = unquote(link_tag["href"].split("&")[0].replace("/url?q=", ""))

                                        if link in fetched_links:
                                            continue

                                        fetched_links.add(link)
                                        title = title_tag.text
                                        description = description_tag.text
                                        description = clean_all_newlines(description)

                                        # Store result in the same format as the API results
                                        search_results.append({
                                            "title": title,
                                            "url": link,
                                            "content": description,
                                            "score": 0.8,
                                            "raw_content": description
                                        })

                                        fetched_results += 1
                                        new_results += 1

                                        if fetched_results >= max_results:
                                            break

                                if new_results == 0:
                                    break

                                start += 10
                                time.sleep(1)  # Delay between pages

                            return search_results

                        except Exception as e:
                            print(f"Error in Google search for '{query}': {str(e)}")
                            return []

                    # Execute search in thread pool
                    loop = asyncio.get_running_loop()
                    search_results = await loop.run_in_executor(
                        executor,
                        lambda: google_search(query, max_results)
                    )

                    # Process the results
                    results = search_results

                # If requested, fetch full page content asynchronously (for both API and web scraping)
                if include_raw_content and results:
                    content_semaphore = asyncio.Semaphore(3)

                    async with aiohttp.ClientSession() as session:
                        fetch_tasks = []

                        async def fetch_full_content(result):
                            async with content_semaphore:
                                url = result['url']
                                headers = {
                                    'User-Agent': get_useragent(),
                                    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8'
                                }

                                try:
                                    await asyncio.sleep(0.2 + random.random() * 0.6)
                                    async with session.get(url, headers=headers, timeout=10) as response:
                                        if response.status == 200:
                                            # Check content type to handle binary files
                                            content_type = response.headers.get('Content-Type', '').lower()

                                            # Handle PDFs and other binary files
                                            if 'application/pdf' in content_type or 'application/octet-stream' in content_type:
                                                # For PDFs, indicate that content is binary and not parsed
                                                result['raw_content'] = f"[Binary content: {content_type}. Content extraction not supported for this file type.]"
                                            else:
                                                try:
                                                    # Try to decode as UTF-8 with replacements for non-UTF8 characters
                                                    html = await response.text(errors='replace')
                                                    soup = BeautifulSoup(html, 'html.parser')
                                                    raw_content = soup.get_text()
                                                    raw_content = re.sub(r'\n+', '\n', raw_content)
                                                    raw_content = re.sub(r'\t+', '\t', raw_content)
                                                    result['raw_content'] = raw_content
                                                except UnicodeDecodeError as ude:
                                                    # Fallback if we still have decoding issues
                                                    result['raw_content'] = f"[Could not decode content: {str(ude)}]"
                                except Exception as e:
                                    print(f"Warning: Failed to fetch content for {url}: {str(e)}")
                                    result['raw_content'] = f"[Error fetching content: {str(e)}]"
                                return result

                        for result in results:
                            fetch_tasks.append(fetch_full_content(result))

                        updated_results = await asyncio.gather(*fetch_tasks)
                        results = updated_results
                        print(f"Fetched full content for {len(results)} results")

                return {
                    "query": query,
                    "follow_up_questions": None,
                    "answer": None,
                    "images": [],
                    "results": results
                }
            except Exception as e:
                print(f"Error in Google search for query '{query}': {str(e)}")
                return {
                    "query": query,
                    "follow_up_questions": None,
                    "answer": None,
                    "images": [],
                    "results": []
                }

    try:
        # Create tasks for all search queries
        search_tasks = [search_single_query(query) for query in search_queries]

        # Execute all searches concurrently
        search_results = await asyncio.gather(*search_tasks)

        return search_results
    finally:
        # Only shut down executor if it was created
        if executor:
            executor.shutdown(wait=False)




@traceable
async def wikipedia_search_async(search_queries: list[str], max_results: int = 5) -> list[dict]:
    """
    异步Wikipedia搜索

    Args:
        search_queries: 搜索查询列表
        max_results: 每个查询的最大结果数

    Returns:
        搜索结果列表
    """
    import asyncio

    def search_wikipedia_sync(query: str):
        try:
            # 设置Wikipedia语言（可根据需要修改）
            wikipedia.set_lang("en")

            # 搜索相关页面
            search_results = wikipedia.search(query, results=max_results)

            results = []
            for title in search_results:
                try:
                    # 获取页面摘要
                    summary = wikipedia.summary(title, sentences=3)
                    page = wikipedia.page(title)
                    print('page.url:',page.url)
                    results.append({
                        'title': title,
                        'url': page.url,
                        'content': summary,
                        'score': 0.8,  # Wikipedia给予中等分数
                        'raw_content': page.content[:4000]  # 限制内容长度
                    })
                except wikipedia.exceptions.DisambiguationError as e:
                    # 处理歧义页面，选择第一个选项
                    try:
                        page = wikipedia.page(e.options[0])
                        summary = wikipedia.summary(e.options[0], sentences=3)
                        results.append({
                            'title': e.options[0],
                            'url': page.url,
                            'content': summary,
                            'score': 0.7,
                            'raw_content': page.content[:4000]
                        })
                    except:
                        continue
                except:
                    continue

            return {'query': query, 'results': results}
        except Exception as e:
            print(f"Wikipedia search error for query '{query}': {e}")
            return {'query': query, 'results': []}

    # 使用线程池执行同步的Wikipedia搜索
    loop = asyncio.get_event_loop()
    tasks = [loop.run_in_executor(None, search_wikipedia_sync, query) for query in search_queries]
    return await asyncio.gather(*tasks)


async def select_and_execute_search(search_api: str, query_list: list[str], params_to_pass: dict) -> str:
    """Select and execute the appropriate search API with multi-source capability.

    Args:
        search_api: Name of the search API to use (now supports 'multi_source' for sequential search)
        query_list: List of search queries to execute
        params_to_pass: Parameters to pass to the search API

    Returns:
        Formatted string containing search results

    Raises:
        ValueError: If an unsupported search API is specified
    """

    # 调试信息：打印当前使用的搜索API
    print(f"🔧 调试信息 - 传入的search_api参数: {search_api}")
    # print(f"🔧 调试信息 - 环境变量SEARCH_API: {os.environ.get('SEARCH_API')}")
    print(f"🔧 调试信息 - 搜索查询: {query_list}")


    # 新增：多数据源顺序检索
    if search_api == "multi_source":

        all_results = []
        print("🔍 开始多数据源检索...")
        # 第一步：搜索本地JSON数据库
        print("📁 步骤1: 搜索本地JSON数据库...")
        local_results = await local_json_search(query_list)
        if local_results and any(result['results'] for result in local_results):
            print(f"✅ 本地数据库找到 {sum(len(r['results']) for r in local_results)} 个结果")
        else:
            print("❌ 本地数据库未找到相关结果")

        # 第二步：搜索Wikipedia
        print("📖 步骤2: 搜索Wikipedia...")
        wiki_results = await wikipedia_search_async(query_list, params_to_pass.get('max_results', 5))
        if wiki_results and any(result['results'] for result in wiki_results):
            print(f"✅ Wikipedia找到 {sum(len(r['results']) for r in wiki_results)} 个结果")
        else:
            print("❌ Wikipedia未找到相关结果")

        # 第三步：搜索Google
        print("🌐 步骤3: 搜索Google...")
        google_results = await google_search_async(query_list, params_to_pass.get('max_results', 5))
        if google_results and any(result['results'] for result in google_results):
            print(f"✅ Google找到 {sum(len(r['results']) for r in google_results)} 个结果")
        else:
            print("❌ Google未找到相关结果")

        for q_i,query in enumerate(query_list):
            local_result = local_results[q_i].get('results', [])
            google_result = google_results[q_i].get('results', [])
            wiki_result = wiki_results[q_i].get('results', [])
            _3_results = []
            _3_results.extend(local_result)
            _3_results.extend(google_result)
            _3_results.extend(wiki_result)
            single_q_result = {
                "query": query,
                'results': _3_results
            }
            all_results.append(single_q_result)

        # 格式化所有结果
        if all_results:
            formatted_results = deduplicate_and_format_sources(all_results, max_tokens_per_source=4000,
                                                               deduplication_strategy="keep_first")
            print(f"🎉 多数据源检索完成")
            return formatted_results
        else:
            return "未在任何数据源中找到相关信息。"
    # 单一数据源搜索逻辑
    elif search_api == "local_json":
        search_results = await local_json_search(query_list)
    elif search_api == "googlesearch":
        search_results = await google_search_async(query_list,params_to_pass.get('max_results', 5))
    elif search_api == "wikipedia":
        search_results = await wikipedia_search_async(query_list, params_to_pass.get('max_results', 5))
    else:
        raise ValueError(f"Unsupported search API: {search_api}")

    return deduplicate_and_format_sources(search_results, max_tokens_per_source=4000,
                                          deduplication_strategy="keep_first")





