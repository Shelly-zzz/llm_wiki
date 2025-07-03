import re
from collections import OrderedDict


def format_adjusting(all_sections: str) -> str:
    """
    Processes a report string to consolidate and re-number sources and update in-text citations.

    Args:
        all_sections: A string containing the entire report with multiple sections and source lists.

    Returns:
        A string with a single, deduplicated, and re-numbered source list at the end,
        and all in-text citations updated to match the new numbering.
    """
    # Split the report into sections. We use a positive lookahead to keep the delimiter '## '.
    sections = re.split(r'(?=\n## )', all_sections.strip())

    # Data structures to hold parsed information
    # { (section_index, old_citation_number): "url" }
    old_citation_to_url = {}
    # { "url": {"desc": "description", "url": "url"} }
    unique_sources_map = OrderedDict()
    # List to hold the content part of each section, without the sources
    content_parts = []

    # 1. PARSE AND EXTRACT SOURCES
    for i, section_text in enumerate(sections):
        # Split content and source block within each section
        parts = re.split(r'### 资料来源\s*', section_text, maxsplit=1)
        content = parts[0]
        content_parts.append(content)

        if len(parts) > 1:
            sources_text = parts[1]
            # Find all source lines. A source line is assumed to start with a number/bracket.
            source_lines = re.findall(r'\[?(\d+)\]?\.?\s*(.*?)[：:]\s*((?:https?://|www\.)\S+|\S+\.\S{2,})', sources_text)

            for old_num_str, desc, url in source_lines:
                old_num = int(old_num_str)
                url = url.strip()
                desc = desc.strip()

                # Map the original citation (scoped by its section) to its URL
                old_citation_to_url[(i, old_num)] = url

                # Add to unique sources, preserving the first description found
                if url not in unique_sources_map:
                    unique_sources_map[url] = {'desc': desc, 'url': url}

    # 2. CREATE NEW SOURCE LIST AND URL-TO-NEW-NUMBER MAPPING
    # { "url": new_number }
    url_to_new_num = {url: i + 1 for i, url in enumerate(unique_sources_map.keys())}

    final_source_list = ["## 资料来源"]
    for url, new_num in url_to_new_num.items():
        desc = unique_sources_map[url]['desc']
        final_source_list.append(f"[{new_num}] {desc}: {url}")

    # 3. REPLACE IN-TEXT CITATIONS AND REBUILD THE REPORT
    processed_content_parts = []
    for i, content in enumerate(content_parts):

        # Define a replacer function (a closure) to get access to the current section index `i`
        def create_replacer(section_index):
            def replacer_func(match):
                # Extract the original number from the citation, e.g., '1' from '[1]' or '[[1]]'
                old_num = int(match.group(1))

                # Find the URL using the old section index and number
                url = old_citation_to_url.get((section_index, old_num))
                if url:
                    # Find the new number using the URL
                    new_num = url_to_new_num.get(url)
                    if new_num:
                        # Return the newly formatted citation
                        return f'[{new_num}]'

                # If lookup fails at any point, return the original match to avoid errors
                return match.group(0)

            return replacer_func

        replacer = create_replacer(i)

        # Use a regex to find all citations, including those with extra brackets like [[1]]
        # It finds one or more brackets, captures the digits inside, then finds one or more closing brackets.
        # This correctly handles [1], [[1]], and applies the replacement to each number in [5][6] sequentially.
        processed_content = re.sub(r'\[+([0-9]+)\]+', replacer, content)
        processed_content = re.sub(r'\]\[', '] [', processed_content)
        processed_content_parts.append(processed_content)

    # Assemble the final report
    final_body = "".join(processed_content_parts).strip()
    final_sources_str = "\n\n" + "\n".join(final_source_list)

    return final_body + final_sources_str

if __name__ == '__main__':
    test_str = """### 资料来源 
1. 测试测试：http://www.test.com
## 性格特征和决策风格 
亚当·希夫，作为著名的美国立法者，以其在公共辩论中的强有力沟通技巧和坚定的态度而著称。他的社交媒体活动经常用作倡导正义和平等的平台，突出其原则立场。例如，希夫对司法部的批评强调了他对问责无关政治身份的信念，这一立场通过他坚持“没有人凌驾于法律之上”体现出来[1]。此外，他积极处理诸如医疗保健可及性和消费者权益等问题，反映出对公众福利的关心[1]。 希夫的决策以在伦理考虑基础上的慎重审议为特点。他的立法历史揭示了与州领导合作以促进经济增长和保护基本自由的一贯主题[1]。他在公共沟通中的方法显示了一种对透明度和保障民主原则的承诺。这些元素融合在一起，定义了他的性格和决策风格，其中饱含对正义、平等和有效治理的奉献。 
### 资料来源
1. 亚当·希夫：postinfo/亚当·希夫.json """
    print(format_adjusting(test_str))