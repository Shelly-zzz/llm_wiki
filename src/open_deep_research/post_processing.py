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
            source_lines = re.findall(
                r'\[?(\d+)\]?\.?\s*(.*?)[：:]\s*((?:https?:\/\/|www\.)\S+(?:\s+\S+)*|\S+(?:\s+\S+)*\.json)',
                sources_text
            )
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
    test_str = """# 巴里·劳德米尔克

巴里·劳德米尔克是佐治亚州第十一选区的美国众议员，在金融服务委员会和众议院行政委员会中担任重要职务。作为共和党研究委员会的核心人物，劳德米尔克推广财政保守主义和国家安全，始终支持减少税收、加强国防和控制移民的措施。他的立法行动通常反映出他的外交政策立场，特别是批评中国的经济影响力并倡导网络安全增强。劳德米尔克的军旅生涯支持了他的决策风格，体现为对宪法的忠诚和果敢的保守领导。然而，他的职业生涯也不乏争议，包括他对1月6日委员会的批评，反映出他分裂的政治立场和更广泛的保守目标。

## 工作职责和政策立场

巴里·劳德米尔克作为佐治亚州第十一国会选区的美国众议员，通过他的委员会任命积极参与立法进程。他被任命到美国众议院金融服务委员会，该委员会处理银行、住房和国家金融政策等问题，以及负责监督众议院运作和联邦选举的众议院行政委员会[1][2]。劳德米尔克也是众议院内一个著名保守派派系——共和党研究委员会的有影响力的成员[1]。

劳德米尔克支持保守的政策立场，积极支持国防、财政保守主义和严格的移民控制。他的投票记录符合这些原则，体现在他支持诸如特朗普总统的减税、扩展能源部门和保卫国界的立法措施[3]。通过公开声明，他强调对以色列的支持以及对与伊朗外交处理的批评，重申了他在全球安全和防御上的立场[4]。他的立法工作始终关注经济增长和维护国家安全，这与他的公开和委员会活动相一致[3][4]。

### 资料来源
[1] 美国众议院书记官办公室 - 巴里·劳德米尔克: https://clerk.house.gov/members/L000583  
[2] 家 - 美国众议员巴里·劳德米尔克: https://loudermilk.house.gov/  
[3] 巴里·劳德米尔克: postinfo\Barry Loudermilk.json  
[4] 投票记录 - 美国众议员巴里·劳德米尔克: https://loudermilk.house.gov/voterecord/  

## 对中国的立场

来自佐治亚州的共和党议员巴里·劳德米尔克对中国持批判态度，专注于经济、安全和立法领域。他表达了对不公平贸易做法的担忧，呼应了更广泛的共和党旨在抵制中国经济战略的目标。劳德米尔克支持特朗普总统的政策，旨在重写全球贸易惯例并加征关税，这突显了他致力于抑制中国经济影响力的承诺[1]。

劳德米尔克还强调国家安全问题，尤其是在网络安全和技术领域。他倡导加强美国的网络安全，以防范来自中国的潜在威胁[2]。此外，劳德米尔克共同发起了旨在限制与中国共产党有关联实体资本访问的立法，展示了他限制中国经济杠杆的积极态度[1]。

通过这些行动和声明，劳德米尔克反映了更广泛的共和党策略，旨在制衡中国日益增强的全球影响力，重点是保护美国经济利益和维护国家安全。他的立法和政策努力旨在追究外国实体，特别是中国公司，责任，强调需要安全和公平的经济实践。

### 资料来源
[1] 巴里·劳德米尔克社交媒体总结: Facebook帖子，2021-2025 - postinfo\Barry Loudermilk.json  
[2] 评估使用制裁解决国家安全和外交政策挑战 | Congress.gov - https://congress.gov/116/chrg/CHRG-116hhrg37927/CHRG-116hhrg37927.htm  

## 人格特征和决策风格

巴里·劳德米尔克以坚定的宪法保守主义和根深蒂固的基督教信仰为特征，经常在立法行动和公开互动中强调这些价值观[3]。他作为美国空军声纳技术员的军事背景在塑造他的方法中发挥了重大作用，重点在于纪律、爱国主义和国家安全。劳德米尔克不断表达对军队和国家象征的钦佩，体现在他的社交媒体帖子庆祝国旗日和纪念军事成就[1]。

劳德米尔克的决策风格是果断的保守主义，这通过他在国际问题上的坚定立场尤其体现在伊朗和以色列上。他频繁强调伊朗核野心构成的存在威胁，并大声支持以色列的自卫权[1][3]。

他对宪法原则的承诺进一步指导他的立法议程，始终将其政策倡议与传统的美国价值观保持一致。这种奉献在他的演讲和公开谈话中有所体现，经常引用宪法忠诚[5]。总的来说，劳德米尔克的决策受到他的军事经验、宪法保守主义和保护国家利益的承诺的强烈影响。

### 资料来源
[1] 巴里·劳德米尔克Facebook帖子摘录: postinfo\Barry Loudermilk.json  
[3] 巴里·劳德米尔克 - 国会议员 - 美国众议院...: https://www.linkedin.com/in/barry-loudermilk-8b246425  
[5] 代表巴里·劳德米尔克 - 佐治亚州，十一，任职中 - 简历"""
    print(format_adjusting(test_str))