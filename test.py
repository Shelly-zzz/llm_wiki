#!/usr/bin/env python3
"""
验证工作流是否正常
"""

import os
import sys
import asyncio
from dotenv import load_dotenv
import uuid

# 加载环境变量
import os
load_dotenv("./.env")

# 添加项目路径
sys.path.insert(0, os.getcwd())


# REPORT_STRUCTURE = """Use this structure to create a report on the user-provided topic:

# 1. Introduction (no research needed)
#    - Brief overview of the topic area

# 2. Main Body Sections:
#    - Each section should focus on a sub-topic of the user-provided topic
   
# 3. Conclusion
#    - Aim for 1 structural element (either a list of table) that distills the main body sections 
#    - Provide a concise summary of the report"""
   
   
REPORT_STRUCTURE = """Use this structure to create a report on the user-provided topic:

1. Introduction
    - Brief introduction of the subject
    
2. Main Body
    - 2.1 Work Responsibilities and Policy Positions
        - Conclusion
        - Evidence
    - 2.2 Stance on China
        - Overall attitude
        - Position on key issues
        - Conclusion
        - Evidence
    - 2.3 Personality Traits and Decision-Making Style
        - Conclusion
        - Evidence
    - 2.4 External Evaluation and Social Controversies
        - Conclusion
        - Evidence
        
3. Conclusion

4. Summary
    - Summarize main findings in a table or list 
    - Concise overall assessment"""
   
thread = {"configurable": {"thread_id": str(uuid.uuid4()),
                           "search_api": "tavily",
                           "planner_provider": "openai",
                           "planner_model": "gpt-4o",
                           # "planner_model_kwargs": {"temperature":0.8}, # if set custom parameters
                           "writer_provider": "openai",
                           "writer_model": "gpt-4o",
                           # "writer_model_kwargs": {"temperature":0.8}, # if set custom parameters
                           "max_search_depth": 2,
                           "report_structure": REPORT_STRUCTURE,
                           }}   

async def quick_test():
    
    try:
        # 导入模块
        from src.open_deep_research.graph import graph
        from src.open_deep_research.state import ReportState
        
        print("导入src.open_deep_research.graph中定义的结构")
        
        # 创建简单测试状态
        test_state = ReportState(
            topic="介绍Barry Loudermilk",
        )
        
        print(f"主题: {test_state['topic']}")
        
        # 运行工作流
        final_state = None
        step_count = 0
        
        print("开始测试工作流...")
        print("Original state:", )
        print(test_state)
        
        async for state in graph.astream(test_state, thread):
            
            step_count += 1
            print('-'*40)
            print(f"📍 步骤 {step_count}")
            print('当前状态:', state.keys())
            
            
            print('-'*40)
            
            final_state = state
            

            
    except Exception as e:
        print(f"❌ 测试出错: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(quick_test())
