


extract_schema_prompt = '''
Given some tables, I want to answer a question: {query}
Given a query, think about what kind of table schema (column names) might contain information to answer this query.
Please answer with a list of column names in JSON format without any additional explanation.
Example:
["column1", "column2", "column3"]
'''

judge_table_prompt = '''
Given a table, I want to answer a question: {query}
Table Information:
{table_info}

Without giving the values of the table, please analyze if this table is relevant to this question, and if it ispossible that this table contains the information to answer the question.

Follow this format strictly:
<Thought>
your analysis about the table and the question
</Thought>

<Answer>
[Only "Yes" or "No"]
</Answer>

Now analyze the current case:
'''