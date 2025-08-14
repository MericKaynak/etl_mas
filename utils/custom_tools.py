
from langchain_experimental.tools.python.tool import PythonREPLTool
from langchain_community.utilities.sql_database import SQLDatabase
from langchain_community.agent_toolkits.sql.toolkit import SQLDatabaseToolkit
from langchain_community.utilities.sql_database import SQLDatabase
from dotenv import load_dotenv
import os

class QuietPythonREPLTool(PythonREPLTool):
    def _run(self, query: str) -> str:
        result = super()._run(query)
        if len(result) > 2000:
            return result[:2000] + "\n... (output was cut, it is too long)"
        return result

def get_postgres_toolkit(llm):
    pg_user = os.getenv("POSTGRES_USER")
    pg_pass = os.getenv("POSTGRES_PASSWORD")
    pg_db = os.getenv("POSTGRES_DB")
    pg_host = os.getenv("POSTGRES_HOST")
    pg_port = os.getenv("POSTGRES_PORT")
    db_uri = f"postgresql+psycopg2://{pg_user}:{pg_pass}@{pg_host}:{pg_port}/{pg_db}"
    db = SQLDatabase.from_uri(db_uri)
    sql_toolkit = SQLDatabaseToolkit(db=db, llm=llm)
    return sql_toolkit.get_tools() 