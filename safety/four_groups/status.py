"""Read-only progress, no model loading and no test text output."""
from .common import *
def main():
 c=db();counts=c.execute("SELECT json_extract(result,'$.model'),json_extract(result,'$.status'),count(*) FROM records WHERE task='generation' GROUP BY 1,2").fetchall()
 audits=c.execute("SELECT task,json_extract(result,'$.status'),count(*) FROM records WHERE task!='generation' GROUP BY 1,2").fetchall()
 state=json.loads((LOCAL/'orchestration.json').read_text()) if (LOCAL/'orchestration.json').exists() else None
 print(json.dumps(dict(generations=counts,audits=audits,orchestration=state),ensure_ascii=False))
if __name__=='__main__':main()
