import subprocess
from .common import *
def main():
 if (HERE/'results.json').exists() and (HERE/'REPORT.md').exists():print('Already completed; no repeated inference');return
 for args in [('generate','--model','original'),('generate','--model','rmu80'),('audit',),('summarize',)]:
  print('START',args,flush=True)
  subprocess.run([sys.executable,'-m','safety.truncation_diagnostic.'+args[0],*args[1:]],cwd=ROOT,check=True)
 print('DIAGNOSTIC_COMPLETE_STOP',flush=True)
if __name__=='__main__':main()
