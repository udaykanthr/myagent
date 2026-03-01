import sys
import os

sys.path.append(os.path.abspath(r"C:\Users\udayk\gitrepos\agentchanti"))
from multi_agent_coder.kb.local.manifest import Manifest

def main():
    db_path = os.path.join(os.getcwd(), ".agentchanti", "kb", "local", "index.db")
    if not os.path.exists(db_path):
        print(f"DB not found at {db_path}")
        return
    
    m = Manifest(db_path)
    files = m.get_all_files()
    if not files:
        print("No files in manifest")
        return
    
    for file_record in files[:3]:
        print(f"File: {file_record.path}")
        print(f"Symbols: {[s.name for s in file_record.symbols][:5]}")

if __name__ == "__main__":
    main()
