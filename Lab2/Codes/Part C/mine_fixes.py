import re, csv, os, sys, subprocess
from pydriller import Repository

REPO_PATH = '/Users/tejasmacipad/Desktop/Third_year/STT/lab2/boxmot'
BUGFIX_RE = re.compile(
    r"\b("
    r"fix|fixed|fixes|hotfix|bug|issue|regression|crash|fail|failure|"
    r"error|assert|hang|leak|overflow|underflow|npe|null|segfault|cve|"
    r"workaround|broken|reproduc(e|ible)|steps?[- ]to[- ]reproduce|"
    r"stack[- ]trace|testcase|coverity"
    r")\b",
    re.IGNORECASE,
)

EXCLUDE_RE= re.compile(r"\b(doc|docs|documentation|readme|typo|spelling|comment)s?\b", re.IGNORECASE)

CODE_EXTS = ('.py', '.c', '.cc', '.cpp', '.cu', '.h', '.hpp')

def merge_check_commit(pathrepo: str, hascom: str) -> list[str]:
    cmd = ["git", "-C", pathrepo, "show", "-m", "--name-only", "--pretty=", hascom]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=False)
    
    if result.returncode != 0:
        return []
        
    paths = [
        line.strip() for line in result.stdout.splitlines() 
        if line.strip() and line.lower().endswith(CODE_EXTS)
    ]
    return sorted(list(set(paths)))


os.makedirs('out', exist_ok=True)
out_path = '/Users/tejasmacipad/Desktop/Third_year/STT/lab2/commits.csv'
try:
    with open(out_path, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['Hash','Message','Hashes of parents','Is a merge commit?','List of modified files'])

        repo = Repository(REPO_PATH)
        count = 0
        for commit in repo.traverse_commits():
            count += 1
            if count % 100 == 0:
                print(f"Processed {count} commits...")
            msg = (commit.msg or '').strip()
            if (not msg or not BUGFIX_RE.search(msg)): continue
            if EXCLUDE_RE.search(msg): continue
            codesfile = []
            if commit.merge:
                codesfile = merge_check_commit(REPO_PATH, commit.hash)
            else:
                for m in commit.modified_files:
                    path = m.new_path or m.old_path or ''
                    if path and path.lower().endswith(CODE_EXTS):
                        codesfile.append(path)
            if not codesfile: continue
            lis_pare = commit.parents or []
            parents_str = ';'.join(lis_pare)
            chkmerge = 'True' if commit.merge else 'False'
            flstr = ';'.join(sorted(list(set(codesfile))))
            w.writerow([commit.hash, msg, parents_str, chkmerge, flstr])
    print(f'done {out_path}')

except Exception as e:
    print(e)


