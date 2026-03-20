import os
import re
import google.auth
from googleapiclient.discovery import build
from tqdm import tqdm
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build

def drive_service_interactive():
    SCOPES = ['https://www.googleapis.com/auth/drive.readonly']
    flow = InstalledAppFlow.from_client_secrets_file(
        'client_secrets.json', SCOPES
    )
    creds = flow.run_local_server(port=0)
    return build("drive", "v3", credentials=creds)

def drive_service(creds=None, readonly=True):
    scopes = ["https://www.googleapis.com/auth/drive.readonly" if readonly
              else "https://www.googleapis.com/auth/drive"]
    if creds is None:
        creds, _ = google.auth.default(scopes=scopes)
    return build("drive", "v3", credentials=creds, cache_discovery=False)


def resolve_component(service, parent_id, name, is_folder):
    q = (
        f"name = '{name}' and "
        f"'{parent_id}' in parents and "
        f"mimeType {( '=' if is_folder else '!=' )} 'application/vnd.google-apps.folder' and "
        "trashed = false"
    )
    r = service.files().list(q=q, fields="files(id)", pageSize=2).execute()["files"]
    if len(r) == 1:
        return r[0]["id"]
    return None


def drive_path_to_id(service, path_str):
    comps = path_str.strip("/").split("/")
    parent = "root"
    for i, comp in enumerate(comps):
        is_folder = i < len(comps) - 1
        cid = resolve_component(service, parent, comp, is_folder)
        if cid is None:
            return None
        parent = cid
    return parent     # ID of final element


def collect_e_ids(local_root, d_regex=r".+", target="e.h5"):
    service = drive_service()
    mapping = {}
    prog = re.compile(d_regex)

    for c in tqdm(os.listdir(local_root), desc="c‑folders"):
        path_c = os.path.join(local_root, c)
        if not os.path.isdir(path_c):
            continue

        for d in os.listdir(path_c):
            if not prog.fullmatch(d):
                continue
            path_d = os.path.join(path_c, d)
            file_path = os.path.join(path_d, target)
            if not os.path.isfile(file_path):
                continue

            # build Drive path "a/b/c/d/e.h5" relative to Drive root
            rel_drive_path = os.path.relpath(file_path, start=local_root).replace(os.sep, "/")
            fid = drive_path_to_id(service, rel_drive_path)
            if fid:
                key = "_".join(rel_drive_path.split("/")[:3])   # a_b_c
                mapping[key] = fid
    return mapping



local_parent = r"I:\\My Drive\\Scene-5\\f-28.0mm"   # adjust to your mount point
ids = collect_e_ids(local_parent, d_regex=r"^stereocal_results_.*$", target = "rectified\\rectified_lefts.h5")   # e.g. d = any name starting with "d"
print(ids)