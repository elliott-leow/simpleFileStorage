#!/usr/bin/env python3
from flask import Flask, render_template, send_from_directory, abort, request, jsonify, session # Added session
import os
import humanize
from datetime import datetime
import urllib.parse
import shutil
from werkzeug.utils import secure_filename
from werkzeug.exceptions import RequestEntityTooLarge
from dotenv import load_dotenv
import json # For loading the config file
import pickle
import numpy as np
import time

# --- Semantic Search (Optional - Ensure search_index.py is present if using) ---
try:
    import search_index
    MODEL_LOADED = search_index.model is not None
    SEMANTIC_INDEX_DATA = search_index.semantic_index if MODEL_LOADED else None
    print(f"Semantic Search Ready: {MODEL_LOADED}")
    if SEMANTIC_INDEX_DATA is None and MODEL_LOADED:
        print("Semantic index not loaded. Use /rebuild-index endpoint.")
except ImportError as e:
    print(f"Could not import search_index module: {e}")
    print("Semantic search functionality will be DISABLED.")
    MODEL_LOADED = False
    SEMANTIC_INDEX_DATA = None
except Exception as e: # Catch broader exceptions during index/model load
     print(f"Error during search_index import/init: {e}")
     MODEL_LOADED = False
     SEMANTIC_INDEX_DATA = None
# --- End Semantic Search ---

load_dotenv()
app = Flask(__name__)

# --- IMPORTANT: Set a Secret Key for Sessions ---
# Generate a good key using: python -c 'import os; print(os.urandom(24))'
# Store it securely, e.g., in .env file or environment variable
app.secret_key = os.getenv("FLASK_SECRET_KEY", "dev-insecure-fallback-key")
if app.secret_key == "dev-insecure-fallback-key":
    print("\n!!! WARNING: Using insecure default Flask secret key. !!!")
    print("!!! Set the FLASK_SECRET_KEY environment variable for secure sessions. !!!\n")


PUBLIC_DIR = os.path.expanduser("~/public")
# Use global key for general uploads and potentially admin actions like rebuild
UPLOAD_API_KEY = os.getenv("KEY")
# Key required for deleting files/folders
DELETE_KEY = os.getenv("DELETE_KEY")
DELETE_KEY_CONFIGURED = bool(DELETE_KEY)

# Ensure PUBLIC_DIR exists
if not os.path.exists(PUBLIC_DIR):
    os.makedirs(PUBLIC_DIR)
    print(f"Created PUBLIC_DIR at {PUBLIC_DIR}")

# --- Folder Protection Configuration ---
FOLDER_KEYS_CONFIG_FILE = 'folder_keys.json'
PROTECTED_FOLDERS = {} # Dictionary to store {normpath: key}

def load_folder_keys():
    """Loads protected folder configurations from JSON file."""
    global PROTECTED_FOLDERS
    protected_config = {}
    try:
        with open(FOLDER_KEYS_CONFIG_FILE, 'r') as f:
            config_data = json.load(f)
            raw_paths = config_data.get('protected_paths', [])
            sorted_paths = sorted(raw_paths, key=lambda x: len(x.get('path', '')), reverse=True)

            for item in sorted_paths:
                path = item.get('path')
                key = item.get('key')
                if path and key:
                    norm_rel_path = os.path.normpath(path.strip('/'))
                    protected_config[norm_rel_path] = key
                else:
                    print(f"Warning: Invalid entry in {FOLDER_KEYS_CONFIG_FILE}: {item}")

            PROTECTED_FOLDERS = protected_config
            print(f"Loaded {len(PROTECTED_FOLDERS)} protected folder configurations.")

    except FileNotFoundError:
        print(f"Info: {FOLDER_KEYS_CONFIG_FILE} not found. No folder-specific keys loaded.")
        PROTECTED_FOLDERS = {}
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {FOLDER_KEYS_CONFIG_FILE}.")
        PROTECTED_FOLDERS = {}
    except Exception as e:
        print(f"Error loading folder keys: {e}")
        PROTECTED_FOLDERS = {}

load_folder_keys() # Load config on startup

# app.py

def get_all_directories(start_path_abs, base_path_abs):
    """Recursively finds all directory relative paths within the start path."""
    dir_list = []
    try:
        for root, dirs, _ in os.walk(start_path_abs, topdown=True):
            # Ensure we don't follow symlinks outside the base path or process unsafe roots
            if not check_path_safety(root): # check_path_safety compares against PUBLIC_DIR
                print(f"Warning: get_all_directories skipping unsafe path: {root}")
                dirs[:] = [] # Don't descend further into this branch
                continue

            # Add the current root (relative to base) to the list, skip the base itself
            rel_root = os.path.relpath(root, base_path_abs)
            if rel_root != '.': # Don't add the root path itself as '.'
                 # Normalize for display (use forward slashes)
                normalized_rel_root = rel_root.replace(os.sep, '/')
                dir_list.append(normalized_rel_root)

            # Filter dirs to prevent descending into potentially unsafe symlinked directories
            safe_dirs = []
            for d in dirs:
                dir_abs_path = os.path.join(root, d)
                if check_path_safety(dir_abs_path):
                    safe_dirs.append(d)
                else:
                    print(f"Warning: get_all_directories skipping descent into unsafe dir: {dir_abs_path}")
            dirs[:] = safe_dirs # Modify dirs in place for os.walk

    except OSError as e:
        print(f"Error walking directory {start_path_abs} in get_all_directories: {e}")

    dir_list.sort()
    return dir_list

def find_files_by_name(query, start_dir_abs, base_public_dir, recursive=False):
    """
    Walks directories and finds files/folders matching the query name substring.
    Returns a list of dictionaries, each containing 'rel_path' and basic info.
    """
    results = []
    query_lower = query.lower()

    # Ensure start_dir is safe and exists (should be checked by caller ideally)
    if not check_path_safety(start_dir_abs) or not os.path.isdir(start_dir_abs):
        return []

    try:
        for root, dirs, files in os.walk(start_dir_abs, topdown=True):
            # Safety check for the root of this iteration (prevent symlink traversal issues)
            if not check_path_safety(root):
                 print(f"Warning: find_files_by_name skipping unsafe path during walk: {root}")
                 dirs[:] = [] # Don't descend further
                 files = []
                 continue

            # If not recursive, modify dirs list to prevent descending further AFTER processing current level
            if not recursive and root != start_dir_abs:
                dirs[:] = [] # Stop descending for next iteration

            current_level_items = sorted(dirs + files, key=lambda s: s.lower())

            for name in current_level_items:
                if query_lower in name.lower():
                    entry_abs_path = os.path.join(root, name)
                    # Double check safety and existence
                    if not check_path_safety(entry_abs_path): continue
                    # Use lexists to handle potentially broken symlinks gracefully
                    if not os.path.lexists(entry_abs_path): continue

                    entry_rel_path = os.path.relpath(entry_abs_path, base_public_dir)
                    # Avoid adding duplicates if walk yields same path multiple times (unlikely but possible)
                    # We'll rely on the caller to handle adding to a final dictionary/set
                    results.append({'rel_path': entry_rel_path, 'abs_path': entry_abs_path, 'name': name})

            # Crucial: If not recursive, break *after* processing the first level
            if not recursive and root == start_dir_abs:
                break # Stop os.walk from going deeper

    except OSError as e:
        print(f"Error during find_files_by_name in {start_dir_abs}: {e}")
        # Return potentially partial results

    return results

# --- Helper Functions ---
def format_info(entry_path, rel_path):
    """Formats file/directory information for the template."""
    try:
        stat_result = os.stat(entry_path)
        is_dir = os.path.isdir(entry_path)
        size = '-' if is_dir else humanize.naturalsize(stat_result.st_size)
        mtime = datetime.fromtimestamp(stat_result.st_mtime).strftime("%Y-%m-%d %H:%M")
        rel_path_encoded = urllib.parse.quote(rel_path)
        # Determine if the item itself requires protection to access content
        is_protected = bool(get_required_key_for_path(rel_path))
        return {
            'is_dir': is_dir,
            'size': size,
            'mtime': mtime,
            'rel_path': rel_path,
            'rel_path_encoded': rel_path_encoded,
            'is_protected': is_protected, # Add protection status
            'error': False
         }
    except OSError as e:
        print(f"Warning: Could not stat {entry_path}: {e}")
        return {
            'is_dir': False, 'size': 'N/A', 'mtime': 'N/A',
            'rel_path': rel_path, 'rel_path_encoded': urllib.parse.quote(rel_path),
            'is_protected': False, 'error': True # Assume not protected if error
        }

def check_path_safety(path_abs):
    """Checks if the absolute path is safely within PUBLIC_DIR."""
    # Ensure comparison with normalized paths including trailing separator if needed
    public_dir_norm = os.path.normpath(PUBLIC_DIR)
    path_abs_norm = os.path.normpath(path_abs)
    # Check if path is exactly the public dir or starts with public dir + separator
    return path_abs_norm == public_dir_norm or \
           path_abs_norm.startswith(public_dir_norm + os.sep)


def get_required_key_for_path(relative_path):
    """Finds the required API key for a given relative path, if any."""
    norm_req_path = os.path.normpath(relative_path.strip('/'))
    # Handle root case explicitly if necessary
    if norm_req_path == '.': norm_req_path = ''

    # Iterate through sorted keys (longest path first implicitly)
    for protected_path, key in PROTECTED_FOLDERS.items():
        if norm_req_path == protected_path or \
           norm_req_path.startswith(protected_path + os.sep):
            return key
    return None # No specific key required

# --- Routes ---

# --- Route to Validate Key and Store in Session ---
@app.route('/validate-key', methods=['POST'])
def validate_access_key():
    """Validates a key for a given path and stores authorization in session."""
    print("--- /validate-key endpoint hit ---") # Debug
    print(f"Request Headers: {request.headers}") # Debug: Check Content-Type
    print(f"Request Raw Data: {request.data}")   # Debug: See the raw bytes
    try:
        data = request.get_json()
        print(f"Received JSON data: {data}") # Debug: See parsed data
        if not data or 'path' not in data or 'key' not in data:
            print("Error: Missing path or key in request.")
            return jsonify(status="error", message="Missing path or key in request."), 400

        target_href_path = data['path'] # Path from href (likely encoded like /private%20docs)
        provided_key = data['key']      # Key entered by user
        print(f"Target Href Path: {target_href_path}, Key: {provided_key}") # Debug

        # Decode the path from the href to get the actual relative path
        try:
            # Strip leading slash which might come from href, before unquoting
            target_rel_path = urllib.parse.unquote(target_href_path.strip('/'))
        except Exception as e:
             print(f"Error unquoting path '{target_href_path}': {e}") # Debug
             return jsonify(status="error", message="Invalid path encoding."), 400
        print(f"Unquoted Relative Path: {target_rel_path}") # Debug

        # Normalize path for consistent lookup and session storage
        norm_target_path = os.path.normpath(target_rel_path)
        if norm_target_path == '.': norm_target_path = '' # Handle root case explicitly
        print(f"Normalized Path for Lookup/Storage: {norm_target_path}") # Debug

        required_key = get_required_key_for_path(norm_target_path)
        print(f"Required Key for path '{norm_target_path}': {required_key}") # Debug

        if not required_key:
            # Path doesn't actually require a key
            print(f"Path '{norm_target_path}' is not protected.") # Debug
            return jsonify(status="success", message="Path is not protected."), 200

        if required_key == provided_key:
            # Key is correct! Store authorization in session as a LIST.
            # Retrieve as list, convert to set for adding, convert back to list for storing
            authorized_paths_list = session.get('authorized_paths', []) # Get list or empty list
            authorized_paths_set = set(authorized_paths_list)         # Convert to set for efficient add/check
            authorized_paths_set.add(norm_target_path)                # Add the newly authorized path
            session['authorized_paths'] = list(authorized_paths_set) # Store back as a LIST
            session.modified = True # Ensure session is saved
            print(f"Session: Granted access to '{norm_target_path}'. Current authorized (stored as list): {session.get('authorized_paths')}") # Debug
            return jsonify(status="success", message="Access granted."), 200
        else:
            # Key is incorrect
            print(f"Session: Denied access to '{norm_target_path}' - incorrect key provided.") # Debug
            return jsonify(status="error", message="Invalid access key provided."), 401 # Unauthorized
    except Exception as e:
        # Catch any unexpected error during processing
        print(f"!!! UNEXPECTED ERROR in /validate-key: {e} !!!") # Debug
        import traceback
        traceback.print_exc() # Print full traceback to console
        return jsonify(status="error", message="Server error during key validation."), 500


@app.route("/", defaults={"path": ""})
@app.route("/<path:path>")
def serve(path):
    """Serves directory listings, files, or search results, checking session permissions."""
    # print(f"--- Serve request for path: '{path}' ---") # Debug
    filename_search_query = request.args.get('search', '').strip()
    smart_query = request.args.get('smart_query', '').strip()
    recursive = request.args.get('recursive', 'true').lower() == 'true'

    entries = {} # Final dictionary of items to display {rel_path: info_dict}
    title = ''
    show_parent = False
    parent_url = '/'
    is_smart_search_results = False
    permission_denied = False # Assume access OK until check fails

    # --- Path Calculation and Base Safety Check ---
    # Use the raw path from the route for joining initially
    current_path_abs = os.path.normpath(os.path.join(PUBLIC_DIR, path))
    if not check_path_safety(current_path_abs):
        print(f"Forbidden: Attempt to access outside PUBLIC_DIR: {current_path_abs}")
        abort(403) # Forbidden (outside PUBLIC_DIR)
    if not os.path.exists(current_path_abs):
         print(f"Not Found: Path does not exist: {current_path_abs}")
         abort(404) # Not Found

    # --- Use Flask Session for Access Control ---
    # Use the unquoted, normalized relative path for session/permission checks
    try:
        relative_path_unquoted = urllib.parse.unquote(path)
    except Exception:
        relative_path_unquoted = path # Fallback if unquoting fails

    norm_current_path = os.path.normpath(relative_path_unquoted.strip('/'))
    if norm_current_path == '.': norm_current_path = '' # Normalize root path

    required_key = get_required_key_for_path(norm_current_path)
    has_session_access = False # Assume no access yet if key is required

    if required_key:
        # Retrieve list from session, convert to set for checking
        authorized_paths_list = session.get('authorized_paths', [])
        authorized_paths_set = set(authorized_paths_list)

        # Check if ANY path in the session covers the current request
        for authorized_path in authorized_paths_set:
            is_root_authorized = authorized_path == ''
            is_current_root = norm_current_path == ''

            if norm_current_path == authorized_path: # Exact match
                has_session_access = True; break
            elif is_root_authorized and not is_current_root: # Root auth covers children
                has_session_access = True; break
            elif not is_root_authorized and norm_current_path.startswith(authorized_path + os.sep): # Parent auth covers children
                has_session_access = True; break

        # If no covering authorization found, deny permission
        if not has_session_access:
            permission_denied = True
            # print(f"Session: Access denied to '{norm_current_path}'. Not covered by authorized paths: {authorized_paths_set}") # Debug
            # Abort only for direct file access denial
            if os.path.isfile(current_path_abs):
                 print(f"Forbidden: Direct access to protected file '{path}' denied. No covering session auth.")
                 abort(403)
        # else:
            # print(f"Session: Access granted to '{norm_current_path}' via covering path in session.") # Debug
    # --- End Session Check ---


    # --- Handle File Request ---
    if os.path.isfile(current_path_abs):
        # If we reach here, permission_denied is False (access granted)
        try:
            # Use original path from route for send_from_directory relative path argument
            return send_from_directory(PUBLIC_DIR, path)
        except Exception as e:
            print(f"Error sending file {current_path_abs} using path '{path}': {e}")
            abort(500)


    # --- Handle Directory/Search ---
    if os.path.isdir(current_path_abs):

        # Determine Title (using unquoted path)
        current_display_path = relative_path_unquoted.strip('/')
        if permission_denied:
            title = f"Access Denied - /{current_display_path}" if current_display_path else "Access Denied - /"
        elif smart_query:
             title = f"Smart Search Results for '{smart_query}'"
             is_smart_search_results = True # Flag used later for sorting/display
        elif filename_search_query:
             title = f"Filename Search '{filename_search_query}'"
             title += f" {'recursively ' if recursive else ''}in /{current_display_path if current_display_path else ''}"
        else:
             title = f"Index of /{current_display_path}" if current_display_path else "Index of /"

        # Build Parent URL (using unquoted path for dirname, then quote result)
        if relative_path_unquoted: # Show parent unless we are at root
            show_parent = True
            parent_path_rel_unquoted = os.path.dirname(relative_path_unquoted).strip('/')
            parent_url = '/' + urllib.parse.quote(parent_path_rel_unquoted if parent_path_rel_unquoted else '')


        # --- Populate Entries (ONLY IF access to view current directory content is NOT denied) ---
        if not permission_denied:

            # --- Smart Search (Content + Name) ---
            if smart_query:
                semantic_results_map = {} # {rel_path: score}
                filename_results_map = {} # {rel_path: {'abs_path':..., 'name':...}}
                temp_entries = {}         # Temporary storage before sorting

                # 1. Semantic Content Search
                if MODEL_LOADED and SEMANTIC_INDEX_DATA is not None:
                    print(f"Performing semantic search for: '{smart_query}'")
                    semantic_raw = search_index.semantic_search(smart_query, SEMANTIC_INDEX_DATA, top_n=50)
                    for res in semantic_raw:
                        semantic_results_map[res['path']] = res['score']
                    print(f"Found {len(semantic_results_map)} semantic results.")
                else:
                     if not MODEL_LOADED: title += " (Semantic Model Error)"
                     elif SEMANTIC_INDEX_DATA is None: title += " (Semantic Index Not Ready)"

                # 2. Filename/Foldername Search (always recursive from root for smart search)
                print(f"Performing filename search for smart query: '{smart_query}' (recursive from root)")
                filename_raw = find_files_by_name(smart_query, PUBLIC_DIR, PUBLIC_DIR, recursive=True)
                for item in filename_raw:
                    if item['rel_path'] not in filename_results_map:
                        filename_results_map[item['rel_path']] = item
                print(f"Found {len(filename_results_map)} filename results for smart query.")

                # 3. Combine and Populate temp_entries, prioritizing semantic results
                for rel_path, score in semantic_results_map.items():
                    abs_path = os.path.normpath(os.path.join(PUBLIC_DIR, rel_path))
                    # Basic safety/existence check
                    if not check_path_safety(abs_path) or not os.path.exists(abs_path): continue
                    if rel_path not in temp_entries:
                        info = format_info(abs_path, rel_path) # format_info adds is_protected
                        if not info['error']:
                            info['score'] = f"{score:.2f}"
                            info['display_name'] = os.path.basename(rel_path)
                            info['matched_name'] = rel_path in filename_results_map # Check if name also matched
                            temp_entries[rel_path] = info

                # Add filename results not already present
                for rel_path, item_data in filename_results_map.items():
                     if rel_path not in temp_entries:
                          abs_path = item_data['abs_path']
                          if not os.path.exists(abs_path) and not os.path.islink(abs_path): continue
                          info = format_info(abs_path, rel_path)
                          if not info['error']:
                               info['score'] = None
                               info['display_name'] = item_data['name']
                               info['matched_name'] = True
                               temp_entries[rel_path] = info

                # 4. Sorting for Smart Search (Score > Name Match > Alpha)
                entries_list = list(temp_entries.items())
                def sort_key(item_tuple):
                    _rel_path, info_dict = item_tuple
                    try: score_val = float(info_dict.get('score', -1.0))
                    except (ValueError, TypeError): score_val = -1.0
                    name_match_bonus = 1 if info_dict.get('matched_name', False) else 0
                    display_name_lower = info_dict.get('display_name', '').lower()
                    return (-score_val, -name_match_bonus, display_name_lower) # Sort Score DESC, NameMatch DESC, Name ASC

                sorted_entries_list = sorted(entries_list, key=sort_key)
                entries = dict(sorted_entries_list) # Assign sorted results to final entries dict

            # --- Filename Filter Search ---
            elif filename_search_query:
                # Use current path absolute for starting directory
                search_start_abs = current_path_abs
                print(f"Performing filename filter: '{filename_search_query}', Recursive: {recursive}, Start: {search_start_abs}")

                # Use the helper function
                filename_results_raw = find_files_by_name(filename_search_query, search_start_abs, PUBLIC_DIR, recursive=recursive)

                # Populate final entries dictionary directly
                for item in filename_results_raw:
                    rel_path = item['rel_path']
                    if rel_path not in entries: # Prevent duplicates
                        abs_path = item['abs_path']
                        info = format_info(abs_path, rel_path) # format_info adds is_protected
                        if not info['error']:
                            info['display_name'] = item['name']
                            info['score'] = None # No score
                            info['matched_name'] = True # Matched name by definition
                            entries[rel_path] = info
                # Note: These results will be sorted alphabetically later

            # --- Normal Directory Listing ---
            else:
                 try:
                     for name in sorted(os.listdir(current_path_abs), key=lambda s: s.lower()):
                         ent_path_abs = os.path.join(current_path_abs, name)
                         if not check_path_safety(ent_path_abs) or not os.path.lexists(ent_path_abs): continue
                         # Use the normalized, unquoted current path for joining
                         ent_path_rel = os.path.normpath(os.path.join(norm_current_path, name)).strip('/')
                         if ent_path_rel == '.': ent_path_rel = '' # Handle case where join results in '.'

                         if ent_path_rel not in entries: # Should not happen with listdir, but safe check
                            info = format_info(ent_path_abs, ent_path_rel) # format_info adds is_protected
                            if not info['error']:
                                info['display_name'] = name
                                entries[ent_path_rel] = info # Use unique relative path as key
                 except OSError as e:
                      print(f"Error listing directory {current_path_abs}: {e}")
                      title += " (Error Listing Directory)"
                 # Note: These results will be sorted alphabetically later

        # --- End Populating Entries ---

        # --- Prepare for Template ---
        # Smart search results are already custom-sorted.
        # Filter search and directory listing need alphabetical sort by display name.
        if is_smart_search_results:
             # 'entries' dictionary already holds the custom-sorted results
             sorted_entries_for_template = entries
        else:
             # Sort filename filter or directory listing results alphabetically
             sorted_entries_for_template = dict(sorted(entries.items(), key=lambda item: item[1]['display_name'].lower()))

        return render_template(
            'index.html',
            title=title,
            entries=sorted_entries_for_template,
            show_parent=show_parent,
            parent_url=parent_url,
            search_query=filename_search_query,
            smart_query=smart_query,
            recursive=recursive,
            is_smart_search_results=is_smart_search_results,
            semantic_search_enabled=MODEL_LOADED,
            permission_denied=permission_denied, # Pass denied status
            current_path=norm_current_path, # Pass normalized, unquoted path
            delete_key_configured=DELETE_KEY_CONFIGURED # Pass delete key status
        )
    else:
        # Path exists, is safe, but not a file or directory? Unexpected.
        print(f"Error: Path is not a file or directory: {current_path_abs}")
        abort(500)

@app.route("/health")
def health_check():
    """Provides basic health information about the server."""
    # ... (keep existing health check code) ...
    try:
        disk_usage = shutil.disk_usage(PUBLIC_DIR)
        health_info = {
            "status": "ok",
            "timestamp": datetime.now().isoformat(),
            "public_dir": PUBLIC_DIR,
            "disk_total": humanize.naturalsize(disk_usage.total, binary=True),
            "disk_used": humanize.naturalsize(disk_usage.used, binary=True),
            "disk_free": humanize.naturalsize(disk_usage.free, binary=True),
            "disk_percent_used": f"{(disk_usage.used / disk_usage.total) * 100:.1f}%"
        }
        return jsonify(health_info)
    except Exception as e:
        print(f"Health check failed: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/upload/<path:filename>", methods=['POST'])
def upload_file(filename):
    """Handles file uploads via POST request, checking folder permissions using X-Upload-Key header."""
    provided_upload_key = request.headers.get('X-Upload-Key')

    # --- Sanitize Path ---
    safe_parts = []
    try:
         # Ensure filename is treated as a relative path for splitting
         path_parts = filename.strip('/').split('/')
         for part in path_parts:
             clean_part = secure_filename(part)
             if clean_part and clean_part != '..': # Extra check for '..' just in case
                 safe_parts.append(clean_part)
             elif part == '.': continue # Allow '.' segment if needed (usually not)
             else:
                 print(f"Forbidden: Invalid path component '{part}' in upload path '{filename}'")
                 abort(400)
    except Exception as e:
        print(f"Error sanitizing upload filename '{filename}': {e}")
        abort(400)

    if not safe_parts:
         print(f"Forbidden: Upload path '{filename}' resulted in empty safe path.")
         abort(400)

    safe_relative_path = os.path.join(*safe_parts)
    destination_abs_path = os.path.normpath(os.path.join(PUBLIC_DIR, safe_relative_path))

    # --- Security: Final Path Check ---
    if not check_path_safety(destination_abs_path):
        print(f"Forbidden: Sanitized path '{destination_abs_path}' is outside PUBLIC_DIR.")
        abort(403)

    # --- Authentication Check for Upload Target Directory ---
    target_dir_rel_path = os.path.dirname(safe_relative_path)
    required_key_for_upload = get_required_key_for_path(target_dir_rel_path)

    auth_ok = False
    if required_key_for_upload:
        if provided_upload_key == required_key_for_upload: auth_ok = True
        else: print(f"Unauthorized upload to protected dir '{target_dir_rel_path}'. Incorrect/missing folder key in X-Upload-Key.")
    else:
        if provided_upload_key == UPLOAD_API_KEY: auth_ok = True
        else: print(f"Unauthorized upload to non-protected dir '{target_dir_rel_path}'. Incorrect/missing global key in X-Upload-Key.")

    if not auth_ok: abort(401)

    # --- Create Directories & Write File ---
    try:
        destination_dir = os.path.dirname(destination_abs_path)
        os.makedirs(destination_dir, exist_ok=True)
        if not request.data: abort(400) # Bad Request: No data
        # Handle large file uploads if MAX_CONTENT_LENGTH is set
        # ...
        with open(destination_abs_path, 'wb') as f:
            f.write(request.data)
        print(f"File uploaded successfully to {destination_abs_path}")
        # !!! Trigger index update here if implemented !!!
        return jsonify({"status": "success", "message": "File uploaded successfully.", "filename": safe_relative_path}), 201
    except RequestEntityTooLarge: abort(413)
    except IOError as e: print(f"IOError writing file {destination_abs_path}: {e}"); abort(500)
    except Exception as e: print(f"Unexpected error during upload: {e}"); abort(500)


@app.route('/upload-ui')
def upload_ui():
    """Serves the simple HTML upload interface."""
    destination_dirs = get_all_directories(PUBLIC_DIR, PUBLIC_DIR)
    return render_template('upload.html',
                           title="Upload File",
                           destination_dirs=destination_dirs # Pass the list here
                           )


@app.route('/rebuild-index', methods=['POST'])
def trigger_rebuild_index():
    """Manually triggers a rebuild of the semantic search index. Requires GLOBAL key."""
    # Use the global key for this admin action
    provided_key = request.headers.get('X-Upload-Key') # Or a dedicated admin key header like 'X-Admin-Key'
    if not provided_key or provided_key != UPLOAD_API_KEY:
        print("Unauthorized index rebuild attempt (requires global key).")
        return jsonify(status="error", message="Unauthorized"), 401

    if not MODEL_LOADED:
        return jsonify(status="error", message="Semantic model not loaded, cannot rebuild index."), 500

    print("Manual index rebuild requested...")
    global SEMANTIC_INDEX_DATA # Allow modification
    try:
        # Ensure search_index module and function exist
        if not hasattr(search_index, 'build_index'):
             return jsonify(status="error", message="build_index function not found."), 500

        new_index_data = search_index.build_index(PUBLIC_DIR)
        if new_index_data is not None: # Check for successful build (might return None on error)
            SEMANTIC_INDEX_DATA = new_index_data
            message = "Semantic index rebuild completed successfully."
            print(message)
            return jsonify(status="success", message=message), 200
        else:
             message = "Semantic index rebuild failed or index is empty."
             print(message)
             return jsonify(status="error", message=message), 500
    except Exception as e:
        print(f"Error during manual index rebuild: {e}")
        return jsonify(status="error", message=f"Rebuild failed: {e}"), 500


# --- API Endpoint for Directory Listing ---
@app.route('/api/list-dirs', methods=['POST'])
def api_list_dirs():
    """API endpoint to list subdirectories within a given path."""
    data = request.get_json()
    relative_req_path = data.get('path', '').strip('/') if data else '' # Default to root

    # --- Path Calculation and Safety Check ---
    # Ensure normalization happens *before* permission checks
    norm_relative_req_path = os.path.normpath(relative_req_path)
    if norm_relative_req_path == '.': norm_relative_req_path = '' # Handle root normalization

    # Use normalized path for checks from here
    current_path_abs = os.path.normpath(os.path.join(PUBLIC_DIR, norm_relative_req_path))
    if not check_path_safety(current_path_abs):
        print(f"API Forbidden: Attempt to list outside PUBLIC_DIR: {current_path_abs}")
        return jsonify(error="Access forbidden."), 403
    if not os.path.isdir(current_path_abs): # Ensure it's a directory
        print(f"API Not Found: Path is not a directory: {current_path_abs}")
        return jsonify(error="Path not found or is not a directory."), 404

    # --- Check Session Permission for Requested Path ---
    print(f"[API /api/list-dirs] Checking access for normalized path: '{norm_relative_req_path}'") # DEBUG
    required_key = get_required_key_for_path(norm_relative_req_path)
    has_session_access = False
    if required_key:
        authorized_paths_list = session.get('authorized_paths', [])
        authorized_paths_set = set(authorized_paths_list)
        print(f"[API /api/list-dirs] Path requires key. Session authorized paths: {authorized_paths_set}") # DEBUG
        # Check if any authorized path covers the requested path (simplified check)
        for authorized_path in authorized_paths_set:
             # Ensure authorized_path from session is also normalized (should be already by /validate-key)
             # norm_authorized_path = os.path.normpath(authorized_path) # Probably redundant but safe
             is_root_authorized = authorized_path == ''
             # Compare normalized requested path with paths from session
             if norm_relative_req_path == authorized_path or \
                (is_root_authorized and norm_relative_req_path != '') or \
                (not is_root_authorized and norm_relative_req_path.startswith(authorized_path + os.sep)):
                  print(f"[API /api/list-dirs] Access GRANTED via session path: '{authorized_path}'") # DEBUG
                  has_session_access = True; break
        if not has_session_access:
            print(f"[API /api/list-dirs] Access DENIED for path: '{norm_relative_req_path}'. Returning 401.") # DEBUG
            # Return a specific error type the frontend can potentially handle (e.g., prompt for key)
            # Pass back the *original* relative_req_path for the password prompt label if needed, or normalized?
            # Let's pass normalized for consistency in the API response structure
            return jsonify(error="Authentication required to view this folder.", requires_key=True, path=norm_relative_req_path), 401
    # --- End Session Check ---
    print(f"[API /api/list-dirs] Access appears GRANTED (or not needed) for: '{norm_relative_req_path}'") # DEBUG

    subdirs_data = [] # Changed from subdirs list to list of dicts
    try:
        for name in sorted(os.listdir(current_path_abs), key=lambda s: s.lower()):
            entry_path_abs = os.path.join(current_path_abs, name)
            if not check_path_safety(entry_path_abs): continue

            if os.path.isdir(entry_path_abs): # We only care about directories
                 # Calculate relative path for the subdirectory to check its protection
                 # Use the normalized requested path as the base for joining
                item_rel_path = os.path.normpath(os.path.join(norm_relative_req_path, name))
                is_item_protected = bool(get_required_key_for_path(item_rel_path))
                subdirs_data.append({
                     'name': name,
                     'is_protected': is_item_protected
                 })
    except OSError as e:
        print(f"API Error listing directory {current_path_abs}: {e}")
        return jsonify(error=f"Error listing directory: {e}"), 500

    return jsonify(subdirs=subdirs_data, current_path=norm_relative_req_path) # Return normalized path

# --- End API Endpoint ---

# --- API Endpoint for Deleting Items ---
@app.route('/api/delete-items', methods=['POST'])
def api_delete_items():
    """API endpoint to delete files or folders."""
    if not DELETE_KEY_CONFIGURED:
        return jsonify(error="Deletion feature not configured on server."), 501 # Not Implemented

    provided_key = request.headers.get('X-Delete-Key')
    if not provided_key or provided_key != DELETE_KEY:
        print("API Delete Unauthorized: Incorrect or missing X-Delete-Key.")
        return jsonify(error="Unauthorized. Invalid delete key."), 401

    data = request.get_json()
    if not data or 'items_to_delete' not in data or not isinstance(data['items_to_delete'], list):
        return jsonify(error="Invalid request body. Expected {'items_to_delete': [...]}."), 400

    items_to_delete = data['items_to_delete']
    success_count = 0
    fail_count = 0
    errors = []

    for item_rel_path in items_to_delete:
        if not isinstance(item_rel_path, str):
            fail_count += 1
            errors.append({"path": "(invalid format)", "error": "Invalid item format in list."})    
            continue # Skip non-string paths
            
        norm_item_rel_path = os.path.normpath(item_rel_path.strip('/'))
        if norm_item_rel_path == '.': norm_item_rel_path = '' # Handle root case if passed

        item_abs_path = os.path.normpath(os.path.join(PUBLIC_DIR, norm_item_rel_path))

        # --- Security / Safety Checks ---
        if not check_path_safety(item_abs_path):
            print(f"API Delete Forbidden: Attempt to delete outside PUBLIC_DIR: {item_abs_path}")
            fail_count += 1
            errors.append({"path": norm_item_rel_path, "error": "Access forbidden (path outside public area)."})
            continue
        # Prevent deleting the root directory itself
        if item_abs_path == os.path.normpath(PUBLIC_DIR):
             print(f"API Delete Forbidden: Attempt to delete root PUBLIC_DIR.")
             fail_count += 1
             errors.append({"path": "/", "error": "Cannot delete the root directory."})    
             continue
        
        if not os.path.lexists(item_abs_path): # Use lexists to handle broken symlinks gracefully
            print(f"API Delete Warning: Item not found: {item_abs_path}")
            fail_count += 1
            errors.append({"path": norm_item_rel_path, "error": "Item not found."})    
            continue

        # --- Attempt Deletion ---
        try:
            if os.path.isdir(item_abs_path) and not os.path.islink(item_abs_path):
                shutil.rmtree(item_abs_path)
                print(f"API Delete Success (Directory): {item_abs_path}")
            else: # Files or symlinks
                os.remove(item_abs_path)
                print(f"API Delete Success (File/Link): {item_abs_path}")
            success_count += 1
        except OSError as e:
            print(f"API Delete Error for {item_abs_path}: {e}")
            fail_count += 1
            errors.append({"path": norm_item_rel_path, "error": f"OS error during deletion: {e.strerror}"})
        except Exception as e:
            print(f"API Delete Unexpected Error for {item_abs_path}: {e}")
            fail_count += 1
            errors.append({"path": norm_item_rel_path, "error": f"Unexpected error during deletion: {str(e)}"})

    # --- Return Result ---
    response = {
        "success_count": success_count,
        "fail_count": fail_count,
        "errors": errors
    }
    status_code = 200 if fail_count == 0 else 207 # 207 Multi-Status if there were errors

    return jsonify(response), status_code
# --- End Delete API Endpoint ---

# --- Error Handlers ---
@app.errorhandler(404)
def page_not_found(e):
    return render_template('error.html', error_code=404, error_message="Page Not Found", error_description=str(e)), 404

@app.errorhandler(403)
def forbidden(e):
     return render_template('error.html', error_code=403, error_message="Forbidden", error_description="You do not have permission to access this resource."), 403

@app.errorhandler(401)
def unauthorized(e):
     return render_template('error.html', error_code=401, error_message="Unauthorized", error_description="Authentication required or failed."), 401

@app.errorhandler(500)
def internal_server_error(e):
     return render_template('error.html', error_code=500, error_message="Internal Server Error", error_description="An unexpected error occurred."), 500

@app.errorhandler(413)
@app.errorhandler(RequestEntityTooLarge)
def request_entity_too_large(error):
    # Custom JSON response for this specific error if needed by JS, else use generic handler
    # return jsonify(status="error", message="File too large"), 413
     return render_template('error.html', error_code=413, error_message="Payload Too Large", error_description="The file you tried to upload exceeds the maximum allowed size."), 413


# --- Main Execution ---
if __name__ == "__main__":
    print("-" * 50)
    print(f"Starting File Browser...")
    print(f"Serving files from: {PUBLIC_DIR}")
    print(f"Global Upload API Key Configured: {'Yes' if UPLOAD_API_KEY else 'NO'}")
    print(f"Loaded {len(PROTECTED_FOLDERS)} protected folder keys from {FOLDER_KEYS_CONFIG_FILE}.")
    print(f"Semantic Search Model Loaded: {MODEL_LOADED}")
    if MODEL_LOADED and SEMANTIC_INDEX_DATA is None:
         print("Semantic index file not found or invalid. Use POST /rebuild-index (with global API key) to build it.")
    elif MODEL_LOADED:
         print(f"Semantic index loaded with {SEMANTIC_INDEX_DATA.get('embeddings', np.array([])).shape[0]} embeddings.")
    print(f"Delete Key Configured: {DELETE_KEY_CONFIGURED}") # Log delete key status
    if app.secret_key == "dev-insecure-fallback-key":
        print("!!! Flask Session Secret Key is INSECURE !!!")
    print("-" * 50)
    # Set debug=False for production environments
    app.run(host="0.0.0.0", port=8000, debug=True) # Turn off debug in production