# Setup
```pip install Flask humanize Werkzeug python-dotenv sentence-transformers torch pypdf scikit-learn```

## API Documentation

### Action Endpoints

### POST /upload/<path:filename>
Purpose: Uploads a single file to the specified path.
Method: POST
Authentication: Requires an X-Upload-Key header.
If uploading to a protected directory, this must be the folder-specific key.
Otherwise, this must be the global KEY from the .env file.
Request Body: The raw binary data of the file to upload.
Example CURL:
```
  # Upload 'local_image.jpg' to the '/images' folder on the server
  curl -X POST http://localhost:8000/upload/images/new_image.jpg \
       -H "X-Upload-Key: UPLOAD_KEY" \
       -H "Content-Type: application/octet-stream" \
       --data-binary @"/path/to/your/local_image.jpg"
```

### POST /validate-key
Purpose: Validates an access key for a specific protected path and grants access by setting a session cookie.
Method: POST
Authentication: The key provided in the JSON body.
Request Body (JSON):
```
  {
      "path": "/protected_folder",
      "key": "folder_specific_key"
  }
```
Example CURL
```
  curl -X POST http://localhost:8000/validate-key \
       -H "Content-Type: application/json" \
       -d '{"path": "/private", "key": "KEY"}' \
       -c cookies.txt # Save session cookie for later use
```

### POST /rebuild-index
Purpose: Triggers a full rebuild of the semantic search index for all supported files.
Method: POST
Authentication: Requires the global KEY from the .env file in the X-Upload-Key header.
Example curl:
```
  curl -X POST http://localhost:8000/rebuild-index -H "X-Upload-Key: UPLOAD_KEY"
```

### POST /api/list-dirs
Purpose: Lists subdirectories within a given path. Used by the "Browse..." modal on the upload page.
Method: POST
Authentication: Session-based. If the path is protected, it requires a valid session cookie from /validate-key.
Request Body (JSON):
```
  {
      "path": "path/to/list"
  }
```
Example CURL:
```
  # List directories in the root
  curl -X POST http://localhost:8000/api/list-dirs \
       -H "Content-Type: application/json" \
       -d '{"path": ""}'

  # List directories in a protected folder (requires prior auth)
  curl -X POST http://localhost:8000/api/list-dirs \
       -H "Content-Type: application/json" \
       -d '{"path": "private"}' \
       -b cookies.txt # Use the saved session cookie
```

### POST /api/delete-items
Purpose: Deletes one or more files or folders. This action is permanent.
Method: POST
Authentication: Requires the DELETE_KEY from the .env file in the X-Delete-Key header.
Request Body (JSON):
```
  {
      "items_to_delete": ["file.txt", "folder_to_delete/"]
  }
```
Example CURL:
```
  curl -X POST http://localhost:8000/api/delete-items \
       -H "X-Delete-Key: DELETE_KEY" \
       -H "Content-Type: application/json" \
       -d '{"items_to_delete": ["old_report.pdf", "temporary_folder"]}'
```

### POST /api/toggle-hidden
Purpose: Hides a folder from listings or makes it visible again.
Method: POST
Authentication: Requires the HIDDEN_KEY from the .env file, provided in the JSON body.
Request Body (JSON):
```
  {
      "path": "folder/to/toggle",
      "key": "the_hidden_key",
      "hide": true
  }
```
Example CURL:
```
  # Hide the 'images' folder
  curl -X POST http://localhost:8000/api/toggle-hidden \
       -H "Content-Type: application/json" \
       -d '{"path": "images", "key": "HIDE_KEY", "hide": true}'

  # Unhide the 'images' folder
  curl -X POST http://localhost:8000/api/toggle-hidden \
       -H "Content-Type: application/json" \
       -d '{"path": "images", "key": "HIDE_KEY", "hide": false}'
```

### System Endpoints

### GET /health
Purpose: Provides a health check of the server, including disk usage.
Method: GET
Authentication: None.
Example CURL:
```
  curl http://localhost:8000/health
```
