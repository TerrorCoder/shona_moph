import toml
import gspread
from google.oauth2.service_account import Credentials

# Load secrets from .streamlit/secrets.toml
with open('.streamlit/secrets.toml', 'r') as f:
    secrets = toml.load(f)

# Test Google Sheets connection
def test_google_sheets():
    try:
        # Get credentials from secrets
        creds_dict = secrets.get("gcp_service_account", None)
        if not creds_dict:
            return "No GCP service account credentials found in secrets."

        # Setup credentials
        scopes = [
            "https://www.googleapis.com/auth/spreadsheets",
            "https://www.googleapis.com/auth/drive"
        ]
        creds = Credentials.from_service_account_info(creds_dict, scopes=scopes)
        client = gspread.authorize(creds)

        # Try to open or create spreadsheet
        sheet_name = secrets.get("sheet_name", "Shona Analysis Log")
        try:
            spreadsheet = client.open(sheet_name)
            return f"Successfully connected to existing spreadsheet: {sheet_name}"
        except gspread.SpreadsheetNotFound:
            spreadsheet = client.create(sheet_name)
            spreadsheet.share('', perm_type='anyone', role='reader')
            return f"Successfully created new spreadsheet: {sheet_name}"

    except Exception as e:
        return f"Error setting up Google Sheets: {str(e)}"

result = test_google_sheets()
print(result)