import pandas as pd
import sqlite3
import os

DB_PATH = "db/sqlite.db"

def clean_columns(df):
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")
    return df


def process_file(file_path, conn):
    file = os.path.basename(file_path)
    table_name = os.path.splitext(file)[0].replace(" ", "_").lower()

    try:
        if file.endswith(".csv"):
            df = pd.read_csv(file_path)
            df = clean_columns(df)
            df.to_sql(table_name, conn, index=False, if_exists="replace")

        elif file.endswith((".xlsx", ".xls")):
            xls = pd.ExcelFile(file_path)

            for sheet in xls.sheet_names:
                df = pd.read_excel(xls, sheet_name=sheet)
                df = clean_columns(df)

                sheet_table = f"{table_name}_{sheet}".lower().replace(" ", "_")
                df.to_sql(sheet_table, conn, index=False, if_exists="replace")

        print(f"✅ Loaded: {file}")

    except Exception as e:
        print(f"❌ Error loading {file}: {e}")


def load_data(DATA_PATH):
    conn = sqlite3.connect(DB_PATH)

    # 🔥 Case 1: DATA_PATH is a directory
    if os.path.isdir(DATA_PATH):
        for file in os.listdir(DATA_PATH):
            process_file(os.path.join(DATA_PATH, file), conn)

    # 🔥 Case 2: DATA_PATH is a single file
    elif os.path.isfile(DATA_PATH):
        process_file(DATA_PATH, conn)

    else:
        print("❌ Invalid path")

    conn.close()


#load_data("/Users/sabarisury/Desktop/RAG_IRS_CHAT_BOT/DATA/tax_brackets_2026.csv")