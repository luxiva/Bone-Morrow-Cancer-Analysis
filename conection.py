# conection.py
import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy.engine import URL


def load_data():
    """
    Connect to SQL Server and load the bone tumor feature matrix.
    Returns:
        pd.DataFrame
    """
    server_name = r"AJ\ARIAN"
    database_name = "bone_tumor"
    driver = "ODBC Driver 17 for SQL Server"

    connection_string = (
        f"Driver={{{driver}}};"
        f"Server={server_name};"
        f"Database={database_name};"
        "Trusted_Connection=yes;"
        "Encrypt=yes;"
        "TrustServerCertificate=yes;"
    )

    connection_url = URL.create(
        "mssql+pyodbc",
        query={"odbc_connect": connection_string}
    )

    engine = create_engine(connection_url, pool_pre_ping=True)

    query = """
    SELECT
        f.*,
        CASE b.outcome_status
            WHEN 'NED' THEN 0
            WHEN 'AWD' THEN 1
            WHEN 'D' THEN 2
        END AS outcome_label
    FROM dbo.vw_feature_matrix f
    JOIN dbo.Bone_Tumor1 b
        ON f.patient_id = b.patient_id;
    """

    df = pd.read_sql(query, engine)
    return df


if __name__ == "__main__":
    df = load_data()
    print("Data loaded successfully")
    print("Shape:", df.shape)
    print(df.head())
