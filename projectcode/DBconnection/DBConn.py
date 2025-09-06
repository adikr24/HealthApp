import psycopg2

# Database connection details (still hardcoded for this example)
# In a real application, consider using environment variables or a config file.
DB_NAME = "FoundationFood"  # Replace with your actual database name
DB_USER = "postgres"        # Replace with your actual PostgreSQL username
DB_PASSWORD = "superuser"   # Replace with your actual PostgreSQL password
DB_HOST = "localhost"       # Usually 'localhost' for local server
DB_PORT = "5432"            # Default PostgreSQL port

def get_db_connection():
    """Establishes and returns a PostgreSQL database connection."""
    try:
        conn = psycopg2.connect(
            dbname=DB_NAME,
            user=DB_USER,
            password=DB_PASSWORD,
            host=DB_HOST,
            port=DB_PORT
        )
        print('Connected to the PostgreSQL server.')
        return conn
    except psycopg2.Error as e:
        print(f"Database connection error: {e}")
        return None

def close_db_connection(conn):
    """Closes the PostgreSQL database connection."""
    if conn is not None:
        conn.close()
        print('Database connection closed.')

def execute_query_and_fetch(conn, query, params=None):
    """Executes a given SQL query and fetches all results."""
    cur = None
    try:
        cur = conn.cursor()
        cur.execute(query, params) # Use parameters here
        rows = cur.fetchall()
        cur.close()
        return rows
    except psycopg2.Error as e:
        print(f"Query execution error: {e}")
        if cur:
            cur.close()
        return None

def get_avg_nutrient_for_food(conn, food_name_pattern):
    """
    Fetches the average nutrient amounts for a specified food based on a pattern.
    The food_name_pattern replaces the hardcoded food name in the original query.
    The pattern for LIKE will be %food_name_pattern%.
    """
    sql_query = """
    SELECT
        AVG(CAST(NULLIF(fn.amount, 'null') AS NUMERIC)) AS average_amount,
        nutrient.name AS nutrient_name,
        nutrient.unit_name
    FROM
        foundationfoodinformation.food AS food
    INNER JOIN
        foundationfoodinformation.food_nutrient AS fn ON food.fdc_id = fn.fdc_id
    INNER JOIN
        foundationfoodinformation.nutrient AS nutrient ON fn.nutrient_id = nutrient.id
    WHERE
        LOWER(food.description) LIKE LOWER(%s) -- Parameterized for food_name_pattern
        AND CAST(NULLIF(fn.amount, 'null') AS NUMERIC) > 0
    GROUP BY
        nutrient.name,
        nutrient.unit_name
    ORDER BY
        nutrient.name;
    """
    # Pass the pattern as a tuple for psycopg2 parameter substitution.
    # The '%' wildcard characters are part of the string passed to the parameter.
    rows = execute_query_and_fetch(conn, sql_query, (f'%{food_name_pattern}%',))
    return rows

# The if __name__ == '__main__': block has been removed as requested.
# No code will execute automatically when this file is imported.
